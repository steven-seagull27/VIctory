//! Synthetic row permutations on hosts supporting waves of at least 32 lanes.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const THREADS: u32 = 128;
const OLD: u32 = 42;
const BASE: u32 = 0x1234_0000;

fn native(cross: bool, fi: bool, bc: bool, alias: bool, wave64: bool, before_shuffle: &[u32]) -> NativeProgram {
    let mov = |dst: u32, src: u32| 0x7E00_0200 | dst << 17 | src;
    let smov = |dst: u32, src: u32| 0xBE80_0300 | dst << 16 | src;
    let mut words = vec![
        smov(5, 128),
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0 (output byte offset)
        0x4A0A_00FF,
        BASE, // v_add_nc_u32 v5, BASE, v0
        mov(6, if alias { 256 + 5 } else { 128 + OLD }),
    ];
    words.push(if wave64 { 0xBEFE_0402 } else { smov(126, 2) }); // exec = s[2:3] / s2
    words.extend_from_slice(before_shuffle);
    words.extend([
        0xD400_0006
            | (if cross { 0x378 } else { 0x377 }) << 16
            | u32::from(fi) << 11
            | u32::from(bc) << 12,
        (256 + if alias { 6 } else { 5 }) | (1 << 18),
    ]); // v_permlane[x]16_b32 v6, v5/v6, s0, s1
    words.push(if wave64 { 0xBEFE_04C1 } else { smov(126, 193) }); // restore EXEC
    words.extend([
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 6 << 8 | 7, // buffer_store_dword v6, v7, s[8:11], 0 offen
        0xBF81_0000,
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lowered(native: &NativeProgram, software: bool) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(&p, native).expect("io");
    let parts = lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    };
    let result = lower::lower_with(&parts, software)?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: result.dxil,
        user_data: result.user_data,
        samplers: result.samplers,
        static_samplers: Vec::new(),
        root_cbvs: result.root_cbvs,
        root_srvs: result.root_srvs,
        root_uavs: result.root_uavs,
        memory: result.memory,
        memory_writes: result.memory_writes,
        view_addresses: result.view_addresses,
        heap: result.heap,
        heap_writes: result.heap_writes,
        view_marks: false,
    };
    match validator::Validator::load(validator::Validator::default_path().as_deref()) {
        Ok(mut v) => {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation");
            assert_eq!(copy[4..20], shader.dxil[4..20], "container hash");
        }
        Err(e) => eprintln!("DXIL validator unavailable: {e}"),
    }
    Ok(shader)
}

#[test]
fn permlane_fetch_inactive_outside_the_host_branch_is_rejected() {
    // Skip the permutation when EXEC is empty. Structurisation turns this
    // into a host lane branch, so FI cannot read lanes that took the skip.
    let native = native(true, true, false, false, false, &[0xBF88_0002]); // s_cbranch_execz +2 (to the EXEC restore)
    assert!(matches!(
        lowered(&native, false),
        Err(ShaderError::Unsupported {
            what: "a permlane fetch from lanes outside the host branch",
            ..
        })
    ));
}

#[test]
fn permlane_fetch_inactive_after_reconvergence_is_allowed() {
    // The branch only skips a scalar nop; every host lane meets at the shuffle.
    let native = native(true, true, false, false, false, &[0xBF88_0001, 0xBF80_0000]);
    lowered(&native, false).expect("fetch-inactive after the lanes reconverge");
}

#[test]
fn permlane_fetch_inactive_inside_a_uniform_branch_is_allowed() {
    let native = native(true, true, false, false, false, &[0xBF06_8000, 0xBF85_0002]); // scalar compare; skip shuffle on SCC
    lowered(&native, false).expect("all host lanes take the scalar branch together");
}

#[test]
fn permlane_fetch_inactive_under_a_computed_uniform_scalar_is_allowed() {
    let native = native(true, true, false, false, false, &[
        0x800c_8100, // s_add_u32 s12, s0, 1
        0xbf06_800c, // s_cmp_eq_u32 s12, 0
        0xbf85_0002, // skip the FI shuffle on SCC
    ]);
    lowered(&native, false).expect("the scalar argument and literal agree across host lanes");
}

#[test]
fn permlane_fetch_inactive_after_a_divergent_scalar_phi_is_rejected() {
    let native = native(true, true, false, false, false, &[
        0xbe8c_0380, // s_mov_b32 s12, 0
        0xbf88_0001, // s_cbranch_execz skips s12 = 1
        0xbe8c_0381, // s_mov_b32 s12, 1
        0xbefe_03c1, // restore all lanes
        0xbf06_800c, // s_cmp_eq_u32 s12, 0
        0xbf85_0002, // skip the FI shuffle on SCC
    ]);
    assert!(matches!(
        lowered(&native, false),
        Err(ShaderError::Unsupported {
            what: "a permlane fetch from lanes outside the host branch",
            ..
        })
    ));
}

#[test]
fn permlane_fetch_inactive_after_a_lane_abort_is_rejected() {
    // A lane-dependent header test with an unconditional back edge. The
    // emitter's iteration limit returns the looping lanes from the shader;
    // they never meet the lanes which reached the shuffle normally.
    let native = native(
        true, true, false, false, false,
        &[
            0x7D88_0003, // v_cmp_gt_u32 vcc, s3, v0
            0xBF86_0002, // s_cbranch_vccz -> shuffle
            0xBF80_0000, // s_nop 0
            0xBF82_FFFC, // s_branch -> compare
        ],
    );
    assert!(matches!(
        lowered(&native, false),
        Err(ShaderError::Unsupported {
            what: "a permlane fetch from lanes outside the host branch",
            ..
        })
    ));
}

#[test]
fn permlane_rows_and_inactive_sources() {
    check_permlane_rows(&[]);
}

/// FI=0 can execute under a lane-dependent branch: an executing lane
/// whose source took the other branch keeps its destination or zeroes it.
#[test]
fn permlane_fi_zero_under_divergent_exec() {
    // Skip the shuffle when EXEC is empty.
    check_permlane_rows(&[0xBF88_0002]);
}

/// Enter the shuffle on the taken EXEC edge, opposite to the skip above.
#[test]
fn permlane_on_the_taken_exec_edge() {
    // s_cbranch_execnz -> shuffle; s_branch -> restore.
    check_permlane_rows(&[0xBF89_0001, 0xBF82_0002]);
}

#[test]
fn warp_permlane_declines_partial_waves_before_dispatch() {
    for wave64 in [false, true] {
        let wave = if wave64 { 64 } else { 32 };
        for threads in [[19, 1, 1], [1, 3, 11], [65, 1, 1], [4, 4, 2], [4, 4, 4]] {
            let mut program = native(true, false, false, false, wave64, &[]);
            let StageIo::Cs(io) = &mut program.io else {
                unreachable!()
            };
            io.threads = threads;
            let result = lowered(&program, true);
            if threads.into_iter().product::<u32>().is_multiple_of(wave) {
                result.expect("fully populated software waves");
            } else {
                assert!(matches!(
                    result,
                    Err(ShaderError::Unsupported {
                        what: "a WARP permlane with a partially populated wave",
                        ..
                    })
                ));
            }
            lowered(&program, false).expect("physical adapter translation");
        }
    }
}

#[test]
fn warp_permlane_declines_an_uncaptured_nested_edge() {
    // Take an EXEC branch, then a scalar branch, before the shuffle. A
    // ballot in the second branch cannot recover the entry participation.
    let program = native(
        true,
        false,
        false,
        false,
        false,
        &[0xBF88_0004, 0xBF06_8000, 0xBF85_0002],
    );
    assert!(matches!(
        lowered(&program, true),
        Err(ShaderError::Unsupported {
            what: "a WARP permlane without a known incoming lane mask",
            ..
        })
    ));
    lowered(&program, false).expect("direct FI=0 under a nested branch");
}

fn check_permlane_rows(before_shuffle: &[u32]) {
    let divergent = !before_shuffle.is_empty();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    let mut total_wrong = 0;
    for (backend, warp) in [("hardware", false), ("WARP", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {backend}; skipped");
            continue;
        };
        let mut options =
            windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: options is the correctly sized output for this feature query.
        unsafe {
            host.device
                .CheckFeatureSupport(
                    windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS1,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .expect("wave capabilities");
        }
        eprintln!(
            "{backend} waves {}..{}",
            options.WaveLaneCountMin, options.WaveLaneCountMax
        );
        if options.WaveLaneCountMax < 32 {
            eprintln!("{backend} cannot hold two 16-lane rows; GPU checks skipped");
            continue;
        }
        ran += 1;
        eprintln!("permlane: testing row and EXEC cases on {backend}");
        for wave64 in [false, true] {
            let wave = if wave64 { 64 } else { 32 };
            if wave < options.WaveLaneCountMin || wave > options.WaveLaneCountMax {
                eprintln!("{backend} does not support the requested console wave size; skipped");
                continue;
            }
            let mut wrong = 0;
            let mut compared = 0;
            let mut dispatches = 0;
            let fi_cases: &[bool] = if divergent { &[false] } else { &[false, true] };
            for cross in [false, true] {
                for &fi in fi_cases {
                    for bc in [false, true] {
                        for alias in [false, true] {
                            let program = native(cross, fi, bc, alias, wave64, before_shuffle);
                            let shader = lowered(&program, host.is_software()).expect("lower permlane");
                            let (root, pso) =
                                host.compute_pipeline(&shader).expect("permlane pipeline");
                            for selectors in [
                                [0x7654_3210u32, 0xFEDC_BA98], // identity
                                [0x89AB_CDEF, 0x0123_4567],    // reverse
                                [0, 0],                        // broadcast lane 0
                                [u32::MAX, u32::MAX],          // broadcast lane 15
                                [0xF10E_2D3C, 0x4B5A_6978],    // arbitrary; both selector words
                            ] {
                                for mask in [u32::MAX, 0x5555_5555, 0xFFFF_0000, 1, 0] {
                                    let constants = [selectors[0], selectors[1], mask, mask];
                                    let mut ud = vec![0; shader.user_data_dwords() as usize];
                                    for u in &shader.user_data {
                                        ud[u.index as usize] = if u.root == RootType::Constants {
                                            constants[u.dword as usize]
                                        } else {
                                            0
                                        };
                                    }
                                    let out = host
                                        .dispatch(
                                            &root,
                                            &pso,
                                            &Dispatch {
                                                roots: vec![(0, Root::Constants(ud))],
                                                views: &[View::RawUav(THREADS)],
                                                samplers: &[],
                                                groups: [1, 1, 1],
                                            },
                                        )
                                        .expect("dispatch permlane");
                                    dispatches += 1;
                                    for t in 0..THREADS {
                                        let pos = t % 16;
                                        let packed =
                                            u64::from(selectors[0]) | u64::from(selectors[1]) << 32;
                                        let pick = (packed >> (4 * pos)) as u32 & 15;
                                        let src = ((t & !15) ^ if cross { 16 } else { 0 }) | pick;
                                        let old = if alias { BASE + t } else { OLD };
                                        let want = if mask >> (t % 32) & 1 == 0 {
                                            old
                                        } else if fi || mask >> (src % 32) & 1 != 0 {
                                            BASE + src
                                        } else if bc {
                                            0
                                        } else {
                                            old
                                        };
                                        let got = out[0][t as usize];
                                        if got != want {
                                            if wrong == 0 {
                                                eprintln!(
                                                    "{backend} wave{wave} cross={cross} fi={fi} bc={bc} alias={alias} selectors={selectors:x?} mask={mask:x} lane={t} source={src} got={got:#x} expected={want:#x}"
                                                );
                                            }
                                            wrong += 1;
                                        }
                                        compared += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            eprintln!("{backend} wave{wave} divergent={divergent}: {wrong}/{compared} wrong across {dispatches} dispatches");
            total_wrong += wrong;
        }
    }
    if ran == 0 {
        eprintln!("no SM 6.6 device with 32-lane rows; GPU checks skipped");
    }
    assert_eq!(total_wrong, 0, "permlane lane readbacks");
}
