//! Synthetic GFX10 programs of what the 2020 compiler does not produce,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host GPU and WARP. Gathers at a level (`image_gather4_l`, `_l_o`), which
//! DXIL lacks: at level 0 the host's own gather (`image_gather4_lz`) of the
//! same coordinates is the reference on the hardware adapter (RDNA2, as the
//! console): it rounds the exact product of coordinate and size to 1/256 of
//! a texel, half up, which WARP does not. Lane permutes (`ds_bpermute_b32`).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Per lane (32, wave32): the record [offset word, u, v, level] at entry 1
/// of the table at s0 (a 2D texture at entry 0, a raw UAV at entry 2); the
/// level converted from an integer (so it is whole). Gathers red at the
/// level, with the offset, and at level 0 (`lz`) into 12 dwords at lane × 48.
fn program(whole: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0502,
        0xFA00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        // The culling shaders' sampler: border, transparent black, levels
        // 0-12, linear filters.
        0xB008_01B6, // s_movk_i32 s8, 0x1b6
        0xBE89_03FF,
        0x00C0_0000, // s_mov_b32 s9, 0xc00000
        0xBE8A_03FF,
        0x1850_0000, // s_mov_b32 s10, 0x18500000
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x3420_0084, // v_lshlrev_b32 v16, 4, v0
        0x1628_00B0, // v_mul_u32_u24 v20, 48, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE038_1000,
        0x8005_0C10, // buffer_load_dwordx4 v[12:15], v16, s[20:23], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
    ];
    if whole {
        w.push(0x7E1E_0D0F); // v_cvt_f32_u32 v15, v15
    }
    w.extend([
        0xF110_0108,
        0x0043_040D, // image_gather4_l v[4:7], v[13:15], s[12:19], s[8:11]
        0xF150_0108,
        0x0043_080C, // image_gather4_l_o v[8:11], v[12:15], s[12:19], s[8:11]
        0xF11C_0108,
        0x0043_100D, // image_gather4_lz v[16:19], v[13:14], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0414, // buffer_store_dwordx4 v[4:7], v20, s[40:43], 0 offen
        0xE078_1010,
        0x800A_0814, // buffer_store_dwordx4 v[8:11], v20, s[40:43], 0 offen offset:16
        0xE078_1020,
        0x800A_1014, // buffer_store_dwordx4 v[16:19], v20, s[40:43], 0 offen offset:32
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    build_shader(&compute_program(words))
}

fn compute_program(words: &[u32]) -> NativeProgram {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Srv, 8, 4, 16),
            range(ResourceType::Uav, 16, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn build_shader(native: &NativeProgram) -> Result<HostShader, crate::ShaderError> {
    let lowered = lowered(native)?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| super::StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    })
}

fn lowered(native: &NativeProgram) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
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
    })
}

/// The first texel of a footprint (see `Lower::gather_first`).
fn first(u: f32, size: u32) -> i64 {
    let u = if u.is_nan() { 0.0 } else { u.clamp(-2.0, 2.0) };
    ((f64::from(u) * f64::from(size) * 512.0).floor() as i64 - 255) >> 9
}

#[test]
fn gathers_at_a_level_match_the_hardware() {
    // 24×16 then 12×8, 6×4, 3×2; red = level × 1000 + y × 32 + x + 1.
    let mut texels = Vec::new();
    for l in 0..4 {
        let (w, h) = (24 >> l, 16 >> l);
        for y in 0..h {
            for x in 0..w {
                texels.push([(l * 1000 + y * 32 + x + 1) as f32, 0.0, 0.0, 1.0]);
            }
        }
    }
    let texel = |l: u32, x: i64, y: i64| -> f32 {
        let (w, h) = (24i64 >> l, 16i64 >> l);
        let skip: i64 = (0..l).map(|k| (24i64 >> k) * (16 >> k)).sum();
        if (0..w).contains(&x) && (0..h).contains(&y) {
            texels[(skip + y * w + x) as usize][0]
        } else {
            0.0
        }
    };
    // Records: lanes 0-15 at level 0 where rounding decides (coordinates
    // whose f32 product with 24 lands on the other side of a texel's
    // threshold than the exact one, ties of 1/512 on the 16 texels of y,
    // NaN, infinities, zeros); lanes 16-31 at other levels and offsets.
    let mut records: Vec<[u32; 4]> = Vec::new();
    let rounded = |u: f32| ((f64::from(u * 24.0) * 512.0).floor() as i64 - 255) >> 9;
    let mut straddling = (1..24).filter_map(|i| {
        let u0 = ((f64::from(i) + 127.5 / 256.0) / 24.0) as f32;
        (-1024i32..1024)
            .map(|d| f32::from_bits(u0.to_bits().wrapping_add_signed(d)))
            .find(|&u| rounded(u) != first(u, 24))
    });
    for k in 0..16 {
        let tie = (f64::from(k - 3) + 0.5 - 1.0 / 512.0) / 16.0;
        let u = match k {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => -0.0,
            _ => straddling.next().unwrap(),
        };
        records.push([0, u.to_bits(), (tie as f32).to_bits(), 0]);
    }
    for k in 0..16u32 {
        let level = [1, 2, 3, 7][k as usize % 4];
        let offset = [0x0000, 0x3F01, 0x0302, 0x3E3D][k as usize / 4];
        let (u, v) = (
            (k as f32 * 0.77 + 0.3) / 13.0 - 0.1,
            (k as f32 * 0.41 + 0.2) / 9.0,
        );
        records.push([offset, u.to_bits(), v.to_bits(), level]);
    }
    let input: Vec<u32> = records.iter().flatten().copied().collect();
    let shader = build(&program(true)).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let views = [
        View::Mipped(24, 16, 4, &texels),
        View::Buffer {
            data: &input,
            stride: Stride::Structured(16),
        },
        View::RawUav(32 * 12),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        ran += 1;
        for (lane, r) in records.iter().enumerate() {
            let got = |k: usize| -> Vec<f32> {
                out[0][lane * 12 + 4 * k..][..4]
                    .iter()
                    .map(|&b| f32::from_bits(b))
                    .collect()
            };
            let [offset, u, v, level] = *r;
            let level = level.min(3);
            let (w, h) = (24 >> level, 16 >> level);
            let at = |ox: i64, oy: i64| {
                let (x, y) = (
                    first(f32::from_bits(u), w) + ox,
                    first(f32::from_bits(v), h) + oy,
                );
                vec![
                    texel(level, x, y + 1),
                    texel(level, x + 1, y + 1),
                    texel(level, x + 1, y),
                    texel(level, x, y),
                ]
            };
            let sext = |b: u32| i64::from((b as i32) << 26 >> 26);
            assert_eq!(got(0), at(0, 0), "{label}: lane {lane}");
            assert_eq!(
                got(1),
                at(sext(offset), sext(offset >> 8)),
                "{label}: lane {lane}, offset"
            );
            // WARP's own gather truncates where the hardware rounds.
            if lane < 16 && !warp {
                assert_eq!(got(2), got(0), "{label}: lane {lane} against its gather");
            }
        }
    }
    assert!(ran > 0, "no device");
}

#[test]
fn gathers_at_a_level_that_may_be_a_fraction_stop() {
    let Err(crate::ShaderError::Unsupported { what, .. }) = build(&program(false)) else {
        panic!("translated a gather at a level read from memory");
    };
    assert_eq!(what, "a gather at a level that may not be whole");
}

#[test]
fn permutes_read_the_lane_named_and_zero_from_lanes_left_out() {
    // Lanes below 24 run the permute; each reads lane (13 × lane + 5) mod 32
    // (its address in dwords, offset 20 bytes) of 1000 + lane.
    let shader = build(&[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x1602_00B4, // v_mul_u32_u24 v1, 52, v0
        0x4A04_00FF,
        0x0000_03E8, // v_add_nc_u32 v2, 1000, v0
        0x7E06_02FF,
        0x0000_DEAD, // v_mov_b32 v3, 0xdead
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xDACC_0014,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ])
    .expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..32)
        .map(|l| match (13 * l + 5) % 32 {
            _ if l >= 24 => 0xDEAD,
            src if src < 24 => 1000 + src,
            _ => 0,
        })
        .collect();
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // The program declares the console's wave of 32, which WARP refuses.
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            assert!(warp, "{label}: no pipeline for a wave of 32");
            eprintln!("{label}: no pipeline for a wave of 32; skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
    }
}

#[test]
fn ordered_pixel_sections_use_rasterizer_ordered_views() {
    // A pixel program's ordered section (POPS) as the title's compiler
    // writes it: the collision id after the primitive mask (s10, user data
    // ending at s9), a wait loop on the exiting wave id, a read-modify-write
    // of a UAV, the end of the section.
    let code: Vec<u32> = vec![
        0xBE84_0308, // s_mov_b32 s4, s8
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x9000_9F0A, // s_lshr_b32 s0, s10, 31
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        0x8102_81EF, // s_add_i32 s2, src_pops_exiting_wave_id, 1
        0xBF08_8502, // s_cmp_gt_u32 s2, 5
        0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
        0x8880_0200, // s_or_b64 s[0:1], s[0:1], s[2:3]
        0x87EA_007E, // s_and_b64 vcc, exec, s[0:1]
        0xBF87_0004, // s_cbranch_vccnz +4
        0xBF8E_0001, // s_sleep 1
        0x8100_81EF, // s_add_i32 s0, src_pops_exiting_wave_id, 1
        0xBF0A_8500, // s_cmp_lt_u32 s0, 5
        0xBF84_FFFC, // s_cbranch_scc0 -4
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_3F88,
        0x0003_0200, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm dlc glc
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2
        0xF020_1F08,
        0x0003_0200, // image_store v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        0xF800_1890,
        0x0000_0000, // exp null off, off, off, off done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = |db_shader_control| NativeProgram {
        stage: HwStage::Ps,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 8,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 8,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let l = lowered(&native(1 << 16)).expect("lower");
    assert!(l.rovs);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Without the ordered mode the collision id is no value the host has.
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowered(&native(0)) else {
        panic!("translated a collision id outside an ordered section");
    };
    assert_eq!(what, "a hardware SGPR input");
}

/// Every lane observes the same scalar branch. Lane 0 sets the only VCC bit;
/// a subsequent CMPX enables lane 1, leaving the VCC bit in an inactive lane.
#[test]
fn vcc_branches_take_one_scalar_path_for_the_wave() {
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut mistakes = 0;
    let mut ran = 0;
    let mut unsupported = 0;
    for (name, branch, nonzero) in [("vccz", 0xbf86_0001, false), ("vccnz", 0xbf87_0001, true)] {
        for (comparison, any) in [(0x7d84_0080, true), (0x7d84_00ad, false)] {
            let shader = build(&[
                0xbe84_0300,
                0xbe85_0380, // s[4:5] = descriptor table
                0xf408_0a02,
                0xfa00_0040, // output UAV at byte 64
                0x3420_0082, // v16 = 4 * lane
                0xbe90_038b, // s16 = 11 (taken path)
                comparison,  // vcc = lane == 0 (or lane == 45)
                0x7da4_0081, // exec = lane == 1
                branch,
                0xbe90_0397, // s16 = 23 (fallthrough path)
                0xbefe_03c1, // restore all lanes
                0x7e06_0210, // v3 = s16
                0xbf8c_c07f, // wait for the output descriptor
                0xe070_1000,
                0x800a_0310, // store v3 at lane * 4
                0xbf81_0000,
            ])
            .expect("lower");
            let mut dxil = shader.dxil.clone();
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut dxil).expect("DXIL validation failed");
            }
            let expected = if any == nonzero { 11 } else { 23 };
            for (adapter, warp) in [("hardware", false), ("warp", true)] {
                let host = match Host::new(warp) {
                    Ok(host) => host,
                    Err(error) => {
                        eprintln!("{adapter}: no SM 6.6 device ({error})");
                        continue;
                    }
                };
                if !supports_guest_wave(&host, adapter, 32) {
                    unsupported += 1;
                    continue;
                }
                let (root, pso) = host.compute_pipeline(&shader).expect("compute pipeline");
                let ud = vec![0; shader.user_data_dwords().max(1) as usize];
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .expect("dispatch");
                ran += 1;
                assert_eq!(out[0].len(), 32);
                let mismatches = out[0].iter().filter(|&&got| got != expected).count();
                eprintln!(
                    "MIXED {adapter} {name} any={any} want={expected} mismatches={mismatches}/32 values={:?}",
                    out[0]
                );
                mistakes += mismatches;
            }
        }
    }
    assert!(ran > 0 || unsupported > 0, "no SM 6.6 hardware or WARP device available");
    assert_eq!(mistakes, 0);
}

/// Two full guest waves in one group independently choose their scalar path.
#[test]
fn vcc_branches_isolate_two_full_guest_waves() {
    check_group_vcc_branches(&[(32, 64), (64, 128)]);
}

/// The last real invocation can set VCC in a partially occupied second wave.
#[test]
fn vcc_branches_isolate_a_partial_second_guest_wave() {
    check_group_vcc_branches(&[(32, 45), (64, 85)]);
}

/// A runtime root constant selects the only invocation that sets VCC. Narrow
/// EXEC to lane 1 of each wave, then restore it to observe every lane's result.
fn check_group_vcc_branches(groups: &[(u32, u32)]) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut available = false;
    for (adapter, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {adapter} device; skipped");
            continue;
        };
        available = true;
        for &(wave, threads) in groups {
            if !supports_guest_wave(&host, adapter, wave) {
                continue;
            }
            for (name, branch, nonzero) in
                [("vccz", 0xbf86_0001, false), ("vccnz", 0xbf87_0001, true)]
            {
                let mut native = compute_program(&[
                    0xbe84_0300,
                    0xbe85_0380, // s[4:5] = descriptor table
                    0xf408_0a02,
                    0xfa00_0040,                    // output UAV at byte 64
                    0x3420_0082,                    // v16 = 4 * group thread id
                    0x3622_0000 | (128 + wave - 1), // v17 = group thread id & (wave - 1)
                    0xbe90_038b,                    // s16 = 11 (taken)
                    0x7d84_0002,                    // v_cmp_eq_u32 vcc, s2, v0
                    0x7da4_2281,                    // v_cmpx_eq_u32 exec, 1, v17
                    branch,
                    0xbe90_0397, // s16 = 23 (fallthrough)
                    if wave == 64 { 0xbefe_04c1 } else { 0xbefe_03c1 }, // restore EXEC
                    0x7e06_0210,
                    0xbf8c_c07f, // v3 = s16, wait for the output descriptor
                    0xe070_1000,
                    0x800a_0310, // store v3 at 4 * group thread id
                    0xbf81_0000,
                ]);
                native.wave64 = wave == 64;
                native.io = StageIo::Cs(ComputeIo {
                    rsrc2: 3 << 1,
                    threads: [threads, 1, 1],
                });
                native.layout.push(RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 2,
                    slot: 1,
                });
                let shader = build_shader(&native).expect("lower");
                if let Ok(mut validator) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    validator
                        .validate(&mut shader.dxil.clone())
                        .expect("DXIL validation");
                }
                let (root, pso) = host.compute_pipeline(&shader).expect("compute pipeline");
                let views = [
                    View::Texture(1, 1, &[[0.0; 4]]),
                    View::Buffer {
                        data: &[0; 4],
                        stride: Stride::Structured(16),
                    },
                    View::RawUav(threads + 4),
                ];
                // First wave (including its highest lane), last real invocation
                // of wave two, and a missing invocation just outside the group.
                for target in [0, wave - 1, threads - 1, threads] {
                    let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
                    for word in &shader.user_data {
                        if word.root == RootType::Constants {
                            ud[word.index as usize] = target;
                        }
                    }
                    let out = host
                        .dispatch(
                            &root,
                            &pso,
                            &Dispatch {
                                roots: vec![(0, Root::Constants(ud))],
                                views: &views,
                                samplers: &[],
                                groups: [1, 1, 1],
                            },
                        )
                        .expect("dispatch");
                    let expected: Vec<u32> = (0..threads)
                        .map(|id| {
                            let any = target < threads && id / wave == target / wave;
                            if any == nonzero { 11 } else { 23 }
                        })
                        .collect();
                    let (actual, tail) = out[0].split_at(threads as usize);
                    assert_eq!(tail, &[0; 4], "output beyond group");
                    assert_eq!(
                        actual, expected,
                        "{adapter} wave{wave} group={threads} {name} target={target}"
                    );
                }
                eprintln!("{adapter} Wave{wave} group={threads} {name}: all four readbacks matched");
            }
        }
    }
    assert!(available, "no SM 6.6 hardware or WARP device available");
}

fn supports_guest_wave(host: &Host, adapter: &str, wave: u32) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: options has the size and layout requested by the feature.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut options).cast(),
                std::mem::size_of_val(&options) as u32,
            )
            .expect("D3D12 wave options");
    }
    let supported = options.WaveOps.as_bool()
        && (options.WaveLaneCountMin..=options.WaveLaneCountMax).contains(&wave);
    if !supported {
        eprintln!(
            "{adapter}: Wave{wave} unavailable (wave ops {}, lane range {}..={}); no dispatch",
            options.WaveOps.as_bool(), options.WaveLaneCountMin, options.WaveLaneCountMax
        );
    }
    supported
}

/// A scalar write inside an EXEC-shaped branch is visible to the whole
/// guest wave. A normal DXIL phi would retain it only in the writing lanes.
#[test]
fn vcc_from_a_divergent_scalar_join_is_refused() {
    let result = build(&[
        0xbe84_0300, 0xbe85_0380, // s[4:5] = descriptor table
        0xf408_0a02, 0xfa00_0040, // output UAV
        0x3420_0082, // v16 = 4 * lane
        0xbeea_0380, // s_mov_b32 vcc_lo, 0
        0x7da4_0080, // exec = lane == 0
        0xbf88_0001, // s_cbranch_execz skips the scalar write
        0xbeea_0381, // s_mov_b32 vcc_lo, 1
        0xbefe_03c1, // restore EXEC
        0xbe90_038b, // s16 = 11
        0xbf87_0001, // s_cbranch_vccnz skips the following move
        0xbe90_0397, // s16 = 23
        0x7e06_0210, // v3 = s16
        0xbf8c_c07f, 0xe070_1000, 0x800a_0310, // store v3 at lane * 4
        0xbf81_0000,
    ]);
    assert!(matches!(result, Err(crate::ShaderError::Unsupported {
        what: "a scalar VCC branch depending on a divergent scalar join", ..
    })));
}

/// Non-compute stages where the guest wave width cannot be established keep
/// the per-lane decision for `v_cmp; s_and vcc, exec, vcc; s_cbranch_vcc*`
/// rather than refusing to translate.
#[test]
fn pixel_shader_vcc_branch_uses_per_lane_decision() {
    for branch in [
        0xBF86_0001u32, // s_cbranch_vccz +1
        0xBF87_0001u32, // s_cbranch_vccnz +1
    ] {
        let code = [
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0x87EA_6A7E, // s_and_b64 vcc, exec, vcc
            branch,
            0x7E04_02F2, // v_mov_b32 v2, 1.0
            0xF800_1890,
            0x0000_0000, // exp null off, off, off, off done vm
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: true,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: Vec::new(),
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let l = lowered(&native).expect("pixel shader with VCC branch should translate");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}
