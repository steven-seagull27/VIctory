//! Synthetic scalar mask branch with a single selected lane in a partially active console wave.
use std::path::PathBuf;
use std::process::Command;

use super::host::{Dispatch, Host, Root, View};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

#[derive(Clone, Copy, Debug)]
enum Program {
    DxcReference,
    Branch,
    StoreOnScc0,
    StoreOnScc1,
}

fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(0xbf81_0000);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn program(kind: Program, lanes: u32) -> NativeProgram {
    let wave64 = lanes == 64;
    let save = if wave64 { 0xbe8c_246a } else { 0xbe8c_3c6a }; // s_and_saveexec_b64/b32 s12, vcc
    let mask = if wave64 { 0x8792_7e6a } else { 0x8712_7e6a }; // s_and_b64/b32 s18, vcc, exec
    let restore = if wave64 { 0xbefe_040c } else { 0xbefe_030c }; // s_mov_b64/b32 exec, s12
    let body = match kind {
        Program::Branch => vec![
            0xbf8c_0000, // s_waitcnt 0
            0x7d88_0000, // v_cmp_gt_u32 vcc, s0, v0  (lane < active count)
            save,
            0xbf88_0009, // s_cbranch_execz to the restore
            0x7d84_0001, // v_cmp_eq_u32 vcc, s1, v0  (lane == selected)
            mask,
            0xbf84_0002, // s_cbranch_scc0 to the value 1
            0x7e04_0282, // v_mov_b32 v2, 2
            0xbf82_0001, // s_branch to the common store
            0x7e04_0281, // v_mov_b32 v2, 1
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xe070_1000,
            0x8002_0201, // buffer_store_dword v2 at v1
            restore,
        ],
        Program::StoreOnScc0 | Program::StoreOnScc1 => {
            let (branch, fallthrough, taken) = match kind {
                Program::StoreOnScc0 => (0xbf84_0004, 2, 1),
                _ => (0xbf85_0004, 1, 2),
            };
            vec![
                0xbf8c_0000, // s_waitcnt 0
                0x3402_0082, // v_lshlrev_b32 v1, 2, v0
                0x7d88_0000, // v_cmp_gt_u32 vcc, s0, v0
                save,
                0xbf88_000a, // s_cbranch_execz to the restore
                0x7d84_0001, // v_cmp_eq_u32 vcc, s1, v0
                mask,
                branch,
                0x7e04_0280 | fallthrough, // v_mov_b32 v2, fallthrough value
                0xe070_1000,
                0x8002_0201,         // buffer_store_dword v2 at v1
                0xbf82_0003,         // s_branch to the restore
                0x7e04_0280 | taken, // v_mov_b32 v2, taken value
                0xe070_1000,
                0x8002_0201, // buffer_store_dword v2 at v1
                restore,
            ]
        }
        Program::DxcReference => unreachable!(),
    };
    native_of(&body, wave64, lanes)
}

fn reference(lanes: u32) -> Vec<u8> {
    let dxc = std::env::var_os("VICTORY_PC_DXC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            super::validator::Validator::default_path()
                .expect("Windows SDK or VICTORY_PC_DXC")
                .with_file_name("dxc.exe")
        });
    let hlsl =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/emit/testdata/branch-mask-scc.hlsl");
    let output = std::env::temp_dir().join(format!(
        "victory-mask-scc-{}-{lanes}.dxil",
        std::process::id()
    ));
    let status = Command::new(dxc)
        .args(["-T", "cs_6_6", "-E", "main", "-D"])
        .arg(format!("LANES={lanes}"))
        .arg("-Fo")
        .arg(&output)
        .arg(hlsl)
        .status()
        .expect("run PC DXC");
    assert!(status.success(), "DXC failed");
    let bytes = std::fs::read(&output).expect("reference DXIL");
    std::fs::remove_file(output).unwrap();
    bytes
}

#[test]
#[ignore = "requires a D3D12 device and PC DXC"]
fn every_active_lane_observes_the_scc_of_a_mask_operation() {
    let host = Host::new(false).expect("D3D12 adapter");
    let mut total_wrong = 0;
    for lanes in [32, 64] {
        let cases = if lanes == 32 {
            [
                (0, 0),
                (1, 0),
                (16, 15),
                (16, 16),
                (24, 7),
                (24, 21),
                (24, 25),
                (32, 31),
                (32, 32),
            ]
        } else {
            [
                (0, 0),
                (1, 0),
                (32, 31),
                (32, 32),
                (48, 7),
                (48, 37),
                (48, 49),
                (64, 63),
                (64, 64),
            ]
        };
        for kind in [
            Program::DxcReference,
            Program::Branch,
            Program::StoreOnScc0,
            Program::StoreOnScc1,
        ] {
            let (root, pso, shader) = match kind {
                Program::DxcReference => {
                    let bytes = reference(lanes);
                    let root = host.root_signature_bytes(&bytes).unwrap();
                    let pso = host.compute_pipeline_with(&root, &bytes).unwrap();
                    (root, pso, None)
                }
                _ => {
                    let n = program(kind, lanes);
                    let (p, st) =
                        crate::range::build(&n.patched_code(), n.wave64, n.vgprs).unwrap();
                    let st = st.unwrap();
                    let res = crate::resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
                    let io = crate::io::raise(&p, &n).unwrap();
                    let bytes = crate::dxbc::build(&[]);
                    let c = crate::Container::parse(&bytes).unwrap();
                    let shader = super::emit(&c, &n, &p, &st, &io, &res).unwrap();
                    let (root, pso) = host.compute_pipeline(&shader).unwrap();
                    (root, pso, Some(shader))
                }
            };
            let mut wrong = 0;
            let mut inactive_wrong = 0;
            for (below, selected) in cases {
                let roots = match &shader {
                    Some(shader) => {
                        let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
                        for u in &shader.user_data {
                            if u.root == RootType::Constants && u.slot == 0 && u.dword < 2 {
                                data[u.index as usize] = [below, selected][u.dword as usize];
                            }
                        }
                        vec![(0, Root::Constants(data))]
                    }
                    None => vec![
                        (0, Root::Constants(vec![below, selected])),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 0,
                            },
                        ),
                    ],
                };
                let outputs = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots,
                            views: &[View::RawUav(lanes)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap();
                assert_eq!(outputs[0].len(), lanes as usize);
                for (lane, &word) in outputs[0].iter().enumerate() {
                    let active = (lane as u32) < below;
                    let expected = if active {
                        1 + u32::from(selected < below)
                    } else {
                        0
                    };
                    if word != expected {
                        if wrong == 0 {
                            eprintln!(
                                "  {kind:?}: active={below} selected={selected} lane={lane} read={word} expected={expected}"
                            );
                        }
                        wrong += 1;
                        if !active {
                            inactive_wrong += 1;
                        }
                    }
                }
            }
            eprintln!(
                "Wave{lanes} {kind:?}: {wrong}/{} incorrect ({inactive_wrong} inactive)",
                lanes * cases.len() as u32
            );
            if matches!(kind, Program::DxcReference) {
                assert_eq!(wrong, 0, "DXC reference");
            }
            total_wrong += wrong;
        }
    }
    assert_eq!(total_wrong, 0, "translated branch-only mask SCC");
}
