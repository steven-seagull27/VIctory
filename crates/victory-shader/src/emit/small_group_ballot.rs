//! A standalone DXC shader for comparing native Wave32/Wave64 with the
//! guest first-lane test. Reports every adapter and supported width.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, validator};
use crate::native::HwStage;
use std::path::{Path, PathBuf};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
};

fn compile(dxc: &Path, width: u32, threads: u32) -> HostShader {
    let output = std::env::temp_dir().join(format!(
        "victory-ballot-{}-{width}-{threads}.dxil",
        std::process::id()
    ));
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/emit/small_group_ballot.hlsl");
    let result = std::process::Command::new(dxc)
        .args(["-E", "main", "-T", "cs_6_6", "-O3", "-WX", "-D"])
        .arg(format!("WIDTH={width}"))
        .arg("-D")
        .arg(format!("THREADS={threads}"))
        .arg("-Fo")
        .arg(&output)
        .arg(input)
        .output()
        .expect("run the Windows PC dxc.exe");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let dxil = std::fs::read(&output).expect("DXC output");
    std::fs::remove_file(output).unwrap();
    HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: Vec::new(),
        static_samplers: Vec::new(),
        samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: true,
        heap_writes: true,
        view_marks: false,
    }
}

#[test]
#[ignore = "prints a native adapter diagnostic; needs the Windows SDK's PC dxc.exe"]
fn native_ballot_in_a_partial_wave() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let dxc = std::env::var_os("VICTORY_PC_DXC")
        .map(PathBuf::from)
        .or_else(|| validator::Validator::default_path().map(|p| p.with_file_name("dxc.exe")))
        .expect("set VICTORY_PC_DXC to the Windows PC dxc.exe");
    let mut hosts = Host::each_hardware();
    if let Ok(warp) = Host::new(true) {
        hosts.push(("WARP".into(), warp));
    }
    let mut failures = Vec::new();
    let mut runs = 0;
    for (name, host) in hosts {
        let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: the initialized options structure matches the D3D12 feature.
        unsafe {
            host.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    (&raw mut caps).cast(),
                    size_of_val(&caps) as u32,
                )
                .unwrap();
        }
        for width in [32, 64] {
            if !caps.WaveOps.as_bool()
                || !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&width)
            {
                eprintln!(
                    "{name}: skips Wave{width} ({}..{})",
                    caps.WaveLaneCountMin, caps.WaveLaneCountMax
                );
                continue;
            }
            for threads in [32, 64] {
                let shader = compile(&dxc, width, threads);
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .expect("fixed-width pipeline");
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            views: &[View::RawUav(8 * threads)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .expect("dispatch and read back");
                let mut wrong = 0;
                let (rows, rest) = out[0].as_chunks::<8>();
                assert!(rest.is_empty());
                assert_eq!(rows.len(), threads as usize);
                for (tid, &[id, lane, actual, lo, hi, first, got, fixed]) in rows.iter().enumerate()
                {
                    assert_eq!((id, actual), (tid as u32, width));
                    assert!(lane < width);
                    // This lane's own predicate bit must match on *any* lane-to-thread mapping.
                    let word = if lane < 32 { lo } else { hi };
                    let own_bit = (word >> (lane % 32)) & 1;
                    if own_bit != u32::from(tid >= 7) || (lane == 7 && fixed != 100 + id) {
                        wrong += 1;
                    }
                    if tid == 7 {
                        eprintln!(
                            "{name} Wave{width}, group {threads}: tid=7 lane={lane} ballot={lo:08x} {hi:08x} first={first} elected_read={got} fixed_read_7={fixed}"
                        );
                    }
                }
                eprintln!(
                    "{name} Wave{width}, group {threads}: {wrong}/{threads} wrong self checks"
                );
                if wrong != 0 {
                    failures.push((name.clone(), width, threads, wrong));
                }
                runs += 1;
            }
        }
    }
    assert!(runs > 0, "no adapter ran a fixed Wave32 or Wave64 pipeline");
    assert!(failures.is_empty(), "{failures:?}");
}
