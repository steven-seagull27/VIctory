use super::*;
use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::core::s;

/// Compile or load a real compute PSO, then dispatch it and check all 32 words.
fn dispatch(key: &str, tag: u32, cached: bool) {
    let host = crate::host::host().unwrap();
    let hlsl = format!(
        r#"RWByteAddressBuffer dst : register(u0);
           [RootSignature("UAV(u0)")] [numthreads(32,1,1)]
           void main(uint lane : SV_GroupIndex) {{ dst.Store(4*lane, {tag}u+lane); }}"#
    );
    let mut blob = None;
    // SAFETY: the source, root signature and shader blob outlive the calls using them.
    let (root, pso) = unsafe {
        D3DCompile(
            hlsl.as_ptr().cast(),
            hlsl.len(),
            None,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut blob,
            None,
        )
        .unwrap();
        let blob = blob.unwrap();
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let loaded = LOADED.load(Ordering::Relaxed);
        let pso = pipeline(
            key,
            |lib, name| lib.LoadComputePipeline(name, &desc),
            || host.device.CreateComputePipelineState(&desc),
        )
        .unwrap();
        assert_eq!(
            LOADED.load(Ordering::Relaxed) - loaded,
            u64::from(cached),
            "{key} load"
        );
        (root, pso)
    };
    let (output, mapped) = host.shared_buffer(32 * 4).unwrap();
    let list = host.new_list().unwrap();
    // SAFETY: an open list, matching root signature and live mapped UAV.
    unsafe {
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootUnorderedAccessView(0, output.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    host.execute(&[&list]);
    // SAFETY: execute has waited for all 32 words of the mapped UAV.
    let words = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), 32) };
    assert!(words.iter().copied().eq((tag..).take(32)), "{key} readback");
}

fn phase(dir: &Path, step: &str, busy: bool) {
    use_pipeline_cache(dir.to_owned());
    match step {
        "seed" => {
            dispatch("alpha", 0x100, false);
            dispatch("gamma", 0x300, false);
            for key in ["alpha", "gamma"] {
                note_prebuilt(key.to_owned());
            }
            note_used("alpha");
            save(true);
            let binary = dir.join(LIBRARY_FILE);
            let before = std::fs::read(&binary).unwrap();
            let stamp = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
            File::options()
                .write(true)
                .open(&binary)
                .unwrap()
                .set_modified(stamp)
                .unwrap();
            dispatch("gamma", 0x300, true);
            note_used("gamma");
            save(true);
            let untouched = std::fs::metadata(&binary).unwrap().modified().unwrap() == stamp;
            assert_eq!(std::fs::read(&binary).unwrap(), before);
            assert!(
                std::fs::read_to_string(dir.join(EARLY_FILE))
                    .unwrap()
                    .lines()
                    .any(|k| k == "gamma")
            );
            eprintln!(
                "early-only save: binary untouched={untouched}, {} bytes",
                before.len()
            );
        }
        "add" => {
            assert!(prebuilt("alpha") && prebuilt("gamma"));
            assert!(wanted_early("alpha") && wanted_early("gamma"));
            dispatch("beta", 0x200, false);
            note_prebuilt("beta".to_owned());
            note_used("beta");
            // Hold the existing early list open without FILE_SHARE_DELETE so the
            // atomic replacement fails once. The binary and manifest stay writable.
            let reader = busy.then(|| {
                OpenOptions::new()
                    .read(true)
                    .share_mode(1 | 2)
                    .open(dir.join(EARLY_FILE))
                    .unwrap()
            });
            save(true);
            let recorded = std::fs::read_to_string(dir.join(EARLY_FILE))
                .unwrap()
                .lines()
                .any(|k| k == "beta");
            assert_eq!(recorded, !busy);
            drop(reader);
            save(true);
        }
        "reopen" => {
            for (key, tag) in [("alpha", 0x100), ("beta", 0x200), ("gamma", 0x300)] {
                dispatch(key, tag, true);
            }
            let valid = ["alpha", "beta", "gamma"].map(prebuilt);
            let early = ["alpha", "beta", "gamma"].map(wanted_early);
            eprintln!(
                "reopen: busy={busy} cached dispatch=96/96 early={early:?} prebuilt={valid:?}"
            );
            assert_eq!(early, [true; 3]);
            assert_eq!(valid, [true; 3]);
        }
        _ => unreachable!(),
    }
}

#[test]
#[ignore = "opens a real D3D12 pipeline library in six fresh test processes"]
fn cache_manifest_survives_early_list_retry() {
    const STEP: &str = "VICTORY_TEST_LIBRARY_SAVE_STEP";
    const DIR_ENV: &str = "VICTORY_TEST_LIBRARY_SAVE_DIRECTORY";
    const BUSY: &str = "VICTORY_TEST_LIBRARY_SAVE_BUSY";
    if let Some(step) = std::env::var_os(STEP) {
        let dir = PathBuf::from(std::env::var_os(DIR_ENV).unwrap());
        return phase(
            &dir,
            step.to_str().unwrap(),
            std::env::var_os(BUSY).is_some(),
        );
    }
    // Cache state is held in process-wide OnceLocks. Each child gets a real restart.
    let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
    else {
        eprintln!("no D3D12 device; skipped");
        return;
    };
    if open(&device, Vec::new()).is_none() {
        eprintln!("no D3D12 pipeline library; skipped");
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let thread = std::thread::current();
    let test = thread.name().expect("libtest names the test thread");
    for busy in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        for step in ["seed", "add", "reopen"] {
            let mut child = std::process::Command::new(&exe);
            child
                .args([
                    "--exact",
                    test,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(STEP, step)
                .env(DIR_ENV, dir.path())
                .env_remove(BUSY);
            if busy {
                child.env(BUSY, "1");
            }
            assert!(
                child.status().unwrap().success(),
                "{step}, early list busy={busy}"
            );
        }
    }
}
