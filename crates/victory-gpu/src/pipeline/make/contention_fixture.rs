//! Builds a small graphics pipeline through the guest API and checks first use
//! with 32 simultaneous callers. Set VICTORY_PROBE_DXC to the SDK compiler
//! if dxc.exe is outside PATH. PROBE_CORES enables timing: comma-separated
//! decimal affinity masks, one logical processor on each of eight physical
//! cores. Each worker owns a guest command list. The timing unit is wall
//! nanoseconds divided by all successful lookups across those workers.

use super::super::*;
use super::{Ordering, PROBE_BUILDS};
use crate::object::{IUNKNOWN, S_OK, slot};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::Instant;
use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};

// This synthetic VS/PS pair creates a valid host pipeline. The lookup loop
// does not record GPU draws, so shader work does not enter the timings.
fn shader_codes() -> &'static (Vec<u8>, Vec<u8>) {
    static CODES: std::sync::OnceLock<(Vec<u8>, Vec<u8>)> = std::sync::OnceLock::new();
    CODES.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("lookup.hlsl");
        std::fs::write(
            &source,
            r#"
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.25, 1);
}
float4 ps() : SV_Target { return float4(0.5, 0.25, 0.75, 1); }
"#,
        )
        .unwrap();
        let compile = |stage: &str| {
            let path = dir.path().join(format!("{stage}.dxil"));
            let dxc = std::env::var_os("VICTORY_PROBE_DXC").unwrap_or_else(|| "dxc.exe".into());
            let built = std::process::Command::new(dxc)
                .arg("-T")
                .arg(format!("{stage}_6_6"))
                .args(["-E", stage, "-Fo"])
                .arg(&path)
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                built.status.success(),
                "{}",
                String::from_utf8_lossy(&built.stderr)
            );
            std::fs::read(path).unwrap()
        };
        (compile("vs"), compile("ps"))
    })
}

fn prime_translation(shader: &Arc<Shader>) {
    shader.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: match shader.stage {
                    "vertex" => victory_shader::HwStage::Vs,
                    "pixel" => victory_shader::HwStage::Ps,
                    other => panic!("stage {other}"),
                },
                dxil: shader.container.clone(),
                user_data: vec![],
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
                root_uavs: vec![],
                memory: false,
                memory_writes: false,
                view_addresses: false,
                heap: false,
                heap_writes: false,
                view_marks: false,
            },
            used: vec![],
        })
    });
}

fn graphics_stream() -> Vec<u8> {
    let (vs, ps) = shader_codes();
    let stage = |kind: u64, code: &[u8]| [kind, code.as_ptr() as u64, code.len() as u64];
    let mut bytes: Vec<u8> = [stage(1, vs), stage(2, ps)]
        .into_iter()
        .flatten()
        .flat_map(u64::to_le_bytes)
        .collect();
    // Triangle topology, one RGBA8_UNORM target and D32_FLOAT depth.
    let state = [14u32, 3, 15, 28, 0, 0, 0, 0, 0, 0, 0, 1, 16, 40];
    bytes.extend(state.into_iter().flat_map(u32::to_le_bytes));
    bytes
}

fn guest() -> usize {
    let bytes = graphics_stream();
    let mut p = std::ptr::null();
    let desc = StreamDesc {
        size: bytes.len(),
        stream: bytes.as_ptr(),
    };
    assert_eq!(
        create_stream(crate::device::device(), &desc, &PIPELINE_IIDS[0], &mut p),
        S_OK
    );
    p as usize
}

fn command_list() -> (usize, usize) {
    let mut list = std::ptr::null();
    let mut allocator = std::ptr::null();
    let device = crate::device::device();
    assert_eq!(
        crate::list::create_allocator(device, 0, &IUNKNOWN, &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            device,
            0,
            0,
            allocator,
            std::ptr::null(),
            &IUNKNOWN,
            &mut list
        ),
        S_OK
    );
    assert!(crate::list::host_list(list).is_some());
    assert!(!crate::list::has_newer_table(list));
    (list as usize, allocator as usize)
}

fn release(p: usize) {
    let p = p as This;
    assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
}

fn raced() {
    for completed in [false, true] {
        for _ in 0..16 {
            let p = guest();
            let pending = &pipeline(p as This).host;
            if completed {
                pending.make();
            }
            assert!(!pending.used.load(Ordering::Relaxed));
            let builds = PROBE_BUILDS.load(Ordering::Relaxed);
            let calls = library::PROBE_RECORDS.load(Ordering::Relaxed);
            let binds = PROBE_BINDS.load(Ordering::Relaxed);
            let barrier = Barrier::new(32);
            let results = std::thread::scope(|scope| {
                let mut handles = vec![];
                for _ in 0..32 {
                    handles.push(scope.spawn(|| {
                        barrier.wait();
                        bound(p as This);
                        let first = host_pipeline(p as This);
                        if completed {
                            assert!(first.is_some());
                        }
                        let until = Instant::now() + std::time::Duration::from_secs(8);
                        let host = first.unwrap_or_else(|| {
                            loop {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                                if let Some(host) = host_pipeline(p as This) {
                                    break host;
                                }
                                assert!(Instant::now() < until, "worker did not finish");
                            }
                        });
                        assert!(!host.compute && !host.mesh);
                        host as *const _ as usize
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert!(results.iter().all(|n| *n == results[0]));
            assert_eq!(
                PROBE_BUILDS.load(Ordering::Relaxed) - builds,
                u64::from(!completed)
            );
            assert_eq!(library::PROBE_RECORDS.load(Ordering::Relaxed) - calls, 1);
            assert_eq!(PROBE_BINDS.load(Ordering::Relaxed) - binds, 1);
            assert_eq!(pending.queued.load(Ordering::Relaxed), !completed);
            release(p);
        }
        println!(
            "RACE precompleted={completed} objects=16 threads=32 all_identical=true note_calls=16 builds={}",
            if completed { 0 } else { 16 }
        );
    }
}

#[test]
#[ignore = "manual CPU contention benchmark"]
fn actual_guest_pipeline_lookups() {
    let (vs, ps) = shader_codes();
    for (stage, code) in [("vertex", vs), ("pixel", ps)] {
        prime_translation(&shader(stage, code));
    }
    let host = crate::host::host().unwrap();
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    // SAFETY: the live device owns this LUID, and the factory returns the matching adapter.
    let name = unsafe {
        let fac: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let a: IDXGIAdapter1 = fac.EnumAdapterByLuid(host.device.GetAdapterLuid()).unwrap();
        String::from_utf16_lossy(&a.GetDesc1().unwrap().Description)
            .trim_end_matches('\0')
            .to_owned()
    };
    println!("DEVICE {name}");
    raced();
    if std::env::var_os("PROBE_CORES").is_none() {
        return;
    }
    let masks: Vec<usize> = std::env::var("PROBE_CORES")
        .unwrap()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    assert!(masks.len() >= 8);
    let objects: Vec<_> = (0..8).map(|_| guest()).collect();
    let lists: Vec<_> = (0..8).map(|_| command_list()).collect();
    for &p in &objects {
        pipeline(p as This).host.make();
        assert!(host_pipeline(p as This).is_some());
        bound(p as This);
    }
    let notes = library::PROBE_RECORDS.load(Ordering::Relaxed);
    let builds = PROBE_BUILDS.load(Ordering::Relaxed);
    for (threads, private, period) in [
        (1, false, 0),
        (4, false, 0),
        (8, false, 0),
        (8, true, 0),
        (8, false, 16),
        (8, false, 1),
        (8, true, 1),
    ] {
        let its: usize = if threads == 1 { 3_000_000 } else { 320_000 };
        let mut elapsed = vec![];
        for _ in 0..7 {
            let begin = Barrier::new(threads + 1);
            let go = std::sync::atomic::AtomicBool::new(false);
            let done = Barrier::new(threads + 1);
            let time = std::thread::scope(|scope| {
                for i in 0..threads {
                    let (p, l, mask) = (objects[if private { i } else { 0 }], lists[i].0, masks[i]);
                    let (begin, go, done) = (&begin, &go, &done);
                    scope.spawn(move || {
                        assert_ne!(
                            // SAFETY: this is the current worker's pseudo-handle and an OS affinity mask.
                            unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) },
                            0
                        );
                        let (p, l) = (p as This, l as This);
                        let bind = slot::<extern "system" fn(This, This)>(l, 0xc8);
                        begin.wait();
                        while !go.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                        if period == 0 {
                            for _ in 0..its {
                                black_box(host_pipeline(black_box(p)).unwrap());
                            }
                        } else if period == 1 {
                            for _ in 0..its {
                                bind(l, p);
                                black_box(host_pipeline(black_box(p)).unwrap());
                            }
                        } else {
                            for _ in 0..its / 16 {
                                bind(l, p);
                                for _ in 0..16 {
                                    black_box(host_pipeline(black_box(p)).unwrap());
                                }
                            }
                        }
                        done.wait();
                    });
                }
                begin.wait();
                let start = Instant::now();
                go.store(true, Ordering::Release);
                done.wait();
                start.elapsed().as_secs_f64() * 1e9 / (threads * its) as f64
            });
            elapsed.push(time);
        }
        println!("TIMES {threads} private={private} period={period} per_thread={its} {elapsed:?}");
    }
    assert_eq!(library::PROBE_RECORDS.load(Ordering::Relaxed), notes);
    assert_eq!(PROBE_BUILDS.load(Ordering::Relaxed), builds);
    for (l, a) in lists {
        release(l);
        release(a);
    }
    for p in objects {
        release(p);
    }
}
