use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, UserData};
use crate::native::{HwStage, RootType};
use std::path::PathBuf;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

#[test]
#[ignore = "run tests/permlane-native/run.py to supply independently compiled shaders"]
fn compare_indices() {
    let dir = PathBuf::from(std::env::var_os("PERMLANE_CONTROL_DIR").expect("DXIL directory"));
    let warp = std::env::var_os("PERMLANE_CONTROL_WARP").is_some();
    let host = Host::new(warp).expect("D3D12 device");
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the feature query writes the supplied structure. The adapter LUID
    // comes from the same live D3D12 device that executes this dispatch.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
            .unwrap();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        eprintln!(
            "{} (wave range {}–{})",
            String::from_utf16_lossy(&desc.Description).trim_end_matches('\0'),
            caps.WaveLaneCountMin,
            caps.WaveLaneCountMax
        );
    }
    if warp {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
        let name: Vec<_> = "d3d10warp.dll".encode_utf16().chain([0]).collect();
        let mut path = vec![0u16; 32768];
        // SAFETY: the module name is terminated and the output buffer is live.
        let len = unsafe {
            let module = GetModuleHandleW(name.as_ptr());
            assert!(!module.is_null());
            GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize
        };
        assert!(len > 0 && len < path.len());
        let loaded = PathBuf::from(String::from_utf16(&path[..len]).unwrap())
            .canonicalize()
            .unwrap();
        eprintln!("Loaded WARP: {}", loaded.display());
        let expected = PathBuf::from(std::env::var_os("PERMLANE_CONTROL_WARP").unwrap());
        assert_eq!(loaded, expected.canonicalize().unwrap());
    }
    let mut failures = 0;
    for width in [32, 64] {
        if !caps.WaveOps.as_bool()
            || !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&width)
        {
            eprintln!("Wave{width} skipped");
            continue;
        }
        for opt in ["O3", "Od"] {
            let shader = HostShader {
                stage: HwStage::Cs,
                dxil: std::fs::read(dir.join(format!("{width}-{opt}.dxil"))).unwrap(),
                user_data: (0..4)
                    .map(|d| UserData {
                        index: d,
                        slot: 0,
                        root: RootType::Constants,
                        dword: d,
                    })
                    .collect(),
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
                root_uavs: vec![],
                memory: false,
                memory_writes: false,
                view_addresses: false,
                heap: true,
                heap_writes: true,
                view_marks: false,
            };
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let output = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(
                            0,
                            Root::Constants(vec![0xf10e2d3c, 0x4b5a6978, 0x55555555, 0x55555555]),
                        )],
                        views: &[View::RawUav(128 * 12)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(output[0].len(), 128 * 12);
            let mut active = 0;
            let mut wrong = 0;
            for record in output[0].chunks_exact(12) {
                let [
                    _,
                    lanes,
                    source,
                    selected,
                    _,
                    _,
                    guarded,
                    direct,
                    lo,
                    hi,
                    source_enabled,
                    enabled,
                ] = *record
                else {
                    unreachable!()
                };
                assert_eq!(lanes, width);
                assert_eq!((lo, hi), (u32::MAX, if width == 64 { u32::MAX } else { 0 }));
                if source_enabled == 1 && enabled == 1 {
                    assert_eq!(selected, source);
                    active += 1;
                    wrong += usize::from(guarded != direct);
                }
            }
            assert!(active > 0);
            let r = &output[0][4 * 12..5 * 12];
            eprintln!(
                "Wave{width} -{opt}: thread 4 lane={} source={} selected={} guarded={:#010x} direct={:#010x}; {wrong}/{active} enabled reads differ",
                r[0], r[2], r[3], r[6], r[7]
            );
            failures += wrong;
        }
    }
    assert_eq!(
        failures, 0,
        "guarded and direct shuffles of the same active source differ"
    );
}
