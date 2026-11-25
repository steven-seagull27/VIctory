use super::*;
use crate::draw::State;
use victory_shader::emit::{HostShader, UserData};
use victory_shader::native::{HwStage, NativeProgram, RootParam, RootType, StageIo};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

const SIZE: u32 = 16;
struct Fixtures {
    vertex: Vec<u8>,
    signature: Vec<u8>,
}

fn compile_fixtures() -> Fixtures {
    let dxc = victory_shader::emit::validator::Validator::default_path()
        .map(|path| path.with_file_name("dxc.exe"))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| "dxc.exe".into());
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("fixture.hlsl");
    std::fs::write(&source, include_str!("fixture.hlsl")).unwrap();
    let compile = |entry: &str, profile: &str| {
        let output = dir.path().join(format!("{entry}.dxil"));
        let result = std::process::Command::new(&dxc)
            .args(["-T", profile, "-E", entry, "-Fo"])
            .arg(&output)
            .arg(&source)
            .output()
            .expect("Windows SDK dxc.exe required");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::read(output).unwrap()
    };
    Fixtures {
        vertex: compile("fullscreen", "vs_6_0"),
        signature: compile("color_signature", "ps_6_0"),
    }
}

fn pixel(dword: u32, signature: &[u8]) -> HostShader {
    let words = [
        0x7e00_0200 | dword, // v_mov_b32 v0, s[dword]
        0x0802_00f2,         // v_sub_f32 v1, 1.0, v0
        0x7e04_0280,         // v_mov_b32 v2, 0
        0x7e06_02f2,         // v_mov_b32 v3, 1.0
        0xf800_180f,
        0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
        0xbf81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        used: vec![],
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: dword + 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let sig = victory_shader::Container::parse(signature).unwrap();
    let ir = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&ir).unwrap();
    let res =
        victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&ir, &native).unwrap();
    let host = victory_shader::emit::emit(&sig, &native, &ir, &st, &io, &res).unwrap();
    assert_eq!(
        host.user_data,
        [UserData {
            index: dword,
            slot: 0,
            root: RootType::Constants,
            dword
        }]
    );
    host
}

fn stage(name: &'static str, host: HostShader) -> Arc<Shader> {
    // Test setup supplies both translated stages to the regular pipeline builder.
    let shader = Shader::new(name, &host.dxil);
    assert!(
        shader
            .translated
            .set(Some(Translated { host, used: vec![] }))
            .is_ok()
    );
    Arc::new(shader)
}

fn pipeline(dword: u32, fixtures: &Fixtures) -> crate::object::This {
    let ps = pixel(dword, &fixtures.signature);
    let vs = HostShader {
        stage: HwStage::Vs,
        dxil: fixtures.vertex.clone(),
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
    };
    let mut graphics = GraphicsState::default();
    graphics.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    graphics.targets.NumRenderTargets = 1;
    graphics.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    graphics.depth.DepthEnable = false.into();
    graphics.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
    let mut out = std::ptr::null();
    assert_eq!(
        super::super::hand_out(
            vec![stage("vertex", vs), stage("pixel", ps)],
            Some(Box::new(graphics)),
            &super::super::PIPELINE_IIDS[0],
            &mut out
        ),
        crate::object::S_OK
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while super::super::host_pipeline(out).is_none() {
        assert!(
            std::time::Instant::now() < deadline,
            "PSO readiness timeout"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    out
}

fn pattern(dword: u32, fixtures: &Fixtures) -> bool {
    let host = host().unwrap();
    let p = pipeline(dword, fixtures);
    let mut state = State::default();
    state.set_pipeline(p);
    let list = host.new_list().unwrap();
    state.set_topology(&list, 4);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: SIZE.into(),
        Height: SIZE,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut target: Option<ID3D12Resource> = None;
    // SAFETY: unshared resources and an open list, held through the synchronous submission.
    unsafe {
        host.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
            .unwrap();
    }
    let target = target.unwrap();
    let rtv = host.rtv(&target, None);
    let viewport = D3D12_VIEWPORT {
        Width: SIZE as f32,
        Height: SIZE as f32,
        MaxDepth: 1.,
        ..Default::default()
    };
    crate::draw::viewports(&list, &[viewport]);
    unsafe {
        list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.list
            .ClearRenderTargetView(rtv, &[0., 0., 1., 1.], None);
    }
    let before = crate::draw::drawn();
    let _ = crate::draw::take_dropped();
    let mut constants = vec![0.25f32.to_bits(); dword as usize + 1];
    for (left, right, value) in [(0, 8, 1f32), (8, 16, 0f32)] {
        constants[dword as usize] = value.to_bits();
        state.set_constants(false, 0, &constants, 0);
        crate::draw::scissors(
            &list,
            &[RECT {
                left,
                right,
                top: 0,
                bottom: 16,
            }],
        );
        state.draw(&list, 3, 1, 0, 0);
    }
    let draws = crate::draw::drawn() - before;
    let drops = crate::draw::take_dropped();
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut fp),
            None,
            None,
            Some(&mut total),
        )
    };
    let (back, data) = host.shared_buffer(total).unwrap();
    unsafe {
        list.list.ResourceBarrier(&[crate::host::transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&target),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&back),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: fp,
            },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    list.close();
    host.execute(&[&list]);
    let mut pixels = [0; 4];
    let mut mismatches = 0;
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let pixel = unsafe {
                data.add(y * fp.Footprint.RowPitch as usize + x * 4)
                    .cast::<[u8; 4]>()
                    .read_unaligned()
            };
            let color = match pixel {
                [255, 0, 0, 255] => 0,
                [0, 255, 0, 255] => 1,
                [0, 0, 255, 255] => 2,
                _ => 3,
            };
            pixels[color] += 1;
            mismatches += u32::from(color != usize::from(x >= 8));
        }
    }
    crate::object::release(p);
    eprintln!(
        "PIXELS dwords={} draws={draws} drops={drops} red={} green={} clear={} other={} mismatches={mismatches}",
        dword + 1,
        pixels[0],
        pixels[1],
        pixels[2],
        pixels[3]
    );
    draws == 2 && drops == 0 && mismatches == 0
}

#[test]
#[ignore = "submits actual D3D12 draws and reads every pixel"]
fn root_constants_draw_two_colors() {
    let host = host().expect("D3D12 device required");
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&desc.Description);
        eprintln!(
            "DEVICE {} vendor={:04x} device={:04x} luid={:08x}:{:08x} flags={}",
            name.trim_end_matches('\0'),
            desc.VendorId,
            desc.DeviceId,
            desc.AdapterLuid.HighPart,
            desc.AdapterLuid.LowPart,
            desc.Flags
        );
        if desc.Flags & 2 != 0 {
            use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
            let dll: Vec<u16> = "d3d10warp.dll".encode_utf16().chain([0]).collect();
            let module = GetModuleHandleW(dll.as_ptr());
            assert!(!module.is_null());
            let mut path = vec![0u16; 32768];
            let len = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
            assert!(len > 0 && len < path.len());
            eprintln!("WARP_DLL {}", String::from_utf16_lossy(&path[..len]));
        }
    }
    let fixtures = compile_fixtures();
    let control = pattern(3, &fixtures);
    let fifth = pattern(4, &fixtures);
    assert!(control, "the four-dword control must render");
    assert!(fifth, "the five-dword case must render");
}
