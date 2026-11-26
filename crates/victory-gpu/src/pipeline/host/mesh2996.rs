// Test-only adaptation of andrewpozdnakov7-jpg's translated fixture in #2996.
use super::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

#[test]
fn direct_24x1x1() {
    run([24, 1, 1], true, [1; 24]);
}

#[test]
fn indirect_24x1x1() {
    run([24, 1, 1], false, [1; 24]);
}

#[test]
fn indirect_3x2x4() {
    let mut expected = [1; 24];
    if std::env::var("MESH2996_EXPECT_X_ONLY").as_deref() == Ok("1") {
        expected = [0; 24];
        expected[..3].fill(1);
    }
    run([3, 2, 4], false, expected);
}

fn run(grid: [u32; 3], direct: bool, expected: [u32; 24]) {
    use crate::object::{S_OK, This, guid, slot};
    use victory_shader::native::{MeshIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, emit,
        io, ir, resource, structure,
    };

    let host = host().expect("a D3D12 host is required");
    assert!(host.mesh_shaders(), "mesh shaders are required");
    // SAFETY: look up the adapter of the same live D3D12 device used for submission.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    let name = String::from_utf16_lossy(&adapter.Description);
    eprintln!(
        "MESH_DEVICE {} vendor={:04x} device={:04x} software={}",
        name.trim_end_matches(char::from(0)),
        adapter.VendorId,
        adapter.DeviceId,
        adapter.Flags & 2 != 0
    );
    assert_eq!(
        adapter.Flags & 2 != 0,
        std::env::var("VICTORY_D3D12_WARP").as_deref() == Ok("1"),
        "the selected adapter must match the requested backend"
    );
    let words = [
        0xbe8d_0380u32, // s_mov_b32 s13, 0
        0xf408_0206,
        0xfa00_0000, // s_load_dwordx4 s[8:11], s[12:13], 0
        0x7e00_0505, // v_readfirstlane_b32 s0, v5 (mesh group)
        0x7e00_0200, // v_mov_b32 v0, s0
        0x3400_0082, // v_lshlrev_b32 v0, 2, v0 (dword to byte)
        0x7e02_0281, // v_mov_b32 v1, 1
        0xe0c8_1000,
        0x8002_0100, // buffer_atomic_add v1, v0, s[8:11], 0 offen
        0xbf81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 6,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
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
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 1,
            max_vertices: 3,
            max_primitives: 1,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let container = Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let program = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let structured = structure::structure(&program).unwrap();
    let resources =
        resource::track(&program, &native.layout, native.stage, Some(&structured)).unwrap();
    let shader_io = io::raise(&program, &native).unwrap();
    assert!(
        shader_io
            .inputs
            .iter()
            .any(|&(_, input)| input == io::Input::MeshGroup)
    );
    let shader = emit::emit(
        &container,
        &native,
        &program,
        &structured,
        &shader_io,
        &resources,
    )
    .unwrap();
    let shader = Shader::for_test(
        "mesh",
        Translated {
            host: shader,
            used: native.used,
        },
    );

    let null = std::ptr::null();
    let mut graphics = GraphicsState {
        topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ..Default::default()
    };
    graphics.depth.DepthEnable = false.into();
    let mut pipeline = null;
    assert_eq!(
        crate::pipeline::hand_out(
            vec![shader],
            Some(Box::new(graphics)),
            &crate::pipeline::PIPELINE_IIDS[0],
            &mut pipeline,
        ),
        S_OK
    );
    let mut heap = null;
    let desc: [u32; 8] = [0, 1, 1, 0, 0, 0, 0, 0];
    let iid = ID3D12DescriptorHeap::IID;
    let iid = guid(iid.data1, iid.data2, iid.data3, iid.data4);
    assert_eq!(
        crate::descriptor::create_heap(null, desc.as_ptr().cast(), &iid, &mut heap),
        S_OK
    );
    let handle = crate::descriptor::heap(heap).start;
    let output = crate::resource::test_buffer(128);
    let args = crate::resource::test_buffer(12);
    let (out, _) = crate::resource::buffer_bytes(output).unwrap();
    let (arg, _) = crate::resource::buffer_bytes(args).unwrap();
    // SAFETY: both pointers name the new committed title buffers above.
    unsafe {
        out.cast::<[u32; 24]>().write([0; 24]);
        arg.cast::<[u32; 3]>().write(grid);
    }
    let raw_uav = [39, 1, 0, 0, 24, 0, 0, 0, 1, 0];
    crate::descriptor::create_uav(null, output, null, &raw_uav, handle);
    let list = host.new_list().unwrap();
    let mut state = crate::draw::State::default();
    state.set_pipeline(pipeline);
    state.set_heaps(&[heap]);
    state.set_table(false, 1, handle);
    if direct {
        assert_eq!(&grid[1..], &[1, 1]);
        state.dispatch_mesh(&list, grid[0]);
    } else {
        assert!(state.execute_indirect(&list, (10, 12, 1), arg as u64, None));
    }
    list.close();
    eprintln!("MESH_SUBMIT direct={direct} grid={grid:?}");
    host.execute(&[&list]);
    // SAFETY: the synchronous submission completed and the buffer is live.
    let observed = unsafe { out.cast::<[u32; 24]>().read() };
    eprintln!("MESH_READBACK direct={direct} grid={grid:?} observed={observed:?}");
    drop(state);
    drop(list);
    for object in [output, args, heap, pipeline] {
        slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
    }
    assert_eq!(observed, expected);
}
