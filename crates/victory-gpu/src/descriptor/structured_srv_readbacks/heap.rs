use super::*;
use victory_shader::emit::{HostShader, VIEW_STRIDES};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// Existing HEAP_ELEMENT_WAVE32 fixture from victory-shader::emit::stride_tests.
/// Lanes index the heap SRV by element and write its second dword to a raw UAV.
const HEAP_ELEMENT: &[u32] = &[
    0x8F04_8500,
    0xBE86_0301,
    0xBE87_0380,
    0xF408_0203,
    0x0800_0000,
    0xBE8C_0302,
    0xBE8D_0380,
    0xF408_0406,
    0xFA00_0000,
    0x3402_0082,
    0xBF8C_C07F,
    0xE030_2004,
    0x8002_0200,
    0xBF8C_3F70,
    0xE070_1000,
    0x8004_0201,
    0xBF81_0000,
];

fn shader() -> HostShader {
    let layout = [RootType::Constants, RootType::ResourceHeap, RootType::Table]
        .into_iter()
        .enumerate()
        .map(|(i, root)| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: i as u8,
            slot: i as u8,
        })
        .collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: HEAP_ELEMENT.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let ir = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&ir).unwrap();
    let res =
        victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&ir, &native).unwrap();
    let signatures = victory_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    victory_shader::emit::emit(&signatures, &native, &ir, &st, &io, &res).unwrap()
}

fn mirror_srv(heap: &HostHeap, slot: u32, address: u64, first: u32, stride: u32, bytes: u32) {
    let mut resource = [0; 14];
    resource[0] = 1; // buffer
    resource[4] = bytes;
    let mut words = [0; 16];
    words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
    words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    words[4] = first;
    words[6] = 32; // elements
    words[7] = stride;
    words[10] = address as u32;
    words[11] = (address >> 32) as u32;
    let mut record = [0u32; 8];
    let handle = record.as_mut_ptr() as u64;
    crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &words, handle);
    assert!(heap.mirror(slot, handle, true).is_some());
}

#[repr(align(4096))]
struct Page([u32; 1024]);

fn with_heap_reader(check: impl FnOnce(&HostHeap, &dyn Fn(u32) -> [u32; 32])) {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    let host = host().expect("D3D12 host required");
    // SAFETY: query this actual D3D12 device's associated DXGI adapter.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    let name = String::from_utf16_lossy(&adapter.Description);
    eprintln!("DEVICE {}", name.trim_end_matches('\0'));
    let cache_dir = tempfile::tempdir().unwrap();
    crate::shader_cache::use_shader_cache(cache_dir.path().to_owned()).unwrap();
    let heap = HostHeap::new(false, 800_000).expect("allocate production view heap");
    let limit = victory_shader::emit::host_heap_views();
    let null = victory_shader::emit::null_buffer_view();
    // SAFETY: a plain D3D12 descriptor query.
    let total = unsafe { heap.heap.GetDesc() }.NumDescriptors;
    eprintln!(
        "HEAP allocated={total} shader_limit={limit} null_srv={null} raw_structured={}",
        victory_shader::emit::host_raw_structured()
    );
    assert!(total >= limit && null < total);
    assert!(null > heap.count + heap.dynamic);
    assert!(victory_shader::emit::host_raw_structured());
    let shader = shader();
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    let device = victory_shader::emit::host::Host {
        device: host.device.clone(),
    };
    let (root, pso) = device.compute_pipeline(&shader).unwrap();
    let (output, mapped) = host.shared_buffer(32 * 4).unwrap();
    heap.raw_view_at(0, Some(&output), 0, 32 * 4, true);
    let read = |slot: u32| {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                RootType::Constants => slot,
                RootType::Table => 0,
                other => panic!("{other:?}"),
            };
        }
        let list = host.new_list().unwrap();
        crate::bindless::refresh(&heap.viewed, |shadow, address, len| {
            shadow.refresh(&list, address as u64, len as u64);
        });
        // SAFETY: all referenced resources and heaps remain live through the waited submission.
        // The output has 32 mapped dwords. The completed prior submission does not use it.
        unsafe {
            std::ptr::write_bytes(mapped, 0xa5, 32 * 4);
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.list.SetComputeRootShaderResourceView(
                1,
                heap.strides().unwrap().GetGPUVirtualAddress(),
            );
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: execute waits for the fence and the mapping contains 32 dwords.
        unsafe {
            host.device.GetDeviceRemovedReason().unwrap();
            mapped.cast::<[u32; 32]>().read_unaligned()
        }
    };
    check(&heap, &read);
}

#[test]
#[ignore = "requires an isolated D3D12 device process and a GPU readback"]
fn capped_runtime_heap_reads_null_and_structured_views() {
    with_heap_reader(|heap, read| {
        let mut input = Box::new(Page([0; 1024]));
        for (i, e) in input.0.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            e.copy_from_slice(&[100 + i as u32, 200 + i as u32, 300 + i as u32]);
        }
        let address = input.0.as_ptr() as u64;
        // Same 32 logical elements as FirstElement=1, relocated to offset zero.
        let mut control = Box::new(Page([0; 1024]));
        control.0[..96].copy_from_slice(&input.0[3..99]);
        let control_address = control.0.as_ptr() as u64;
        crate::shadow::register(address as usize, address, 4096);
        crate::shadow::register(control_address as usize, control_address, 4096);
        for (slot, first) in [(1, 0), (2, 4), (3, 1), (4, 2), (5, 3)] {
            mirror_srv(heap, slot, address, first, 12, 4096);
        }
        mirror_srv(heap, 6, control_address, 0, 12, 4096);
        let mut misses = 0;
        for (label, slot, expected_first) in [
            ("base", 1, Some(0)),
            ("aligned_48", 2, Some(4)),
            ("offset_12", 3, Some(1)),
            ("offset_24", 4, Some(2)),
            ("offset_36", 5, Some(3)),
            ("relocated_control", 6, Some(1)),
            ("missing", 10, None),
        ] {
            let values = read(slot);
            let expected: Vec<u32> = (0..32)
                .map(|i| expected_first.map_or(0, |f| 200 + f + i))
                .collect();
            let wrong = values.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!("READ {label} mismatches={wrong}/32 actual={values:?} expected={expected:?}");
            misses += wrong;
        }
        crate::shadow::unregister(address as usize, address);
        crate::shadow::unregister(control_address as usize, control_address);
        assert_eq!(misses, 0);
    });
}

#[test]
#[ignore = "requires an isolated D3D12 device process and a GPU readback"]
fn one_heap_slot_can_change_stride_alignment_and_nullness() {
    with_heap_reader(|heap, read| {
        let mut input = Box::new(Page([0; 1024]));
        for (i, word) in input.0.iter_mut().enumerate() {
            *word = 0x5080_0000 + i as u32;
        }
        let address = input.0.as_ptr() as u64;
        crate::shadow::register(address as usize, address, 4096);
        let mut misses = 0;
        let slot = 7;
        let empty = [0u32; 8];
        // (stride, first element, byte offset of placed resource within the parent).
        for (case, (stride, first, placed)) in [
            (12u32, 0u32, 0u32),
            (12, 1, 0),
            (20, 1, 0),
            (8, 1, 0),
            (12, 4, 0),
            (0, 0, 0),
            (28, 1, 0),
            (16, 1, 0),
            (12, 0, 12),
            (12, 2, 12),
            (20, 1, 20),
            (12, 1, 24),
            (32, 3, 0),
            (0, 0, 0),
            (12, 0, 0),
        ]
        .into_iter()
        .enumerate()
        {
            if stride == 0 {
                assert!(heap.mirror(slot, empty.as_ptr() as u64, true).is_none());
            } else {
                assert!(placed + (first + 32) * stride <= 4096);
                mirror_srv(
                    heap,
                    slot,
                    address + u64::from(placed),
                    first,
                    stride,
                    4096 - placed,
                );
            }
            let values = read(slot);
            let expected: Vec<u32> = (0..32)
                .map(|lane| {
                    if stride == 0 {
                        0
                    } else {
                        0x5080_0000 + (placed + (first + lane) * stride + 4) / 4
                    }
                })
                .collect();
            let wrong = values.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!(
                "REBIND {case} stride={stride} first={first} placed={placed} wrong={wrong}/32 actual={values:?} expected={expected:?}"
            );
            misses += wrong;
        }
        crate::shadow::unregister(address as usize, address);
        assert_eq!(misses, 0);
    });
}
