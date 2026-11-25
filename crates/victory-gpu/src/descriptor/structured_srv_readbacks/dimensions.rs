use super::*;
use victory_shader::emit::{HostShader, VIEW_STRIDES};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

// Two ordinary descriptor tables, each accessed at a fixed index. Lane i
// returns record i's last field and descriptor dword 2 (the record count).
fn shader() -> HostShader {
    let words: &[u32] = &[
        0xBE86_0301,
        0xBE87_0380, // s[6:7] = input table from s1
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0
        0xBE8C_0302,
        0xBE8D_0380, // s[12:13] = output table from s2
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (lane * 8)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2008,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:8
        0x7E06_020A, // v_mov_b32 v3,s10
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0201, // store v2 to output byte v1
        0xE070_1004,
        0x8004_0301, // store v3 to output byte v1+4
        0xBF81_0000, // s_endpgm
    ];
    let layout = [1, 2]
        .into_iter()
        .map(|slot| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: slot,
            slot,
        })
        .collect();
    let used = [(1, ResourceType::Srv, 12), (2, ResourceType::Uav, 0)]
        .into_iter()
        .map(|(slot, kind, stride)| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: slot,
            slot,
        })
        .collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let ir = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&ir).unwrap();
    let res =
        victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&ir, &native).unwrap();
    let container = victory_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    victory_shader::emit::emit(&container, &native, &ir, &st, &io, &res).unwrap()
}

#[test]
#[ignore = "requires a D3D12 device"]
fn reports_record_count_and_payload() {
    let host = host().unwrap();
    // SAFETY: query this actual D3D12 device's associated DXGI adapter.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    eprintln!(
        "DEVICE {}",
        String::from_utf16_lossy(&adapter.Description).trim_end_matches('\0')
    );
    let cache_dir = tempfile::tempdir().unwrap();
    crate::shader_cache::use_shader_cache(cache_dir.path().to_owned()).unwrap();
    let heap = HostHeap::new(false, 128).unwrap();
    assert!(victory_shader::emit::host_raw_structured());
    let shader = shader();
    assert!(shader.root_srvs.is_empty() || shader.root_srvs == [VIEW_STRIDES]);
    let device = victory_shader::emit::host::Host {
        device: host.device.clone(),
    };
    let (root, pso) = device.compute_pipeline(&shader).unwrap();
    let (output, mapped) = host.shared_buffer(256).unwrap();
    heap.raw_view_at(0, Some(&output), 0, 256, true);

    #[repr(align(4096))]
    struct Allocation([u32; 108]);
    let data = Box::new(Allocation(std::array::from_fn(|i| 0x5120_0000 + i as u32)));
    let address = data.0.as_ptr() as u64;
    crate::shadow::register(address as usize, address, 432);
    let mut total_count = 0;
    let mut total_payload = 0;
    for first in 0..5usize {
        let mut resource = [0; 14];
        resource[0] = 1;
        resource[4] = 432;
        let mut words = [0; 16];
        words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        words[4] = first as u32;
        words[6] = 32;
        words[7] = 12;
        words[10] = address as u32;
        words[11] = (address >> 32) as u32;
        let mut record = [0u32; 8];
        let handle = record.as_mut_ptr() as u64;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &words, handle);
        let slot = 1 + first as u32;
        let view = heap.mirror(slot, handle, true).unwrap();
        assert_eq!(
            buffer_range(&view),
            Some((address + (first * 12) as u64, 384))
        );
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table);
            ud[u.index as usize] = match u.slot {
                1 => slot * 32,
                2 => 0,
                other => panic!("unexpected slot {other}"),
            };
        }
        let list = host.new_list().unwrap();
        crate::bindless::refresh(&heap.viewed, |shadow, address, len| {
            shadow.refresh(&list, address as u64, len as u64);
        });
        // SAFETY: references remain live through the waited submission. Each
        // output pair has its own lane, and 32 eight-byte pairs fit the mapping.
        unsafe {
            std::ptr::write_bytes(mapped, 0x55, 256);
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            if !shader.root_srvs.is_empty() {
                list.list.SetComputeRootShaderResourceView(
                    1,
                    heap.strides().unwrap().GetGPUVirtualAddress(),
                );
            }
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: waited fence, live mapping and plain device status.
        let values = unsafe {
            host.device.GetDeviceRemovedReason().unwrap();
            std::slice::from_raw_parts(mapped.cast::<u32>(), 64)
        };
        let counts: Vec<_> = values.as_chunks::<2>().0.iter().map(|p| p[1]).collect();
        let payloads: Vec<_> = values.as_chunks::<2>().0.iter().map(|p| p[0]).collect();
        let want: Vec<_> = (0..32).map(|lane| data.0[(first + lane) * 3 + 2]).collect();
        let wrong_count = counts.iter().filter(|&&n| n != 32).count();
        let wrong_payload = payloads.iter().zip(&want).filter(|(a, b)| a != b).count();
        eprintln!(
            "READ first={first} count_wrong={wrong_count}/32 payload_wrong={wrong_payload}/32 counts={counts:?} payloads={payloads:?} expected_payloads={want:?}"
        );
        total_count += wrong_count;
        total_payload += wrong_payload;
    }
    crate::shadow::unregister(address as usize, address);
    eprintln!("TOTAL count_wrong={total_count}/160 payload_wrong={total_payload}/160");
    assert_eq!((total_count, total_payload), (0, 0));
}
