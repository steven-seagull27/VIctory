use super::*;
use std::ptr::null;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_SRV_DIMENSION_BUFFER,
    D3D12_UAV_DIMENSION_BUFFER,
};
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{IUnknown, Interface};

/// Read what the guest shader sees both on the table's first use and after
/// replacing an existing buffer view. Each slot spans two address words,
/// a record count and a packed format/type word.
#[test]
fn adjacent_buffer_descriptors_report_their_fields_after_a_rewrite() {
    let Some(device) = crate::host::host() else {
        eprintln!("no D3D12 device; skipped buffer descriptor readback");
        return;
    };
    let mut guest = null();
    assert_eq!(
        create(
            HeapDesc {
                kind: VIEWS_KIND,
                count: 3,
                flags: SHADER_VISIBLE,
                nodes: 0
            },
            Placement {
                address: 0,
                size: 0
            },
            &mut guest,
        ),
        S_OK
    );
    // SAFETY: create handed out one reference to the guest descriptor heap.
    let _heap_owner = unsafe { IUnknown::from_raw(guest.cast_mut().cast()) };
    let guest_heap = heap(guest);
    let input = crate::resource::test_buffer(256);
    // SAFETY: test_buffer hands out one reference to the guest buffer.
    let _input_owner = unsafe { IUnknown::from_raw(input.cast_mut().cast()) };
    let base = crate::resource::buffer_bytes(input).unwrap().0 as u64;
    let handle =
        |slot: u32| guest_heap.start + u64::from(slot * increment_size(null(), VIEWS_KIND));
    let write_srv = |slot: u32, format: DXGI_FORMAT, first: u32, count: u32, padding: u32| {
        // The guest CreateShaderResourceView argument is ten DWORDs, including its
        // padding. The buffer union begins at DWORD 4 with the 64-bit FirstElement.
        let desc = [
            format.0 as u32,
            D3D12_SRV_DIMENSION_BUFFER.0 as u32,
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            padding,
            first,
            0,
            count,
            0,
            0,
            0,
        ];
        create_srv(null(), input, &desc, handle(slot));
    };
    write_srv(1, DXGI_FORMAT_R32_UINT, 4, 5, 0);
    write_srv(2, DXGI_FORMAT_R32_FLOAT, 12, 11, 0);

    let shader = descriptor_reader();
    assert!(shader.view_addresses && shader.root_srvs.is_empty() && !shader.memory);
    let host = victory_shader::emit::host::Host::from_device(device.device.clone());
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let mirror = guest_heap.host().expect("host view heap");
    let addresses = mirror.view_addresses().expect("view-address table");
    let (output, mapped) = device.shared_buffer(32).unwrap();
    mirror.raw_view_at(0, Some(&output), 0, 32, true);
    let dispatch = || {
        let list = device.new_list().unwrap();
        let table_base = vec![0u32; shader.user_data_dwords() as usize];
        // SAFETY: the root table begins at slot 0. The shader reads only the two
        // valid input slots and writes all eight DWORDs of the output. Each
        // submission finishes before rewriting a descriptor or reusing the output.
        unsafe {
            std::ptr::write_bytes(mapped, 0xa5, 32);
            list.list.SetDescriptorHeaps(&[Some(mirror.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRoot32BitConstants(
                0,
                table_base.len() as u32,
                table_base.as_ptr().cast(),
                0,
            );
            list.list.SetComputeRootShaderResourceView(1, addresses);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        device.execute(&[&list]);
        // SAFETY: eight mapped DWORDs written before the submission's fence.
        unsafe { mapped.cast::<[[u32; 4]; 2]>().read_unaligned() }
    };
    let before = dispatch();
    write_srv(1, DXGI_FORMAT_R32_SINT, 20, 7, 0);
    let after = dispatch();
    // AMD RDNA2 ISA, buffer resource descriptor: NUM_RECORDS is DWORD 2,
    // FORMAT is bits 18:12 of DWORD 3. R32_UINT/SINT/FLOAT encode as 20/21/22.
    let fields = |[lo, hi, count, word3]: [u32; 4]| {
        let address = u64::from(lo) | (u64::from(hi & 0xffff) << 32);
        (address.wrapping_sub(base), count, (word3 >> 12) & 0x7f)
    };
    let actual = [before.map(fields), after.map(fields)];
    let expected = [[(16, 5, 20), (48, 11, 22)], [(80, 7, 21), (48, 11, 22)]];
    eprintln!("descriptor readbacks (byte offset, count, FORMAT) before and after: {actual:?}");
    assert_eq!(actual, expected);

    // SRV first element 3, five records. UAV first element 7, nine
    // four-byte records. The expected offsets and GFX10 FORMAT codes
    // below are independent of the runtime's DXGI conversion.
    for (srv, srv_offset, srv_code, uav, uav_code) in [
        (DXGI_FORMAT_R8_UINT, 3, 5, DXGI_FORMAT_R32_UINT, 20),
        (DXGI_FORMAT_R8_SNORM, 3, 2, DXGI_FORMAT_R32_SINT, 21),
        (DXGI_FORMAT_R16_SINT, 6, 12, DXGI_FORMAT_R32_FLOAT, 22),
        (DXGI_FORMAT_R16_FLOAT, 6, 13, DXGI_FORMAT_R32_UINT, 20),
        (DXGI_FORMAT_R8G8_UNORM, 6, 14, DXGI_FORMAT_R32_SINT, 21),
        (DXGI_FORMAT_R16G16_FLOAT, 12, 29, DXGI_FORMAT_R32_FLOAT, 22),
        (DXGI_FORMAT_R32G32_FLOAT, 24, 64, DXGI_FORMAT_R32_UINT, 20),
        (
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            24,
            71,
            DXGI_FORMAT_R32_SINT,
            21,
        ),
        (
            DXGI_FORMAT_R32G32B32A32_UINT,
            48,
            75,
            DXGI_FORMAT_R32_FLOAT,
            22,
        ),
        (
            DXGI_FORMAT_R11G11B10_FLOAT,
            12,
            36,
            DXGI_FORMAT_R32_UINT,
            20,
        ),
        (
            DXGI_FORMAT_R10G10B10A2_UNORM,
            12,
            50,
            DXGI_FORMAT_R32_SINT,
            21,
        ),
        (
            DXGI_FORMAT_R8G8B8A8_UNORM,
            12,
            56,
            DXGI_FORMAT_R32_FLOAT,
            22,
        ),
    ] {
        // If SRV API padding leaks into the guest FORMAT, it reads as 127.
        write_srv(1, srv, 3, 5, 0x0007_f000);
        let desc = [
            uav.0 as u32,
            D3D12_UAV_DIMENSION_BUFFER.0 as u32,
            7,
            0,
            9,
            0,
            0,
            0,
            0,
            0,
        ];
        create_uav(null(), input, null(), &desc, handle(2));
        let actual = dispatch().map(fields);
        let expected = [(srv_offset, 5, srv_code), (28, 9, uav_code)];
        assert_eq!(actual, expected, "SRV {srv:?}, UAV {uav:?}");
    }
}

/// A synthetic one-thread GFX10 shader copies the four descriptor words at
/// table slots 1 and 2 into the raw UAV at slot 0. Neither input is dereferenced.
fn descriptor_reader() -> victory_shader::emit::HostShader {
    use victory_shader::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };

    let mut words = vec![
        0xbe81_0380u32, // s_mov_b32 s1, 0: the table begins at s0
        0xf408_0100,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0 (output UAV)
        0xf408_0200,
        0xfa00_0020, // s_load_dwordx4 s[8:11], s[0:1], 32 (slot 1)
        0xf408_0300,
        0xfa00_0040, // s_load_dwordx4 s[12:15], s[0:1], 64 (slot 2)
        0xbf8c_0000, // s_waitcnt 0
    ];
    for v in 0..8 {
        words.push(0x7e00_0200 | v << 17 | (8 + v)); // v_mov_b32 vN, s[8 + N]
    }
    words.extend([
        0xe078_0000,
        0x8001_0000, // buffer_store_dwordx4 v[0:3], off, s[4:7], 0
        0xe078_0010,
        0x8001_0400, // buffer_store_dwordx4 v[4:7], off, s[4:7], 0 offset:16
        0xbf81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let signatures = victory_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let ir = victory_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = victory_shader::structure::structure(&ir).unwrap();
    let res =
        victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
    let io = victory_shader::io::raise(&ir, &native).unwrap();
    victory_shader::emit::emit(&signatures, &native, &ir, &st, &io, &res).unwrap()
}
