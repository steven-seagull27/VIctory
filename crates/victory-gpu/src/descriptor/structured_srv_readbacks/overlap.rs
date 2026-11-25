use super::*;
use crate::host::Host;
use victory_shader::emit::{HostShader, VIEW_STRIDES};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

const LANES: usize = 32;
const ELEMENTS: usize = 36;
const FIELDS: usize = 3;
const BYTES: usize = ELEMENTS * FIELDS * 4;
const WRITER_SLOT: u32 = 6;

// Each lane addresses element v0 of the heap view at s0. Readers export the
// selected field through a separate raw output UAV in s2. Writers store the
// lane index, distinguishable from the initial values in every backing dword.
fn shader(write: bool, field: usize) -> HostShader {
    let mut code = vec![
        0x8F04_8500, // s_lshl_b32 s4, s0, 5
        0xBE86_0301,
        0xBE87_0380, // s[6:7] = resource heap
        0xF408_0203,
        0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    ];
    let byte = (field * 4) as u32;
    if write {
        code.extend([
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_2000 | byte,
            0x8002_0000, // store v0 to elem v0, s[8:11]
        ]);
    } else {
        code.extend([
            0xBE8C_0302,
            0xBE8D_0380, // s[12:13] = output table
            0xF408_0406,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_2000 | byte,
            0x8002_0200, // load v2 from elem v0, s[8:11]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_1000,
            0x8004_0201, // store v2 to raw byte v1, s[16:19]
        ]);
    }
    code.push(0xBF81_0000); // s_endpgm
    let layout = [RootType::Constants, RootType::ResourceHeap, RootType::Table]
        .into_iter()
        .enumerate()
        .map(|(slot, root)| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: slot as u8,
            slot: slot as u8,
        })
        .collect();
    let output = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 2,
        slot: 2,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used: if write { vec![] } else { vec![output] },
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [LANES as u32, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
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
    let emitted = victory_shader::emit::emit(&signatures, &native, &ir, &st, &io, &res).unwrap();
    assert!(emitted.heap && emitted.heap_writes == write);
    assert_eq!(emitted.root_srvs, [VIEW_STRIDES]);
    emitted
}

fn mirror(heap: &HostHeap, slot: u32, address: u64, first: usize, write: bool) {
    assert!(first + LANES <= ELEMENTS);
    let mut resource = [0; 14];
    resource[0] = 1; // D3D12_RESOURCE_DIMENSION_BUFFER
    resource[4] = BYTES as u32;
    let mut words = [0; 16];
    words[10] = address as u32;
    words[11] = (address >> 32) as u32;
    let mut record = [0u32; 8];
    let handle = record.as_mut_ptr() as u64;
    if write {
        words[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
        words[2] = first as u32;
        words[4] = LANES as u32;
        words[5] = (FIELDS * 4) as u32;
        crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &words, handle);
    } else {
        words[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        words[4] = first as u32;
        words[6] = LANES as u32;
        words[7] = (FIELDS * 4) as u32;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &words, handle);
    }
    let view = heap.mirror(slot, handle, true).unwrap();
    assert_eq!(
        buffer_range(&view),
        Some((
            address + (first * FIELDS * 4) as u64,
            (LANES * FIELDS * 4) as u64
        ))
    );
}

struct Kernel {
    shader: HostShader,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl Kernel {
    fn new(device: &victory_shader::emit::host::Host, write: bool, field: usize) -> Self {
        let shader = shader(write, field);
        let (root, pso) = device.compute_pipeline(&shader).unwrap();
        Self { shader, root, pso }
    }

    fn run(&self, host: &Host, heap: &HostHeap, slot: u32, refresh: bool) {
        let list = host.new_list().unwrap();
        if refresh {
            crate::bindless::refresh(&heap.viewed, |shadow, address, len| {
                shadow.refresh(&list, address as u64, len as u64);
            });
        }
        let mut ud = vec![0; self.shader.user_data_dwords().max(1) as usize];
        for u in &self.shader.user_data {
            ud[u.index as usize] = match u.root {
                RootType::Constants => slot,
                RootType::Table => 0,
                other => panic!("unexpected root {other:?}"),
            };
        }
        // SAFETY: all descriptors, resources and root arguments live through
        // the waited submission. Each buffer implicitly promotes from COMMON.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            list.list
                .SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.list.SetComputeRootShaderResourceView(
                1,
                heap.strides().unwrap().GetGPUVirtualAddress(),
            );
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        // Fence completion and implicit buffer decay precede the next run.
        host.execute(&[&list]);
        // SAFETY: status query on the live device.
        unsafe { host.device.GetDeviceRemovedReason().unwrap() };
    }
}

#[test]
#[ignore = "requires a D3D12 device and an independent GPU readback"]
fn gpu_writes_visible_through_overlapping_views() {
    let host = host().expect("D3D12 device required");
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
    let device = victory_shader::emit::host::Host {
        device: host.device.clone(),
    };
    let writer = Kernel::new(&device, true, FIELDS - 1);
    let readers = (0..FIELDS)
        .map(|field| Kernel::new(&device, false, field))
        .collect::<Vec<_>>();
    let (output, mapped) = host.shared_buffer((LANES * 4) as u64).unwrap();
    heap.raw_view_at(0, Some(&output), 0, (LANES * 4) as u64, true);

    #[repr(align(4096))]
    struct Allocation([u32; ELEMENTS * FIELDS]);
    let initial = std::array::from_fn(|i| 0xa170_0000 + i as u32);
    let allocations: Vec<_> = (0..5).map(|_| Box::new(Allocation(initial))).collect();
    let mut total_wrong = 0;
    for (writer_first, input) in allocations.iter().enumerate() {
        let address = input.0.as_ptr() as u64;
        crate::shadow::register(address as usize, address, BYTES as u64);
        for reader_first in 0..5 {
            mirror(&heap, 1 + reader_first as u32, address, reader_first, false);
        }
        mirror(&heap, WRITER_SLOT, address, writer_first, true);
        writer.run(host, &heap, WRITER_SLOT, true);
        let mut expected = initial;
        for lane in 0..LANES {
            expected[(writer_first + lane) * FIELDS + FIELDS - 1] = lane as u32;
        }
        // No refresh or copy-back between the GPU writer and any reader.
        // A per-view private copy cannot pass by relaying data through the CPU.
        assert_eq!(input.0, initial);
        for reader_first in 0..5 {
            for (field, reader) in readers.iter().enumerate() {
                // SAFETY: the previous submission completed; 32 mapped output dwords.
                unsafe { std::ptr::write_bytes(mapped, 0x55, LANES * 4) };
                reader.run(host, &heap, 1 + reader_first as u32, false);
                // SAFETY: every lane's output is visible after the queue fence.
                let values = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), LANES) };
                let want = (0..LANES)
                    .map(|lane| expected[(reader_first + lane) * FIELDS + field])
                    .collect::<Vec<_>>();
                let wrong = values.iter().zip(&want).filter(|(a, b)| a != b).count();
                let wrong_first = values.iter().zip(&want).position(|(a, b)| a != b);
                eprintln!(
                    "READ writer={writer_first} reader={reader_first} field={field} mismatches={wrong}/32 first_difference={wrong_first:?} last={:#010x} expected_last={:#010x} actual={values:?} expected={want:?}",
                    values[LANES - 1],
                    want[LANES - 1]
                );
                total_wrong += wrong;
            }
        }
        assert_eq!(input.0, initial);
        crate::shadow::unregister(address as usize, address);
    }
    eprintln!("TOTAL mismatches={total_wrong}/2400");
    assert_eq!(total_wrong, 0);
}
