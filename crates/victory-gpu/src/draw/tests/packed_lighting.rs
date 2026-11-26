use super::*;
use crate::object::{guid, slot};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC,
};
use windows::core::Interface;

const TEXELS: usize = 32;

const WRITE_PACKED: &str = r#"
RWTexture2D<uint> target : register(u0);
[RootSignature("DescriptorTable(UAV(u0))")]
[numthreads(32, 1, 1)]
void main(uint i : SV_GroupIndex) {
    uint r = 13 + 7 * i, g = 101 + 9 * i, b = 501 - 11 * i, e = 12 + i % 9;
    target[uint2(27 * i, 15 * i)] = r | (g << 9) | (b << 18) | (e << 27);
}
"#;

const READ_RGB: &str = r#"
Texture2D<float4> source : register(t0);
RWByteAddressBuffer output : register(u1);
[RootSignature("DescriptorTable(SRV(t0)), UAV(u1)")]
[numthreads(32, 1, 1)]
void main(uint i : SV_GroupIndex) {
    output.Store4(16 * i, asuint(source.Load(int3(27 * i, 15 * i, 0))));
}
"#;

// RGB9E5 has three nine-bit unsigned mantissas and one biased exponent.
fn decode(word: u32) -> [u32; 4] {
    let scale = 2.0_f32.powi((word >> 27) as i32 - 24);
    let [r, g, b] = [0, 9, 18].map(|shift| (((word >> shift) & 511u32) as f32 * scale).to_bits());
    [r, g, b, 1.0_f32.to_bits()]
}

#[test]
#[ignore = "requires a D3D12 adapter supporting direct R32_UINT/RGB9E5 casting"]
fn packed_lighting_uav_write_is_visible_to_a_same_list_rgb9e5_srv() {
    let host = crate::host::host().expect("selected D3D12 adapter");
    let write = compile_cs(WRITE_PACKED);
    let read = compile_cs(READ_RGB);
    let slots = Slots::new(2);
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let resource_id = ID3D12Resource::IID;
    let iid = guid(
        resource_id.data1,
        resource_id.data2,
        resource_id.data3,
        resource_id.data4,
    );
    let uav_flag = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let rtv_flag = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    for requested in [DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R32_UINT] {
        for flags in [uav_flag, uav_flag | rtv_flag] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 853,
                Height: 480,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: requested,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let mut title = std::ptr::null();
            assert_eq!(
                crate::resource::create_committed(
                    std::ptr::null(),
                    std::ptr::from_ref(&properties).cast(),
                    0,
                    std::ptr::from_ref(&desc).cast(),
                    0,
                    std::ptr::null(),
                    &iid,
                    &mut title
                ),
                S_OK
            );
            let backing = crate::resource::host_resource(title).unwrap();
            let mut uav = [0; 10];
            uav[0] = DXGI_FORMAT_R32_UINT.0 as u32;
            uav[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
            let uav = View::Unordered(title as usize, 0, Some(uav));
            let mut srv = [0; 10];
            srv[0] = DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0 as u32;
            srv[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
            srv[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv[5] = 1;
            let srv = View::Shader(title as usize, Some(srv));
            let _ = host.log_messages();
            assert_eq!(slots.make(0, &uav), backing, "direct R32_UINT UAV");
            assert_eq!(slots.make(1, &srv), backing, "direct RGB9E5 SRV");
            let (out, data) = host.shared_buffer((TEXELS * 16) as u64).unwrap();
            // SAFETY: this new mapped output holds exactly one float4 per invocation.
            unsafe { data.write_bytes(0xcd, TEXELS * 16) };
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Uav), &uav, false);
            run(&list, &slots, &write, 0, None, (1, 1));
            readable(&list, &used(ResourceType::Srv), &srv, false);
            run(&list, &slots, &read, 1, Some(&out), (1, 1));
            // SAFETY: both dispatches are recorded and the list has one owner.
            unsafe { list.list.Close() }.unwrap();
            assert_eq!(host.log_messages(), None);
            host.execute(&[&list]);
            for i in 0..TEXELS as u32 {
                let word = (13 + 7 * i)
                    | ((101 + 9 * i) << 9)
                    | ((501 - 11 * i) << 18)
                    | ((12 + i % 9) << 27);
                // SAFETY: this texel is in the mapped output and execution has finished.
                let actual = unsafe {
                    data.add(i as usize * 16)
                        .cast::<[u32; 4]>()
                        .read_unaligned()
                };
                assert_eq!(
                    actual,
                    decode(word),
                    "format {requested:?}, flags {flags:?}, sample {i}, packed {word:#010x}"
                );
            }
            eprintln!(
                "RGB9E5 format={} flags={} pixels={TEXELS}/{TEXELS} components=128/128",
                requested.0, flags.0
            );
            assert_eq!(
                slot::<extern "system" fn(This) -> u32>(title, 0x10)(title),
                0
            );
        }
    }
}
