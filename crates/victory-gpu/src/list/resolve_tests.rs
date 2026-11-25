use super::*;
use std::ptr::null;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{IUnknown, Interface};

// Each guest call returns one COM reference. Keep it through the readback,
// including when a failed assertion unwinds the test.
fn create(call: impl FnOnce(*mut This) -> HRESULT) -> IUnknown {
    let mut out = null();
    assert_eq!(call(&mut out), S_OK);
    // SAFETY: the successful call returned a new reference to a guest object.
    unsafe { IUnknown::from_raw(out.cast_mut().cast()) }
}

fn depth_texture(size: u32, samples: u32) -> IUnknown {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size.into(),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT(118),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: samples,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    // The guest heap and resource descriptions have the D3D12 ABI.
    create(|out| {
        crate::resource::create_committed(
            null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null(),
            &PAGEABLE_IIDS[0],
            out,
        )
    })
}

fn check_depth_resolve(size: u32) {
    let Some(host) = crate::host::host() else {
        eprintln!("skipping format-118 resolve (no host D3D12 device)");
        return;
    };
    let allocator = create(|out| create_allocator(null(), 0, &ALLOCATOR_IIDS[0], out));
    let title = create(|out| {
        create_list(
            null(),
            0,
            0,
            allocator.as_raw().cast(),
            null(),
            &LIST_IIDS[1],
            out,
        )
    });
    let src = depth_texture(4, 4);
    let dst = depth_texture(size, 1);
    let (src, dst, title) = (
        src.as_raw().cast(),
        dst.as_raw().cast(),
        title.as_raw().cast(),
    );
    let list = host_list(title).expect("host command list");

    // Empty and full depth occupy two columns each. The single-sample
    // destination starts at 1/3 (0x555555 in D24), distinct from either half.
    for (resource, dimension, initial) in [
        (src, D3D12_DSV_DIMENSION_TEXTURE2DMS, 1.0),
        (dst, D3D12_DSV_DIMENSION_TEXTURE2D, 1.0 / 3.0),
    ] {
        ensure_state(list, resource, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
        let texture = crate::resource::host_resource(resource).expect("host depth texture");
        let view = host.dsv(
            texture,
            Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT_D24_UNORM_S8_UINT,
                ViewDimension: dimension,
                ..Default::default()
            }),
        );
        // SAFETY: live textures in DEPTH_WRITE, with a matching DSV and an in-bounds rectangle.
        unsafe {
            list.list
                .ClearDepthStencilView(view, D3D12_CLEAR_FLAG_DEPTH, initial, 0, None);
            if resource == src {
                let left = RECT {
                    left: 0,
                    top: 0,
                    right: 2,
                    bottom: 4,
                };
                list.list.ClearDepthStencilView(
                    view,
                    D3D12_CLEAR_FLAG_DEPTH,
                    0.0,
                    0,
                    Some(&[left]),
                );
            }
        }
    }

    resolve(title, dst, 0, src, 0, 118);
    ensure_state(list, dst, COPY_SOURCE);
    let result = crate::resource::host_resource(dst).unwrap();
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut bytes = 0;
    // SAFETY: one output footprint for the live destination's depth plane.
    unsafe {
        host.device.GetCopyableFootprints(
            &result.GetDesc(),
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut bytes),
        );
    }
    let (buffer, mapped) = host.shared_buffer(bytes).unwrap();
    // SAFETY: both resources outlive submission. The copy locations borrow
    // their pointers and the footprint describes the complete depth plane.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(result),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        list.list
            .Close()
            .expect("record a format-118 depth resolve");
    }
    host.execute(&[list]);
    for y in 0..size {
        for x in 0..size {
            let expected = if x >= 4 || y >= 4 {
                0x55_5555
            } else if x < 2 {
                0
            } else {
                0xff_ffff
            };
            let offset = footprint.Offset as usize
                + y as usize * footprint.Footprint.RowPitch as usize
                + x as usize * 4;
            // SAFETY: the completed copy fills size × size D24 words at the device's row pitch.
            let depth = unsafe { mapped.add(offset).cast::<u32>().read_unaligned() } & 0xff_ffff;
            assert_eq!(
                depth, expected,
                "depth at ({x}, {y}) in {size}x{size} destination"
            );
        }
    }
    eprintln!(
        "format-118 readback: 16 depth pixels and {} border pixels matched",
        size * size - 16
    );
}

#[test]
fn format_118_resolves_the_whole_depth_plane() {
    check_depth_resolve(4);
}

#[test]
fn format_118_resolves_the_overlapping_depth_region() {
    check_depth_resolve(6);
}
