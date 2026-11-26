use super::*;
use crate::host::HostList;

fn placed(
    buffer: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: all copy resources outlive submission; copy locations borrow them.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    }
}

fn subresource(texture: &ID3D12Resource, sub: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: all copy resources outlive submission; copy locations borrow them.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    }
}

fn move_to(list: &HostList, tex: &Arc<HostTexture>, state: u32) {
    if let Some(b) = list.move_texture(tex, state) {
        // SAFETY: the list is open, and the tracker knows the current state.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
}

fn byte(f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, x: u32, y: u32, size: u32) -> usize {
    f.Offset as usize + y as usize * f.Footprint.RowPitch as usize + x as usize * size as usize
}

/// Format 118's writable twin keeps 16-bit guest depth alongside an 8-bit
/// stencil; the host depth plane stores floats. Cover every 16-bit code,
/// odd widths across the 128-pixel dispatch boundary and non-zero footprints.
#[test]
fn depth_twin_preserves_every_u16_and_the_stencil_across_mips_and_slices() {
    let Some(host) = host() else {
        eprintln!("no D3D12 device; skipped");
        return;
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 259,
        Height: 257,
        DepthOrArraySize: 2,
        MipLevels: 3,
        Format: DXGI_FORMAT_R32G8X24_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: valid single-sample depth texture; the out pointer is ours.
    unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            None,
            &mut resource,
        )
    }
    .unwrap();
    let depth = Arc::new(HostTexture::new(resource.unwrap(), DEPTH_WRITE, Vec::new()));
    let twin = make(118, &depth.texture).expect("a D16/S8 writable twin");
    assert_eq!(twin.routes.iter().filter(|r| r.depth.is_some()).count(), 6);
    assert_eq!(twin.routes.len(), 12);
    // Every offset is a planned footprint inside the shared scratch allocation.
    // SAFETY: the scratch resource is live.
    let size = unsafe { twin.scratch.GetDesc().Width };
    let (source, map) = host.upload_buffer(size).unwrap();
    let (poison, poison_map) = host.upload_buffer(size).unwrap();
    let (forward, fmap) = host.shared_buffer(size).unwrap();
    let (back, bmap) = host.shared_buffer(size).unwrap();
    // SAFETY: two distinct mapped upload buffers of size bytes, before submission.
    let (source_bytes, poison_bytes) = unsafe {
        (
            std::slice::from_raw_parts_mut(map, size as usize),
            std::slice::from_raw_parts_mut(poison_map, size as usize),
        )
    };
    poison_bytes.fill(0xa5);
    source_bytes.fill(0x5a);
    let value = |s: u32, i: u32| (i as u16).wrapping_mul(40503).wrapping_add(17 * s as u16);
    let stencil = |s: u32, i: u32| (i as u8).wrapping_mul(37).wrapping_add(19 * s as u8);
    for r in &twin.routes {
        for y in 0..r.twin.Footprint.Height {
            for x in 0..r.twin.Footprint.Width {
                let i = y * r.twin.Footprint.Width + x;
                if r.plane == 0 {
                    let at = byte(r.twin, x, y, 2);
                    source_bytes[at..at + 2].copy_from_slice(&value(r.twin_sub, i).to_le_bytes());
                } else {
                    source_bytes[byte(r.twin, x, y, 1)] = stencil(r.twin_sub, i);
                }
            }
        }
    }
    let list = host.new_list().unwrap();
    let upload = |source: &ID3D12Resource| {
        for p in &twin.planes {
            move_to(&list, p, COPY_DEST);
        }
        for r in &twin.routes {
            // SAFETY: a full copy of the exact API-provided subresource footprint.
            unsafe {
                list.list.CopyTextureRegion(
                    &subresource(&twin.planes[r.plane].texture, r.twin_sub),
                    0,
                    0,
                    0,
                    &placed(source, r.twin),
                    None,
                );
            }
        }
    };
    upload(&source);
    assert!(twin.copy(&list.list, &depth, true, &mut |t, w| {
        list.move_texture(t, w)
    }));
    move_to(&list, &depth, COPY_SOURCE);
    for r in &twin.routes {
        // SAFETY: full depth/stencil-plane copy; each buffer uses its matching footprint.
        unsafe {
            list.list.CopyTextureRegion(
                &placed(&forward, r.depth.unwrap_or(r.twin)),
                0,
                0,
                0,
                &subresource(&depth.texture, r.depth_sub),
                None,
            );
        }
    }
    // Make a no-op on the way back observable.
    upload(&poison);
    assert!(twin.copy(&list.list, &depth, false, &mut |t, w| {
        list.move_texture(t, w)
    }));
    for p in &twin.planes {
        move_to(&list, p, COPY_SOURCE);
    }
    for r in &twin.routes {
        // SAFETY: full texture-plane copy to its matching footprint.
        unsafe {
            list.list.CopyTextureRegion(
                &placed(&back, r.twin),
                0,
                0,
                0,
                &subresource(&twin.planes[r.plane].texture, r.twin_sub),
                None,
            );
        }
    }
    // SAFETY: all resources remain alive until synchronous execute returns.
    unsafe { list.list.Close() }.unwrap();
    host.execute(&[&list]);
    // SAFETY: completed GPU copy to both size-byte mapped readback buffers.
    let (forward_bytes, back_bytes) = unsafe {
        (
            std::slice::from_raw_parts(fmap, size as usize),
            std::slice::from_raw_parts(bmap, size as usize),
        )
    };
    let mut counts = [0usize; 2];
    for r in &twin.routes {
        for y in 0..r.twin.Footprint.Height {
            for x in 0..r.twin.Footprint.Width {
                let i = y * r.twin.Footprint.Width + x;
                if let Some(d) = r.depth {
                    let at = byte(d, x, y, 4);
                    let f = u32::from_le_bytes(forward_bytes[at..at + 4].try_into().unwrap());
                    let u = value(r.twin_sub, i);
                    let expected = (f32::from(u) * f32::from_bits(0x3780_0080)).to_bits();
                    assert_eq!(f, expected, "depth sub {} ({x},{y})", r.twin_sub);
                    let at = byte(r.twin, x, y, 2);
                    assert_eq!(
                        back_bytes[at..at + 2],
                        u.to_le_bytes(),
                        "twin sub {} ({x},{y})",
                        r.twin_sub
                    );
                } else {
                    let at = byte(r.twin, x, y, 1);
                    let s = stencil(r.twin_sub, i);
                    assert_eq!(
                        (forward_bytes[at], back_bytes[at]),
                        (s, s),
                        "stencil sub {} ({x},{y})",
                        r.twin_sub
                    );
                }
                counts[r.plane] += 1;
            }
        }
    }
    eprintln!(
        "{} depth and {} stencil values matched in both directions",
        counts[0], counts[1]
    );
}
