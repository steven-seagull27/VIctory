use super::*;

/// A native depth clear gives each mip and array slice its own pattern.
/// Read them through the guest list in reverse order, with spare pitch
/// in each row and no spare byte after the final row.
#[test]
fn depth_mips_and_array_slices_reach_their_guest_footprints() {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    let Some(host) = crate::host::host() else {
        return;
    };
    let null = std::ptr::null();
    let create = |desc: &ResourceDesc, state| {
        let mut resource = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                desc,
                state,
                null.cast(),
                &RESOURCE_IIDS[0],
                &mut resource
            ),
            S_OK
        );
        resource
    };
    let depth_value = |sub: usize, high: bool| (2 * sub + if high { 17 } else { 1 }) as f32 / 32.0;
    let mut failures = Vec::new();
    for (resource_format, view_format, bytes) in
        [(40, 40, 4), (39, 40, 4), (55, 55, 2), (53, 55, 2)]
    {
        const DEPTH_WRITE: u32 = 0x10;
        let desc = ResourceDesc {
            flags: 2,
            depth_or_array_size: 2,
            ..texture(131, 9, resource_format, 3)
        };
        let image = create(&desc, DEPTH_WRITE);
        let native = host_resource(image).unwrap();
        let (queue, allocator, list) = texture_copy_list();
        let recording = crate::list::host_list(list).unwrap();
        crate::list::ensure_state(recording, image, DEPTH_WRITE);
        let mut footprints = Vec::new();
        let mut end = 512u64;
        for sub in 0..6 {
            let mip = sub as u32 % 3;
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT(view_format),
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: sub as u32 / 3,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let handle = host.dsv(&native, Some(&view));
            let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            // SAFETY: the live texture has six subresources; the output holds one footprint.
            unsafe {
                host.device.GetCopyableFootprints(
                    &native.GetDesc(),
                    sub as u32,
                    1,
                    0,
                    Some(&mut layout),
                    None,
                    None,
                    None,
                )
            };
            let fp = layout.Footprint;
            let rect = RECT {
                left: (fp.Width / 2) as i32,
                top: (fp.Height / 2) as i32,
                right: fp.Width as i32,
                bottom: fp.Height as i32,
            };
            // SAFETY: this DSV names one live mip/slice in DEPTH_WRITE; rect is inside it.
            unsafe {
                recording.list.ClearDepthStencilView(
                    handle,
                    D3D12_CLEAR_FLAG_DEPTH,
                    depth_value(sub, false),
                    0,
                    None,
                );
                recording.list.ClearDepthStencilView(
                    handle,
                    D3D12_CLEAR_FLAG_DEPTH,
                    depth_value(sub, true),
                    0,
                    Some(&[rect]),
                );
            }
            let footprint = PlacedFootprint {
                offset: end.next_multiple_of(512),
                format: fp.Format.0 as u32,
                width: fp.Width,
                height: fp.Height,
                depth: 1,
                row_pitch: fp.RowPitch + 256,
            };
            end = footprint.offset
                + u64::from(footprint.height - 1) * u64::from(footprint.row_pitch)
                + u64::from(footprint.width) * bytes as u64;
            footprints.push(footprint);
        }
        let target = create(&buffer(end), 0);
        let (out, len) = buffer_bytes(target).unwrap();
        assert_eq!(len, end);
        let mut want = vec![0xEE; len as usize];
        // SAFETY: the live guest buffer spans exactly len bytes.
        unsafe { std::ptr::write_bytes(out, 0xEE, len as usize) };
        type Copy = extern "system" fn(
            This,
            *const TextureLocation,
            u32,
            u32,
            u32,
            *const TextureLocation,
            *const u8,
        );
        let copy = slot::<Copy>(list, 0x80);
        for (sub, &fp) in footprints.iter().enumerate().rev() {
            let source = TextureLocation {
                resource: image,
                kind: 0,
                footprint: PlacedFootprint {
                    offset: sub as u64,
                    ..fp
                },
            };
            let into = TextureLocation {
                resource: target,
                kind: 1,
                footprint: fp,
            };
            copy(list, &into, 0, 0, 0, &source, null.cast());
            for y in 0..fp.height {
                for x in 0..fp.width {
                    let at = fp.offset as usize
                        + y as usize * fp.row_pitch as usize
                        + x as usize * bytes;
                    let depth = depth_value(sub, x >= fp.width / 2 && y >= fp.height / 2);
                    let bits = if bytes == 4 {
                        depth.to_bits()
                    } else {
                        (depth * 65535.0).round() as u32
                    };
                    want[at..at + bytes].copy_from_slice(&bits.to_le_bytes()[..bytes]);
                }
            }
        }
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        // SAFETY: the synchronous submission completed; the guest buffer remains live.
        let read = unsafe { std::slice::from_raw_parts(out, len as usize) };
        if let Some(at) = read
            .iter()
            .zip(&want)
            .position(|(actual, expected)| actual != expected)
        {
            let sub = footprints.iter().rposition(|fp| fp.offset <= at as u64);
            failures.push((resource_format, sub, at, read[at], want[at]));
        }
        for object in [list, allocator, queue, image, target] {
            slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
        }
    }
    assert!(
        failures.is_empty(),
        "(format, subresource, byte, actual, expected): {failures:?}"
    );
}
