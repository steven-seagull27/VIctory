use super::*;
use crate::object::{S_OK, guid, slot};

/// Read the destination texture, including the slice omitted from the packed upload.
fn read_slices(
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    upload: &Upload,
    expected: [u8; 3],
) -> [usize; 3] {
    let host = host().unwrap();
    let list = host.new_list().unwrap();
    let (readback, data) = list.readback(host, staging.size).unwrap();
    // SAFETY: all resources remain alive until execute has completed. Copy locations
    // borrow these references, and the footprints are those reported by the device.
    unsafe {
        if let Some(b) = record_upload(&list, copy, staging, upload) {
            list.list.ResourceBarrier(&[b]);
        }
        if let Some(b) = list.move_texture(copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32) {
            list.list.ResourceBarrier(&[b]);
        }
        for (i, &footprint) in staging.footprints.iter().enumerate() {
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&copy.texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
    }
    list.close();
    host.execute(&[&list]);
    std::array::from_fn(|i| {
        let footprint = staging.footprints[i];
        let mut matched = 0;
        for y in 0..128 {
            for x in 0..128 {
                let at =
                    footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize + x * 4;
                // SAFETY: one RGBA8 texel inside the mapped readback, after the fence.
                let pixel = unsafe { data.add(at).cast::<[u8; 4]>().read() };
                matched += usize::from(pixel == [expected[i]; 4]);
            }
        }
        matched
    })
}

#[test]
fn packed_retry_preserves_old_and_new_writes_across_clean_slice() {
    let Some(host) = host() else {
        eprintln!("no host GPU; skipped");
        return;
    };
    const BYTES: usize = 128 * 128 * 4;
    let address = crate::watch::tests::pool(3);
    #[repr(align(8))]
    struct Words([u32; 14]);
    let desc = Words([
        3,
        0,
        0,
        0,
        128,
        0,
        128,
        3 | (1 << 16),
        28,
        1,
        0,
        0x100,
        0,
        0,
    ]);
    let iid = guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );
    let mut resource = std::ptr::null();
    assert_eq!(
        crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.0.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        ),
        S_OK
    );
    let copy = crate::resource::host_copy(resource).unwrap();
    let staging = staging(resource, &copy.texture).unwrap();
    assert!(staging.tracked.is_some());
    assert_eq!(staging.layout.size as usize, 3 * BYTES);
    let fill = |i, value| {
        let start = staging.layout.offset(0, 0, 0, i).unwrap() as usize;
        assert_eq!(start, i as usize * BYTES);
        // SAFETY: each linear 128x128 slice occupies one page in the test allocation.
        unsafe { std::ptr::write_bytes((address + start) as *mut u8, value, BYTES) };
    };

    for (i, value) in [0x11, 0x22, 0x33].into_iter().enumerate() {
        fill(i as u32, value);
    }
    let allocate = |bytes| {
        host.upload_buffer(bytes)
            .map(|(buffer, data)| (buffer, 0, data))
    };
    let first = staging.refresh_now(allocate).1.unwrap();
    assert_eq!(first.2, [0, 1, 2]);
    let all = [128 * 128; 3];
    assert_eq!(
        read_slices(&copy, &staging, &first, [0x11, 0x22, 0x33]),
        all
    );

    fill(2, 0xc3);
    let mut refused_bytes = 0;
    let (_, refused) = staging.refresh_now(|bytes| {
        refused_bytes = bytes;
        None
    });
    assert!(refused.is_none());
    // A different page changes while the last slice still awaits its retry.
    fill(0, 0xa1);
    let mut retry_bytes = 0;
    let (_, retry) = staging.refresh_now(|bytes| {
        retry_bytes = bytes;
        allocate(bytes)
    });
    let retry = retry.expect("the first slice was just written");
    let matched = read_slices(&copy, &staging, &retry, [0xa1, 0x22, 0xc3]);
    eprintln!(
        "refused {refused_bytes} bytes, retry {retry_bytes} bytes for {:?}, matched texels {matched:?}",
        retry.2
    );
    assert_eq!(matched, all);
    assert_eq!(retry.2, [0, 2]);
    assert_eq!(refused_bytes, BYTES as u64);
    assert_eq!(retry_bytes, (2 * BYTES) as u64);
    let (_, unchanged) = staging.refresh_now(|_| panic!("clean texture requested staging space"));
    assert!(unchanged.is_none());
    drop((staging, copy));
    assert_eq!(
        slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource),
        0
    );
    crate::watch::tests::free(address);
}
