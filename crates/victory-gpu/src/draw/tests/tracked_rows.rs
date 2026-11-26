use super::*;
use crate::watch::PAGE;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

/// Two adjacent live allocations, ordinary committed memory followed by a
/// write-watched pool.
fn mixed_pages() -> usize {
    let flags = MEM_RESERVE | MEM_COMMIT;
    for _ in 0..16 {
        // SAFETY: ask Windows for a candidate address. Another allocator
        // may win it after release, so check both following reservations.
        unsafe {
            let slot = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE);
            assert!(!slot.is_null());
            assert_ne!(VirtualFree(slot, 0, MEM_RELEASE), 0);
            let ordinary = VirtualAlloc(slot, PAGE, flags, PAGE_READWRITE);
            if ordinary.is_null() {
                continue;
            }
            let pool = (ordinary as usize + PAGE) as *mut std::ffi::c_void;
            if !VirtualAlloc(pool, 2 * PAGE, flags | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE).is_null() {
                return ordinary as usize;
            }
            assert_ne!(VirtualFree(ordinary, 0, MEM_RELEASE), 0);
        }
    }
    panic!("could not allocate adjacent watched and ordinary pages");
}

fn check_rows(address: usize, mixed: bool) {
    let host = crate::host::host().unwrap();
    const WIDTH: usize = 64;
    const ROWS_PER_PAGE: usize = PAGE / (WIDTH * 4);
    const HEIGHT: usize = 3 * ROWS_PER_PAGE;
    let fill = |page: usize, value: u8| {
        let at = address + page * PAGE;
        let _writing = crate::watch::writing(at, PAGE);
        // SAFETY: each of the three pages is a live allocation owned by this test.
        unsafe { std::ptr::write_bytes(at as *mut u8, value, PAGE) };
    };
    let object = texture_at_address(address as u64, &texture_words(WIDTH as u32, HEIGHT as u32, 28, 0x100, 0));
    let (_, desc) = crate::resource::texture_memory(object).unwrap();
    let layout = crate::resource::texture_layout(&desc).unwrap();
    let layout = &layout.planes[0].layout;
    assert_eq!(layout.offset(0, 0, ROWS_PER_PAGE as u32, 0).unwrap(), PAGE as u64);
    assert_eq!(layout.size, (3 * PAGE) as u64);
    let copy = crate::resource::host_copy(object).unwrap();
    let staging = crate::texture::staging(object, &copy.texture).unwrap();
    let ordinary = crate::texture::tests::untracked_bytes(&staging);
    if mixed {
        assert_eq!(ordinary, [(address, address + PAGE)]);
    } else {
        assert!(ordinary.is_empty());
    }
    for page in 0..3 {
        fill(page, 0x33);
    }
    let read = compile_cs(READ);
    let slots = Slots::new(1);
    let mut srv = [0; 10];
    srv[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
    let view = View::Shader(object as usize, Some(srv));
    slots.make(0, &view);
    let (output, mapped) = host.shared_buffer((WIDTH * HEIGHT * 16) as u64).unwrap();
    let submit = |expected: [u8; 3], copies| {
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Srv), &view, false);
        run(&list, &slots, &read, 0, Some(&output), (8, HEIGHT as u32 / 8));
        list.close();
        let before = crate::texture::copied(object);
        host.execute(&[&list]);
        // SAFETY: the completed dispatch wrote WIDTH * HEIGHT float4s.
        let pixels = unsafe { std::slice::from_raw_parts(mapped.cast::<[f32; 4]>(), WIDTH * HEIGHT) };
        for (page, group) in pixels.chunks(WIDTH * ROWS_PER_PAGE).enumerate() {
            let want = f32::from(expected[page]) / 255.0;
            let wrong = group.iter().filter(|p| !p.iter().all(|&v| (v - want).abs() <= 1e-6)).count();
            assert_eq!(wrong, 0, "page {page}, expected {expected:02x?}");
        }
        assert_eq!(crate::texture::copied(object) - before, copies);
    };
    submit([0x33; 3], 1);
    if mixed {
        fill(0, 0x99);
        submit([0x99, 0x33, 0x33], 1);
        submit([0x99, 0x33, 0x33], 0);
    } else {
        fill(0, 0x66);
        crate::texture::tests::defer_one_upload(&staging);
        submit([0x66, 0x33, 0x33], 1);
        fill(0, 0x77);
        crate::texture::tests::defer_one_upload(&staging);
        fill(1, 0x88);
        submit([0x77, 0x88, 0x33], 1);
        submit([0x77, 0x88, 0x33], 0);
    }
    drop(staging);
    drop(copy);
    crate::object::release(object);
}

#[test]
fn mixed_pages_and_deferred_rows_reach_the_sampled_texture() {
    if crate::host::host().is_none() {
        return;
    }
    // Both stay allocated, as in the other draw tests: the tracked pages' guards\n    // outlive the textures, and a later test's pool at a freed address would\n    // never see its writes.
    let mixed = mixed_pages();
    let tracked = crate::watch::tests::pool(3);
    check_rows(mixed, true);
    check_rows(tracked, false);
}
