use super::*;
use crate::object::{S_OK, guid, slot};
use windows::core::Interface;

struct TestObject(This);

impl Drop for TestObject {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

#[test]
fn structured_clears_preserve_neighbor_words_in_the_shared_buffer() {
    let Some(host) = crate::host::host() else {
        eprintln!("no D3D12 adapter; skipped");
        return;
    };
    let null = std::ptr::null();
    let mut heap = null;
    let heap_desc = [0u32, 1, 1, 0, 0, 0, 0, 0];
    let g = ID3D12DescriptorHeap::IID;
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            heap_desc.as_ptr().cast(),
            &guid(g.data1, g.data2, g.data3, g.data4),
            &mut heap
        ),
        S_OK
    );
    let heap = TestObject(heap);
    let handle = crate::descriptor::heap(heap.0).start;
    let mut failed = 0;
    for (first, bias) in [(4, 0), (3, 4), (2, 8), (1, 12)] {
        for float in [false, true] {
            for partial in [false, true] {
                let buffer = TestObject(crate::resource::test_buffer(128));
                let (data, _) = crate::resource::buffer_bytes(buffer.0).unwrap();
                let address = data as u64;
                let original: [u32; 32] = std::array::from_fn(|i| 0xa5a5_0000 + i as u32);
                // SAFETY: all 32 words are in the test buffer.
                unsafe { data.cast::<[u32; 32]>().write_unaligned(original) };
                let shadow = crate::shadow::find(address, 128).expect("a shared host copy");
                assert_eq!(shadow.offset(address + u64::from(first * 12)) % 16, bias);
                let view = [0, 1, first, 0, 6, 12, 0, 0, 0, 0];
                crate::descriptor::create_uav(null, buffer.0, null, &view, handle);
                // Seed all guard words on both sides, including words outside the view.
                let init = host.new_list().unwrap();
                shadow.refresh(&init, address, 128);
                init.close();
                host.execute(&[&init]);
                assert_eq!(shadow.read_host(address, 32), original);

                let mut state = State::default();
                state.set_heaps(&[heap.0]);
                let list = host.new_list().unwrap();
                // Zero has the same representation after any float-to-word conversion.
                // The guards remain nonzero so a missing float clear cannot pass.
                let bits = if float { 0.0f32.to_bits() } else { 0xcafe_1234 };
                let rect = [RECT {
                    left: 2,
                    top: 0,
                    right: 6,
                    bottom: 1,
                }];
                let rects = if partial { &rect[..] } else { &[] };
                state.clear_unordered(&list, handle, [bits, 0, 0, 0], float, rects);
                // SAFETY: this test is the sole recorder; submit only a valid list.
                unsafe { list.list.Close() }.expect("a legal clear must close");
                host.execute(&[&list]);
                let mut expected = original;
                let start = first as usize * 3;
                let cleared = if partial {
                    start + 2..start + 6
                } else {
                    start..start + 18
                };
                expected[cleared].fill(bits);
                let gpu = shadow.read_host(address, 32);
                // SAFETY: the submission and its copy-back finished.
                let title = unsafe { data.cast::<[u32; 32]>().read_unaligned() };
                let gpu_wrong = gpu.iter().zip(&expected).filter(|(a, b)| a != b).count();
                let title_wrong = title.iter().zip(&expected).filter(|(a, b)| a != b).count();
                eprintln!(
                    "CLEAR bias={bias} float={float} partial={partial} gpu_wrong={gpu_wrong} title_wrong={title_wrong}"
                );
                failed += gpu_wrong + title_wrong;
            }
        }
    }
    assert_eq!(failed, 0);
}
