use super::*;
use crate::object::{S_OK, slot};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, Interface};

struct TestObject(This);

impl Drop for TestObject {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

fn iid(g: GUID) -> Iid {
    guid(g.data1, g.data2, g.data3, g.data4)
}

fn submit(
    host: &crate::host::Host,
    allocator: This,
    record: impl FnOnce(This, &crate::host::HostList),
) {
    let null = std::ptr::null();
    let mut list = null;
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list
        ),
        S_OK
    );
    let list = TestObject(list);
    let commands = crate::list::host_list(list.0).expect("host command list");
    // Exclude diagnostics already queued during global host or list initialization.
    let _ = host.log_messages();
    record(list.0, commands);
    // SAFETY: this test is the sole recorder and submits only successfully closed lists.
    let closed = unsafe { commands.list.Close() };
    let diagnostic = host.log_messages();
    assert!(
        closed.is_ok() && diagnostic.is_none(),
        "Close={closed:?}, D3D12={diagnostic:?}"
    );
    host.execute(&[commands]);
}

/// The title creates a timestamp *heap* and ends timestamp *queries*. Resolve
/// the same two slots both ways so scheduling does not affect the reference.
#[test]
#[ignore = "requires VICTORY_D3D12_DEBUG=1 and the D3D12 debug layer"]
fn timestamp_heap_resolves_the_same_slots_as_native_d3d12() {
    let host = crate::host::host().expect("D3D12 adapter");
    assert_eq!(std::env::var("VICTORY_D3D12_DEBUG").as_deref(), Ok("1"));
    let _info = host
        .device
        .cast::<ID3D12InfoQueue>()
        .expect("D3D12 debug layer");
    let null = std::ptr::null();
    let desc = Desc {
        kind: D3D12_QUERY_HEAP_TYPE_TIMESTAMP.0 as u32,
        count: 4,
        nodes: 0,
    };
    let mut heap = null;
    assert_eq!(create(null, &desc, &HEAP_IIDS[0], &mut heap), S_OK);
    let heap = TestObject(heap);
    let native_heap = host_heap(heap.0).expect("backing GPU query heap");
    let mut allocator = null;
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    let allocator = TestObject(allocator);
    for index in [1, 2] {
        submit(host, allocator.0, |list, _| {
            let end_query = slot::<extern "system" fn(This, This, u32, u32)>(list, 0xe8);
            end_query(list, heap.0, D3D12_QUERY_TYPE_TIMESTAMP.0 as u32, index);
        });
        if index == 1 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    let result = TestObject(crate::resource::test_buffer(32));
    let (ptr, _) = crate::resource::buffer_bytes(result.0).unwrap();
    let guard = 0xdead_beef_dead_beef;
    // SAFETY: this newly allocated title buffer has room for all four u64s.
    unsafe { ptr.cast::<[u64; 4]>().write_unaligned([guard; 4]) };
    let native = host.readback_buffer(16).expect("native query readback");
    submit(host, allocator.0, |list, commands| {
        let resolve = slot::<extern "system" fn(This, This, u32, u32, u32, This, u64)>(list, 0xf0);
        resolve(
            list,
            heap.0,
            D3D12_QUERY_TYPE_TIMESTAMP.0 as u32,
            1,
            2,
            result.0,
            8,
        );
        // SAFETY: two finished timestamp slots fit in the 16-byte readback buffer.
        unsafe {
            commands.list.ResolveQueryData(
                &native_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                1,
                2,
                &native,
                0,
            );
        }
    });
    let mut mapped = std::ptr::null_mut();
    // SAFETY: submission is complete; both timestamp slots are available to read.
    let (actual, raw) = unsafe {
        native
            .Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 16 }),
                Some(&mut mapped),
            )
            .unwrap();
        let raw = mapped.cast::<[u64; 2]>().read_unaligned();
        native.Unmap(0, None);
        (ptr.cast::<[u64; 4]>().read_unaligned(), raw)
    };
    let hz = host.timestamp_frequency();
    assert!(hz > 0 && raw[1] > raw[0]);
    let expected = raw.map(|ticks| ((u128::from(ticks) * 100_000_000) / u128::from(hz)) as u64);
    eprintln!("QUERY_READBACK host_hz={hz} native={raw:?} guest={actual:?} expected={expected:?}");
    assert_eq!(actual, [guard, expected[0], expected[1], guard]);
}
