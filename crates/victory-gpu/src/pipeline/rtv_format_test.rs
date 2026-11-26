//! Synthetic render-target format mismatch tests through the title entry points.

use super::*;
use crate::object::{S_OK, slot};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{GUID, Interface, PCSTR, s};

fn compile_hlsl(source: &[u8], target: PCSTR) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: live source buffer and output pointers.
    let res = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = res {
        if let Some(err) = errors {
            // SAFETY: error blob owns its bytes.
            let msg = unsafe {
                std::slice::from_raw_parts(err.GetBufferPointer().cast::<u8>(), err.GetBufferSize())
            };
            panic!("D3DCompile failed ({e}): {}", String::from_utf8_lossy(msg));
        }
        panic!("D3DCompile failed: {e}");
    }
    let code = code.unwrap();
    // SAFETY: blob owns its bytes.
    unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            .to_vec()
    }
}

fn synthetic_shader(stage: &'static str, hw: victory_shader::HwStage, dxil: &[u8]) -> Arc<Shader> {
    let s = shader(stage, dxil);
    s.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: hw,
                dxil: dxil.to_vec(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                memory: false,
                memory_writes: false,
                view_addresses: false,
                heap: false,
                heap_writes: false,
                view_marks: false,
            },
            used: Vec::new(),
        })
    });
    s
}

fn wait_pipeline(pso: This, targets: &[DXGI_FORMAT]) -> Arc<host::HostPipeline> {
    let started = std::time::Instant::now();
    loop {
        if let Some(p) = host_pipeline_for(pso, targets) {
            return p;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "timed out waiting for host pipeline variant"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn create_title_rt(format: DXGI_FORMAT, flags: u32) -> This {
    let iid = guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );
    // 2D 4x4 texture with requested resource flags (bit 0 = ALLOW_RENDER_TARGET).
    let desc: [u32; 14] = [
        3,
        0,
        0,
        0,
        4,
        0,
        4,
        1 | (1 << 16),
        format.0 as u32,
        1,
        0,
        0,
        flags,
        0,
    ];
    let props = [1u32, 0, 0, 0, 0];
    let mut out = std::ptr::null();
    let hr = crate::resource::create_committed(
        std::ptr::null(),
        props.as_ptr().cast(),
        0,
        desc.as_ptr().cast(),
        0x4, // RENDER_TARGET
        std::ptr::null(),
        &iid,
        &mut out,
    );
    assert_eq!(hr, S_OK);
    out
}

fn read_back_u32(host: &crate::host::Host, title_rt: This, format: DXGI_FORMAT) -> u32 {
    read_back_4x4(host, title_rt, format)[0]
}

fn read_back_4x4(host: &crate::host::Host, title_rt: This, format: DXGI_FORMAT) -> [u32; 16] {
    let tex = crate::resource::host_resource(title_rt).expect("host texture");
    // SAFETY: a live resource.
    let mut desc = unsafe { tex.GetDesc() };
    // Typeless host textures (e.g. sRGB with UAV) need a concrete footprint format.
    desc.Format = format;
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: single-subresource footprint query.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let (buf, mapped) = host.shared_buffer(total).expect("readback buffer");
    let list = host.new_list().expect("readback list");
    crate::list::ensure_state(&list, title_rt, 0x800); // COPY_SOURCE
    // SAFETY: live resources and open list; copies subresource 0.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&tex),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buf),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    list.close();
    host.execute(&[&list]);
    assert_eq!((desc.Width, desc.Height), (4, 4));
    std::array::from_fn(|i| {
        let at = footprint.Offset as usize
            + (i / 4) * footprint.Footprint.RowPitch as usize
            + 4 * (i % 4);
        // SAFETY: the copy is complete and this 32-bit texel is inside the mapped footprint.
        unsafe { mapped.add(at).cast::<u32>().read_unaligned() }
    })
}

fn drain_d3d12_errors(device: &ID3D12Device) -> Vec<(i32, String)> {
    let Ok(queue) = device.cast::<ID3D12InfoQueue>() else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    // SAFETY: reading stored messages from InfoQueue.
    unsafe {
        for i in 0..queue.GetNumStoredMessages() {
            let mut size = 0;
            if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                continue;
            }
            let mut buf = vec![0u64; size.div_ceil(8)];
            let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if queue.GetMessage(i, Some(msg), &mut size).is_ok() {
                let m = &*msg;
                if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    errors.push((
                        m.ID.0,
                        String::from_utf8_lossy(text)
                            .trim_end_matches('\0')
                            .to_owned(),
                    ));
                }
            }
        }
        queue.ClearStoredMessages();
    }
    errors
}

fn decode_r11g11b10_float(bits: u32) -> [f32; 3] {
    let unpack = |raw: u32, mantissa_bits: u32| -> f32 {
        let exp = (raw >> mantissa_bits) & 0x1f;
        let mant = raw & ((1 << mantissa_bits) - 1);
        if exp == 0 {
            (mant as f32) * 2f32.powi(-(14 + mantissa_bits as i32))
        } else {
            (1.0 + (mant as f32) / (1u32 << mantissa_bits) as f32) * 2f32.powi(exp as i32 - 15)
        }
    };
    [
        unpack(bits & 0x7ff, 6),
        unpack((bits >> 11) & 0x7ff, 6),
        unpack((bits >> 22) & 0x3ff, 5),
    ]
}

fn decode_r10g10b10a2_unorm(bits: u32) -> [f32; 4] {
    [
        (bits & 0x3ff) as f32 / 1023.0,
        ((bits >> 10) & 0x3ff) as f32 / 1023.0,
        ((bits >> 20) & 0x3ff) as f32 / 1023.0,
        ((bits >> 30) & 0x3) as f32 / 3.0,
    ]
}

fn create_graphics_stream_pso(
    vs: &Arc<Shader>,
    ps: &Arc<Shader>,
    rt_formats: &[DXGI_FORMAT],
) -> This {
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth_kind: u32,
        depth: D3D12_DEPTH_STENCIL_DESC,
        topology: [u32; 2],
        targets_kind: u32,
        targets: D3D12_RT_FORMAT_ARRAY,
    }
    let mut targets = D3D12_RT_FORMAT_ARRAY {
        NumRenderTargets: rt_formats.len() as u32,
        ..Default::default()
    };
    targets.RTFormats[..rt_formats.len()].copy_from_slice(rt_formats);
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        depth_kind: 11,
        depth: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            ..Default::default()
        },
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        targets_kind: 15,
        targets,
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let mut pso = std::ptr::null();
    assert_eq!(
        create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    pso
}

/// A graphics pipeline declared with `R10G10B10A2_UNORM` drawing into a bound
/// `R11G11B10_FLOAT` render target (and then switching back to
/// `R10G10B10A2_UNORM` without re-setting the title pipeline) lazily builds a
/// matching host pipeline variant, triggers no D3D12 Error 613, and writes the
/// expected pixels to both targets.
#[test]
fn mismatched_rtv_format_builds_variant_and_draws_cleanly() {
    check_rtv_variant(VariantStart::Prebuilt);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VariantStart {
    Prebuilt,
    FirstDraw,
    SpentBudget,
}

#[test]
#[ignore = "first-use timing: run alone in a fresh process"]
fn unbuilt_rtv_variant_first_draw() {
    check_rtv_variant(VariantStart::FirstDraw);
}

#[test]
#[ignore = "intentionally spends the frame's shared wait budget; run alone"]
fn unbuilt_rtv_variant_with_spent_frame_budget() {
    check_rtv_variant(VariantStart::SpentBudget);
}

fn check_rtv_variant(start: VariantStart) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let vs_bytes = compile_hlsl(
        b"float4 main(uint id : SV_VertexID) : SV_Position { return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.5, 1); }",
        s!("vs_5_1"),
    );
    let ps_bytes = compile_hlsl(
        b"float4 main() : SV_Target0 { return float4(0.5, 0.25, 0.75, 1.0); }",
        s!("ps_5_1"),
    );
    let vs = synthetic_shader("vertex", victory_shader::HwStage::Vs, &vs_bytes);
    let ps = synthetic_shader("pixel", victory_shader::HwStage::Ps, &ps_bytes);
    let pso = create_graphics_stream_pso(&vs, &ps, &[DXGI_FORMAT_R10G10B10A2_UNORM]);

    // Keep a ready matching pipeline as a control in every run.
    let base = wait_pipeline(pso, &[DXGI_FORMAT_R10G10B10A2_UNORM]);
    assert!(pipeline(pso).variants.lock().unwrap().is_empty());
    if start == VariantStart::Prebuilt {
        wait_pipeline(pso, &[DXGI_FORMAT_R11G11B10_FLOAT]);
    } else if start == VariantStart::SpentBudget {
        let pending = &pipeline(pso).host;
        let unqueued = pending.variant(pending.graphics.as_deref().unwrap().targets);
        make::spend_remaining_frame_budget_for_test(&unqueued);
    }

    let rt_r11 = create_title_rt(DXGI_FORMAT_R11G11B10_FLOAT, 1);
    let rt_retry = create_title_rt(DXGI_FORMAT_R11G11B10_FLOAT, 1);
    let rt_r10 = create_title_rt(DXGI_FORMAT_R10G10B10A2_UNORM, 1);

    let null = std::ptr::null();
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    let mut rtv_heap = null;
    let heap_desc: [u32; 8] = [2, 4, 0, 0, 0, 0, 0, 0]; // RTV heap, 4 descriptors
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut rtv_heap,
        ),
        S_OK
    );
    let rtv_start = crate::descriptor::heap(rtv_heap).start;
    let rtv_r11 = rtv_start;
    let rtv_r10 = rtv_start + 128;
    let rtv_retry = rtv_start + 256;
    crate::descriptor::create_rtv(null, rt_r11, std::ptr::null(), rtv_r11);
    crate::descriptor::create_rtv(null, rt_r10, std::ptr::null(), rtv_r10);
    crate::descriptor::create_rtv(null, rt_retry, std::ptr::null(), rtv_retry);

    let mut allocator = null;
    let mut list = null;
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list,
        ),
        S_OK
    );

    let _ = drain_d3d12_errors(&host.device);
    let set_targets =
        slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190);
    let set_pso = slot::<extern "system" fn(This, This)>(list, 0xc8);
    let set_topo = slot::<extern "system" fn(This, u32)>(list, 0xa0);
    let set_vp = slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8);
    let set_sc = slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0);
    let draw = slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60);
    let close = slot::<extern "system" fn(This) -> HRESULT>(list, 0x48);

    // Bind PSO once, draw to mismatched R11G11B10_FLOAT, then switch RTV to matching R10G10B10A2_UNORM.
    set_pso(list, pso);
    set_topo(list, 4); // triangle list
    set_vp(
        list,
        1,
        &D3D12_VIEWPORT {
            Width: 4.0,
            Height: 4.0,
            MaxDepth: 1.0,
            ..Default::default()
        },
    );
    set_sc(
        list,
        1,
        &RECT {
            right: 4,
            bottom: 4,
            ..Default::default()
        },
    );
    let clear =
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const RECT)>(list, 0x1a0);
    // Prime the regular draw machinery with the matching base PSO. Clear
    // this target again so the final switch back must really draw to pass.
    set_targets(list, 1, &rtv_r10, 0, std::ptr::null());
    draw(list, 3, 1, 0, 0);
    clear(list, rtv_r10, &[0.0; 4], 0, std::ptr::null());
    clear(list, rtv_r11, &[0.0; 4], 0, std::ptr::null());
    clear(list, rtv_retry, &[0.0; 4], 0, std::ptr::null());
    set_targets(list, 1, &rtv_r11, 0, std::ptr::null());
    if start != VariantStart::Prebuilt {
        assert!(pipeline(pso).variants.lock().unwrap().is_empty());
    }
    let began = std::time::Instant::now();
    draw(list, 3, 1, 0, 0);
    let first_us = began.elapsed().as_micros();
    // Wait only after the first real draw. A second target makes the retry
    // independent of the first target's contents.
    let variant = wait_pipeline(pso, &[DXGI_FORMAT_R11G11B10_FLOAT]);
    assert_ne!(base.pso.as_raw(), variant.pso.as_raw());
    assert!(Arc::ptr_eq(
        &variant,
        &wait_pipeline(pso, &[DXGI_FORMAT_R11G11B10_FLOAT])
    ));
    set_targets(list, 1, &rtv_retry, 0, std::ptr::null());
    let began = std::time::Instant::now();
    draw(list, 3, 1, 0, 0);
    let cached_us = began.elapsed().as_micros();
    set_targets(list, 1, &rtv_r10, 0, std::ptr::null());
    draw(list, 3, 1, 0, 0);
    assert_eq!(close(list), S_OK);

    let commands = crate::list::host_list(list).unwrap();
    host.execute(&[commands]);

    let errors = drain_d3d12_errors(&host.device);
    assert!(
        !errors
            .iter()
            .any(|(id, _)| *id == D3D12_MESSAGE_ID_RENDER_TARGET_FORMAT_MISMATCH_PIPELINE_STATE.0),
        "unexpected D3D12 Error 613: {errors:?}"
    );

    let first = read_back_4x4(host, rt_r11, DXGI_FORMAT_R11G11B10_FLOAT);
    let cached = read_back_4x4(host, rt_retry, DXGI_FORMAT_R11G11B10_FLOAT);
    let count = |words: &[u32]| {
        words
            .iter()
            .filter(|&&w| decode_r11g11b10_float(w) == [0.5, 0.25, 0.75])
            .count()
    };
    let (fresh, reused) = (count(&first), count(&cached));
    let black = first.iter().filter(|&&w| w == 0).count();
    eprintln!(
        "RTV {start:?}: first={fresh}/16 cached={reused}/16 black={black}/16 first_us={first_us} cached_us={cached_us}"
    );
    assert_eq!(reused, 16);
    assert!(fresh == 16 || (black == 16 && start != VariantStart::Prebuilt));
    let rgba10 =
        decode_r10g10b10a2_unorm(read_back_u32(host, rt_r10, DXGI_FORMAT_R10G10B10A2_UNORM));
    assert!(
        (rgba10[0] - 0.5).abs() < 0.01
            && (rgba10[1] - 0.25).abs() < 0.01
            && (rgba10[2] - 0.75).abs() < 0.01
            && (rgba10[3] - 1.0).abs() < 0.01,
        "R10G10B10A2_UNORM readback mismatch: {rgba10:?}"
    );

    for obj in [list, allocator, rtv_heap, rt_r11, rt_retry, rt_r10, pso] {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(obj, 0x10)(obj), 0);
    }
}

/// Multi-RT draw where slot 0 differs (`R10G10B10A2_UNORM` -> `R11G11B10_FLOAT`,
/// explicit RTV desc) and slot 1 is a typeless-backed `R8G8B8A8_UNORM_SRGB`
/// target (`ALLOW_RENDER_TARGET | ALLOW_UNORDERED_ACCESS`, null RTV desc) bound
/// against a declared `R8G8B8A8_UNORM` slot: both slots get matching formats
/// and slot 1 writes sRGB-encoded bytes rather than linear values.
#[test]
fn multi_rt_and_srgb_mismatch_draws_both_targets_cleanly() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let vs_bytes = compile_hlsl(
        b"float4 main(uint id : SV_VertexID) : SV_Position { return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.5, 1); }",
        s!("vs_5_1"),
    );
    let ps_bytes = compile_hlsl(
        b"struct Out { float4 a : SV_Target0; float4 b : SV_Target1; }; Out main() { Out o; o.a = float4(0.5, 0.25, 0.75, 1.0); o.b = float4(0.214, 0.5, 0.0, 1.0); return o; }",
        s!("ps_5_1"),
    );
    let vs = synthetic_shader("vertex", victory_shader::HwStage::Vs, &vs_bytes);
    let ps = synthetic_shader("pixel", victory_shader::HwStage::Ps, &ps_bytes);
    let pso = create_graphics_stream_pso(
        &vs,
        &ps,
        &[DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM],
    );

    let _ = wait_pipeline(
        pso,
        &[DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB],
    );

    let rt0 = create_title_rt(DXGI_FORMAT_R11G11B10_FLOAT, 1);
    // ALLOW_RENDER_TARGET | ALLOW_UNORDERED_ACCESS forces typeless host backing for sRGB.
    let rt1 = create_title_rt(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 1 | 4);
    assert_eq!(
        // SAFETY: live host texture.
        unsafe { crate::resource::host_resource(rt1).unwrap().GetDesc() }.Format,
        DXGI_FORMAT_R8G8B8A8_TYPELESS
    );

    let null = std::ptr::null();
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    let mut rtv_heap = null;
    let heap_desc: [u32; 8] = [2, 4, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut rtv_heap,
        ),
        S_OK
    );
    let rtv_start = crate::descriptor::heap(rtv_heap).start;
    let rtv0 = rtv_start;
    let rtv1 = rtv_start + 128;
    // Explicit concrete RTV desc on rt0.
    let explicit_r11: [u32; 6] = [DXGI_FORMAT_R11G11B10_FLOAT.0 as u32, 4, 0, 0, 0, 0];
    crate::descriptor::create_rtv(null, rt0, &explicit_r11, rtv0);
    // Null RTV desc on typeless-backed sRGB rt1 (resolves via view_format to R8G8B8A8_UNORM_SRGB).
    crate::descriptor::create_rtv(null, rt1, std::ptr::null(), rtv1);

    let mut allocator = null;
    let mut list = null;
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list,
        ),
        S_OK
    );

    let _ = drain_d3d12_errors(&host.device);
    let set_targets =
        slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190);
    let set_pso = slot::<extern "system" fn(This, This)>(list, 0xc8);
    let set_topo = slot::<extern "system" fn(This, u32)>(list, 0xa0);
    let set_vp = slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8);
    let set_sc = slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0);
    let draw = slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60);
    let close = slot::<extern "system" fn(This) -> HRESULT>(list, 0x48);

    let handles = [rtv0, rtv1];
    set_targets(list, 2, handles.as_ptr(), 0, std::ptr::null());
    set_pso(list, pso);
    set_topo(list, 4);
    set_vp(
        list,
        1,
        &D3D12_VIEWPORT {
            Width: 4.0,
            Height: 4.0,
            MaxDepth: 1.0,
            ..Default::default()
        },
    );
    set_sc(
        list,
        1,
        &RECT {
            right: 4,
            bottom: 4,
            ..Default::default()
        },
    );
    draw(list, 3, 1, 0, 0);
    assert_eq!(close(list), S_OK);

    let commands = crate::list::host_list(list).unwrap();
    host.execute(&[commands]);

    let errors = drain_d3d12_errors(&host.device);
    assert!(
        !errors
            .iter()
            .any(|(id, _)| *id == D3D12_MESSAGE_ID_RENDER_TARGET_FORMAT_MISMATCH_PIPELINE_STATE.0),
        "unexpected D3D12 Error 613: {errors:?}"
    );

    let rgb11 = decode_r11g11b10_float(read_back_u32(host, rt0, DXGI_FORMAT_R11G11B10_FLOAT));
    assert!(
        (rgb11[0] - 0.5).abs() < 0.02
            && (rgb11[1] - 0.25).abs() < 0.02
            && (rgb11[2] - 0.75).abs() < 0.02,
        "slot 0 R11G11B10_FLOAT readback mismatch: {rgb11:?}"
    );
    // Linear 0.214 encodes to ~0.5 sRGB (~127..=129); linear 0.5 encodes to ~0.735 (~186..=189).
    let srgb = read_back_u32(host, rt1, DXGI_FORMAT_R8G8B8A8_UNORM).to_le_bytes();
    assert!(
        (126..=129).contains(&srgb[0])
            && (186..=189).contains(&srgb[1])
            && srgb[2] == 0
            && srgb[3] == 255,
        "slot 1 R8G8B8A8_UNORM_SRGB readback mismatch: {srgb:?}"
    );

    for obj in [list, allocator, rtv_heap, rt0, rt1, pso] {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(obj, 0x10)(obj), 0);
    }
}
