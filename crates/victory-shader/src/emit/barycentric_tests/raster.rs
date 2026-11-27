//! Raster readback for the two center interpolation families.

use super::{LINEAR_CENTER, PERSP_CENTER, element, pixel, validate};
use crate::emit::{
    bitcode::{Attr, Term},
    host::{Draw, Host, Root},
    shader::Shader,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3, D3D12_SHADER_VISIBILITY_PIXEL,
};

use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

/// Forward clip position. The pixel fixture also declares an unused TEXCOORD0 at row 0.
fn vertex_shader() -> Vec<u8> {
    let mut s = Shader::new(false);
    let b = s.entry;
    let mut input = element("POSITION", 0, 0);
    input.usage = 15;
    s.inputs.push(input);
    for (row, (name, kind)) in [("TEXCOORD", 0), ("SV_Position", 3)]
        .into_iter()
        .enumerate()
    {
        let mut output = element(name, kind, row as u32);
        output.usage = 15;
        s.outputs.push(output);
    }
    for col in 0..4 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
        let value = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        for id in 0..2 {
            let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), value];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
    }
    s.m.terminate(b, Term::Ret);
    s.finish()
}

/// Compile the reference from HLSL, independently of VIctory's DXIL writer.
fn compile_reference(entry: &str) -> Vec<u8> {
    use std::{ffi::OsString, path::PathBuf, process::Command};
    let dxc = std::env::var_os("VICTORY_DXC").unwrap_or_else(|| {
        super::super::validator::Validator::default_path()
            .map(|p| p.with_file_name("dxc.exe"))
            .filter(|p| p.is_file())
            .map_or_else(|| OsString::from("dxc.exe"), PathBuf::into_os_string)
    });
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/emit/barycentric_tests/reference.hlsl");
    let output = std::env::temp_dir().join(format!(
        "victory-barycentric-{}-{entry}.dxil",
        std::process::id()
    ));
    let result = Command::new(dxc)
        .args(["-T", "ps_6_6", "-E", entry, "-Fo"])
        .arg(&output)
        .arg(source)
        .output()
        .expect("Windows SDK dxc.exe (or set VICTORY_DXC)");
    assert!(
        result.status.success(),
        "DXC {entry}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(&output).expect("read reference DXIL");
    std::fs::remove_file(output).expect("remove reference DXIL");
    bytes
}

#[derive(Clone, Copy)]
enum Mode {
    Both,
    Perspective,
    Affine,
}

const EPSILON: f32 = 1e-4;

/// Read only pixels strictly inside the triangle. No pixel center is on the diagonal.
fn measure(
    texels: &[[f32; 4]],
    wb: f32,
    wc: f32,
    mode: Mode,
    label: &str,
) -> (Vec<[f32; 4]>, [usize; 4]) {
    assert_eq!(texels.len(), 32);
    let mut interior = Vec::new();
    let mut channel_misses = [0; 4];
    let mut wrong_pixels = 0;
    let mut max_error = 0.0f32;
    for (p, actual) in texels.iter().enumerate() {
        let (x, y) = (p % 8, p / 8);
        let b = (x as f32 + 0.5) / 8.0;
        let c = (y as f32 + 0.5) / 4.0;
        let a = 1.0 - b - c;
        if a < 0.0 {
            continue;
        }
        let inv_w = a + b / wb + c / wc;
        let (pi, pj) = (b / wb / inv_w, c / wc / inv_w);
        let expected = match mode {
            Mode::Both => [pi, pj, b, c],
            Mode::Perspective => [pi, pj, pi, pj],
            Mode::Affine => [b, c, b, c],
        };
        let mut wrong = false;
        for (ch, (v, r)) in actual.iter().zip(expected).enumerate() {
            let error = (v - r).abs();
            if !error.is_finite() || error > EPSILON {
                channel_misses[ch] += 1;
                wrong = true;
            }
            max_error = max_error.max(error);
        }
        wrong_pixels += usize::from(wrong);
        interior.push(*actual);
    }
    assert_eq!(interior.len(), 16);
    eprintln!(
        "{label}: {wrong_pixels}/16 pixels differ from the analytic weights, channel mismatches {channel_misses:?}/16, max error {max_error:.8}, pixel (2,1) {:?}",
        texels[10]
    );
    (interior, channel_misses)
}

#[test]
#[ignore = "D3D12 raster readback with an independent Windows SDK DXC reference"]
fn raw_center_weights_match_native_rasterization() {
    // exp mrt0, v0, v1, v2, v3 done vm; s_endpgm.
    let ps = pixel(
        &[0xF800_180F, 0x0302_0100, 0xBF81_0000],
        PERSP_CENTER | LINEAR_CENTER,
    )
    .expect("translate perspective and linear I/J");
    validate(&ps);
    let vs = vertex_shader();
    let programs = [
        ("guest dual", Mode::Both, ps.dxil.clone()),
        ("DXC dual", Mode::Both, compile_reference("dual")),
        (
            "DXC perspective",
            Mode::Perspective,
            compile_reference("perspective_only"),
        ),
        ("DXC affine", Mode::Affine, compile_reference("affine_only")),
    ];
    let forced_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut adapters = 0;
    for (label, warp) in [("hardware", false), ("WARP", true)] {
        if forced_warp && !warp {
            continue;
        }
        let host = match Host::new(warp) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{label}: {e}; 0 pixels checked");
                continue;
            }
        };
        let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the LUID is the selected device's and caps is the OPTIONS3 output.
        let desc = unsafe {
            host.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS3,
                    (&raw mut caps).cast(),
                    size_of_val(&caps) as u32,
                )
                .expect("D3D12 options 3");
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
                .and_then(|a| a.GetDesc1())
                .expect("selected adapter")
                .Description
        };
        let end = desc.iter().position(|&w| w == 0).unwrap_or(desc.len());
        let name = String::from_utf16_lossy(&desc[..end]);
        if !caps.BarycentricsSupported.as_bool() {
            eprintln!("{name}: barycentrics unsupported; 0 pixels checked");
            continue;
        }
        let root = host
            .root_signature_for(&[(&ps, D3D12_SHADER_VISIBILITY_PIXEL)])
            .unwrap();
        let psos: Vec<_> = programs
            .iter()
            .map(|(label, _, dxil)| {
                host.pipeline_with(&root, &vs, dxil)
                    .unwrap_or_else(|e| panic!("{name}, {label}: {e}"))
            })
            .collect();
        for (case, wb, wc) in [("equal w", 1.0, 1.0), ("unequal w", 2.0, 4.0)] {
            // Fixed on-screen corners (0,0), (8,0), (0,4); only clip w changes.
            let vertices = [-1.0, 1.0, 0.0, 1.0, wb, wb, 0.0, wb, -wc, -wc, 0.0, wc];
            let mut dual = Vec::new();
            for ((label, mode, _), pso) in programs.iter().zip(&psos) {
                let texels = host
                    .draw(
                        &root,
                        pso,
                        &Draw {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            textures: &[],
                            samplers: &[],
                            vertices: &vertices,
                            stride: 16,
                            size: (8, 4),
                            mesh_groups: 0,
                        },
                    )
                    .expect("read back 8x4 RGBA32F");
                let (covered, misses) =
                    measure(&texels, wb, wc, *mode, &format!("{name}, {case}, {label}"));
                if matches!(mode, Mode::Both) {
                    dual.push(covered);
                }
                if !matches!(mode, Mode::Both) || (wb == 1.0 && wc == 1.0) {
                    assert_eq!(misses, [0; 4], "{name}, {case}, {label}");
                }
            }
            let [guest, native] = dual.as_slice() else {
                panic!("each draw needs the translated and native dual-mode programs");
            };
            let mut bit_diffs = 0;
            let mut outside_tolerance = 0;
            for (g, n) in guest.iter().flatten().zip(native.iter().flatten()) {
                bit_diffs += usize::from(g.to_bits() != n.to_bits());
                let error = (g - n).abs();
                outside_tolerance += usize::from(!error.is_finite() || error > EPSILON);
            }
            eprintln!(
                "{name}, {case}: guest vs DXC dual {outside_tolerance}/64 outside tolerance, {bit_diffs}/64 bit differences"
            );
            assert_eq!(outside_tolerance, 0, "{name}, {case}: guest vs DXC");
        }
        adapters += 1;
    }
    assert!(
        adapters > 0,
        "0 barycentrics-capable devices ran the diagnostic"
    );
}
