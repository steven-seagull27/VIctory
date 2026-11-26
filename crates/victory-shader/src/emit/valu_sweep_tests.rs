//! Exploratory sweeps of synthetic shaders against exact references: every
//! GFX10 typed buffer format through `tbuffer_load_format_xyzw` and its d16
//! form, and the f16 ALU operations. Each runs on the hardware GPU and WARP
//! and prints every mismatch before failing; they are `#[ignore]`d because
//! they take a while (`cargo test -p victory-shader --lib sweep -- --ignored
//! --nocapture`). `SWEEP_NORM_ONLY=1` limits the format sweep to the UNORM and
//! SNORM formats. `SWEEP_D16=1` adds the d16 loads.

use super::super::HostShader;
use super::super::host::{Dispatch, Host, Root, Stride, View};
use super::{
    ONE_AT_A_TIME, build_valu_cs, f16_to_f32, is_f16_nan, lower_valu_cs, run_cases, vgpr, vop1,
    vop3,
};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Num {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// The GFX10 buffer formats by number: channel widths from the lowest bits
/// and the numeric format (LLVM `UfmtGFX10`; written out independently of
/// `lower::buffer_format`).
fn formats() -> Vec<(u8, String, &'static [u32], Num)> {
    use Num::*;
    const NORM: &[Num] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Num] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Num] = &[Uint, Sint, Float];
    let families: [(&str, &'static [u32], &[Num]); 14] = [
        ("8", &[8], NORM),
        ("16", &[16], ALL),
        ("8_8", &[8, 8], NORM),
        ("32", &[32], WIDE),
        ("16_16", &[16, 16], ALL),
        ("10_11_11", &[11, 11, 10], ALL),
        ("11_11_10", &[10, 11, 11], ALL),
        ("10_10_10_2", &[2, 10, 10, 10], NORM),
        ("2_10_10_10", &[10, 10, 10, 2], NORM),
        ("8_8_8_8", &[8, 8, 8, 8], NORM),
        ("32_32", &[32, 32], WIDE),
        ("16_16_16_16", &[16, 16, 16, 16], ALL),
        ("32_32_32", &[32, 32, 32], WIDE),
        ("32_32_32_32", &[32, 32, 32, 32], WIDE),
    ];
    let mut v = Vec::new();
    let mut n = 1u8;
    for (name, bits, nums) in families {
        for &num in nums {
            v.push((n, format!("{name}_{num:?}").to_uppercase(), bits, num));
            n += 1;
        }
    }
    v
}

/// One acceptable result: exact bits, or any NaN of the given width.
#[derive(Clone, Copy, Debug)]
enum Want {
    Bits(u32),
    NanF32,
}

/// An f64 rounded to the nearest-even f16 (65520 and up to infinity).
fn f16_rne(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a == 0.0 {
        return sign;
    }
    let e = (((a.to_bits() >> 52) & 0x7FF) as i32 - 1023).max(-14);
    let m = a / 2f64.powi(e - 10);
    let r = m.round_ties_even() as i32;
    sign | (((e + 15) << 10) + r - 1024) as u16
}

/// The f32 RDNA2 gives for channel `field` (`w` bits) of numeric format `num`,
/// as exact f64 (or bits for integers), per the GFX10 ISA's conversions.
enum Exact {
    Int(u32),
    Real(f64),
    F32Bits(u32),
}

fn exact(field: u32, w: u32, num: Num) -> Exact {
    let signed = if w == 32 { field as i32 } else { ((field << (32 - w)) as i32) >> (32 - w) };
    match num {
        Num::Unorm => Exact::Real(f64::from(field) / ((1u64 << w) - 1) as f64),
        Num::Snorm => {
            Exact::Real((f64::from(signed) / ((1u64 << (w - 1)) - 1) as f64).max(-1.0))
        }
        Num::Uscaled => Exact::Real(f64::from(field)),
        Num::Sscaled => Exact::Real(f64::from(signed)),
        Num::Uint => Exact::Int(field),
        Num::Sint => Exact::Int(signed as u32),
        Num::Float if w == 32 => Exact::F32Bits(field),
        Num::Float => {
            let h = if w == 16 { field } else { field << (15 - w) };
            Exact::F32Bits(f16_to_f32(h as u16).to_bits())
        }
    }
}

/// The channels of `format` read from `dwords`, then 0, 0, 0, 1.
fn channels(bits: &[u32], num: Num, dwords: [u32; 4]) -> Vec<Exact> {
    let mut out = Vec::new();
    let mut at = 0;
    for &w in bits {
        let word = dwords[(at / 32) as usize];
        let field = if w == 32 { word } else { (word >> (at % 32)) & ((1 << w) - 1) };
        out.push(exact(field, w, num));
        at += w;
    }
    let int = matches!(num, Num::Uint | Num::Sint);
    while out.len() < 4 {
        out.push(if out.len() == 3 {
            if int { Exact::Int(1) } else { Exact::Real(1.0) }
        } else if int {
            Exact::Int(0)
        } else {
            Exact::Real(0.0)
        });
    }
    out
}

/// What a 32-bit load gives for one channel.
fn want32(x: &Exact) -> Vec<Want> {
    match *x {
        Exact::Int(i) => vec![Want::Bits(i)],
        Exact::Real(r) => vec![Want::Bits((r as f32).to_bits())],
        Exact::F32Bits(b) if f32::from_bits(b).is_nan() => vec![Want::NanF32],
        Exact::F32Bits(b) => vec![Want::Bits(b)],
    }
}

/// RDNA2 D16 buffer loads saturate integer fields and round normalized
/// fields once to nearest. Floating-point fields narrow toward zero (#2064).
/// The sweep accepts any NaN payload as before.
fn want16(x: &Exact, num: Num) -> Option<u16> {
    match *x {
        Exact::Int(i) if num == Num::Sint => {
            Some((i as i32).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as u16)
        }
        Exact::Int(i) => Some(i.min(u32::from(u16::MAX)) as u16),
        Exact::Real(r) => Some(f16_rne(r)),
        Exact::F32Bits(b) => {
            let f = f64::from(f32::from_bits(b));
            let nearest = rne_or_nan(f)?;
            // The adjacent half toward zero has a magnitude one bit pattern smaller.
            Some(nearest - u16::from(f16v(nearest).abs() > f.abs()))
        }
    }
}

fn input_cases() -> Vec<[u32; 4]> {
    let patterns = [
        0x0000_0000, 0xFFFF_FFFF, 0x8000_0000, 0x7FFF_FFFF, 0x0000_0001, 0x8080_8080,
        0x7F7F_7F7F, 0x8181_8181, 0x0101_0101, 0xFFFF_0000, 0x0000_FFFF, 0x8000_7FFF,
        0x7FFF_8000, 0x8001_8001, 0x3C00_BC00, 0x7C00_FC00, 0x7E00_7C01, 0x0001_0401,
        0x3FF0_03FF, 0xC00F_FC00, 0x7FF0_03FF, 0x5555_5555, 0xAAAA_AAAA, 0x3F80_0000,
    ];
    let mut cases: Vec<[u32; 4]> = patterns.iter().map(|&p| [p, p, p, p]).collect();
    // Mixed dwords for the wide formats, then pseudo-random ones.
    for w in patterns.windows(4) {
        cases.push([w[0], w[1], w[2], w[3]]);
    }
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..24 {
        let mut c = [0u32; 4];
        for d in &mut c {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *d = (s >> 16) as u32;
        }
        cases.push(c);
    }
    cases
}

/// Each one-thread group loads its own 16-byte row from the input SRV and
/// writes all xyzw channels to the matching row in a separate raw UAV.
fn format_shader(format: u8, d16: bool) -> HostShader {
    let words = [
        vop1(0x001, 0, 1), // v_mov_b32 v0, s1 (group ID x)
        0x3400_0084,       // v_lshlrev_b32 v0, 4, v0 (byte offset)
        0xBE81_0380,       // s_mov_b32 s1, 0 (high half of table address)
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32
        0xE803_1000 | (u32::from(format) << 19),
        0x8002_0100 | (u32::from(d16) << 21), // tbuffer_load_format[_d16]_xyzw into v1..
        if d16 { 0xE074_1000 } else { 0xE078_1000 },
        0x8003_0100, // buffer_store_dwordx2 / x4 v1.., v0, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let range = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    lower_valu_cs(&NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![range(ResourceType::Srv, 0), range(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 7 | 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    })
    .expect("lower typed buffer sweep")
}

fn format_rows(shader: &HostShader, cases: &[[u32; 4]]) -> Vec<(&'static str, Vec<u32>)> {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut rows = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host
            .compute_pipeline(shader)
            .expect("typed buffer sweep pipeline");
        let count = cases.len() as u32;
        let mut readback = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                    )],
                    views: &[
                        View::Buffer {
                            data: cases.as_flattened(),
                            stride: Stride::Raw,
                        },
                        View::RawUav(4 * count),
                    ],
                    samplers: &[],
                    groups: [count, 1, 1],
                },
            )
            .expect("typed buffer sweep dispatch");
        let data = readback.remove(0);
        assert_eq!(data.len(), 4 * cases.len(), "{label} readback length");
        rows.push((label, data));
    }
    assert!(
        !rows.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    rows
}

/// Mismatches grouped by what failed: key -> (count, first example).
type Findings = BTreeMap<String, (usize, String)>;

fn note(findings: &mut Findings, key: String, example: String) {
    let e = findings.entry(key).or_insert((0, example));
    e.0 += 1;
}

fn print(title: &str, findings: &Findings, checked: usize) {
    println!("==== {title}: {checked} results checked, {} groups mismatch", findings.len());
    for (key, (count, example)) in findings {
        println!("{key}: {count} mismatches; e.g. {example}");
    }
}

#[test]
#[ignore = "sweep: --ignored --nocapture"]
fn sweep_typed_buffer_formats() {
    let d16 = std::env::var_os("SWEEP_D16").is_some();
    let mut cases = input_cases();
    if d16 {
        // The four 16-bit codes whose exact ratio and decoded f32 round to
        // different halves, replicated so every channel sees them (#1940).
        for code in [0xFFAF_u32, 0xFFEF, 0x7FF7, 0x8009] {
            cases.push([code << 16 | code; 4]);
        }
    }
    let mut findings = Findings::new();
    let mut checked = 0;
    for (n, name, bits, num) in formats() {
        let norm = matches!(num, Num::Unorm | Num::Snorm);
        if std::env::var_os("SWEEP_NORM_ONLY").is_some() && !norm {
            continue;
        }
        let refs: Vec<Vec<Exact>> = cases.iter().map(|&c| channels(bits, num, c)).collect();
        let shader = format_shader(n, false);
        for (backend, values) in format_rows(&shader, &cases) {
            for (i, row) in values.as_chunks::<4>().0.iter().enumerate() {
                for (ch, &got) in row.iter().enumerate() {
                    checked += 1;
                    let wants = want32(&refs[i][ch]);
                    let ok = wants.iter().any(|w| match *w {
                        Want::Bits(b) => b == got,
                        Want::NanF32 => f32::from_bits(got).is_nan(),
                    });
                    if !ok {
                        note(
                            &mut findings,
                            format!("{backend} fmt {n} {name} xyzw ch{ch}"),
                            format!("in {:08x?} got {got:#010x} want {wants:x?}", cases[i]),
                        );
                    }
                }
            }
        }
        if !d16 {
            continue;
        }
        let shader = format_shader(n, true);
        for (backend, values) in format_rows(&shader, &cases) {
            for (i, row) in values.as_chunks::<4>().0.iter().enumerate() {
                checked += 2; // two packed dwords per row
                for ch in 0..4 {
                    let g = (row[ch / 2] >> (16 * (ch % 2))) as u16;
                    let want = want16(&refs[i][ch], num);
                    let ok = match want {
                        Some(b) => b == g,
                        None => is_f16_nan(g),
                    };
                    if !ok {
                        note(
                            &mut findings,
                            format!("{backend} fmt {n} {name} d16 ch{ch}"),
                            format!("in {:08x?} got {g:#06x} want {want:x?}", cases[i]),
                        );
                    }
                }
            }
        }
    }
    print("typed buffer formats", &findings, checked);
    assert!(
        findings.is_empty(),
        "{} mismatch groups (printed above)",
        findings.len()
    );
}

/// f16 inputs: specials, then pseudo-random halves of every exponent
/// (`narrow`: exponents -8..=8 only, so an f64 FMA stays exact).
fn f16_inputs(narrow: bool, seed: u64) -> Vec<u16> {
    let mut v = vec![
        0x0000, 0x8000, 0x3C00, 0xBC00, 0x3C01, 0x3BFF, 0x7BFF, 0xFBFF, 0x0001, 0x8001,
        0x03FF, 0x0400, 0x7C00, 0xFC00, 0x7E00, 0x1000, 0x1001, 0x5BFF, 0x6400,
    ];
    if narrow {
        v.retain(|&h| {
            let e = i32::from((h >> 10) & 0x1F) - 15;
            (-8..=8).contains(&e) || h & 0x7FFF == 0
        });
    }
    let mut s = seed;
    while v.len() < 120 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let mut h = (s >> 20) as u16;
        if narrow {
            let e = (s >> 40) % 17; // biased exponent 7..=23
            h = (h & 0x83FF) | ((7 + e as u16) << 10);
        }
        v.push(h);
    }
    v
}

fn f16v(h: u16) -> f64 {
    f64::from(f16_to_f32(h))
}

/// Nearest-even f16 of `x`; NaN as `None` (any NaN accepted).
fn rne_or_nan(x: f64) -> Option<u16> {
    (!x.is_nan()).then(|| f16_rne(x))
}

#[test]
#[ignore = "sweep: --ignored --nocapture"]
fn sweep_f16_alu() {
    let mut findings = Findings::new();
    let mut checked = 0;
    // (name, shader words, operands, packed, exact f64 result per lane)
    type Op = fn(f64, f64, f64) -> f64;
    let vop2 = |op: u32| vec![vop1(0x001, 0, 0x80), op << 25 | 2 << 9 | 0x101];
    let pk = |op: u32, three: bool| {
        let src2 = if three { 0x103u32 << 18 } else { 0 };
        vec![0xCC00_4000 | op << 16, 0x1800_0000 | src2 | 0x102 << 9 | 0x101]
    };
    let ops: Vec<(&str, Vec<u32>, bool, bool, Op)> = vec![
        ("v_add_f16", vop2(0x32), false, false, |a, b, _| a + b),
        ("v_sub_f16", vop2(0x33), false, false, |a, b, _| a - b),
        ("v_mul_f16", vop2(0x35), false, false, |a, b, _| a * b),
        (
            "v_fma_f16",
            [vec![vop1(0x001, 0, 0x80)], vop3(0x34B, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec()]
                .concat(),
            true,
            false,
            |a, b, c| a.mul_add(b, c),
        ),
        ("v_pk_add_f16", pk(0x0F, false), false, true, |a, b, _| a + b),
        ("v_pk_mul_f16", pk(0x10, false), false, true, |a, b, _| a * b),
        ("v_pk_fma_f16", pk(0x0E, true), true, true, |a, b, c| a.mul_add(b, c)),
    ];
    for (name, words, three, packed, f) in ops {
        let narrow = three;
        let (a, b, c) =
            (f16_inputs(narrow, 0x9E37), f16_inputs(narrow, 0x7F4A), f16_inputs(narrow, 0x1234));
        let n = a.len();
        let cases: Vec<[u32; 4]> = (0..n)
            .map(|i| {
                let j = (i * 7 + 3) % n;
                let k = (i * 13 + 5) % n;
                let pack = |lo: u16, hi: u16| u32::from(lo) | u32::from(hi) << 16;
                [pack(a[i], a[k]), pack(b[i], b[j]), pack(c[i], c[j]), 0]
            })
            .collect();
        let shader = build_valu_cs(&words);
        run_cases(&shader, &cases, |backend, i, _, got| {
            checked += 1;
            let lanes = if packed { 2 } else { 1 };
            for lane in 0..lanes {
                let h = |x: u32| (x >> (16 * lane)) as u16;
                let (x, y, z) = (h(cases[i][0]), h(cases[i][1]), h(cases[i][2]));
                let want = rne_or_nan(f(f16v(x), f16v(y), if three { f16v(z) } else { 0.0 }));
                let g = h(got);
                let ok = match want {
                    Some(w) => w == g,
                    None => is_f16_nan(g),
                };
                if !ok {
                    let args = if three {
                        format!("{x:#06x}, {y:#06x}, {z:#06x}")
                    } else {
                        format!("{x:#06x}, {y:#06x}")
                    };
                    note(
                        &mut findings,
                        format!("{backend} {name} lane{lane}"),
                        format!("({args}) = {g:#06x}, want {want:x?}"),
                    );
                }
            }
        });
    }
    // v_cvt_f16_f32 and v_cvt_f32_f16 over pseudo-random bits.
    let mut s = 0xDEAD_BEEF_u64;
    let mut f32s: Vec<u32> = Vec::new();
    while f32s.len() < 200 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Mostly in f16 range (exponent 100..=150), some anywhere.
        let bits = (s >> 16) as u32;
        let e = 100 + ((s >> 50) % 51) as u32;
        f32s.push(if f32s.len().is_multiple_of(5) { bits } else { bits & 0x807F_FFFF | e << 23 });
    }
    let cvt = build_valu_cs(&[vop1(0x001, 0, 0x80), vop1(0x00A, 0, vgpr(1))]);
    let cases: Vec<[u32; 4]> = f32s.iter().map(|&b| [b, 0, 0, 0]).collect();
    run_cases(&cvt, &cases, |backend, i, _, got| {
        checked += 1;
        let x = f64::from(f32::from_bits(cases[i][0]));
        let want = rne_or_nan(x);
        let g = got as u16;
        let ok = match want {
            Some(w) => w == g,
            None => is_f16_nan(g),
        };
        if !ok {
            note(
                &mut findings,
                format!("{backend} v_cvt_f16_f32"),
                format!("({:#010x}) = {g:#06x}, want {want:x?}", cases[i][0]),
            );
        }
    });
    let widen = build_valu_cs(&[vop1(0x00B, 0, vgpr(1))]);
    let halves: Vec<u16> = (0..=0xFFFFu32).step_by(97).map(|h| h as u16).collect();
    let cases: Vec<[u32; 4]> = halves.iter().map(|&h| [u32::from(h), 0, 0, 0]).collect();
    run_cases(&widen, &cases, |backend, i, _, got| {
        checked += 1;
        let want = f16_to_f32(halves[i]);
        let ok = if want.is_nan() { f32::from_bits(got).is_nan() } else { got == want.to_bits() };
        if !ok {
            note(
                &mut findings,
                format!("{backend} v_cvt_f32_f16"),
                format!("({:#06x}) = {got:#010x}, want {:#010x}", halves[i], want.to_bits()),
            );
        }
    });
    print("f16 ALU", &findings, checked);
    assert!(findings.is_empty(), "{} mismatch groups (printed above)", findings.len());
}
