//! d16 format loads of UNORM / SNORM channels: the f16 RDNA2 gives, rounded
//! once to nearest even (see [`Lower::norm_f16`]).

use super::Lower;
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::flags;

impl Lower<'_> {
    /// A UNORM / SNORM channel `x` (its decoded f32 bits) of `w` bits as
    /// f16, rounded once to nearest even as RDNA2 does (`legacyF32ToF16` of
    /// the f32 rounds twice, the second time toward zero, leaving about half
    /// the codes a unit low). `code / (2^w - 1)` repeats the w-bit code in
    /// binary, so its first 32 bits, `(code << 32) / (2^w - 1)`, decide the
    /// rounding and a tie cannot happen. SNORM is `|v| / (2^(w-1) - 1)`.
    pub(super) fn norm_f16(&mut self, x: V, w: u32, snorm: bool) -> V {
        let c = self.cur;
        let bits = if snorm { w - 1 } else { w };
        let d = (1u32 << bits) - 1;
        let (i32t, i64t) = (self.s.t.i32, self.s.m.int(64));
        self.s.flags |= flags::INT64;
        // the code, back from the decoded value (|value| for SNORM)
        let mag = self.s.m.i32(0x7FFF_FFFF);
        let ax = self.s.m.bin(c, Bin::And, x, mag);
        let af = self.as_float(ax);
        let df = self.s.m.f32((d as f32).to_bits());
        let y = self.s.m.bin(c, Bin::Mul, af, df);
        let y = self.unary(26, y); // round to nearest even
        let y = self.s.m.cast(c, Cast::FpToUi, y, i32t);
        // first 32 bits of y / d
        let y64 = self.s.m.cast(c, Cast::ZExt, y, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let y64 = self.s.m.bin(c, Bin::Shl, y64, k32);
        let d64 = self.s.m.int_const(i64t, i64::from(d));
        let p = self.s.m.bin(c, Bin::UDiv, y64, d64);
        let p = self.s.m.cast(c, Cast::Trunc, p, i32t);
        let lz = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[p]);
        // normal: 11 bits after the leading one and the round bit
        let k14 = self.s.m.i32(14);
        let e = self.s.m.bin(c, Bin::Sub, k14, lz);
        let pn = self.s.m.bin(c, Bin::Shl, p, lz);
        let k20 = self.s.m.i32(20);
        let b = self.s.m.bin(c, Bin::LShr, pn, k20);
        let one = self.s.m.i32(1);
        let m = self.s.m.bin(c, Bin::LShr, b, one);
        let r = self.s.m.bin(c, Bin::And, b, one);
        let m = self.s.m.bin(c, Bin::Add, m, r);
        let em1 = self.s.m.bin(c, Bin::Sub, e, one);
        let k10 = self.s.m.i32(10);
        let eh = self.s.m.bin(c, Bin::Shl, em1, k10);
        let normal = self.s.m.bin(c, Bin::Add, eh, m);
        // below 2^-14: a subnormal half
        let k8 = self.s.m.i32(8);
        let k7 = self.s.m.i32(7);
        let s8 = self.s.m.bin(c, Bin::LShr, p, k8);
        let r7 = self.s.m.bin(c, Bin::LShr, p, k7);
        let r7 = self.s.m.bin(c, Bin::And, r7, one);
        let sub = self.s.m.bin(c, Bin::Add, s8, r7);
        let is_normal = self.s.m.cmp(c, pred::SGE, e, one);
        let h = self.s.m.select(c, is_normal, normal, sub);
        // 0 and 1.0 (the latter's expansion has no leading one in 32 bits' reach)
        let zero = self.s.m.i32(0);
        let is0 = self.s.m.cmp(c, pred::EQ, y, zero);
        let h = self.s.m.select(c, is0, zero, h);
        let dv = self.s.m.i32(d);
        let full = self.s.m.cmp(c, pred::IUGE, y, dv);
        let h1 = self.s.m.i32(0x3C00);
        let h = self.s.m.select(c, full, h1, h);
        if !snorm {
            return h;
        }
        let k31 = self.s.m.i32(31);
        let sign = self.s.m.bin(c, Bin::LShr, x, k31);
        let k15 = self.s.m.i32(15);
        let sign = self.s.m.bin(c, Bin::Shl, sign, k15);
        self.s.m.bin(c, Bin::Or, h, sign)
    }
}
