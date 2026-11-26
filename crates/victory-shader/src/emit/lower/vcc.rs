//! The whole-wave VCC branch and the scalar words it tests.

use super::{Atom, Bin, Lower, MaskData, Reg, StageIo, V, Val, Value, Why, pred};
use std::collections::HashSet;

impl Lower<'_> {
    pub(super) fn vcc_nonzero(&mut self, lo: Value, hi: Value, at: usize) -> Result<V, Why> {
        // Stages without an established guest wave width (pixel,
        // vertex, mesh) keep the per-lane decision for lane-mask VCC
        // values so compiler patterns like `v_cmp; s_and vcc, exec,
        // vcc; s_cbranch_vccz` continue to translate.
        let singleton = matches!(&self.x.native.io, StageIo::Cs(cs) if cs.threads == [1, 1, 1]);
        let has_wave_width =
            self.mask_data == MaskData::Console && (singleton || self.s.wave_size.is_some());
        if !has_wave_width && let Some(mask) = self.x.st.mask_bits(lo, (lo != hi).then_some(hi)) {
            return self.boolean(mask, at);
        }
        self.check_vcc_joins(lo, hi)?;
        let low = self.vcc_word(lo)?;
        let bits = if hi == lo {
            low
        } else {
            let high = self.vcc_word(hi)?;
            self.s.m.bin(at, Bin::Or, low, high)
        };
        let zero = self.s.m.i32(0);
        Ok(self.s.m.cmp(at, pred::NE, bits, zero))
    }

    /// Scalar writes can set bits for lanes outside EXEC or even outside a
    /// partially filled wave. Read their dword as written. For a VALU result,
    /// take the ballot at its definition, before any subsequent divergence.
    fn vcc_word(&mut self, v: Value) -> Result<V, Why> {
        let st = self.x.st;
        let mask = st.masks.contains_key(&v) || st.highs.contains_key(&v);
        if !mask || matches!(self.vals.get(&v), Some(Val::I32(_) | Val::Bit(_))) {
            return self.i32_of(v);
        }
        let singleton = matches!(&self.x.native.io, StageIo::Cs(cs) if cs.threads == [1, 1, 1]);
        if self.mask_data != MaskData::Console || !singleton && self.s.wave_size.is_none() {
            return Err("a scalar VCC branch requiring the guest wave width");
        }
        // A ballot cannot include lanes on another host control-flow path.
        // The guest compare's own EXEC may already exclude them.
        let low = st.highs.get(&v).copied().unwrap_or(v);
        if let (Some(&block), Some(&mask)) = (self.def_block.get(&low), st.masks.get(&low))
            && st.active.contains(&low)
            && !singleton
            && st
                .bools
                .support(st.reach[block])
                .iter()
                .any(|a| !matches!(a, Atom::Uniform(_) | Atom::VccNonzero(..)))
            && st.bools.meet(&[(mask, true), (st.reach[block], false)])
        {
            return Err("a scalar VCC branch with mask bits outside its producing path");
        }
        self.mask_dword(v)
    }

    /// A normal DXIL phi can select a different SGPR value in each lane at a
    /// join of the per-lane CFG. That phi cannot represent a scalar VCC word.
    fn check_vcc_joins(&self, lo: Value, hi: Value) -> Result<(), Why> {
        if matches!(&self.x.native.io, StageIo::Cs(cs) if cs.threads == [1, 1, 1]) {
            return Ok(());
        }
        let st = self.x.st;
        let mut values = vec![lo, hi];
        let mut seen = HashSet::new();
        while let Some(v) = values.pop() {
            if !seen.insert(v) {
                continue;
            }
            if let Some(&(b, i)) = self.phi_at.get(&v) {
                let block = &self.x.p.blocks[b];
                for (&pred, &arg) in block.preds.iter().zip(&block.phis[i].args) {
                    let edge = self.lane_edge(pred, b);
                    if !st.bools.meet(&edge) {
                        continue;
                    }
                    for (cond, _) in edge {
                        for atom in st.bools.support(cond) {
                            match atom {
                                Atom::Uniform(scc) => values.push(scc),
                                Atom::VccNonzero(low, high) => values.extend([low, high]),
                                _ => {
                                    return Err(
                                        "a scalar VCC branch depending on a divergent scalar join",
                                    );
                                }
                            }
                        }
                    }
                    values.push(arg);
                }
            } else if let Some(inst) = self.def_inst(v) {
                // The lanes' vector inputs may legitimately differ. A scalar
                // input (including an enclosing EXEC mask) must still be valid.
                values.extend(
                    inst.reads.iter().filter_map(|&(r, value)| {
                        matches!(r, Reg::S(_) | Reg::Scc).then_some(value)
                    }),
                );
            }
        }
        Ok(())
    }
}
