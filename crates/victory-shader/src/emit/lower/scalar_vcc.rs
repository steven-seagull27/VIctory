use super::{Atom, Lower, Reg, StageIo, Value, Why};
use std::collections::HashSet;

impl Lower<'_> {
    /// An SGPR copied into VCC may already be a per-lane DXIL phi. A
    /// whole-word comparison cannot turn that phi back into a wave value.
    pub(super) fn check_scalar_vcc_joins(&self, lo: Value, hi: Value) -> Result<(), Why> {
        if self.phi_at.is_empty()
            || matches!(&self.x.native.io, StageIo::Cs(cs) if cs.threads == [1, 1, 1])
        {
            return Ok(());
        }
        let st = self.x.st;
        let mut pending = vec![lo];
        if self.x.native.wave64 && lo != hi {
            pending.push(hi);
        }
        let mut seen = HashSet::new();
        while let Some(value) = pending.pop() {
            if !seen.insert(value) {
                continue;
            }
            if let Some(&(b, i)) = self.phi_at.get(&value) {
                let block = &self.x.p.blocks[b];
                for (&pred, &arg) in block.preds.iter().zip(&block.phis[i].args) {
                    let edge = self.lane_edge(pred, b);
                    if !st.bools.meet(&edge) {
                        continue;
                    }
                    for (condition, _) in edge {
                        for atom in st.bools.support(condition) {
                            match atom {
                                Atom::Uniform(scc) => pending.push(scc),
                                Atom::Nonzero(low, high) => pending.extend([low, high]),
                                _ => {
                                    return Err(
                                        "a scalar VCC branch depending on a divergent scalar join",
                                    );
                                }
                            }
                        }
                    }
                    pending.push(arg);
                }
            } else if let Some(inst) = self.def_inst(value) {
                pending.extend(inst.reads.iter().filter_map(|&(reg, input)| {
                    matches!(reg, Reg::S(_) | Reg::Scc).then_some(input)
                }));
            }
        }
        Ok(())
    }
}
