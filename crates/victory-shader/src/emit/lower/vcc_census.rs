//! An opt-in breakdown of the two scalar-VCC refusals. Annotate the existing
//! error path so dead instructions, lowering retries and successful fallback
//! translations do not count as final failures.

use super::{Fields, Lower, Operand, Value, Why};
use crate::ShaderError;
use std::cell::Cell;

const JOIN: Why = "a scalar VCC branch depending on a divergent scalar join";
const PATH: Why = "a scalar VCC branch with mask bits outside its producing path";
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; }

/// Run one translation and return its optional fixed VCC refusal category.
/// The returned translation and ordinary error reason are unchanged.
pub fn with_vcc_refusal_bucket<T>(
    translate: impl FnOnce() -> Result<T, ShaderError>,
) -> (Result<T, ShaderError>, Option<&'static str>) {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.set(self.0);
        }
    }
    let _restore = Restore(ENABLED.replace(true));
    match translate() {
        Err(ShaderError::Unsupported { offset, what }) => {
            let reason = if what.starts_with("join/") {
                JOIN
            } else if what.starts_with("producing-path/") {
                PATH
            } else {
                return (Err(ShaderError::Unsupported { offset, what }), None);
            };
            (
                Err(ShaderError::Unsupported {
                    offset,
                    what: reason,
                }),
                Some(what),
            )
        }
        other => (other, None),
    }
}

impl Lower<'_> {
    pub(super) fn vcc_join_refusal(
        &self,
        v: Value,
        lo: Value,
        hi: Value,
        b: usize,
        i: usize,
    ) -> Why {
        if !ENABLED.get() {
            return JOIN;
        }
        let block = &self.x.p.blocks[b];
        let live: Vec<_> = block
            .preds
            .iter()
            .copied()
            .zip(&block.phis[i].args)
            .filter(|&(pred, _)| !self.never_taken(pred, b))
            .collect();
        let args = || live.iter().map(|&(_, &v)| v);
        let immediate = |arg| {
            self.def_inst(arg).is_some_and(|inst| {
                let Fields::Sop { src, imm, .. } = &inst.fields else {
                    return false;
                };
                match inst.name() {
                    "s_movk_i32" => imm.is_some(),
                    "s_mov_b32" | "s_mov_b64" => match src.first() {
                        Some(Operand::Const(k)) => crate::resource::inline(*k).is_some(),
                        Some(Operand::Literal(_)) => true,
                        _ => false,
                    },
                    _ => false,
                }
            })
        };
        // Classify the first phi whose incoming edge fails the original check.
        // Prefer a back-edge or nested join before classifying definitions.
        let bucket = if live.iter().any(|&(p, _)| self.rpo[p] >= self.rpo[b]) {
            ["join/direct/back-edge", "join/dependency/back-edge"]
        } else if args().any(|a| self.phi_at.contains_key(&a)) {
            ["join/direct/phi-input", "join/dependency/phi-input"]
        } else if args().all(immediate) {
            [
                "join/direct/immediate-moves",
                "join/dependency/immediate-moves",
            ]
        } else if args().any(|a| {
            self.def_inst(a)
                .is_some_and(|inst| matches!(inst.fields, Fields::Smem { .. }))
        }) {
            ["join/direct/scalar-memory", "join/dependency/scalar-memory"]
        } else if args().any(|a| {
            self.def_inst(a)
                .is_some_and(|inst| matches!(inst.fields, Fields::Valu(_)))
        }) {
            ["join/direct/valu-input", "join/dependency/valu-input"]
        } else {
            ["join/direct/other-inputs", "join/dependency/other-inputs"]
        };
        bucket[usize::from(v != lo && v != hi)]
    }

    pub(super) fn vcc_path_refusal(&self, low: Value) -> Why {
        if !ENABLED.get() {
            return PATH;
        }
        match self.def_inst(low) {
            Some(i) if i.name().starts_with("v_cmpx") => "producing-path/compare-writing-exec",
            Some(i) if i.name().starts_with("v_cmp") => "producing-path/compare",
            Some(i) if matches!(i.fields, Fields::Valu(_)) => "producing-path/other-valu",
            Some(i) if matches!(i.fields, Fields::Sop { .. }) => "producing-path/scalar-operation",
            _ => "producing-path/other-definition",
        }
    }
}
