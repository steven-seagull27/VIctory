//! Diagnostic provenance at the point where a lane-mask refusal escapes.

use super::{Atom, Lower, Stop, V, Value, Why, unsupported};
#[cfg(feature = "diagnostics")]
use super::{Inst, LANE_MASK, Parts};
#[cfg(feature = "diagnostics")]
use std::collections::HashMap;

#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy)]
struct Source {
    value: Value,
    /// Numeric conversions on the failing path, including any earlier
    /// instruction which propagated the same refusal.
    numeric: u8,
}

#[cfg(feature = "diagnostics")]
struct Leaf {
    atom: Atom,
    value: Value,
    tracked: bool,
    lane_bit: bool,
}

#[cfg(feature = "diagnostics")]
pub(super) struct MaskTrace {
    enabled: bool,
    read: Option<Source>,
    leaf: Option<Leaf>,
    from: HashMap<Value, Source>,
    /// Only refused phi arguments survive a new lowering pass.
    joins: HashMap<Value, Source>,
}

#[cfg(feature = "diagnostics")]
impl MaskTrace {
    pub(super) fn new() -> Self {
        Self {
            enabled: std::env::var("VICTORY_TRACE_LANE_MASK").as_deref() == Ok("1"),
            read: None,
            leaf: None,
            from: HashMap::new(),
            joins: HashMap::new(),
        }
    }

    pub(super) fn begin_pass(&mut self) {
        self.clear_read();
        self.from.clone_from(&self.joins);
    }

    pub(super) fn clear_read(&mut self) {
        self.read = None;
        self.leaf = None;
    }

    pub(super) fn missing(&mut self, value: Value) {
        if self.enabled {
            self.read = Some(
                self.from
                    .get(&value)
                    .copied()
                    .unwrap_or(Source { value, numeric: 0 }),
            );
        }
    }

    fn numeric(&mut self, value: Value, flag: u8) {
        if self.read.is_none() {
            self.missing(value);
        }
        if let Some(source) = &mut self.read {
            source.numeric |= flag;
        }
    }

    pub(super) fn propagate(&mut self, inst: &Inst) {
        if let Some(from) = self.read {
            for &(_, written) in &inst.writes {
                self.from.insert(written, from);
            }
        }
    }

    fn report(&self, parts: &Parts<'_>, offset: usize, site: &str) {
        if !self.enabled {
            return;
        }
        let instructions = || parts.p.blocks.iter().flat_map(|b| &b.insts);
        let op = instructions()
            .find(|i| i.machine.offset == offset)
            .map_or("block", |i| i.name());
        let stage = parts.native.stage;
        let leaf = self
            .leaf
            .as_ref()
            .map(|leaf| {
                let defined = instructions()
                    .find(|i| i.writes.iter().any(|&(_, w)| w == leaf.value))
                    .map_or_else(
                        || "input or phi".to_owned(),
                        |i| format!("{} at {:#x}", i.name(), i.machine.offset),
                    );
                format!(
                    "; leaf {:?} defined by {defined}, structural mask={}, lane bit={}",
                    leaf.atom, leaf.tracked, leaf.lane_bit
                )
            })
            .unwrap_or_default();
        let producer = self
            .read
            .and_then(|s| instructions().find(|i| i.writes.iter().any(|&(_, w)| w == s.value)));
        let numeric = self.read.map_or(0, |s| s.numeric);
        let helpers: Vec<_> = [(1, "i32_of"), (2, "mask_dword"), (4, "mask_number")]
            .into_iter()
            .filter_map(|(bit, name)| (numeric & bit != 0).then_some(name))
            .collect();
        if let Some(source) = producer {
            eprintln!(
                "lane-mask trace: {stage:?} {op} at {offset:#x} [{site}] from {} at {:#x}; numeric helpers {helpers:?}{leaf}",
                source.name(),
                source.machine.offset
            );
        } else {
            eprintln!(
                "lane-mask trace: {stage:?} {op} at {offset:#x} [{site}] has no traced instruction producer; numeric helpers {helpers:?}{leaf}"
            );
        }
    }
}

impl Lower<'_> {
    /// Print the decoded operands and their SSA roles at selected machine
    /// byte offsets. Only called when an opted-in mask refusal escapes.
    #[cfg(feature = "diagnostics")]
    fn trace_mask_operands(&self, offsets: &str) {
        use crate::ir::Reg;
        let offsets: std::collections::BTreeSet<_> = offsets
            .split(',')
            .map(str::trim)
            .filter_map(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .collect();
        let p = self.x.p;
        let st = self.x.st;
        for (block, b) in p.blocks.iter().enumerate() {
            for inst in &b.insts {
                if !offsets.contains(&inst.machine.offset) {
                    continue;
                }
                eprintln!(
                    "mask operands: wave{} b{block} at {:#x} {}",
                    if p.wave64 { 64 } else { 32 },
                    inst.machine.offset,
                    inst.machine.text(p.wave64)
                );
                for (role, operands) in [("read", &inst.reads), ("write", &inst.writes)] {
                    for &(reg, value) in operands {
                        let pair = match reg {
                            Reg::S(n) if p.wave64 && n % 2 == 0 => operands
                                .iter()
                                .find(|&&(r, _)| r == Reg::S(n + 1))
                                .and_then(|&(_, hi)| st.mask_bits(value, Some(hi))),
                            _ => None,
                        };
                        let stored = match self.vals.get(&value) {
                            Some(super::Val::I32(_)) => "i32",
                            Some(super::Val::Bit(_)) => "bit",
                            Some(super::Val::Missing(why)) => why,
                            None => "none",
                        };
                        eprintln!(
                            "  {role} {reg:?}={value:?} entry={} mask={:?} other={:?} high_of={:?} pair={pair:?} active={} lane_bit={} stored={stored}",
                            p.inputs.contains(&(reg, value)),
                            st.masks.get(&value),
                            st.other.get(&value),
                            st.highs.get(&value),
                            st.active.contains(&value),
                            self.lane_bits.contains_key(&value)
                        );
                    }
                }
            }
        }
    }

    pub(super) fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        let result = self.atom_inner(atom, at);
        #[cfg(feature = "diagnostics")]
        if matches!(result, Err(LANE_MASK))
            && self.mask_trace.enabled
            && self.mask_trace.leaf.is_none()
        {
            let value = match atom {
                Atom::Lane(v)
                | Atom::Uniform(v)
                | Atom::Quad(v)
                | Atom::Phi(v)
                | Atom::Opaque(v) => Some(v),
                Atom::Constant(_) => None,
            };
            if let Some(value) = value {
                self.mask_trace.leaf = Some(Leaf {
                    atom,
                    value,
                    tracked: self.x.st.masks.contains_key(&value)
                        || self.x.st.highs.contains_key(&value),
                    lane_bit: self.lane_bits.contains_key(&value),
                });
            }
        }
        result
    }

    pub(super) fn i32_of(&mut self, value: Value) -> Result<V, Why> {
        let result = self.i32_of_inner(value);
        #[cfg(feature = "diagnostics")]
        if matches!(result, Err(LANE_MASK)) {
            self.mask_trace.numeric(value, 1);
        }
        result
    }

    pub(super) fn mask_dword(&mut self, value: Value) -> Result<V, Why> {
        let result = self.mask_dword_inner(value);
        #[cfg(feature = "diagnostics")]
        if matches!(result, Err(LANE_MASK)) {
            self.mask_trace.numeric(value, 2);
        }
        result
    }

    pub(super) fn mask_number(&mut self, value: Value) -> Result<(V, V), Why> {
        let result = self.mask_number_inner(value);
        #[cfg(feature = "diagnostics")]
        if matches!(result, Err(LANE_MASK)) {
            self.mask_trace.numeric(value, 4);
        }
        result
    }

    pub(super) fn traced_refusal(&self, offset: usize, why: Why, site: &'static str) -> Stop {
        #[cfg(feature = "diagnostics")]
        if why == LANE_MASK && self.mask_trace.enabled {
            self.mask_trace.report(self.x, offset, site);
            if let Ok(offsets) = std::env::var("VICTORY_TRACE_LANE_MASK_AT") {
                self.trace_mask_operands(&offsets);
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = site;
        unsupported(offset, why)
    }

    pub(super) fn trace_retry(&mut self, phi: Value, arg: Value, why: Why) -> Stop {
        #[cfg(feature = "diagnostics")]
        if why == LANE_MASK {
            if self.mask_trace.read.is_none() {
                self.mask_trace.missing(arg);
            }
            if let Some(source) = self.mask_trace.read {
                self.mask_trace.joins.insert(phi, source);
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = arg;
        Stop::Retry(phi, why)
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;
    use crate::ir::{Reg, build};

    #[test]
    fn a_missing_scc_keeps_its_producer_across_a_scalar_select() {
        let words = [
            0x7d88_0090u32, // v_cmp_gt_u32 vcc, 16, v0
            0xbe88_3c6a,    // s_and_saveexec_b32 s8, vcc
            0x8507_8081,    // s_cselect_b32 s7, 1, 0
            0xbf81_0000,    // s_endpgm
        ];
        let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let program = build(&code, false, 1).unwrap();
        let instructions = &program.blocks[0].insts;
        let scc = instructions[1].write(Reg::Scc).unwrap();
        let selected = instructions[2].write(Reg::S(7)).unwrap();
        let mut trace = MaskTrace {
            enabled: true,
            ..MaskTrace::new()
        };
        trace.missing(scc);
        trace.propagate(&instructions[2]);
        trace.clear_read();
        trace.numeric(selected, 1);
        assert_eq!(trace.read.unwrap().value, scc);
        assert_eq!(trace.read.unwrap().numeric, 1);
        trace.begin_pass();
        assert!(trace.read.is_none() && trace.from.is_empty());
    }
}
