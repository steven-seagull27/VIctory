//! Row permutations, host participation, and banked compute lane reads on WARP.

use super::{
    Atom, Attr, Bin, Inst, LaneExit, Lower, Operand, Program, Reg, Structured, V, Why, pred,
};
use crate::gfx10::{Fields, Valu};
use crate::ir::Value;
use std::collections::HashSet;

/// A guest SCC can depend on an SGPR phi selected by a divergent host
/// branch. Prove ordinary scalar expressions from compute SGPR inputs.
/// Phis and wave operations require their own participation proof.
fn host_uniform_scalars(p: &Program, order: &[usize]) -> HashSet<Value> {
    let mut uniform: HashSet<_> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| matches!(reg, Reg::S(0..=105)).then_some(v))
        .collect();
    for &b in order {
        for inst in &p.blocks[b].insts {
            let Fields::Sop { src, .. } = &inst.fields else {
                continue;
            };
            let name = inst.name();
            let pure = name.starts_with("s_cmp_")
                || name.starts_with("s_cmpk_")
                || matches!(
                    name,
                    "s_mov_b32"
                        | "s_mov_b64"
                        | "s_movk_i32"
                        | "s_add_u32"
                        | "s_sub_u32"
                        | "s_add_i32"
                        | "s_sub_i32"
                        | "s_and_b32"
                        | "s_or_b32"
                        | "s_xor_b32"
                        | "s_lshl_b32"
                        | "s_lshr_b32"
                        | "s_ashr_i32"
                        | "s_cselect_b32"
                        | "s_cselect_b64"
                );
            if pure
                && src.iter().all(|x| match x {
                    Operand::Scalar { code: 0..=105, .. } | Operand::Literal(_) => true,
                    Operand::Const(k) => crate::resource::inline(*k).is_some(),
                    _ => false,
                })
                && inst.reads.iter().all(|(_, v)| uniform.contains(v))
            {
                uniform.extend(
                    inst.writes
                        .iter()
                        .filter_map(|&(r, v)| matches!(r, Reg::S(0..=105) | Reg::Scc).then_some(v)),
                );
            }
        }
    }
    uniform
}

/// Blocks reached by every host lane: a fetch-inactive shuffle cannot
/// recover registers of invocations absent from a divergent host branch.
fn all_lane_blocks(p: &Program, st: &Structured) -> Option<Vec<bool>> {
    let exit_block = st.exits.len();
    let n = exit_block + 1;
    let mut successors = vec![Vec::new(); n];
    let mut reaches_end = vec![false; n];
    reaches_end[exit_block] = true;
    let mut rpo = vec![usize::MAX; exit_block];
    let order = crate::ir::reverse_postorder(&p.blocks);
    for (index, &block) in order.iter().enumerate() {
        rpo[block] = index;
    }
    let mut aborts = vec![false; exit_block];
    for (b, exit) in st.exits.iter().enumerate() {
        match *exit {
            LaneExit::Jump(t) => {
                successors[b].push(t);
                if rpo[b] != usize::MAX && rpo[t] <= rpo[b] {
                    // `Lower::exit` adds a per-lane return at the iteration
                    // limit of an unconditional back edge. It bypasses the
                    // loop's normal exit and every later reconvergence point.
                    successors[b].push(exit_block);
                    aborts[b] = true;
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                successors[b].extend([taken, not_taken]);
            }
            LaneExit::End => successors[b].push(exit_block),
            LaneExit::Indirect => return None,
        }
    }
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            if !reaches_end[b] && successors[b].iter().any(|&q| reaches_end[q]) {
                reaches_end[b] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Post-dominance without any path to the exit would be vacuous.
    if reaches_end.iter().any(|&reaches| !reaches) {
        return None;
    }
    let words = n.div_ceil(64);
    let mut post = vec![vec![u64::MAX; words]; n];
    for (b, next) in successors.iter().enumerate() {
        if next.is_empty() {
            post[b].fill(0);
            post[b][b / 64] |= 1 << (b % 64);
        }
    }
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            if successors[b].is_empty() {
                continue;
            }
            let mut row = vec![u64::MAX; words];
            for &q in &successors[b] {
                for (word, &other) in row.iter_mut().zip(&post[q]) {
                    *word &= other;
                }
            }
            row[b / 64] |= 1 << (b % 64);
            if row != post[b] {
                post[b] = row;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut all = vec![true; n];
    let uniform = host_uniform_scalars(p, &order);
    for (branch, exit) in st.exits.iter().enumerate() {
        let divergent = match *exit {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // Lower::exit also tests a per-lane iteration counter on a back edge.
                rpo[taken] <= rpo[branch]
                    || rpo[not_taken] <= rpo[branch]
                    || !st
                        .bools
                        .support(when)
                        .iter()
                        .all(|a| matches!(a, Atom::Uniform(v) if uniform.contains(v)))
            }
            _ => false,
        };
        if !divergent && !aborts[branch] {
            continue;
        }
        let mut seen = vec![false; n];
        let mut pending = successors[branch].clone();
        while let Some(b) = pending.pop() {
            if seen[b] || post[branch][b / 64] >> (b % 64) & 1 != 0 {
                continue;
            }
            seen[b] = true;
            all[b] = false;
            pending.extend(successors[b].iter().copied());
        }
    }
    all.truncate(exit_block);
    Some(all)
}

impl Lower<'_> {
    pub(super) fn active_lane_words(&mut self, enabled: V) -> [V; 2] {
        let c = self.cur;
        let four = self
            .s
            .m
            .named_struct("dx.types.fouri32", vec![self.s.t.i32; 4]);
        let mask = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[enabled]);
        [self.s.m.extract(c, mask, 0), self.s.m.extract(c, mask, 1)]
    }

    /// On WARP 1.0.22, divergent WaveReadLaneAt can select the wrong
    /// four-lane group. Hold the group constant in each read, then select
    /// the requested group's result. See #2949.
    pub(super) fn read_lane_on_warp(&mut self, value: V, lane: V, active: [V; 2]) -> V {
        let c = self.cur;
        let me = self.lane_index();
        let i32t = self.s.t.i32;
        let first = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[me]);
        let inner_mask = self.s.m.i32(3);
        let outer_mask = self.s.m.i32(!3);
        let inner = self.s.m.bin(c, Bin::And, lane, inner_mask);
        let outer = self.s.m.bin(c, Bin::And, lane, outer_mask);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let bank_mask = self.s.m.i32(15);
        let bit = self.s.m.bin(c, Bin::Shl, one, inner);
        let mut result = zero;
        for start in (0..self.wave()).step_by(4) {
            let base = self.s.m.i32(start);
            let shift = self.s.m.i32(start % 32);
            let mask = self
                .s
                .m
                .bin(c, Bin::LShr, active[(start / 32) as usize], shift);
            let mask = self.s.m.bin(c, Bin::And, mask, bank_mask);
            let nonempty = self.s.m.cmp(c, pred::NE, mask, zero);
            let possible = self.s.m.bin(c, Bin::And, mask, bit);
            let live = self.s.m.cmp(c, pred::NE, possible, zero);
            let fallback = self
                .s
                .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[mask]);
            let selected = self.s.m.select(c, live, inner, fallback);
            let bank_lane = self.s.m.bin(c, Bin::Or, selected, base);
            // An unselected read also needs a live host source. Use the
            // wave's first lane for a group with no active source.
            let source = self.s.m.select(c, nonempty, bank_lane, first);
            let read = self.read_lane(value, source);
            let requested = self.s.m.cmp(c, pred::EQ, outer, base);
            result = self.s.m.select(c, requested, read, result);
        }
        result
    }

    fn all_lanes_at(&self, block: usize) -> bool {
        self.permlane_full
            .get_or_init(|| all_lane_blocks(self.x.p, self.x.st))
            .as_ref()
            .is_some_and(|all| all[block])
    }

    /// Capture the participation of a single forward edge while the whole
    /// wave can still vote. A ballot issued inside the split on WARP can
    /// include lanes which took the other edge.
    pub(super) fn record_permlane_edge(&mut self, from: usize, to: usize, cond: V, taken: bool) {
        let target = &self.x.p.blocks[to];
        if target.preds != [from] || self.rpo[to] <= self.rpo[from] {
            return;
        }
        let reads_edge = target
            .insts
            .iter()
            .zip(&self.x.st.guards[to])
            .any(|(i, guard)| {
                guard.is_none()
                    && matches!(i.name(), "v_permlane16_b32" | "v_permlanex16_b32")
                    && matches!(&i.fields, Fields::Valu(v) if v.opsel & 1 == 0)
            });
        if !reads_edge || !self.all_lanes_at(from) {
            return;
        }
        let enabled = if taken {
            cond
        } else {
            let (no, yes) = (self.s.m.bool(false), self.s.m.bool(true));
            self.s.m.select(self.cur, cond, no, yes)
        };
        let lanes = self.active_lane_words(enabled);
        self.permlane_edges.insert(to, lanes);
    }

    pub(super) fn permlane(&mut self, inst: &Inst, v: &Valu, guard: Option<V>) -> Result<(), Why> {
        let c = self.cur;
        // RDNA2 ISA 12.12: each nibble of {src2, src1} selects a lane
        // in this row of 16, or the other row of the same 32-lane half.
        // OP_SEL[0:1] are fetch-inactive and bound-control, not half selects.
        if v.abs != 0 || v.neg != 0 || v.opsel & !3 != 0 {
            return Err("permlane input modifiers");
        }
        self.cross_lane(false)?;
        let fetch_inactive = v.opsel & 1 != 0;
        if self.warp_permute {
            let group = self.s.threads.expect("cross_lane checked compute");
            let lanes = group.into_iter().map(u64::from).product::<u64>();
            if !lanes.is_multiple_of(u64::from(self.wave())) {
                return Err("a WARP permlane with a partially populated wave");
            }
        }
        if fetch_inactive && !self.all_lanes_at(self.cur_ir) {
            // DXIL cannot read the last VGPR value of a host invocation
            // that has left through a lane branch. A guard keeps it here.
            return Err("a permlane fetch from lanes outside the host branch");
        }
        let (x, lo, hi) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
        let lane = self.lane_index();
        let (k7, k8, k15, k16, k31, k32, k4, one, zero) = (
            self.s.m.i32(7),
            self.s.m.i32(8),
            self.s.m.i32(15),
            self.s.m.i32(16),
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(4),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let pos = self.s.m.bin(c, Bin::And, lane, k15);
        let lower = self.s.m.cmp(c, pred::IULT, pos, k8);
        let packed = self.s.m.select(c, lower, lo, hi);
        let shift = self.s.m.bin(c, Bin::And, pos, k7);
        let shift = self.s.m.bin(c, Bin::Mul, shift, k4);
        let pick = self.s.m.bin(c, Bin::LShr, packed, shift);
        let pick = self.s.m.bin(c, Bin::And, pick, k15);
        let not15 = self.s.m.i32(!15);
        let mut row = self.s.m.bin(c, Bin::And, lane, not15);
        if inst.name() == "v_permlanex16_b32" {
            row = self.s.m.bin(c, Bin::Xor, row, k16);
        }
        let src = self.s.m.bin(c, Bin::Or, row, pick);
        let r = if fetch_inactive {
            if self.warp_permute {
                // FI reads all participating host lanes, including
                // sources disabled only by the guest EXEC mask.
                let on = self.s.m.bool(true);
                let host = self.active_lane_words(on);
                self.read_lane_on_warp(x, src, host)
            } else {
                self.read_lane(x, src)
            }
        } else {
            // Ballot the console EXEC guard as well as the host's active
            // invocations. Never shuffle from an inactive host lane.
            let edge = if self.warp_permute && guard.is_none() {
                self.permlane_edges.get(&self.cur_ir).copied()
            } else {
                None
            };
            if self.warp_permute && edge.is_none() && !self.all_lanes_at(self.cur_ir) {
                return Err("a WARP permlane without a known incoming lane mask");
            }
            let [lo, hi] = edge.unwrap_or_else(|| {
                let enabled = guard.unwrap_or_else(|| self.s.m.bool(true));
                self.active_lane_words(enabled)
            });
            let lower = self.s.m.cmp(c, pred::IULT, src, k32);
            let word = self.s.m.select(c, lower, lo, hi);
            let bit = self.s.m.bin(c, Bin::And, src, k31);
            let bit = self.s.m.bin(c, Bin::LShr, word, bit);
            let bit = self.s.m.bin(c, Bin::And, bit, one);
            let active = self.s.m.cmp(c, pred::NE, bit, zero);
            let safe = self.s.m.select(c, active, src, lane);
            let read = if self.warp_permute {
                self.read_lane_on_warp(x, safe, [lo, hi])
            } else {
                self.read_lane(x, safe)
            };
            let fallback = if v.opsel & 2 != 0 {
                zero
            } else {
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a permlane result outside a VGPR");
                };
                self.i32_of(inst.read(Reg::V(reg)).ok_or("a permlane old value")?)?
            };
            self.s.m.select(c, active, read, fallback)
        };
        self.write_i(inst, r, guard)
    }
}
