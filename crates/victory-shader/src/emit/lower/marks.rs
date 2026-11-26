//! Which lanes of a wave set a write or read mark (module notes of `emit`,
//! "Write marks", "Read marks").

use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use super::Lower;

impl Lower<'_> {
    /// Whether this lane sets mark bit `n` (its load and atomic go to the
    /// bitmap in system memory): lanes with the first active lane's bit
    /// leave it to that lane. Most lanes of a wave mark the same page, so
    /// one load crosses the bus instead of 32. In pixel shaders the lane
    /// that leads is the first that is no helper: a helper's writes are
    /// dropped.
    pub(super) fn mark_lead(&mut self, n: V) -> Option<V> {
        self.s.flags |= flags::WAVE_OPS;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let skip_helpers = if self.s.pixel {
            let helper = self
                .s
                .op(self.cur, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
            let no = self.s.m.bool(false);
            let visible = self.s.m.cmp(self.cur, pred::EQ, helper, no);
            self.begin_guarded(Some(visible))
        } else {
            None
        };
        // Only lanes that can perform the write take part in this election.
        // An all-helper pixel wave skips both the election and the mark.
        let c = self.cur;
        let f = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[n]);
        let first = self
            .s
            .op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        let differ = self.s.m.cmp(c, pred::NE, n, f);
        let mark = self.s.m.bin(c, Bin::Or, differ, first);
        let no = self.s.m.bool(false);
        Some(self.end_guarded(skip_helpers, &[(mark, no)])[0])
    }
}
