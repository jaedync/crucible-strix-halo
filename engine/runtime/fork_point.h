// The fork point at a release. When an open of another session takes a slot, the
// slot's tail goes to the host or the disk (Release), and the slot's points leave. The longest general point below
// the tail can go to disk too, as a derived state (StateDisk::SpillDerived): its full blocks are the tail's blocks, so
// it costs one state file. A later prompt that leaves the tail inside the last turn's output resumes there.
// A fork point is a tail-class state: it restores with the tail contract (contract A, docs/SESSION_REPLAY.md).
#pragma once
#include "slot_checkpoints.h"
#include <algorithm>
#include <optional>
#include <span>

namespace crucible {
struct ForkRule {
  size_t min_tokens = 16384;  // the disk tier's min_tokens: a shorter state is cheaper to prefill
  size_t min_gain = 8192;     // rows above the disk state that the turn restored (else the point gains too little)
  size_t tail_slack = 256;    // rows below the tail (kLedgerSlack, miss_ledger.h): a nearer point is the tail itself
  size_t grid = 4;            // the CRUCIBLE_EXACT_RESUME grid (slot_checkpoints.h)
};

// The point of `slot` to write beside the tail `tail_key`, or nullopt. A candidate is an End, Pre or Grid point of
// `tenant` with no pending GDN rows, whose key is a prefix of the tail, on the grid, at least min_tokens long, longer
// than `keep` (what the open keeps of the slot), at least tail_slack rows below the tail, and at least min_gain rows
// above floor(P) (the longest disk state at or below P that the tail extends). The longest candidate wins.
template<class Floor>
std::optional<int> PickForkPoint(const SlotCheckpoints& points, int slot, std::span<const int> tail_key,
                                 const StateTenant& tenant, size_t keep, Floor floor, const ForkRule& rule = {}) {
  std::optional<int> best;
  size_t best_rows = 0;
  points.ForEach(slot, [&](int index, const SlotCheckpoint& point) {
    const auto& key = point.key.empty() ? point.hist : point.key;
    const size_t rows = key.size();
    const bool general = point.kind == CheckpointKind::End || point.kind == CheckpointKind::Pre ||
                         point.kind == CheckpointKind::Grid;
    if (!general || point.tenant != tenant || point.replay != 0 || rows <= best_rows) return;
    if (rows % rule.grid || rows < rule.min_tokens || rows <= keep || rows + rule.tail_slack > tail_key.size()) return;
    if (!std::equal(key.begin(), key.end(), tail_key.begin())) return;
    if (rows - std::min(size_t(floor(rows)), rows) < rule.min_gain) return;
    best = index;
    best_rows = rows;
  });
  return best;
}
}  // namespace crucible
