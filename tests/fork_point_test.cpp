// The fork point at a release: which slot point goes to disk beside the tail.
#include "../engine/runtime/fork_point.h"
#include <cassert>
#include <numeric>
using namespace crucible;

static std::vector<int> Tokens(size_t count) {
  std::vector<int> ids(count);
  std::iota(ids.begin(), ids.end(), 1);
  return ids;
}
static SlotCheckpoint Point(const std::vector<int>& ids, size_t length, CheckpointKind kind,
                            const StateTenant& tenant = kLocalTenant, int replay = 0) {
  SlotCheckpoint point{std::vector<int>(ids.begin(), ids.begin() + std::ptrdiff_t(length)), {}, replay, false, kind, 1};
  point.tenant = tenant;
  if (kind == CheckpointKind::Frontier) point.frontier = {length, false, false};
  return point;
}
static size_t None(size_t) { return 0; }

int main() {
  const auto tail = Tokens(30000);
  StateTenant other{};
  other[0] = 7;
  // The longest End, Pre or Grid point wins; a Frontier point keeps its own handover.
  {
    SlotCheckpoints c(1);
    c.Store(0, 0, Point(tail, 20000, CheckpointKind::End));
    c.Store(0, 1, Point(tail, 24000, CheckpointKind::Grid));
    c.Store(0, 2, Point(tail, 28000, CheckpointKind::Frontier));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, None) == 1);
    c.Store(0, 1, Point(tail, 22000, CheckpointKind::Pre));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, None) == 1);
    c.Store(0, 1, Point(tail, 18000, CheckpointKind::Pre));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, None) == 0);
  }
  // Excluded: another tenant, a Start point, pending GDN rows, off the 4-row grid, below min_tokens, a key that is not a
  // prefix of the tail, too near the tail end.
  {
    auto off = tail;
    off[100] = -1;
    for (const auto& point : {Point(tail, 24000, CheckpointKind::End, other), Point(tail, 24000, CheckpointKind::Start),
                              Point(tail, 24000, CheckpointKind::End, kLocalTenant, 3),
                              Point(tail, 24001, CheckpointKind::End), Point(tail, 16380, CheckpointKind::End),
                              Point(off, 24000, CheckpointKind::End), Point(tail, 29800, CheckpointKind::Grid)}) {
      SlotCheckpoints c(1);
      c.Store(0, 0, point);
      assert(!PickForkPoint(c, 0, tail, kLocalTenant, 0, None));
    }
    SlotCheckpoints c(1);
    c.Store(0, 0, Point(tail, 16384, CheckpointKind::End));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, None) == 0);  // min_tokens itself is allowed
    c.Store(0, 0, Point(tail, 29744, CheckpointKind::End));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, None) == 0);  // tail - P = 256 is allowed
  }
  // The floor: a point less than 8,192 rows above the disk state that the turn restored gains too little.
  {
    SlotCheckpoints c(1);
    c.Store(0, 0, Point(tail, 20000, CheckpointKind::End));
    c.Store(0, 1, Point(tail, 24000, CheckpointKind::Grid));
    assert(!PickForkPoint(c, 0, tail, kLocalTenant, 0, [](size_t p) { return p >= 20000 ? size_t(20000) : 0; }));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, [](size_t p) { return p >= 20000 ? size_t(15808) : 0; }) == 1);
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 0, [](size_t p) { return p > 20000 ? size_t(20000) : 0; }) == 0);
  }
  // A point at or below what the open keeps (a resume from a point at or above it) is not a fork point.
  {
    SlotCheckpoints c(1);
    c.Store(0, 0, Point(tail, 20000, CheckpointKind::End));
    c.Store(0, 1, Point(tail, 24000, CheckpointKind::Grid));
    assert(PickForkPoint(c, 0, tail, kLocalTenant, 20000, None) == 1);  // the 20,000-row point stays on the slot
    assert(!PickForkPoint(c, 0, tail, kLocalTenant, 24000, None));
    assert(!PickForkPoint(c, 0, tail, kLocalTenant, SIZE_MAX, None));
  }
  // An empty slot: none.
  {
    SlotCheckpoints c(2);
    assert(!PickForkPoint(c, 1, tail, kLocalTenant, 0, None));
  }
  return 0;
}
