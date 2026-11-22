#include "../engine/runtime/slot_checkpoints.h"
#include <cassert>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
using namespace crucible;

static std::vector<int> Tokens(size_t count, int first = 1) {
  std::vector<int> ids(count); std::iota(ids.begin(), ids.end(), first); return ids;
}
static std::span<const int> Below(const std::vector<int>& ids, size_t length) { return {ids.data(), length}; }
static SlotCheckpoint Point(const std::vector<int>& ids, size_t length, CheckpointKind kind, uint64_t owner) {
  return {std::vector<int>(ids.begin(), ids.begin() + std::ptrdiff_t(length)), {}, 0, false, kind, owner};
}

static SlotCheckpoint Frontier(const std::vector<int>& ids, size_t length, uint64_t owner, bool speculative = false) {
  auto point = Point(ids, length, CheckpointKind::Frontier, owner);
  point.frontier = {length, speculative, false};
  return point;
}

// Frontier points.
static void Frontiers() {
  const auto prompt = Tokens(100);
  SlotCheckpoints c(2);
  // A Frontier point never takes a general buffer through Victim, and a general point never takes the Frontier
  // buffer.
  bool threw = false;
  try { c.Victim(0, Below(prompt, 8), CheckpointKind::Frontier, 1, kLocalTenant); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  // Request 1: its confirmed frontier (8) takes the Frontier buffer, its header grid point (16) an empty general
  // buffer (MEDIUM-4: both stay).
  auto pick = c.FrontierVictim(0, Below(prompt, 8), 1, kLocalTenant);
  assert(!pick.held && pick.index == kFrontierCheckpoint);
  c.Store(0, *pick.index, Frontier(prompt, 8, 1));
  pick = c.FrontierVictim(0, Below(prompt, 16), 1, kLocalTenant);
  assert(!pick.held && pick.index == 0);
  c.Store(0, 0, Frontier(prompt, 16, 1, true));
  assert((c.Frontiers(0) == std::vector<int>{0, kFrontierCheckpoint}) && c.FrontierLength(0) == 16);
  pick = c.FrontierVictim(0, Below(prompt, 16), 1, kLocalTenant);
  assert(pick.held && pick.index == 0);  // the slot holds it already, in buffer 0
  // The End point of the prompt takes the other general buffer; a third frontier of the same request then finds no
  // buffer: the host copies it.
  c.Store(0, *c.Victim(0, Below(prompt, 20), CheckpointKind::End, 1, kLocalTenant), Point(prompt, 20, CheckpointKind::End, 1));
  pick = c.FrontierVictim(0, Below(prompt, 24), 1, kLocalTenant);
  assert(!pick.held && !pick.index);
  // A general point leaves the shortest general point: here the header Frontier point (16) before the End point.
  assert(*c.Victim(0, Below(prompt, 40), CheckpointKind::Grid, 2, kLocalTenant) == 0);
  bool stored = false;
  try { c.Store(0, kFrontierCheckpoint, Point(prompt, 40, CheckpointKind::Grid, 2)); } catch (const std::invalid_argument&) {
    stored = true;
  }
  assert(stored);
  // The next request's frontier replaces the older request's point in the Frontier buffer (a prefix of it).
  pick = c.FrontierVictim(0, Below(prompt, 32), 2, kLocalTenant);
  assert(!pick.held && pick.index == kFrontierCheckpoint);
  c.Store(0, kFrontierCheckpoint, Frontier(prompt, 32, 2, true));
  // Best with the host rule: a refused Frontier point leaves the next longest point.
  assert(*c.Best(0, prompt, kLocalTenant) == kFrontierCheckpoint);
  const auto host_rule = [](const SlotCheckpoint& point) { return point.kind != CheckpointKind::Frontier; };
  assert(c.At(0, *c.Best(0, prompt, kLocalTenant, host_rule)).hist.size() == 20);
  // A hit confirms a speculative point; Confirm leaves other kinds alone.
  assert(c.At(0, kFrontierCheckpoint).frontier.speculative);
  c.Confirm(0, kFrontierCheckpoint);
  c.Confirm(0, 1);
  assert(!c.At(0, kFrontierCheckpoint).frontier.speculative && c.At(0, 1).kind == CheckpointKind::End);
  // HIGH-1: a Frontier point needs a whole 128-row pool block and prefill chunk, and no pending rows.
  assert(SlotFrontierGrid(4096, 4096) && SlotFrontierGrid(256, 128) && SlotFrontierGrid(128, 8));
  assert(!SlotFrontierGrid(8, 8) && !SlotFrontierGrid(136, 8) && !SlotFrontierGrid(0, 8) && !SlotFrontierGrid(4096, 0));
  assert(!SlotFrontierGrid(4096, 4096, 3) && !SlotFrontierGrid(2048, 4096) && !SlotFrontierGrid(300, 100));
  // Keep drops Frontier points as any other point.
  c.Keep(0, 24);
  assert(c.Frontiers(0) == std::vector<int>{0} && c.FrontierLength(0) == 16 && c.FrontierLength(1) == 0);
}

// W1 (design section 3.2): a point keeps its tenant. Best and Common find nothing of another tenant; a point
// of another tenant with the same key is stale (Victim replaces it, FrontierVictim does not call it held).
static void TenantPoints() {
  const auto prompt = Tokens(100);
  StateTenant a{}, b{};
  a[0] = 1;
  b[0] = 2;
  SlotCheckpoints c(1);
  auto end = Point(prompt, 20, CheckpointKind::End, 1);
  end.tenant = a;
  c.Store(0, *c.Victim(0, Below(prompt, 20), CheckpointKind::End, 1, a), end);
  auto frontier = Frontier(prompt, 16, 1);
  frontier.tenant = a;
  c.Store(0, kFrontierCheckpoint, frontier);
  assert(c.Best(0, prompt, a) && c.Common(0, prompt, a) == 20);
  assert(!c.Best(0, prompt, b) && c.Common(0, prompt, b) == 0 && !c.Best(0, prompt, kLocalTenant));
  assert(!c.Victim(0, Below(prompt, 20), CheckpointKind::End, 2, a));  // A's slot holds it
  assert(c.Victim(0, Below(prompt, 20), CheckpointKind::End, 2, b) == 0);  // B's state with A's key: replace A's
  assert(c.FrontierVictim(0, Below(prompt, 16), 1, a).held);
  const auto pick = c.FrontierVictim(0, Below(prompt, 16), 2, b);
  assert(!pick.held && pick.index == kFrontierCheckpoint);
  assert(c.At(0, kFrontierCheckpoint).tenant == a && SlotCheckpoint{}.tenant == kLocalTenant);
}

// Re-send fix A: an End point that keeps the prompt's outputs (the first token's greedy ID, its logits
// row, the MTP carry) serves an identical prompt with no prompt row. Every other point needs one more row.
// Fix B: a Pre point (a chunk end at the final chunk's core cut) serves a prompt that extends it.
static void EndOutputs() {
  const auto prompt = Tokens(60);
  const auto outputs = std::make_shared<const PromptOutputs>(
      PromptOutputs{7, std::make_shared<const std::vector<float>>(4, 0.f), true});
  {
    SlotCheckpoints c(1);
    c.Store(0, 0, Point(prompt, 60, CheckpointKind::End, 1));
    assert(!c.Best(0, prompt, kLocalTenant) && c.Common(0, prompt, kLocalTenant) == 0);  // no outputs: the last row must run
    auto end = Point(prompt, 60, CheckpointKind::End, 1);
    end.outputs = outputs;
    c.Store(0, 0, end);
    assert(*c.Best(0, prompt, kLocalTenant) == 0 && c.Common(0, prompt, kLocalTenant) == 60);
    assert(*c.Best(0, Tokens(70), kLocalTenant) == 0);  // a longer prompt resumes from it as before
    assert(!c.Best(0, Tokens(59), kLocalTenant));       // a shorter prompt does not
    auto other = prompt; other[59] = -1;  // the same length with another last token
    assert(!c.Best(0, other, kLocalTenant));
    // A host rule that refuses it leaves no point.
    assert(!c.Best(0, prompt, kLocalTenant, [](const SlotCheckpoint& point) { return point.kind != CheckpointKind::End; }));
    c.Keep(0, 60);  // a restore at the prompt end keeps the point
    assert(*c.Best(0, prompt, kLocalTenant) == 0);
    c.Keep(0, 59);
    assert(!c.Best(0, prompt, kLocalTenant));
  }
  // Start, Grid, Pre and Frontier points of the prompt's length stay refused, also with outputs.
  for (const auto kind : {CheckpointKind::Start, CheckpointKind::Grid, CheckpointKind::Pre, CheckpointKind::Frontier}) {
    SlotCheckpoints c(1);
    auto point = kind == CheckpointKind::Frontier ? Frontier(prompt, 60, 1) : Point(prompt, 60, kind, 1);
    point.outputs = outputs;
    c.Store(0, kind == CheckpointKind::Frontier ? kFrontierCheckpoint : 0, point);
    assert(!c.Best(0, prompt, kLocalTenant));
  }
  // A cold prompt keeps its Pre point and its End point in the two general buffers. An extended prompt that
  // repeats the rows below the Pre point resumes there; an identical one resumes at the End point.
  {
    SlotCheckpoints c(1);
    assert(*c.Victim(0, Below(prompt, 48), CheckpointKind::Pre, 1, kLocalTenant) == 0);
    c.Store(0, 0, Point(prompt, 48, CheckpointKind::Pre, 1));
    auto end = Point(prompt, 60, CheckpointKind::End, 1);
    end.outputs = outputs;
    assert(*c.Victim(0, Below(prompt, 60), CheckpointKind::End, 1, kLocalTenant) == 1);
    c.Store(0, 1, end);
    auto extended = prompt; extended[52] = -3; extended.insert(extended.end(), {1, 2, 3});
    assert(*c.Best(0, extended, kLocalTenant) == 0 && c.Common(0, extended, kLocalTenant) == 48);
    assert(*c.Best(0, prompt, kLocalTenant) == 1);
    assert(!c.Victim(0, Below(prompt, 60), CheckpointKind::End, 2, kLocalTenant));  // the re-send saves no second End point
  }
}

// cache-thinkcap: the event order of a think-capped chat turn on one slot. The first
// job keeps its Pre point (2,432) and End point (2,499); the continuation resumes at the Pre point (Keep drops the End
// point) and saves its own Pre (2,524) and End (2,588) points, past the client's boundary (2,494). With the boundary the
// End save leaves the continuation's Pre point, so the client's next prompt resumes at 2,432; without it (the old rule,
// boundary 0) the shortest point leaves and the next prompt finds no point.
static void BoundaryKeepsTheClientsPoint() {
  auto continuation = Tokens(2588);
  const std::vector<int> prompt(continuation.begin(), continuation.begin() + 2499);
  for (const size_t boundary : {size_t(0), size_t(2494)}) {
    SlotCheckpoints c(1);
    c.Store(0, *c.Victim(0, Below(prompt, 2432), CheckpointKind::Pre, 1, kLocalTenant), Point(prompt, 2432, CheckpointKind::Pre, 1));
    c.Store(0, *c.Victim(0, Below(prompt, 2499), CheckpointKind::End, 1, kLocalTenant), Point(prompt, 2499, CheckpointKind::End, 1));
    c.Keep(0, 2432);
    const auto pre = c.Victim(0, Below(continuation, 2524), CheckpointKind::Pre, 2, kLocalTenant, boundary);
    assert(pre == 1);  // an empty buffer first
    c.Store(0, *pre, Point(continuation, 2524, CheckpointKind::Pre, 2));
    const auto end = c.Victim(0, Below(continuation, 2588), CheckpointKind::End, 2, kLocalTenant, boundary);
    assert(end == (boundary ? 1 : 0));
    c.Store(0, *end, Point(continuation, 2588, CheckpointKind::End, 2));
    const auto best = c.Best(0, prompt, kLocalTenant);
    assert(boundary ? best && c.At(0, *best).hist.size() == 2432 : !best);
    assert(c.At(0, *c.Best(0, Tokens(2700), kLocalTenant)).hist.size() == 2588);  // the tool loop resumes at the End point
  }
  // The kept point is the newest at or below the boundary, of the request's tenant; a later grid point of the same
  // request does not replace it; with one general point in all (the other buffer empty) the empty one goes first.
  SlotCheckpoints c(1);
  const auto ids = Tokens(300);
  c.Store(0, 0, Point(ids, 100, CheckpointKind::Grid, 6));
  c.Store(0, 1, Point(ids, 120, CheckpointKind::Grid, 7));
  assert(c.KeptPoint(0, 110, kLocalTenant) == 0 && c.KeptPoint(0, 120, kLocalTenant) == 1 && !c.KeptPoint(0, 99, kLocalTenant));
  assert(!c.KeptPoint(0, 0, kLocalTenant));
  StateTenant other{};
  other[0] = 9;
  assert(!c.KeptPoint(0, 200, other));
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::Grid, 7, kLocalTenant) == 1);       // the old rule: its grid point
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::Grid, 7, kLocalTenant, 110) == 1);  // 100 stays: the same
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::Grid, 7, kLocalTenant, 130) == 0);  // 120 stays, 100 goes
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::End, 8, kLocalTenant) == 0);        // the old rule: the shortest
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::End, 8, kLocalTenant, 110) == 1);   // 100 stays, 120 goes
  c.Keep(0, 110);
  assert(c.Victim(0, Below(ids, 140), CheckpointKind::End, 8, kLocalTenant, 110) == 1);   // an empty buffer first
}

// The old victim rule (boundary 0), as it was before cache-thinkcap: the reference of the property test below.
static std::optional<int> OldVictim(const std::array<std::optional<SlotCheckpoint>, kSlotCheckpoints>& held,
                                    std::span<const int> key, CheckpointKind kind, uint64_t owner) {
  for (int i = 0; i < kSlotCheckpoints; ++i)
    if (held[i] && held[i]->key.size() == key.size()) {
      if (std::equal(key.begin(), key.end(), held[i]->key.begin())) return std::nullopt;
      if (i < kGeneralCheckpoints) return i;
    }
  if (kind == CheckpointKind::Grid)
    for (int i = 0; i < kGeneralCheckpoints; ++i)
      if (held[i] && held[i]->owner == owner && held[i]->kind == CheckpointKind::Grid) return i;
  int victim = 0;
  for (int i = 0; i < kGeneralCheckpoints; ++i) {
    if (!held[i]) return i;
    if (held[i]->hist.size() < held[victim]->hist.size()) victim = i;
  }
  return victim;
}

// Property (cache-thinkcap): random event orders on one slot, as the session calls the store. Requests restore at a
// held point (Keep), reset or restore from the host (Keep 0), run rows past the cursor and save Start, Grid, Pre, End and
// Frontier points; a save names the request's boundary (0 for a client prompt, KeepBoundary for a continuation).
// After every save:
//   - the new point is held, every held point is a prefix of the slot's rows;
//   - boundary 0: the victim is the old rule's (normal requests keep the old behavior);
//   - boundary B: the newest point at or below B before the save, or a newer one at or below B, is still held (the
//     store has room: two general buffers, one for that point and one for the new point).
static void BoundaryProperty() {
  std::mt19937_64 random(20261118);
  int saves = 0, kept_checks = 0;
  for (int run = 0; run < 400; ++run) {
    SlotCheckpoints c(1);
    std::vector<int> rows;  // the slot's rows
    std::array<std::optional<SlotCheckpoint>, kSlotCheckpoints> mirror{};  // what the store holds, by index
    uint64_t owner = 0;
    size_t boundary = 0;
    const auto held_points = [&] {
      std::vector<std::pair<int, size_t>> found;
      c.ForEach(0, [&](int index, const SlotCheckpoint& point) { found.emplace_back(index, point.hist.size()); });
      return found;
    };
    for (int event = 0; event < 60; ++event) {
      const int what = int(random() % 10);
      if (what == 0) {  // reset, or a host restore
        c.Keep(0, 0);
        rows.clear();
        mirror = {};
      } else if (what <= 2) {  // a new request: it restores at a held point (or keeps the rows), with a boundary
        const auto points = held_points();
        size_t keep = rows.size();
        if (!points.empty() && random() % 3) keep = points[random() % points.size()].second;
        c.Keep(0, keep);
        rows.resize(keep);
        for (auto& item : mirror) if (item && item->hist.size() > keep) item.reset();
        ++owner;
        // A client prompt (boundary 0 for KeepBoundary), or a continuation whose client boundary lies near the restore.
        boundary = random() % 2 ? 0 : keep + random() % 9 - std::min<size_t>(keep, 4);
        if (boundary > 0 && boundary < keep / 2) boundary = keep;
      } else {  // the request runs rows and saves a point at the cursor
        const size_t more = 1 + random() % 40;
        for (size_t i = 0; i < more; ++i) rows.push_back(int(random() % 4));
        const std::span<const int> key(rows.data(), rows.size());
        if (what == 9) {  // a Frontier point
          const auto pick = c.FrontierVictim(0, key, owner, kLocalTenant);
          if (!pick.held && pick.index) {
            auto point = Frontier(rows, rows.size(), owner);
            c.Store(0, *pick.index, point);
            mirror[size_t(*pick.index)] = point;
          }
          continue;
        }
        const CheckpointKind kinds[] = {CheckpointKind::Start, CheckpointKind::Grid, CheckpointKind::Pre,
                                        CheckpointKind::End};
        const CheckpointKind kind = kinds[random() % 4];
        std::optional<size_t> before;  // the newest point at or below the boundary
        for (int i = 0; i < kGeneralCheckpoints; ++i)
          if (boundary && mirror[i] && mirror[i]->hist.size() <= boundary)
            before = std::max(before.value_or(0), mirror[i]->hist.size());
        const auto victim = c.Victim(0, key, kind, owner, kLocalTenant, boundary);
        if (!boundary) assert(victim == OldVictim(mirror, key, kind, owner));
        if (!victim) continue;
        ++saves;
        auto point = Point(rows, rows.size(), kind, owner);
        c.Store(0, *victim, point);
        mirror[size_t(*victim)] = point;
        assert(c.At(0, *victim).hist == rows);
        if (before) {
          ++kept_checks;
          bool kept = false;
          for (int i = 0; i < kGeneralCheckpoints; ++i)
            kept |= mirror[i] && mirror[i]->hist.size() >= *before && mirror[i]->hist.size() <= boundary;
          assert(kept);
        }
      }
      c.ForEach(0, [&](int index, const SlotCheckpoint& point) {
        assert(mirror[size_t(index)] && mirror[size_t(index)]->hist == point.hist);
        assert(point.hist.size() <= rows.size() && std::equal(point.hist.begin(), point.hist.end(), rows.begin()));
      });
    }
  }
  assert(saves > 5000 && kept_checks > 1000);
  std::printf("BoundaryProperty: %d saves, %d with a kept point\n", saves, kept_checks);
}

int main() {
  BoundaryKeepsTheClientsPoint();
  BoundaryProperty();
  TenantPoints();
  static_assert(kGeneralCheckpoints == 2 && kSlotCheckpoints == 3, "a development branch T4 and one Frontier buffer");
  // An earlier change: fix A's logits rows per stream: one per general buffer (End points) and the in-flight row of
  // a final chunk (Request::outputs) before its End save.
  static_assert(EndLogitsBytes(4, 248320) == size_t(4) * 3 * 248320 * sizeof(float));
  const auto prompt = Tokens(100);
  SlotCheckpoints c(2);
  // One replayed turn takes three points in this order: its start on the tail (20), its last grid point (40) and
  // its prompt end (60). With two buffers the turn start leaves at the prompt end (it is the shortest).
  for (const auto& [length, kind] : {std::pair{20, CheckpointKind::Start}, {40, CheckpointKind::Grid},
                                     {60, CheckpointKind::End}}) {
    const auto victim = c.Victim(0, Below(prompt, size_t(length)), kind, 1, kLocalTenant);
    assert(victim);
    c.Store(0, *victim, Point(prompt, size_t(length), kind, 1));
  }
  assert(c.At(0, 0).hist.size() == 60 && c.At(0, 0).kind == CheckpointKind::End);
  assert(c.At(0, 1).hist.size() == 40 && c.At(0, 1).kind == CheckpointKind::Grid);
  // A later grid point of the same request replaces its earlier one.
  assert(*c.Victim(0, Below(prompt, 50), CheckpointKind::Grid, 1, kLocalTenant) == 1);
  c.Store(0, 1, Point(prompt, 50, CheckpointKind::Grid, 1));
  // The slot already holds that state: nothing to copy.
  assert(!c.Victim(0, Below(prompt, 60), CheckpointKind::Start, 2, kLocalTenant));
  // Same length, other key: not the same state. Under the invariant (every point is a prefix of
  // the slot's rows, Keep) this cannot happen; if it does, the stale point is the one to replace.
  auto stale = prompt; stale[7] += 1 << 20;
  assert(*c.Victim(0, Below(stale, 60), CheckpointKind::Start, 2, kLocalTenant) == 0);
  // The longest checkpoint that a prompt extends by at least one token.
  assert(*c.Best(0, prompt, kLocalTenant) == 0 && c.At(0, 0).hist.size() == 60);
  auto split = prompt; split[55] = -1;  // a token split inside the previous answer
  assert(c.At(0, *c.Best(0, split, kLocalTenant)).hist.size() == 50 && c.Common(0, split, kLocalTenant) == 50);
  // A retry of the same request: the prompt-end point needs one more row, so the grid point serves it, as
  // with three points.
  const std::vector<int> retry(prompt.begin(), prompt.begin() + 60);
  assert(c.At(0, *c.Best(0, retry, kLocalTenant)).hist.size() == 50);
  // What two points give up: an edit between the turn start and the grid point (here at row 30) finds no point;
  // three points resumed it at the turn start (20). The host frontiers still serve it.
  auto edit = prompt; edit[30] = -4;
  assert(!c.Best(0, edit, kLocalTenant) && c.Common(0, edit, kLocalTenant) == 0);
  assert(!c.Best(0, std::vector<int>(prompt.begin(), prompt.begin() + 20), kLocalTenant));  // the last row must run
  assert(!c.Best(1, prompt, kLocalTenant) && c.Common(1, prompt, kLocalTenant) == 0);
  auto other = prompt; other[3] = -2;
  assert(!c.Best(0, other, kLocalTenant));
  // A new request: the shortest checkpoint leaves first.
  assert(*c.Victim(0, Below(prompt, 70), CheckpointKind::End, 2, kLocalTenant) == 1);
  // A restore at 50 keeps the rows below it; a reset or host restore keeps none.
  c.Keep(0, 50);
  assert(c.At(0, *c.Best(0, prompt, kLocalTenant)).hist.size() == 50);
  c.Keep(0, 0);
  assert(!c.Best(0, prompt, kLocalTenant) && *c.Victim(0, Below(prompt, 10), CheckpointKind::End, 3, kLocalTenant) == 0);
  bool threw = false;
  try { c.Store(0, 2, Point(prompt, 10, CheckpointKind::End, 3)); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  // Points compare cache keys: the same tokens under another key (another image) do not match.
  auto keyed = Point(prompt, 30, CheckpointKind::End, 4);
  keyed.key = keyed.hist;
  keyed.key[5] += 1 << 20;
  c.Store(1, 0, keyed);
  assert(!c.Best(1, prompt, kLocalTenant) && *c.Best(1, keyed.key, kLocalTenant) == 0 && c.At(1, 0).hist.size() == 30);
  auto stored = c.At(1, 0).key;
  stored.insert(stored.end(), prompt.begin() + 30, prompt.end());
  assert(c.Common(1, stored, kLocalTenant) == 30);
  c.Store(0, 0, Point(prompt, 10, CheckpointKind::End, 5));  // an empty key is the tokens
  assert(c.At(0, 0).key == c.At(0, 0).hist);
  threw = false;
  keyed.key.pop_back();
  try { c.Store(1, 1, keyed); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  Frontiers();
  EndOutputs();
  return 0;
}
