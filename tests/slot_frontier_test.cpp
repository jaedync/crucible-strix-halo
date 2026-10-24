// slot frontiers in BatchSession. A session frontier of a normal turn stays in
// the slot as a Frontier point; it resumes as a host frontier hit does; any open that does not keep it copies it to
// the host first (the guard, HIGH-1); a resume from it drops the slot's tail (HIGH-2). Fake hooks keep a host cache
// of token keys, as batch_session_test does.
#include "../engine/serve/batch_session.h"
#include "../engine/runtime/sha256.h"
#include <cstdio>
#include <utility>
#include <cassert>
#include <map>
#include <set>
#include <string>

struct MockEngine {
  int selected = 0, prefill_rows = 4, launch_rows = 8;
  std::map<int, std::vector<int>> histories;
  std::vector<int> last;
  std::vector<int> chunks;  // rows per Forward call; negative for Prefill (the bulk path)
  bool in_prefill = false;
  int Context() const { return 64; }
  std::vector<int> capacities;
  int PrefillRows() const { return prefill_rows; }
  int PrefillLaunchRows() const { return launch_rows; }
  void ConfigureServeStreams(int count, uint64_t = 0, int = 0, int = 0) { capacities.assign(size_t(count), Context()); }
  int StreamCapacity(int slot) const { return capacities.at(size_t(slot)); }
  void SelectStream(int slot) { selected = slot; }
  void Reset() { histories[selected].clear(); }
  std::vector<int> Forward(const int* ids, int rows, int, int) {
    last.clear();
    chunks.push_back(in_prefill ? -rows : rows);
    for (int row = 0; row < rows; ++row) {
      histories[selected].push_back(ids[row]);
      last.push_back((ids[row] + 1) % 16);
    }
    return last;
  }
  std::vector<int> Prefill(const int* ids, int rows, const std::string&, bool predict,
                           const crucible::PrefillBetween* = nullptr, int = 0) {
    in_prefill = true;
    const auto output = Forward(ids, rows, 0, rows);
    in_prefill = false;
    last = {output.back()};
    return predict ? last : std::vector<int>{};
  }
  std::vector<int> ForwardBatch(const std::vector<int>& ids, const std::vector<int>& slots) {
    last.clear();
    for (size_t row = 0; row < ids.size(); ++row) {
      histories[slots[row]].push_back(ids[row]);
      last.push_back((ids[row] + 1) % 16);
    }
    return last;
  }
  int MaxPassRows() const { return crucible::kPackedRows; }
  std::vector<int> ForwardVerifySpans(const std::vector<crucible::VerifySpan>&) { return {}; }
  std::vector<float> CopyLogits(int, int row = 0) const {
    std::vector<float> scores(16);
    for (int id = 0; id < 16; ++id) scores[size_t(id)] = -float(std::abs(last[size_t(row)] - id));
    return scores;
  }
};

// The host side: frontiers and tails by key, the frontier rule of BatchCacheFrontiers (shared, own, header), and the
// hooks of BatchPrefixCache with a log.
struct FakeCache {
  static constexpr size_t kRows = 4;
  MockEngine& engine;
  bool slot_frontiers = true, refuse_fit = false, host_tail_first = false;
  // frontier-copy: what the copy hook does. kRefuse and kThrow leave the donor's stream selected on purpose, so the
  // session must select the target again.
  enum class Copy { kCopy, kRefuse, kThrow, kOff } copy = Copy::kCopy;
  std::vector<int> restore_slots;  // the selected stream at each host restore
  std::map<std::vector<int>, bool> host;  // key -> is a tail
  // W1: the tenant of a host entry, as BatchPrefixCache's policy keeps it (an entry without one: any tenant).
  std::map<std::vector<int>, crucible::StateTenant> owner;
  std::map<std::pair<int, int>, std::vector<int>> buffers;
  std::vector<std::string> log;
  struct Frontier { size_t position; bool shared, speculative; };
  static std::vector<Frontier> Frontiers(const crucible::ServeRequest& r) {
    const size_t length = r.ids.size(), boundary = size_t(r.cache_prefix_tokens);
    const size_t own = (boundary ? std::min(boundary, length - 1) : length - 1) / kRows * kRows;
    const size_t common = r.shared_prefix_tokens ? std::min(size_t(r.shared_prefix_tokens), own) / kRows * kRows : 0;
    std::vector<Frontier> out;
    if (common && common != own) out.push_back({common, true, false});
    if (own) out.push_back({own, common == own, false});
    const size_t header = (length - 1) / kRows * kRows;
    if (boundary && header > own && header > common) out.push_back({header, false, true});
    return out;
  }
  size_t Longest(const crucible::ServeRequest& r, bool tails) const {
    const size_t stable = r.cache_prefix_tokens ? size_t(r.cache_prefix_tokens) : r.ids.size();
    size_t best = 0;
    for (const auto& [key, tail] : host)
      if (tail == tails && key.size() < r.ids.size() && (tails || key.size() <= stable) && key.size() > best &&
          (!owner.contains(key) || owner.at(key) == r.tenant) && std::equal(key.begin(), key.end(), r.ids.begin()))
        best = key.size();
    return best;
  }
  void Write(const crucible::ServeRequest& r, size_t length) {
    engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(length));
  }
  crucible::BatchCacheHooks Hooks() {
    crucible::BatchCacheHooks hooks;
    hooks.restore = [this](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      restore_slots.push_back(engine.selected);
      if (r.cache_control != "auto") return 0;
      if (const size_t tail = Longest(r, true)) {
        Write(r, tail);
        log.push_back("restore tail " + std::to_string(tail));
        return {tail, false, 0, true};
      }
      const size_t best = Longest(r, false);
      log.push_back("restore " + std::to_string(best));
      if (best) Write(r, best);
      return best;
    };
    hooks.peek = [this](const crucible::ServeRequest& r) { return std::max(Longest(r, true), Longest(r, false)); };
    hooks.capture = [this](const crucible::ServeRequest& r, size_t cursor) {
      for (const auto& frontier : Frontiers(r))
        if (frontier.position == cursor) {
          host[std::vector<int>(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(cursor))] = false;
          owner[std::vector<int>(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(cursor))] = r.tenant;
          log.push_back("capture " + std::to_string(cursor));
        }
    };
    hooks.next_capture = [](const crucible::ServeRequest& r, size_t cursor) {
      size_t next = SIZE_MAX;
      for (const auto& frontier : Frontiers(r)) if (frontier.position > cursor) next = std::min(next, frontier.position);
      return next;
    };
    hooks.tail = [this](const crucible::ServeRequest& r, std::span<const int> generated, int,
                        crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
      std::vector<int> hist(r.ids);
      hist.insert(hist.end(), generated.begin(), generated.end() - 1);
      engine.histories[engine.selected] = hist;
      log.push_back("tail " + std::to_string(hist.size()));
      if (r.cache_tail == "skip") return std::nullopt;  // ConversationTailState found no tail
      held[engine.selected] = hist;
      held_tenant[engine.selected] = r.tenant;
      return crucible::BatchResident{hist.size(), 0, false};
    };
    hooks.resume = [this](const crucible::ServeRequest& r, crucible::BatchResident resident) -> crucible::BatchRestore {
      if (host_tail_first) {
        const size_t tail = Longest(r, true);
        Write(r, tail);
        log.push_back("resume host tail " + std::to_string(tail));
        return {tail, false, 0, true};
      }
      log.push_back("resume slot " + std::to_string(resident.length));
      return {resident.length, false, 0, true, true};
    };
    hooks.release = [this](int slot, bool resumed) {
      if (held.contains(slot)) host[held[slot]] = true, owner[held[slot]] = held_tenant[slot];
      held.erase(slot);
      log.push_back("release slot=" + std::to_string(slot) + " resumed=" + std::to_string(resumed));
    };
    hooks.checkpoint_save = [this](int slot, int index) {
      assert(slot == engine.selected);
      buffers[{slot, index}] = engine.histories[slot];
    };
    hooks.checkpoint_load = [this](int slot, int index, const crucible::SlotCheckpoint& point) {
      const auto& stream = engine.histories[slot];
      assert(buffers.at({slot, index}) == point.hist && stream.size() >= point.hist.size() &&
             std::equal(point.hist.begin(), point.hist.end(), stream.begin()));
      engine.histories[slot] = point.hist;
      log.push_back("load slot=" + std::to_string(slot) + " at " + std::to_string(point.hist.size()) +
                    (point.kind == crucible::CheckpointKind::Frontier ? " frontier" : ""));
    };
    if (!slot_frontiers) return hooks;
    hooks.frontier = [this](const crucible::ServeRequest& r, size_t cursor,
                            const std::function<bool(const crucible::SlotFrontier&)>& keep) {
      for (const auto& frontier : Frontiers(r)) {
        if (frontier.position != cursor) continue;
        if (frontier.shared || !keep({size_t(r.cache_prefix_tokens), frontier.speculative, false})) return false;
        log.push_back("keep " + std::to_string(cursor) + (frontier.speculative ? " speculative" : ""));
        return true;
      }
      return false;
    };
    hooks.frontier_fits = [this](const crucible::ServeRequest& r, const crucible::SlotCheckpoint& point) {
      const size_t stable = r.cache_prefix_tokens ? size_t(r.cache_prefix_tokens) : r.ids.size();
      return !refuse_fit && point.hist.size() <= std::min(stable, r.ids.size() - 1) && point.hist.size() % kRows == 0;
    };
    hooks.handover = [this](int slot, int index, const crucible::SlotCheckpoint& point) {
      // The rows below the point are still on the slot, and the buffer holds the state at it.
      const auto& stream = engine.histories[slot];
      assert(slot == engine.selected && buffers.at({slot, index}) == point.hist && stream.size() >= point.hist.size() &&
             std::equal(point.hist.begin(), point.hist.end(), stream.begin()));
      host[point.hist] = false;
      owner[point.hist] = point.tenant;
      log.push_back("handover slot=" + std::to_string(slot) + " at " + std::to_string(point.hist.size()) +
                    (point.frontier.speculative ? " speculative" : "") +
                    (point.tenant != crucible::kLocalTenant ? " tenant" : ""));
    };
    hooks.drop = [this](int slot) {
      held.erase(slot);
      log.push_back("drop slot=" + std::to_string(slot));
    };
    hooks.resume_host = [this](const crucible::ServeRequest&, crucible::BatchResident) { return host_tail_first; };
    if (copy == Copy::kOff) return hooks;
    // frontier-copy: a transient host copy of donor point (from, index), restored into the selected (target) slot.
    hooks.frontier_copy = [this](const crucible::ServeRequest&, int from, int index,
                                 const crucible::SlotCheckpoint& point) -> crucible::BatchRestore {
      const int target = engine.selected;
      const auto& stream = engine.histories[from];
      assert(from != target && point.kind == crucible::CheckpointKind::Frontier && buffers.at({from, index}) == point.hist &&
             stream.size() >= point.hist.size() && std::equal(point.hist.begin(), point.hist.end(), stream.begin()));
      engine.SelectStream(from);  // as FrontierCopy reads the donor's stream
      const std::string names = " slot=" + std::to_string(target) + " from=" + std::to_string(from);
      if (copy == Copy::kThrow) throw std::runtime_error("copy failed");
      if (copy == Copy::kRefuse) {
        log.push_back("copy refused" + names);
        return 0;
      }
      engine.SelectStream(target);
      engine.histories[target] = point.hist;
      log.push_back("copy" + names + " at " + std::to_string(point.hist.size()));
      return point.hist.size();
    };
    return hooks;
  }
  std::map<int, std::vector<int>> held;
  std::map<int, crucible::StateTenant> held_tenant;
};

crucible::ServeRequest Request(std::vector<int> ids, size_t boundary, int max_tokens = 3) {
  crucible::ServeRequest r;
  r.ids = std::move(ids);
  r.max_tokens = max_tokens;
  r.spec = 0;
  r.cache_prefix_tokens = int(boundary);
  return r;
}
std::vector<int> Tokens(int first, size_t count) {
  std::vector<int> ids(count);
  for (size_t i = 0; i < count; ++i) ids[i] = (first + int(i)) % 16;
  return ids;
}
struct Turn { std::vector<int> out; size_t cached = 0; bool checkpoint = false, copy = false; int slot = -1; };
Turn Finish(crucible::BatchSession<MockEngine>& session, MockEngine& engine, uint64_t id,
            const crucible::ServeRequest& r, bool cancel = false) {
  Turn turn;
  auto events = session.Open(id, r);
  turn.slot = engine.selected;
  for (;;) {
    for (const auto& event : events) {
      if (event.token) turn.out.push_back(*event.token);
      if (event.done) {
        turn.cached = event.cached_tokens;
        turn.checkpoint = event.checkpoint_resume;
        turn.copy = event.frontier_copy;
        return turn;
      }
    }
    if (cancel) {
      session.Close(id);
      return turn;
    }
    events = session.Step({id});
  }
}
bool Has(const std::vector<std::string>& log, const std::string& line) {
  return std::find(log.begin(), log.end(), line) != log.end();
}
size_t At(const std::vector<std::string>& log, const std::string& line) {
  return size_t(std::find(log.begin(), log.end(), line) - log.begin());
}

// The trace digest of LocalTenantUnchanged, recorded on train 23 (7b3106e8).
const std::string kLocalTrace23 = "9f706d43f82fea667f9a9024c4c1e4ed056123bd5ff084b372670c54c34ef030";

// Turn 1 of session A: 21 tokens, stable boundary 13. Its frontiers: 12 (confirmed) and 20 (the header grid point,
// speculative). Turn 2 drops the output (Pi): it repeats the header and leaves the tail at row 20.
const std::vector<int> kA1 = Tokens(1, 21);
std::vector<int> NextOf(const std::vector<int>& turn, size_t keep, int first, size_t more) {
  std::vector<int> next(turn.begin(), turn.begin() + std::ptrdiff_t(keep));
  const auto tail = Tokens(first, more);
  next.insert(next.end(), tail.begin(), tail.end());
  return next;
}

// The frontiers stay in the slot, and the next turn resumes from the header point with the chunks of a host hit
// (MEDIUM-1): the same launches, captures and output as the host path, no head forward, and the tail is dropped.
void StoreAndResume() {
  std::vector<std::vector<int>> chunks;
  std::vector<std::vector<int>> outputs;
  for (const bool slot : {true, false}) {
    MockEngine engine;
    FakeCache cache{engine};
    cache.slot_frontiers = slot;
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    const auto t1 = Finish(session, engine, 1, Request(kA1, 13));
    if (slot) {
      assert(Has(cache.log, "keep 12") && Has(cache.log, "keep 20 speculative") && !Has(cache.log, "capture 12"));
      assert(cache.host.empty());  // no host copy of a live session's frontier
      const auto points = session.FrontierPoints();
      assert(points.size() == 2 && std::get<2>(points[0]).hist.size() == 20 && std::get<2>(points[1]).hist.size() == 12);
      assert(std::get<2>(points[0]).frontier.speculative && !std::get<2>(points[1]).frontier.speculative);
    } else {
      assert(Has(cache.log, "capture 12") && Has(cache.log, "capture 20") && cache.host.size() == 2);
    }
    engine.chunks.clear();
    cache.log.clear();
    const auto next = NextOf(kA1, 20, 9, 9);  // 29 tokens: the header repeated, then the new user message
    const auto t2 = Finish(session, engine, 2, Request(next, 24));
    assert(t2.cached == 20 && t2.checkpoint == slot);
    if (slot) {
      assert(cache.log.front() == "drop slot=0" && Has(cache.log, "load slot=0 at 20 frontier"));
      assert(!Has(cache.log, "release slot=0 resumed=0") && cache.host.empty());
      // The hit confirms the header point; turn 2's own frontier (24) replaced turn 1's points in the Frontier buffer.
      const auto points = session.FrontierPoints();
      assert(!points.empty() && std::get<2>(points[0]).hist.size() == 28);
    } else {
      assert(cache.log.front() == "release slot=0 resumed=0" && Has(cache.log, "restore 20"));
    }
    chunks.push_back(engine.chunks);
    outputs.push_back(t1.out);
    outputs.push_back(t2.out);
  }
  // The first chunk after the restore is a bulk chunk on both paths (no small-row head forward).
  assert(chunks[0] == chunks[1] && !chunks[0].empty() && chunks[0].front() < 0);
  assert(outputs[0] == outputs[2] && outputs[1] == outputs[3]);
}

// the server reads a restore from the opened event when it cancels a job before its done event (a capped
// think turn with max_tokens 1). A deferred open that resumes from a Frontier point names the point's length there.
void OpenedNamesTheFrontier() {
  MockEngine engine;
  FakeCache cache{engine};
  crucible::BatchSession<MockEngine> session(engine, 1);
  session.SetCacheHooks(cache.Hooks());
  Finish(session, engine, 1, Request(kA1, 13));
  cache.log.clear();
  const auto opened = session.Open(2, Request(NextOf(kA1, 20, 9, 9), 24, 1), true);
  assert(opened.size() == 1 && !opened[0].done && !opened[0].token && opened[0].cached_tokens == 20);
  assert(Has(cache.log, "load slot=0 at 20 frontier"));
  session.Close(2);  // the cancel: no done event
}

// MEDIUM-2: a Frontier point that the host rule refuses is not loaded; the open then writes the slot from the host
// path, so the guard copies the point first.
void PickRule() {
  MockEngine engine;
  FakeCache cache{engine};
  crucible::BatchSession<MockEngine> session(engine, 1);
  session.SetCacheHooks(cache.Hooks());
  Finish(session, engine, 1, Request(kA1, 13));
  cache.log.clear();
  cache.refuse_fit = true;
  const auto t2 = Finish(session, engine, 2, Request(NextOf(kA1, 20, 9, 9), 24));
  assert(!Has(cache.log, "load slot=0 at 20 frontier") && !t2.checkpoint);
  assert(At(cache.log, "handover slot=0 at 20 speculative") < At(cache.log, "handover slot=0 at 12"));
  assert(At(cache.log, "handover slot=0 at 12") < At(cache.log, "release slot=0 resumed=0"));
  assert(Has(cache.log, "restore 20") && t2.cached == 20);  // the host copy serves the turn
  // the slot choice uses the same rule. A slot whose only matching points the rule refuses is
  // not "the session's own slot": an empty slot comes first, and the points stay.
  MockEngine engine2;
  FakeCache cache2{engine2};
  crucible::BatchSession<MockEngine> two(engine2, 2);
  two.SetCacheHooks(cache2.Hooks());
  Finish(two, engine2, 1, Request(kA1, 13));
  cache2.refuse_fit = true;
  cache2.log.clear();
  assert(Finish(two, engine2, 2, Request(NextOf(kA1, 20, 9, 9), 24)).slot == 1);
  assert(!Has(cache2.log, "handover slot=0 at 20 speculative"));
}

// HIGH-1: every open that writes over a Frontier point copies it first, the longest first, before the release of
// the slot's tail and before any restore.
void GuardCases() {
  // Another session takes the only slot.
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    Finish(session, engine, 1, Request(kA1, 13));
    cache.log.clear();
    Finish(session, engine, 2, Request(Tokens(7, 18), 9));
    assert(cache.log.size() >= 4 && cache.log[0] == "handover slot=0 at 20 speculative" &&
           cache.log[1] == "handover slot=0 at 12" && cache.log[2] == "release slot=0 resumed=0" &&
           cache.log[3] == "restore 0");
    // Session A comes back to the slot: the host copy of its header point serves it.
    cache.log.clear();
    const auto t3 = Finish(session, engine, 3, Request(NextOf(kA1, 20, 9, 9), 24));
    assert(t3.cached == 20 && !t3.checkpoint && Has(cache.log, "restore 20"));
  }
  // A cancel (Close: no tail, as FailRequest) and a skipped tail leave points and no held tail. The slot is not
  // empty: the next session takes the empty slot; once both slots hold states, the open of the slot with the smaller
  // loss copies its points.
  for (const bool cancel : {true, false}) {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    auto a = Request(kA1, 13, 6);
    if (!cancel) a.cache_tail = "skip";
    const auto t1 = Finish(session, engine, 1, a, cancel);
    assert(t1.slot == 0 && session.FrontierPoints().size() == 2 && !cache.held.contains(0));
    cache.log.clear();
    const auto t2 = Finish(session, engine, 2, Request(Tokens(7, 30), 25));  // frontiers 24 and 28 on slot 1
    assert(t2.slot == 1 && !Has(cache.log, "handover slot=0 at 20 speculative"));
    cache.log.clear();
    const auto t3 = Finish(session, engine, 3, Request(Tokens(11, 18), 9));
    assert(t3.slot == 0 && cache.log[0] == "handover slot=0 at 20 speculative" && cache.log[1] == "handover slot=0 at 12");
  }
  // A slot resume keeps the points (they lie below the tail): no copy. If the resume restores a host tail instead
  // (HostTailFirst), the points go to the host first.
  for (const bool host_first : {false, true}) {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    Finish(session, engine, 1, Request(kA1, 13));
    std::vector<int> again = cache.held.at(0);
    if (host_first) {
      auto longer = again;
      longer.push_back(5);
      cache.host[longer] = true;  // a longer host tail of the same session
      cache.host_tail_first = true;
    }
    again.insert(again.end(), {5, 6, 7});
    cache.log.clear();
    Finish(session, engine, 2, Request(again, 0));
    if (host_first) {
      assert(cache.log[0] == "handover slot=0 at 20 speculative" && cache.log[1] == "handover slot=0 at 12");
      assert(Has(cache.log, "resume host tail 24"));
    } else {
      assert(!Has(cache.log, "handover slot=0 at 20 speculative") && Has(cache.log, "resume slot 23"));
      assert(cache.log[0] == "release slot=0 resumed=1");
    }
  }
  // A shorter checkpoint load: an edit after row 12 resumes at the confirmed point (12); the header point (20) above
  // it goes to the host first, and the confirmed point stays.
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    Finish(session, engine, 1, Request(kA1, 13));
    cache.log.clear();
    const auto edit = NextOf(kA1, 14, 3, 10);
    const auto t2 = Finish(session, engine, 2, Request(edit, 13));
    assert(cache.log[0] == "handover slot=0 at 20 speculative" && !Has(cache.log, "handover slot=0 at 12"));
    assert(Has(cache.log, "load slot=0 at 12 frontier") && t2.cached == 12 && t2.checkpoint);
    // The release rule: this open resumed from the slot's own Frontier point, so the tail left without a copy.
    assert(Has(cache.log, "drop slot=0") && !Has(cache.log, "release slot=0 resumed=0"));
  }
  // A bypass request keeps no Frontier point, and its open on a slot with points copies them.
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    Finish(session, engine, 1, Request(kA1, 13));
    cache.log.clear();
    auto bypass = Request(Tokens(4, 21), 13);
    bypass.cache_control = "bypass";
    Finish(session, engine, 2, bypass);
    assert(cache.log[0] == "handover slot=0 at 20 speculative" && cache.log[1] == "handover slot=0 at 12");
    assert(!Has(cache.log, "keep 12") && session.FrontierPoints().empty());
  }
  // A shared frontier stays a host copy (other sessions restore it); the session frontier stays in the slot.
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    auto shared = Request(kA1, 13);
    shared.shared_prefix_tokens = 8;
    Finish(session, engine, 1, shared);
    assert(Has(cache.log, "capture 8") && Has(cache.log, "keep 12") && cache.host.size() == 1);
  }
}

// HIGH-1 and MEDIUM-5: among the slots that other sessions left, the open takes the smallest loss first (no
// Frontier point, then the shortest), then the oldest; an empty slot comes before them.
void SmallestLoss() {
  MockEngine engine;
  FakeCache cache{engine};
  crucible::BatchSession<MockEngine> session(engine, 2);
  session.SetCacheHooks(cache.Hooks());
  // Slot 0: session A, frontiers up to 20. Slot 1, later: session B with a short prompt, frontier 8 only.
  assert(Finish(session, engine, 1, Request(kA1, 13)).slot == 0);
  assert(Finish(session, engine, 2, Request(Tokens(5, 10), 0)).slot == 1);
  // A third session: LRU would take slot 0 (left first); the smallest loss is slot 1 (8 against 20).
  cache.log.clear();
  assert(Finish(session, engine, 3, Request(Tokens(9, 11), 0)).slot == 1);
  assert(cache.log[0] == "handover slot=1 at 8");
  // A bypass request leaves slot 1 with no point: a fourth session takes it (no loss), not slot 0.
  auto bypass = Request(Tokens(2, 9), 0);
  bypass.cache_control = "bypass";
  assert(Finish(session, engine, 4, bypass).slot == 1);
  cache.log.clear();
  assert(Finish(session, engine, 5, Request(Tokens(3, 9), 0)).slot == 1);
  assert(!Has(cache.log, "handover slot=0 at 20 speculative"));
}

// Without the slot frontier hooks the session keeps the host path (no Frontier point, no guard).
void HooksOff() {
  MockEngine engine;
  FakeCache cache{engine};
  cache.slot_frontiers = false;
  crucible::BatchSession<MockEngine> session(engine, 1);
  session.SetCacheHooks(cache.Hooks());
  Finish(session, engine, 1, Request(kA1, 13));
  assert(session.FrontierPoints().empty() && Has(cache.log, "capture 12"));
}

// frontier-copy: request `id` opens and runs its prompt, then decodes (its slot is
// busy). Returns its slot.
int Busy(crucible::BatchSession<MockEngine>& session, MockEngine& engine, uint64_t id, const crucible::ServeRequest& r) {
  auto events = session.Open(id, r, true);
  const int slot = engine.selected;
  while (std::none_of(events.begin(), events.end(), [](const auto& event) { return event.token.has_value(); }))
    events = session.Prefill(id);
  return slot;
}
std::vector<int> Join(std::vector<int> head, const std::vector<int>& tail) {
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}
std::vector<int> Head(const std::vector<int>& ids, size_t count) {
  return {ids.begin(), ids.begin() + std::ptrdiff_t(count)};
}

// Session A decodes on slot 0 and holds points 12 and 20 (speculative). Session B shares A's first 13 tokens: it
// misses the host, so it restores a copy of A's point 12 on slot 1, with the chunks and output of a host hit at 12,
// and keeps its own point at 12. A's points stay.
void DonorCopy() {
  const auto b = Request(Join(Head(kA1, 13), Tokens(9, 8)), 13);
  std::vector<std::vector<int>> chunks, outputs;
  for (const bool copy : {true, false}) {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    assert(Busy(session, engine, 1, Request(kA1, 13)) == 0);
    if (!copy) cache.host[Head(kA1, 12)] = false;  // the host path: the frontier is a host entry
    engine.chunks.clear();
    cache.log.clear();
    cache.restore_slots.clear();
    const auto turn = Finish(session, engine, 2, b);
    assert(turn.slot == 1 && turn.cached == 12 && turn.copy == copy);
    if (copy) {
      assert(cache.log.front() == "copy slot=1 from=0 at 12" && At(cache.log, "copy slot=1 from=0 at 12") < At(cache.log, "keep 12"));
      assert(cache.restore_slots.empty() && !Has(cache.log, "handover slot=0 at 12"));
      std::map<int, std::vector<size_t>> points;
      for (const auto& [slot, index, point] : session.FrontierPoints()) points[slot].push_back(point.hist.size());
      assert(points[0] == std::vector<size_t>({20, 12}) && points[1].size() == 2 && points[1].back() == 12);
    } else {
      assert(Has(cache.log, "restore 12") && !Has(cache.log, "copy slot=1 from=0 at 12"));
    }
    chunks.push_back(engine.chunks);
    outputs.push_back(turn.out);
    // A still decodes on its slot.
    auto events = session.Step({1});
    assert(!events.empty());
  }
  assert(chunks[0] == chunks[1] && !chunks[0].empty() && chunks[0].front() < 0);
  assert(outputs[0] == outputs[1]);
}

// The longest donor point wins, over all busy slots. A copy hit confirms a speculative donor point, as a Frontier
// hit does.
void DonorChoice() {
  MockEngine engine;
  FakeCache cache{engine};
  crucible::BatchSession<MockEngine> session(engine, 3);
  session.SetCacheHooks(cache.Hooks());
  assert(Busy(session, engine, 1, Request(kA1, 13)) == 0);
  // E shares 17 tokens with A: it copies A's 12, then keeps its own 16 (and its header 20) on slot 1.
  assert(Busy(session, engine, 2, Request(Join(Head(kA1, 17), Tokens(3, 4)), 17)) == 1);
  assert(Has(cache.log, "copy slot=1 from=0 at 12") && Has(cache.log, "keep 16"));
  cache.log.clear();
  assert(Finish(session, engine, 3, Request(Join(Head(kA1, 17), Tokens(7, 4)), 17)).cached == 16);
  assert(cache.log.front() == "copy slot=2 from=1 at 16");
  // F repeats A's header point (speculative): the copy confirms it.
  MockEngine engine2;
  FakeCache cache2{engine2};
  crucible::BatchSession<MockEngine> two(engine2, 2);
  two.SetCacheHooks(cache2.Hooks());
  Busy(two, engine2, 1, Request(kA1, 13));
  const auto speculative = [&] {
    for (const auto& [slot, index, point] : two.FrontierPoints())
      if (slot == 0 && point.hist.size() == 20) return point.frontier.speculative;
    assert(false);
    return false;
  };
  assert(speculative());
  assert(Finish(two, engine2, 2, Request(Join(Head(kA1, 21), Tokens(9, 4)), 21)).cached == 20);
  assert(Has(cache2.log, "copy slot=1 from=0 at 20") && !speculative());
}

// No donor: the host holds as much; the host rule refuses; bypass; another image (same pad tokens, another digest);
// a Start point (not a Frontier point) on the busy slot; the slot frontier hooks off; no copy hook.
void NoDonor() {
  const auto b = Request(Join(Head(kA1, 13), Tokens(9, 8)), 13);
  const auto run = [&](auto setup, crucible::ServeRequest request, bool frontiers = true,
                       FakeCache::Copy copy = FakeCache::Copy::kCopy) {
    MockEngine engine;
    FakeCache cache{engine};
    cache.slot_frontiers = frontiers;
    cache.copy = copy;
    auto hooks = cache.Hooks();
    // Image rows key by image digest: token 5 is the pad token, and stop_ids carries a test digest.
    hooks.key = [](const crucible::ServeRequest& r, std::span<const int> tokens) {
      std::vector<int> key(tokens.begin(), tokens.end());
      for (int& id : key) if (id == 5 && r.stop_ids.size() > 2) id = -r.stop_ids.back();
      return key;
    };
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(hooks);
    setup(session, engine, cache);
    cache.log.clear();
    Finish(session, engine, 9, request);
    for (const auto& line : cache.log) assert(line.rfind("copy", 0) != 0);
  };
  const auto busy = [](crucible::ServeRequest a) {
    return [a](crucible::BatchSession<MockEngine>& session, MockEngine& engine, FakeCache&) { Busy(session, engine, 1, a); };
  };
  run([&](auto& session, auto& engine, FakeCache& cache) {
    Busy(session, engine, 1, Request(kA1, 13));
    cache.host[Head(kA1, 12)] = false;
  }, b);
  run([&](auto& session, auto& engine, FakeCache& cache) {
    Busy(session, engine, 1, Request(kA1, 13));
    cache.refuse_fit = true;
  }, b);
  auto bypass = b;
  bypass.cache_control = "bypass";
  run(busy(Request(kA1, 13)), bypass);
  auto image_a = Request(kA1, 13), image_b = b;
  image_a.stop_ids.push_back(1000);
  image_b.stop_ids.push_back(2000);
  run(busy(image_a), image_b);
  run([&](auto& session, auto& engine, FakeCache& cache) {
    // A starts on a host tail of 16 rows: its slot holds a Start point at 16 and no Frontier point yet.
    cache.host[Head(kA1, 16)] = true;
    session.Open(1, Request(kA1, 13), true);
    assert(Has(cache.log, "restore tail 16"));
    cache.host.clear();
    (void)engine;
  }, Request(Join(Head(kA1, 17), Tokens(9, 4)), 17));
  run(busy(Request(kA1, 13)), b, false);
  run(busy(Request(kA1, 13)), b, true, FakeCache::Copy::kOff);
}

// a copy hook that selects the donor and then refuses, or throws, never leaves the donor
// selected: the fallback restore writes the target slot, and an error leaves the target selected.
void CopyFallback() {
  {
    MockEngine engine;
    FakeCache cache{engine};
    cache.copy = FakeCache::Copy::kRefuse;
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    Busy(session, engine, 1, Request(kA1, 13));
    cache.log.clear();
    const auto turn = Finish(session, engine, 2, Request(Join(Head(kA1, 13), Tokens(9, 8)), 13));
    assert(cache.log.front() == "copy refused slot=1 from=0" && cache.log[1] == "restore 0");
    assert(cache.restore_slots.back() == 1 && turn.cached == 0 && !turn.copy && Has(cache.log, "keep 12"));
  }
  {
    MockEngine engine;
    FakeCache cache{engine};
    cache.copy = FakeCache::Copy::kThrow;
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    Busy(session, engine, 1, Request(kA1, 13));
    try {
      session.Open(2, Request(Join(Head(kA1, 13), Tokens(9, 8)), 13));
      assert(false);
    } catch (const std::runtime_error&) {}
    assert(engine.selected == 1);
  }
}

// two donors of the same length give the lowest slot; the slot's own point of the same length wins
// over a donor (a slot resume, no copy); a resident tail wins over a longer donor point.
void DonorEdges() {
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 3);
    session.SetCacheHooks(cache.Hooks());
    assert(Busy(session, engine, 1, Request(kA1, 13)) == 0);
    assert(Busy(session, engine, 2, Request(Join(Head(kA1, 13), Tokens(9, 8)), 13)) == 1);
    assert(Has(cache.log, "copy slot=1 from=0 at 12"));
    cache.log.clear();
    const auto turn = Finish(session, engine, 3, Request(Join(Head(kA1, 13), Tokens(7, 8)), 13));
    assert(turn.slot == 2 && turn.copy && cache.log.front() == "copy slot=2 from=0 at 12");
  }
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    assert(Busy(session, engine, 1, Request(kA1, 13)) == 0);
    assert(Finish(session, engine, 2, Request(Join(Head(kA1, 13), Tokens(9, 8)), 13)).copy);
    cache.log.clear();
    const auto turn = Finish(session, engine, 3, Request(Join(Head(kA1, 13), Tokens(5, 8)), 13));
    assert(turn.slot == 1 && turn.cached == 12 && turn.checkpoint && !turn.copy);
    for (const auto& line : cache.log) assert(line.rfind("copy", 0) != 0);
  }
  {
    // The tail that turn 1 of A leaves in its slot (the mock output is the same in each session).
    std::vector<int> again;
    {
      MockEngine engine;
      FakeCache cache{engine};
      crucible::BatchSession<MockEngine> session(engine, 1);
      session.SetCacheHooks(cache.Hooks());
      Finish(session, engine, 1, Request(kA1, 13));
      again = cache.held.at(0);
    }
    again.insert(again.end(), {5, 6, 7});
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    // Slot 0 decodes a longer turn of the same text: it holds Frontier points above row 20 of `again`.
    assert(Busy(session, engine, 1, Request(Join(again, Tokens(3, 8)), again.size())) == 0);
    size_t above = 0;
    for (const auto& [slot, index, point] : session.FrontierPoints())
      if (slot == 0 && point.hist.size() <= again.size()) above = std::max(above, point.hist.size());
    assert(above > 20);
    assert(Finish(session, engine, 2, Request(kA1, 13)).slot == 1 && cache.held.at(1) == Head(again, again.size() - 3));
    cache.log.clear();
    const auto turn = Finish(session, engine, 3, Request(again, 0));
    assert(turn.slot == 1 && !turn.copy && Has(cache.log, "resume slot 23"));
    for (const auto& line : cache.log) assert(line.rfind("copy", 0) != 0);
  }
}

// W1 LocalTenantUnchanged: a fixed trace of opens over three slots (host restores, slot resumes, checkpoint
// and Frontier resumes, frontier copies from a busy or idle slot, handovers). Its decision log is the log of train 23
// (7b3106e8): with the local tenant only, the tenant key changes no decision. A changed log prints it.
std::string TraceDigest(const std::vector<std::string>& lines) {
  crucible::Sha256 hash;
  for (const auto& line : lines) {
    hash.Update(reinterpret_cast<const uint8_t*>(line.data()), line.size());
    hash.Update(reinterpret_cast<const uint8_t*>("\n"), 1);
  }
  return hash.HexDigest();
}
void LocalTenantUnchanged() {
  MockEngine engine;
  FakeCache cache{engine};
  crucible::BatchSession<MockEngine> session(engine, 3);
  session.SetCacheHooks(cache.Hooks());
  uint32_t seed = 20261019;
  const auto next = [&] { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
  std::vector<std::vector<int>> history(4);
  std::vector<std::string> trace;
  uint64_t id = 1, busy = 0;
  const auto drain = [&](uint64_t request) {
    std::vector<int> out;
    for (auto events = session.Step({request});; events = session.Step({request})) {
      bool done = false;
      for (const auto& event : events) {
        if (event.token) out.push_back(*event.token);
        done |= event.done;
      }
      if (done) break;
    }
    trace.push_back("drain " + std::to_string(request) + " out=" + std::to_string(out.size()));
  };
  for (int step = 0; step < 160; ++step) {
    const int s = int(next() % 4);
    auto& h = history[size_t(s)];
    if (h.empty() || h.size() > 40) h = Tokens(int(s % 2 * 5 + next() % 3), 9 + next() % 8);
    const size_t drop = next() % 3 == 0 ? std::min<size_t>(h.size() - 1, next() % 6) : 0;
    auto ids = Head(h, h.size() - drop);
    ids = Join(ids, Tokens(int(next() % 16), 2 + next() % 6));
    const size_t boundary = ids.size() - 1 - next() % 3;
    const auto r = Request(ids, boundary, 2 + int(next() % 3));
    if (!busy && next() % 4 == 0) {
      busy = id++;
      trace.push_back("busy " + std::to_string(busy) + " slot=" + std::to_string(Busy(session, engine, busy, r)));
      h = ids;
      continue;
    }
    const auto turn = Finish(session, engine, id, r);
    trace.push_back("turn " + std::to_string(id++) + " s=" + std::to_string(s) + " slot=" + std::to_string(turn.slot) +
                    " cached=" + std::to_string(turn.cached) + " checkpoint=" + std::to_string(turn.checkpoint) +
                    " copy=" + std::to_string(turn.copy) + " out=" + std::to_string(turn.out.size()));
    h = Join(ids, Head(turn.out, turn.out.size() - 1));
    if (busy && next() % 2 == 0) drain(std::exchange(busy, 0));
  }
  if (busy) drain(busy);
  for (const auto& [slot, index, point] : session.FrontierPoints())
    trace.push_back("point slot=" + std::to_string(slot) + " index=" + std::to_string(index) + " at " +
                    std::to_string(point.hist.size()));
  trace.insert(trace.end(), cache.log.begin(), cache.log.end());
  const std::string digest = TraceDigest(trace);
  if (digest != kLocalTrace23) {
    for (const auto& line : trace) std::printf("%s\n", line.c_str());
    std::printf("trace digest %s\n", digest.c_str());
    std::fflush(stdout);
  }
  assert(digest == kLocalTrace23);
}

// TenantsNeverShareSlot (W1, design sections 3.2 and 3.5): a slot that held tenant A. Tenant B, with a prompt
// that repeats A's tail, gets no slot resume, no checkpoint or Frontier resume and no copy: the open writes the slot
// from the host path (the restore hook, which resets the slot's MTP rows and base in BatchPrefixCache). A's tail and
// Frontier points go to the host first, as A's. A returning A still finds them, and on two slots B takes the empty
// slot while A takes its own (Affinity).
crucible::StateTenant TenantOf(uint8_t first) {
  crucible::StateTenant tenant{};
  tenant[0] = first;
  return tenant;
}
crucible::ServeRequest Of(crucible::ServeRequest r, const crucible::StateTenant& tenant) {
  r.tenant = tenant;
  return r;
}
void TenantsNeverShareSlot() {
  const auto a = TenantOf(1), b = TenantOf(2);
  {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    const auto t1 = Finish(session, engine, 1, Of(Request(kA1, 13), a));
    const auto next = Join(Join(kA1, Head(t1.out, t1.out.size() - 1)), Tokens(3, 4));
    assert(session.FrontierPoints().size() == 2 && cache.held.contains(0));
    cache.log.clear();
    cache.restore_slots.clear();
    const auto tb = Finish(session, engine, 2, Of(Request(next, next.size() - 1), b));
    assert(tb.cached == 0 && !tb.checkpoint && !tb.copy && cache.restore_slots == std::vector<int>{0});
    for (const auto& line : cache.log) assert(!line.starts_with("resume") && !line.starts_with("load") && !line.starts_with("copy"));
    assert(Has(cache.log, "handover slot=0 at 20 speculative tenant") && Has(cache.log, "handover slot=0 at 12 tenant"));
    assert(Has(cache.log, "release slot=0 resumed=0") && cache.owner.at(Head(next, 23)) == a);
    for (const auto& [slot, index, point] : session.FrontierPoints()) assert(point.tenant == b);
    // A returns: its tail is a host entry of A now (B's turn wrote the slot).
    cache.log.clear();
    const auto ta = Finish(session, engine, 3, Of(Request(next, next.size() - 1), a));
    assert(ta.cached == 23 && Has(cache.log, "restore tail 23"));
  }
  {
    // The same B turn with tenant A: slot resume (the control).
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 1);
    session.SetCacheHooks(cache.Hooks());
    const auto t1 = Finish(session, engine, 1, Of(Request(kA1, 13), a));
    const auto next = Join(Join(kA1, Head(t1.out, t1.out.size() - 1)), Tokens(3, 4));
    cache.log.clear();
    const auto t2 = Finish(session, engine, 2, Of(Request(next, next.size() - 1), a));
    assert(t2.cached == 23 && Has(cache.log, "resume slot 23"));
  }
  {
    // Two slots: B takes the empty slot (A's slot ranks as left longest ago), and A takes its own slot back.
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    const auto t1 = Finish(session, engine, 1, Of(Request(kA1, 13), a));
    assert(t1.slot == 0);
    const auto next = Join(Join(kA1, Head(t1.out, t1.out.size() - 1)), Tokens(3, 4));
    const auto tb = Finish(session, engine, 2, Of(Request(next, next.size() - 1), b));
    assert(tb.slot == 1 && tb.cached == 0 && !tb.copy);
    cache.log.clear();
    const auto ta = Finish(session, engine, 3, Of(Request(next, next.size() - 1), a));
    assert(ta.slot == 0 && ta.cached == 23 && Has(cache.log, "resume slot 23"));
  }
}

// W1 frontier copy: a copy from slot S is allowed only when S's point has the request's tenant. A busy slot
// of tenant A holds a Frontier point; another session of A copies it, a session of B does not (it prefills).
void TenantsNeverCopyFrontier() {
  const auto a = TenantOf(1), b = TenantOf(2);
  for (const bool same : {true, false}) {
    MockEngine engine;
    FakeCache cache{engine};
    crucible::BatchSession<MockEngine> session(engine, 2);
    session.SetCacheHooks(cache.Hooks());
    assert(Busy(session, engine, 1, Of(Request(kA1, 13), a)) == 0);
    cache.log.clear();
    const auto turn = Finish(session, engine, 2, Of(Request(Join(Head(kA1, 13), Tokens(9, 8)), 13), same ? a : b));
    assert(turn.slot == 1 && turn.copy == same && turn.cached == (same ? 12u : 0u));
    assert(Has(cache.log, "copy slot=1 from=0 at 12") == same && Has(cache.log, "restore 0") != same);
    // B's own Frontier point on slot 1 is B's; A's point on slot 0 stays A's.
    for (const auto& [slot, index, point] : session.FrontierPoints()) assert(point.tenant == (slot == 0 || same ? a : b));
    assert(!session.Step({1}).empty());
  }
}

int main() {
  TenantsNeverShareSlot();
  TenantsNeverCopyFrontier();
  LocalTenantUnchanged();
  DonorCopy();
  DonorChoice();
  NoDonor();
  CopyFallback();
  DonorEdges();
  StoreAndResume();
  OpenedNamesTheFrontier();
  PickRule();
  GuardCases();
  SmallestLoss();
  HooksOff();
  std::puts("slot_frontier_test passed");
  return 0;
}
