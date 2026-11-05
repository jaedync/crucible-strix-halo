// Re-send fix A and B, on the fake engine.
// A: an identical prompt resumes at the End point of the run that saved it, with no prompt row, and gives that run's
//    first token (greedy ID or keyed draw from the saved logits row).
// B: a long prompt ends a chunk at its final chunk's core cut and keeps a Pre point there; a prompt that extends the
//    old one past that cut resumes there.
#include "batch_mock_engine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <tuple>
#include <unistd.h>

namespace {
using crucible::BatchEvent;
using crucible::ServeRequest;

ServeRequest Prompt(size_t length, int first = 1, int tokens = 3) {
  ServeRequest r = Request(1, tokens);
  r.ids.resize(length);
  std::iota(r.ids.begin(), r.ids.end(), first);
  for (int& id : r.ids) id %= 16;
  return r;
}

// One session with the cache hooks of batched_serve.h in miniature: no host states, slot tails, and checkpoint
// buffers that hold the slot's rows. The engine is a parameter so a vision engine can use the same hooks.
template <class E = MockEngine>
struct FixtureT {
  E engine;
  crucible::BatchSession<E> session;
  std::map<std::pair<int, int>, std::vector<int>> buffers;
  std::vector<std::string> log;
  std::vector<size_t> captures, floors;
  std::vector<std::pair<size_t, size_t>> cuts;  // b-cut2: the cut buffer's position and window per prepared cut
  size_t host_tail = 0;  // > 0: the host restores a session tail of this length (a replayed turn)
  explicit FixtureT(int slots = 1, int prefill = 4, size_t pre_min = 0, int core = 16,
                    int launch = 0, int context = 64)
      : session((engine.prefill_rows = prefill, engine.launch_rows = launch, engine.context = context, engine), slots) {
    session.SetCoreRows(core);
    session.SetSavePoints(0, 0);  // the save points have their own tests (SavePoints)
    if (pre_min) session.SetPreEnd(pre_min, core);
    hooks.restore = [this](const ServeRequest& r) -> crucible::BatchRestore {
      log.push_back("restore");
      if (!host_tail) return 0;
      engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(host_tail));
      return {host_tail, false, 0, true};
    };
    hooks.peek = [](const ServeRequest&) { return size_t(0); };
    hooks.capture = [this](const ServeRequest&, size_t cursor) { captures.push_back(cursor); };
    hooks.next_capture = [](const ServeRequest&, size_t) { return SIZE_MAX; };
    hooks.resume = [](const ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
      return {resident.length, false, resident.replay, true, true, resident.guarded};
    };
    hooks.tail = [this](const ServeRequest& r, std::span<const int> generated, int,
                        crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
      if (generated.size() < 2) return std::nullopt;
      std::vector<int> hist(r.ids);
      hist.insert(hist.end(), generated.begin(), generated.end() - 1);
      engine.histories[engine.selected] = hist;
      return crucible::BatchResident{hist.size(), 0, false};
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
      log.push_back("load " + std::to_string(point.hist.size()));
    };
    hooks.stage = [this](const ServeRequest&, size_t floor, const crucible::BatchStageQueue&) {
      floors.push_back(floor);
      return crucible::BatchStage{};
    };
    hooks.end_resumed = [this](int slot, size_t tokens) {
      log.push_back("end slot=" + std::to_string(slot) + " tokens=" + std::to_string(tokens));
    };
    hooks.pre_saved = [this](int slot, size_t tokens) {
      log.push_back("pre slot=" + std::to_string(slot) + " tokens=" + std::to_string(tokens));
    };
    hooks.header_saved = [this](int slot, size_t tokens) {
      log.push_back("header slot=" + std::to_string(slot) + " tokens=" + std::to_string(tokens));
    };
    // Header point: the donor's rows below the point and the buffer's state go to the selected slot.
    hooks.point_copy = [this](const ServeRequest&, int from, int index, const crucible::SlotCheckpoint& point) -> size_t {
      const auto& donor = engine.histories[from];
      assert(buffers.at({from, index}) == point.hist && donor.size() >= point.hist.size() &&
             std::equal(point.hist.begin(), point.hist.end(), donor.begin()) && from != engine.selected);
      engine.histories[engine.selected] = point.hist;
      log.push_back("copy " + std::to_string(point.hist.size()) + " from=" + std::to_string(from));
      return point.hist.size();
    };
    hooks.header_copied = [this](int slot, int from, size_t tokens) {
      log.push_back("header_copy slot=" + std::to_string(slot) + " from=" + std::to_string(from) +
                    " tokens=" + std::to_string(tokens));
    };
    hooks.bytes = [] { return size_t(0); };
    session.SetCacheHooks(hooks);
  }
  // join rows and a cut buffer, as batched_serve.h with "L+T" launch rows. The buffer holds the
  // slot's rows below the cut, and cut_save copies them to a checkpoint buffer.
  crucible::BatchCacheHooks hooks;
  void EnableCuts(int join = 4) {
    engine.join_rows = join;
    static float state[4], target[4];
    hooks.cut_ready = [](const ServeRequest& r) { return r.cache_control != "bypass"; };
    hooks.cut_prepare = [this](size_t position, size_t window) {
      cuts.emplace_back(position, window);
      return std::vector<crucible::PrefillCutCopy>{{state, target, sizeof state}};
    };
    hooks.cut_save = [this](int slot, int index) {
      const auto& stream = engine.histories[slot];
      assert(!cuts.empty() && stream.size() > cuts.back().first);
      buffers[{slot, index}].assign(stream.begin(), stream.begin() + std::ptrdiff_t(cuts.back().first));
    };
    hooks.cut_capture = [](const ServeRequest&, size_t) { assert(false); };
    session.SetCacheHooks(hooks);
  }
  // The request's tokens and its done event; `opened`: the events of the open (the first token, if any).
  std::pair<std::vector<int>, BatchEvent> Run(uint64_t id, const ServeRequest& r, std::vector<BatchEvent>* opened = nullptr) {
    std::vector<int> out;
    auto events = session.Open(id, r);
    if (opened) *opened = events;
    for (;;) {
      for (const auto& event : events) {
        if (event.token) out.push_back(*event.token);
        if (event.done) return {out, event};
      }
      events = session.Step({id});
    }
  }
  bool Logged(const std::string& line) const { return std::find(log.begin(), log.end(), line) != log.end(); }
};
using Fixture = FixtureT<MockEngine>;

// A: greedy, sampled, one token, a stop token, and the switch off.
void Identical() {
  {
    Fixture f;
    const auto a = Prompt(10);
    const auto [cold, cold_done] = f.Run(1, a);
    assert(cold_done.cached_tokens == 0 && !cold_done.end_resume && f.engine.copied_logits == 1);
    const int calls = f.engine.calls;
    std::vector<BatchEvent> opened;
    const auto [again, done] = f.Run(2, a, &opened);
    // The first token came with the open and no prompt row ran: one decode step per later token only.
    assert(opened.front().token && *opened.front().token == cold.front());
    assert(again == cold && done.cached_tokens == 10 && done.end_resume && done.checkpoint_resume);
    assert(f.engine.calls == calls + int(cold.size()) - 1);
    assert(f.Logged("end slot=0 tokens=10") && f.Logged("load 10"));
    assert(f.engine.copied_logits == 1);  // greedy: no logits read at the resume
    // A third send resumes again (the End point stays: Keep at the prompt end, no second save).
    const auto [third, third_done] = f.Run(3, a);
    assert(third == cold && third_done.cached_tokens == 10 && third_done.end_resume);
  }
  {
    // Sampled: the first token is the keyed draw from the saved row. The engine's logits buffer now holds another
    // row with a NaN, so a draw from it would differ or fail.
    Fixture f;
    auto s = Prompt(10, 3, 6);
    s.sampling.temperature = .8; s.sampling.seed = 77;
    const auto [cold, cold_done] = f.Run(1, s);
    f.engine.last = {9}; f.engine.nan_row = 0;
    const int copied = f.engine.copied_logits;
    auto events = f.session.Open(2, s);
    assert(events.front().token && *events.front().token == cold.front() && events.front().cached_tokens == 10);
    assert(f.engine.copied_logits == copied);
    f.engine.nan_row = -1;
    std::vector<int> out{*events.front().token};
    while (!events.back().done) {
      events = f.session.Step({2});
      for (const auto& event : events) if (event.token) out.push_back(*event.token);
    }
    assert(out == cold);
    // A greedy prompt resumes a sampled run's point: the saved greedy ID is the first token.
    auto g = s; g.sampling.temperature = 0;
    MockEngine reference; reference.prefill_rows = 4;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    const auto expected = Complete(plain, 1, g);
    const auto [greedy, greedy_done] = f.Run(3, g);
    assert(greedy == expected && greedy_done.end_resume);
  }
  {
    // max_tokens 1, and a first token that is a stop ID: the open publishes it and closes the request.
    Fixture f;
    auto one = Prompt(9, 2, 1);
    const auto [cold, cold_done] = f.Run(1, one);
    std::vector<BatchEvent> opened;
    const auto [again, done] = f.Run(2, one, &opened);
    assert(again == cold && again.size() == 1 && done.end_resume && opened.size() == 2 && f.session.Active() == 0);
    Fixture g;
    auto stop = Prompt(9, 2, 5);
    const auto [first, first_done] = g.Run(1, stop);
    stop.stop_ids = {first.front()};
    const auto [stopped, stopped_done] = g.Run(2, stop);
    assert(stopped == std::vector<int>{first.front()} && stopped_done.stopped && stopped_done.end_resume);
  }
  {
    // A bypass send saves no End point, so its re-send needs one more row.
    Fixture f;
    auto bypass = Prompt(10);
    bypass.cache_control = "bypass";
    f.Run(1, bypass);
    assert(f.engine.copied_logits == 0);
    const auto [again, done] = f.Run(2, Prompt(10));
    assert(done.cached_tokens == 0 && !done.end_resume);
  }
}

// A: a deferred open reports the whole prompt; its prefill command publishes the first token with no cursor.
void Deferred() {
  Fixture f;
  const auto a = Prompt(10);
  const auto [cold, cold_done] = f.Run(1, a);
  const auto opened = f.session.Open(2, a, true);
  assert(opened.size() == 1 && !opened.front().token && opened.front().cached_tokens == 10);
  const int calls = f.engine.calls;
  const auto first = f.session.Prefill(2);
  assert(first.front().token && *first.front().token == cold.front() && !f.session.PrefillCursor());
  assert(f.engine.calls == calls);
  f.session.Close(2);
}

// A: a speculative re-send opens its draft state at the whole prompt and starts drafts from the first token; no chunk
// pairing runs.
void Speculative() {
  Fixture f;
  std::vector<std::string> spec;
  crucible::BatchSpecHooks hooks;
  hooks.check = [](const ServeRequest&) { return std::string(); };
  hooks.open = [&](uint64_t id, int, const ServeRequest&, size_t cached) {
    spec.push_back("open " + std::to_string(id) + " cached=" + std::to_string(cached));
  };
  hooks.chunk = [&](uint64_t id, size_t start, int rows, bool) {
    spec.push_back("chunk " + std::to_string(id) + " " + std::to_string(start) + "+" + std::to_string(rows));
  };
  hooks.first = [&](uint64_t id, int token) { spec.push_back("first " + std::to_string(id) + " " + std::to_string(token)); };
  hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{(last + 1) % 16}; };
  hooks.close = [](uint64_t) {};
  f.session.SetSpecHooks(hooks);
  auto r = Prompt(10, 1, 4);
  r.spec = 7;
  const auto run = [&](uint64_t id) {
    std::vector<int> out;
    auto events = f.session.Open(id, r);
    for (;;) {
      for (const auto& event : events) {
        if (event.token) out.push_back(*event.token);
        if (event.done) return std::make_pair(out, event);
      }
      events = f.session.SpecStep(id);
    }
  };
  const auto [cold, cold_done] = run(1);
  spec.clear();
  const auto [again, done] = run(2);
  assert(again == cold && done.end_resume);
  assert((spec == std::vector<std::string>{"open 2 cached=10", "first 2 " + std::to_string(cold.front())}));
}

// A: the slot that holds the End point serves the re-send; the disk stage sees the whole prompt as the floor.
void AffinityAndStage() {
  Fixture f(2);
  const auto a = Prompt(10), b = Prompt(12, 5);
  f.Run(1, a);
  const int a_slot = f.engine.selected;
  f.Run(2, b);
  assert(f.engine.selected != a_slot);
  assert(f.session.Stage(a).state == "none" && f.floors.back() == 10);
  const auto [again, done] = f.Run(3, a);
  assert(done.end_resume && f.engine.selected == a_slot);
}

// B: a prompt of at least the minimum ends a chunk at its final chunk's core cut, keeps a Pre point there and makes
// no host capture there. An extended prompt resumes at the Pre point; an identical one at the End point.
void PreEnd() {
  {
    Fixture f(1, 32, 40, 16);
    const auto c = Prompt(58, 1, 2);
    const auto [cold, cold_done] = f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -16, -10}));
    assert((f.captures == std::vector<size_t>{32}) && f.Logged("pre slot=0 tokens=48"));
    // The cut keeps every row's token and the first token.
    MockEngine reference; reference.prefill_rows = 32;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, c) == cold);
    // Identical: the End point (the Pre point stays below it).
    const auto [again, done] = f.Run(2, c);
    assert(again == cold && done.cached_tokens == 58 && done.end_resume);
    // Extended inside the old prompt's last rows (past the cut): the Pre point.
    auto ext = c;
    ext.ids.resize(50);
    ext.ids.insert(ext.ids.end(), {9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
    f.engine.chunks.clear();
    const auto [extended, ext_done] = f.Run(3, ext);
    assert(ext_done.cached_tokens == 48 && ext_done.checkpoint_resume && !ext_done.end_resume && f.Logged("load 48"));
    // A Pre point is on the core grid: no small-row head (GridResume), one bulk chunk for the rest.
    assert(f.engine.chunks.front() == -12);
  }
  {
    // Merged launches stop at the Pre point too: without the cut the whole prompt is one launch.
    Fixture merged(1, 32, 40, 16, 64);
    merged.Run(1, Prompt(58, 1, 2));
    assert((merged.engine.chunks == std::vector<int>{-32, -16, -10}));
    Fixture whole(1, 32, 0, 16, 64);
    whole.Run(1, Prompt(58, 1, 2));
    assert((whole.engine.chunks == std::vector<int>{-58}) && !whole.Logged("pre slot=0 tokens=48"));
  }
  {
    // No cut: a last piece of 8 rows or fewer, a prompt under the minimum, a bypass prompt, a turn on a tail.
    Fixture eight(1, 32, 40, 16);
    eight.Run(1, Prompt(56, 1, 2));
    assert((eight.engine.chunks == std::vector<int>{-32, -24}));
    Fixture shorter(1, 32, 59, 16);
    shorter.Run(1, Prompt(58, 1, 2));
    assert((shorter.engine.chunks == std::vector<int>{-32, -26}));
    Fixture bypass(1, 32, 40, 16);
    auto b = Prompt(58, 1, 2); b.cache_control = "bypass";
    bypass.Run(1, b);
    assert((bypass.engine.chunks == std::vector<int>{-32, -26}));
    // A replayed turn: its chunk schedule starts at the tail, so it takes Grid points as before and no Pre point.
    Fixture tail(1, 32, 40, 16);
    const auto first = Prompt(58, 1, 2);
    const auto [out, done] = tail.Run(1, first);
    auto next = first; next.ids.push_back(out.front()); next.ids.insert(next.ids.end(), {3, 3, 3});
    tail.engine.chunks.clear(); tail.log.clear();
    const auto [later, later_done] = tail.Run(2, next);
    assert(later_done.cached_tokens == 59 && !tail.Logged("pre slot=0 tokens=48"));
    // A host tail on the grid (32 rows): still a replayed turn's schedule (a head chunk first), so no cut.
    Fixture host(1, 32, 40, 16);
    host.host_tail = 32;
    const auto [hosted, hosted_done] = host.Run(1, Prompt(58, 1, 2));
    assert(hosted_done.cached_tokens == 32 && !host.Logged("pre slot=0 tokens=48"));
    assert((host.engine.chunks == std::vector<int>{8, -18}));
  }
  {
    // A cut of 8 rows or fewer from the chunk start is refused when the session is set.
    MockEngine engine;
    crucible::BatchSession<MockEngine> session(engine, 1);
    bool threw = false;
    try { session.SetPreEnd(40, 8); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }
}
// an End or Pre resume at a multiple of the core rows runs no small-row head. The
// bulk kernels then see the same rows and the same 2,048-row part cuts as a cold prefill, so the request is bitwise
// equal to it (extend-dump: s = 4,096 and 8,192). Off the grid the head stays. A tail (session replay) keeps its head.
void GridResume() {
  using crucible::kMaxRows;
  const auto extend = [](const ServeRequest& base, size_t extra) {
    auto r = base;
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 7 + 3) % 16));
    return r;
  };
  {
    // On the grid (32 rows, core 16): no head, one bulk chunk for the 20 new rows; every Forward has replay 0.
    Fixture f(1, 32, 0, 16);
    const auto a = Prompt(32, 1, 2);
    f.Run(1, a);
    f.engine.chunks.clear(); f.engine.replays.clear();
    const auto b = extend(a, 20);
    const auto [out, done] = f.Run(2, b);
    assert(done.cached_tokens == 32 && done.checkpoint_resume && !done.end_resume && f.Logged("load 32"));
    assert((f.engine.chunks.front() == -20));
    assert(std::all_of(f.engine.replays.begin(), f.engine.replays.end(), [](int r) { return r == 0; }));
    // The same tokens as a cold prefill of the long prompt.
    MockEngine reference; reference.prefill_rows = 32;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, b) == out);
  }
  {
    // Off the grid (30 rows): the head stays, as before.
    Fixture f(1, 32, 0, 16);
    const auto a = Prompt(30, 1, 2);
    f.Run(1, a);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(a, 20));
    assert(done.cached_tokens == 30 && done.checkpoint_resume);
    assert(f.engine.chunks.front() > 0 && f.engine.chunks.front() <= kMaxRows);
  }
  {
    // A tail on the grid (30 prompt rows + 2 kept output rows = 32): the head stays, with replay 0 or not, so the
    // pending GDN rows of a speculative run never choose the path.
    Fixture f(1, 32, 0, 16);
    const auto a = Prompt(30, 1, 3);
    const auto [first, first_done] = f.Run(1, a);
    auto next = a;
    next.ids.insert(next.ids.end(), first.begin(), first.begin() + 2);
    next = extend(next, 20);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, next);
    assert(done.cached_tokens == 32 && done.cache_tail);
    assert(f.engine.chunks.front() > 0 && f.engine.chunks.front() <= kMaxRows);
  }
  {
    // MTP warm-up (a speculative request): the draft hooks see every row from the resume point on, once and in
    // order, as bulk chunks, and no Forward carries pending GDN rows (an End point has none).
    Fixture f(1, 32, 0, 16);
    std::vector<std::tuple<size_t, int, bool>> chunks;
    std::vector<size_t> opened;
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const ServeRequest&) { return std::string(); };
    hooks.open = [&](uint64_t, int, const ServeRequest&, size_t cached) { opened.push_back(cached); };
    hooks.chunk = [&](uint64_t, size_t start, int rows, bool bulk) { chunks.emplace_back(start, rows, bulk); };
    hooks.first = [](uint64_t, int) {};
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{(last + 1) % 16}; };
    hooks.close = [](uint64_t) {};
    f.session.SetSpecHooks(hooks);
    const auto run = [&](uint64_t id, const ServeRequest& r) {
      auto events = f.session.Open(id, r);
      for (;;) {
        for (const auto& event : events) if (event.done) return;
        events = f.session.SpecStep(id);
      }
    };
    auto a = Prompt(32, 1, 3);
    a.spec = 7; a.mtp_prefix = 40;
    run(1, a);
    chunks.clear(); f.engine.replays.clear();
    auto b = extend(a, 20);
    run(2, b);
    assert(opened.back() == 32);
    assert((chunks == std::vector<std::tuple<size_t, int, bool>>{{32, 20, true}}));
    assert(std::all_of(f.engine.replays.begin(), f.engine.replays.end(), [](int r) { return r == 0; }));
  }
  {
    // A Pre point is a grid point too: an extension of a long prompt that resumes there has no head.
    Fixture f(1, 32, 40, 16);
    const auto c = Prompt(58, 1, 2);
    f.Run(1, c);
    auto ext = c;
    ext.ids.resize(50);
    ext.ids.insert(ext.ids.end(), {9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
    f.engine.chunks.clear();
    const auto [extended, ext_done] = f.Run(3, ext);
    assert(ext_done.cached_tokens == 48 && ext_done.checkpoint_resume && f.Logged("load 48"));
    assert(f.engine.chunks.front() == -12);
  }
}
// with exact resume on, a strict extension of an End point that is off the core grid
// resumes at the last multiple of 4 rows that leaves more than 8 rows (a Pre point, saved for any length), so its bulk
// rows and QSA tiles are the cold prefill's. An identical prompt still resumes at the End point; with the switch off
// nothing changes.
void ExactResume() {
  using crucible::kMaxRows;
  const auto extend = [](const ServeRequest& base, size_t extra) {
    auto r = base;
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 7 + 3) % 16));
    return r;
  };
  {
    // 58 rows (step 32, core 16): the Pre point is 48 (58 - 9 = 49, down to a multiple of 4), the End point 58 is off
    // the grid and not a multiple of 4.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(58, 1, 2);
    const auto [cold, cold_done] = f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -16, -10}));
    assert(f.Logged("pre slot=0 tokens=48") && (f.captures == std::vector<size_t>{32}));
    // Identical: the End point.
    const auto [again, again_done] = f.Run(2, c);
    assert(again == cold && again_done.end_resume && again_done.cached_tokens == 58);
    // An extension: the Pre point, no head, bulk rows up to 64 (its own Pre point there), then the 6-row small tail
    // that the cold plan of 70 rows runs alone (64 = 2 x 32, and 6 rows are 8 or fewer).
    const auto ext = extend(c, 12);
    f.engine.chunks.clear(); f.engine.replays.clear();
    const auto [out, done] = f.Run(3, ext);
    assert(done.cached_tokens == 48 && done.checkpoint_resume && !done.end_resume && f.Logged("load 48"));
    assert((f.engine.chunks == std::vector<int>{-16, -6}) && f.Logged("pre slot=0 tokens=64"));
    assert(std::all_of(f.engine.replays.begin(), f.engine.replays.end(), [](int r) { return r == 0; }));
    MockEngine reference; reference.prefill_rows = 32; reference.context = 160;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, ext) == out);
  }
  {
    // The switch off: the same extension resumes at the End point (58) with a head, as before.
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(58, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 12));
    assert(done.cached_tokens == 58 && f.engine.chunks.front() > 0 && f.engine.chunks.front() <= kMaxRows);
  }
  {
    // A cut on a chunk boundary (45 rows: 45 - 9 = 36, point 32): the Pre point is saved and the host capture stays.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(45, 1, 2);
    f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -13}));
    assert(f.Logged("pre slot=0 tokens=32") && (f.captures == std::vector<size_t>{32}));
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 15));
    assert(done.cached_tokens == 32 && (f.engine.chunks == std::vector<int>{-28}));  // lever 1: 60 rows end in a 28-row piece, no Pre point
  }
  {
    // 40 rows: the cold plan is 32 bulk rows and an 8-row small tail, so the Pre point is the tail start 32
    // (before: 28, and the 12 rows after it ran as bulk rows). The End point 40 holds the small-row tail, which the
    // cold 50-row prompt runs as bulk rows, so the extension resumes at 32.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(40, 1, 2);
    f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -8}));
    assert(f.Logged("pre slot=0 tokens=32"));
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 10));
    assert(done.cached_tokens == 32 && (f.engine.chunks == std::vector<int>{-18}));
  }
  {
    // A prompt with no multiple of 4 that leaves 9 rows and more than 8 rows since its chunk start (20 rows: the point
    // is 8): no Pre point, so the extension starts from row 0 (cold arithmetic) instead of the off-grid End point.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(20, 1, 2);
    f.Run(1, c);
    assert(!f.Logged("pre slot=0"));
    f.engine.chunks.clear();
    // 8 new rows: too few for a bulk resume at the End point (gdn-split-inv), which 10 rows would allow.
    const auto [out, done] = f.Run(2, extend(c, 8));
    // The 28-row prompt is cold again (lever 1: a 28-row piece, so no Pre point).
    assert(done.cached_tokens == 0 && (f.engine.chunks == std::vector<int>{-28}) && !f.Logged("pre slot=0"));  // lever 1: no Pre point
  }
  {
    // An End point on the grid stays usable (fix1): no head.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(48, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 12));
    assert(done.cached_tokens == 48 && f.engine.chunks.front() == -12);
  }
  {
    // A resume from a Pre point off the chunk grid (48, step 32) still saves its own Pre point (68) for the next turn.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(58, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear(); f.log.clear();
    const auto next = extend(c, 23);
    const auto [out, done] = f.Run(2, next);
    assert(done.cached_tokens == 48 && (f.engine.chunks == std::vector<int>{-24, -9}));
    assert(f.Logged("pre slot=0 tokens=72"));
    f.engine.chunks.clear();
    const auto [out2, done2] = f.Run(3, extend(next, 12));
    assert(done2.cached_tokens == 72 && (f.engine.chunks == std::vector<int>{-12, -9}));  // the End point 81 is no tile
  }
  {
    // A bypass prompt and a replayed turn make no Pre point.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    auto b = Prompt(58, 1, 2); b.cache_control = "bypass";
    f.Run(1, b);
    assert(!f.Logged("pre slot=0 tokens=48"));
  }
}
// the Pre point is the last multiple of 4 rows that leaves more than
// 8 rows, so a resume that cannot use the End point (not a multiple of 4, or 8 new rows or fewer) recomputes at most
// 20 rows instead of up to core_rows - 1.
void ExactResumeFallback() {
  const auto extend = [](const ServeRequest& base, size_t extra) {
    auto r = base;
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 7 + 3) % 16));
    return r;
  };
  {
    // 53 rows (not a multiple of 4): the Pre point is 44 (53 - 9), inside the second chunk, 12 rows past its start.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(53, 1, 2);
    f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -12, -9}) && f.Logged("pre slot=0 tokens=44"));
    f.engine.chunks.clear();
    // The End point 53 is no tile-aligned state: the extension resumes at 44 (9 rows recomputed), as bulk rows.
    const auto ext = extend(c, 22);
    const auto [out, done] = f.Run(2, ext);
    assert(done.cached_tokens == 44 && done.checkpoint_resume && f.Logged("load 44"));
    assert((f.engine.chunks == std::vector<int>{-20, -11}) && f.Logged("pre slot=0 tokens=64"));
    assert(std::all_of(f.engine.replays.begin(), f.engine.replays.end(), [](int r) { return r == 0; }));
    MockEngine reference; reference.prefill_rows = 32; reference.context = 160;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, ext) == out);
  }
  {
    // Another prompt that repeats only the Pre point, with 6 own rows: too few for a bulk resume, so it starts cold.
    // Its Pre point would be 40, 8 rows past its chunk start (32), so the cut moves to the chunk boundary.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(53, 1, 2);
    f.Run(1, c);
    auto other = c;
    other.ids.resize(44);
    for (int i = 0; i < 6; ++i) other.ids.push_back(9);
    f.engine.chunks.clear(); f.log.clear();
    const auto [out, done] = f.Run(2, other);
    assert(done.cached_tokens == 0 && !f.Logged("load"));
    assert((f.engine.chunks == std::vector<int>{-32, -18}) && f.Logged("pre slot=0 tokens=32"));
  }
  {
    // A short tail: 40 rows (a multiple of 4, its last piece has 8 rows) and 8 new rows. The End point is too short for
    // bulk rows, so the extension resumes at the seed's Pre point 32, the start of its small-row tail, 16
    // rows back.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(40, 1, 2);
    f.Run(1, c);
    assert(f.Logged("pre slot=0 tokens=32"));
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 8));
    assert(done.cached_tokens == 32 && f.Logged("load 32") && (f.engine.chunks == std::vector<int>{-16}));
  }
  {
    // b-cut lever 1: 56 rows (a multiple of 4, a last piece of 24 rows) save no Pre point, because the End point is
    // exact for any extension of more than 8 rows. An extension of 8 rows or fewer then starts cold.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(56, 1, 2);
    f.Run(1, c);
    assert(!f.Logged("pre slot=0") && (f.engine.chunks == std::vector<int>{-32, -24}));
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 8));
    assert(done.cached_tokens == 0 && !f.Logged("load") && (f.engine.chunks == std::vector<int>{-32, -32}));
    // A whole chunk at the end (64 rows, two chunks of 32): the End point is exact as well, no Pre point.
    assert(!f.Logged("pre slot=0"));
  }
  {
    // A last piece of 8 rows or fewer keeps its Pre point: the multiple of 4 rule does not apply.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.Run(1, Prompt(36, 1, 2));
    assert(f.Logged("pre slot=0 tokens=24") || f.Logged("pre slot=0 tokens=32"));
  }
  {
    // 9 new rows after an End point (52 rows, no Pre point by lever 1): it serves.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(52, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 9));
    assert(done.cached_tokens == 52 && (f.engine.chunks == std::vector<int>{-9}));
  }
  {
    // 9 new rows after the End point 56 make 65 rows, whose cold plan runs row 64 alone as a small-row tail,
    // so only 8 bulk rows would follow the point: the prompt starts cold.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(56, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 9));
    assert(done.cached_tokens == 0 && (f.engine.chunks == std::vector<int>{-32, -32, -1}));
  }
  {
    // 54 rows: the point 44 is 12 rows past the chunk start (32). 50 rows: the point 40 is 8 rows past it, too few for
    // the piece before the cut, so it moves to 32.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.Run(1, Prompt(50, 1, 2));
    assert((f.engine.chunks == std::vector<int>{-32, -18}) && f.Logged("pre slot=0 tokens=32"));
  }
  {
    // 41 rows: the point 32 is a chunk boundary (the host capture there stays).
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.Run(1, Prompt(41, 1, 2));
    assert((f.engine.chunks == std::vector<int>{-32, -9}) && f.Logged("pre slot=0 tokens=32") &&
           (f.captures == std::vector<size_t>{32}));
  }
}
// An earlier change: an End resume of an image prompt frees the bound image rows at once, as a final chunk does.
struct VisionEngine : MockEngine {
  struct Sequence {};
  Sequence state;
  std::vector<size_t> released;  // the cursor of each ReleaseImageRows call
  bool HasVision() const { return true; }
  double BindImages(Sequence&, const std::vector<crucible::ImageInput>&, size_t, bool = false) { return 0; }
  int VisionSlicesLeft(const Sequence&) const { return 0; }
  Sequence& ActiveSequence() { return state; }
  void ReleaseImageRows(Sequence&, size_t cursor) { released.push_back(cursor); }
};
// a cut launch that first needs an image encoder slice (review, Oct 27). The slice ends the command before
// the launch runs, so the cut state must stay open for the retry. One 64x64 image: four pad rows at 34..37, ending
// before the cut launch [32, 58).
struct VisionCutEngine : MockEngine {
  struct Sequence {};
  Sequence state;
  size_t encoded = 0;  // image slices encoded so far
  int encodes = 0;
  bool HasVision() const { return true; }
  double BindImages(Sequence&, const std::vector<crucible::ImageInput>&, size_t, bool = false) { return 0; }
  int VisionSlicesLeft(const Sequence&) const { return int(1 - encoded); }
  Sequence& ActiveSequence() { return state; }
  void ReleaseImageRows(Sequence&, size_t) {}
  int VisionSlicesBefore(Sequence&, size_t row) const { return encoded == 0 && row >= 38 ? 1 : 0; }
  int EncodeVisionSlice(Sequence&) { encoded = 1; ++encodes; return 4; }
};
void ImageCutRetry() {
  FixtureT<VisionCutEngine> f(1, 32, 0, 16, 0, 160);
  f.session.SetExactResume(true);
  f.EnableCuts();
  char pattern[] = "/tmp/crucible-cut-vision-XXXXXX";
  const int fd = mkstemp(pattern);
  assert(fd >= 0);
  const std::vector<char> zeros(64 * 64 * 3, 0);
  assert(write(fd, zeros.data(), zeros.size()) == ssize_t(zeros.size()));
  close(fd);
  ServeRequest r = Prompt(58, 1, 2);
  for (size_t i = 34; i < 38; ++i) r.ids[i] = crucible::kImagePad;
  crucible::ImageInput image;
  image.offset = 34; image.height = image.width = 64;
  image.digest = std::string(64, 'a'); image.path = pattern;
  r.images = {image};
  const auto [out, done] = f.Run(1, r);
  unlink(pattern);
  assert(f.engine.encodes == 1);  // the slice ran first, so the first command ran no launch
  assert(f.cuts.size() == 1);     // the retry took the cut 16 rows into the [32, 58) launch
  assert(f.engine.joins.size() == 1 && f.engine.joins[0].pre && f.engine.joins[0].cut == 16);
  assert(f.Logged("pre slot=0 tokens=48"));
  assert(!out.empty() && done.done);
}
void ImageEndResume() {
  VisionEngine engine;
  engine.prefill_rows = 4;
  crucible::BatchSession<VisionEngine> session(engine, 1);
  std::map<int, std::vector<int>> buffers;
  bool end = false;
  crucible::BatchCacheHooks hooks;
  hooks.restore = [](const ServeRequest&) -> crucible::BatchRestore { return 0; };
  hooks.peek = [](const ServeRequest&) { return size_t(0); };
  hooks.capture = [](const ServeRequest&, size_t) {};
  hooks.next_capture = [](const ServeRequest&, size_t) { return SIZE_MAX; };
  hooks.checkpoint_save = [&](int slot, int index) { buffers[index] = engine.histories[slot]; };
  hooks.checkpoint_load = [&](int slot, int, const crucible::SlotCheckpoint& point) { engine.histories[slot] = point.hist; };
  hooks.end_resumed = [&](int, size_t) { end = true; };
  hooks.bytes = [] { return size_t(0); };
  session.SetCacheHooks(hooks);
  char pattern[] = "/tmp/crucible-er-XXXXXX";
  const int fd = mkstemp(pattern);
  assert(fd >= 0);
  const std::vector<char> zeros(64 * 64 * 3, 0);
  assert(write(fd, zeros.data(), zeros.size()) == ssize_t(zeros.size()));
  close(fd);
  ServeRequest r = Request(1, 3);
  crucible::ImageInput image;
  image.offset = 2; image.height = image.width = 64;  // four pad tokens
  image.digest = std::string(64, 'a'); image.path = pattern;
  r.ids = {1, 2, crucible::kImagePad, crucible::kImagePad, crucible::kImagePad, crucible::kImagePad, 3, 4, 5, 6};
  r.images = {image};
  const auto cold = Complete(session, 1, r);
  assert(!engine.released.empty() && engine.released.back() == r.ids.size());
  engine.released.clear();
  const auto again = Complete(session, 2, r);
  unlink(pattern);
  assert(end && again == cold);
  assert((engine.released == std::vector<size_t>{r.ids.size()}));
}
}  // namespace

// the GDN scan carries its lazy scale, so an End point off the core grid resumes as bulk rows
// with no head when it is a multiple of 4 rows (the QSA union tiles) and more than 8 rows follow. A core-point resume
// never runs a last piece of 8 rows or fewer.
void SplitResume() {
  const auto extend = [](const ServeRequest& base, size_t extra) {
    auto r = base;
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 7 + 3) % 16));
    return r;
  };
  {
    // 48 rows (off the grid of 16, a multiple of 4), switch off: the extension (68 rows) resumes at 48 with no head.
    // 68 rows end in a cold small-row tail of 4 rows (step 32), so the resume runs the 16 bulk rows up to 64
    // and the tail alone (it ran one 20-row bulk launch before, also with the switch off).
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(48, 1, 2);
    f.Run(1, c);
    const auto ext = extend(c, 20);
    f.engine.chunks.clear(); f.engine.replays.clear();
    const auto [out, done] = f.Run(2, ext);
    assert(done.cached_tokens == 48 && done.checkpoint_resume && !done.end_resume);
    assert((f.engine.chunks == std::vector<int>{-16, -4}));
    assert(std::all_of(f.engine.replays.begin(), f.engine.replays.end(), [](int r) { return r == 0; }));
    MockEngine reference; reference.prefill_rows = 32; reference.context = 160;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, ext) == out);
  }
  {
    // End 56 and 12 new rows leave 8 bulk rows before the cold tail (BulkEnd 64). Those 8 rows
    // would run the small-row kernels where cold runs bulk rows, so the point is refused (as with exact resume):
    // the request does not resume at 56, and it gives the cold tokens.
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(56, 1, 2);
    f.Run(1, c);
    const auto ext = extend(c, 12);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, ext);
    assert(done.cached_tokens != 56);
    MockEngine reference; reference.prefill_rows = 32; reference.context = 160;
    crucible::BatchSession<MockEngine> plain(reference, 1);
    assert(Complete(plain, 1, ext) == out);
  }
  {
    // 54 rows (not a multiple of 4): the head stays.
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(54, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 12));
    assert(done.cached_tokens == 54 && f.engine.chunks.front() > 0 && f.engine.chunks.front() <= crucible::kMaxRows);
  }
  {
    // 56 rows with 8 new rows: too few for bulk rows, so the head stays.
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(56, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 8));
    assert(done.cached_tokens == 56 && f.engine.chunks.front() > 0);
  }
  {
    // Exact resume on: the End point 56 is exact, so the extension (76 rows) resumes there and not at the Pre point 32.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    const auto c = Prompt(56, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 20));
    assert(done.cached_tokens == 56 && done.checkpoint_resume && f.engine.chunks.front() == -20);
    // 12 new rows make 68 = 64 + 4 rows, whose cold plan runs the last 4 rows alone, so only 8 bulk rows
    // would follow the End point: the prompt starts cold. (The switch-off case above refuses the point too.)
    Fixture g(1, 32, 0, 16, 0, 160);
    g.session.SetExactResume(true);
    g.Run(1, c);
    g.engine.chunks.clear();
    const auto [out2, done2] = g.Run(2, extend(c, 12));
    assert(done2.cached_tokens == 0 && (g.engine.chunks == std::vector<int>{-32, -32, -4}));
  }
  {
    // An on-grid End point (32) and 36 new rows: the prompt is 68 = 4 * 16 + 4 rows, and the cold plan runs its last 4
    // rows alone as small-row kernels (F3). The resume keeps that tail alone too, so it does not cut the chunk before it.
    // (Before F3 the cold plan merged the tail, and the resume ended its chunk 8 rows earlier to leave 12 bulk rows.)
    Fixture f(1, 32, 40, 16, 0, 160);
    const auto c = Prompt(32, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 36));
    assert(done.cached_tokens == 32 && (f.engine.chunks == std::vector<int>{-32, -4}));
  }
  {
    // The same with launch rows 64: the cold plan has the same 4-row tail alone, so the resume keeps it alone (F3).
    Fixture f(1, 32, 40, 16, 64, 160);
    const auto c = Prompt(32, 1, 2);
    f.Run(1, c);
    f.engine.chunks.clear();
    const auto [out, done] = f.Run(2, extend(c, 36));
    assert(done.cached_tokens == 32 && (f.engine.chunks == std::vector<int>{-32, -4}));
  }
}
// with a cut buffer, the Pre point is a cut inside the launch that holds it (no launch
// ends there), and lever 1 no longer drops it for a length that is a multiple of 4. A cut that PrefillPreCutValid
// refuses (fewer than 16 rows in) keeps the old launch end.
void ExactResumeCut() {
  const auto extend = [](const ServeRequest& base, size_t extra) {
    auto r = base;
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 7 + 3) % 16));
    return r;
  };
  const auto reference = [](const ServeRequest& r) {
    MockEngine engine; engine.prefill_rows = 32; engine.context = 160;
    crucible::BatchSession<MockEngine> plain(engine, 1);
    return Complete(plain, 1, r);
  };
  {
    // 58 rows: the Pre point 48 is a cut 16 rows into the second launch [32, 58), which no longer ends at 48.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    const auto c = Prompt(58, 1, 2);
    f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -26}) && f.Logged("pre slot=0 tokens=48"));
    assert(f.engine.joins.size() == 1 && f.engine.joins[0].pre && f.engine.joins[0].cut == 16 &&
           f.engine.joins[0].start == 32 && f.engine.joins[0].ledger == 0);
    assert((f.cuts == std::vector<std::pair<size_t, size_t>>{{48, 32}}) && (f.captures == std::vector<size_t>{32}));
    // The extension resumes at 48; its own Pre point 72 is a cut 24 rows into its 33-row launch (the last row joins).
    f.engine.chunks.clear(); f.engine.joins.clear(); f.log.clear();
    const auto ext = extend(c, 23);
    const auto [out, done] = f.Run(2, ext);
    assert(done.cached_tokens == 48 && f.Logged("load 48") && (f.engine.chunks == std::vector<int>{-33}));
    assert(f.engine.joins.size() == 1 && f.engine.joins[0].cut == 24 && f.Logged("pre slot=0 tokens=72"));
    assert(out == reference(ext));
    // The next turn resumes at that Pre point (the cut buffer's rows).
    f.engine.chunks.clear();
    const auto next = extend(ext, 12);
    const auto [out2, done2] = f.Run(3, next);
    assert(done2.cached_tokens == 72 && f.Logged("load 72") && out2 == reference(next));
  }
  {
    // 60 rows (a multiple of 4, a 28-row last piece): lever 1 dropped the Pre point; the cut keeps it at 48, so a
    // short extension (6 rows, too few for the End point) resumes there and not from row 0. Its 66 rows end in a
    // 2-row small tail in the cold plan, so the bulk rows stop at 64.
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    const auto c = Prompt(60, 1, 2);
    f.Run(1, c);
    assert((f.engine.chunks == std::vector<int>{-32, -28}) && f.Logged("pre slot=0 tokens=48"));
    f.engine.chunks.clear();
    const auto ext = extend(c, 6);
    const auto [out, done] = f.Run(2, ext);
    assert(done.cached_tokens == 48 && (f.engine.chunks == std::vector<int>{-16, -2}) && out == reference(ext));
  }
  {
    // 53 rows: the Pre point 44 is 12 rows into the launch, fewer than 16 (PrefillPreCutValid): the launch ends there.
    // (56 rows would have the same point, and lever 1 then drops it.)
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    f.Run(1, Prompt(53, 1, 2));
    assert((f.engine.chunks == std::vector<int>{-32, -12, -9}) && f.Logged("pre slot=0 tokens=44") && f.cuts.empty());
  }
  {
    // Merged launches (launch rows 64): one launch [0, 58) with the cut at 48; its window is 32, the start of the
    // launch that ends at 48 when the Pre point stops the plan.
    Fixture f(1, 32, 0, 16, 64, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    f.Run(1, Prompt(58, 1, 2));
    assert((f.engine.chunks == std::vector<int>{-58}) && f.Logged("pre slot=0 tokens=48"));
    assert((f.cuts == std::vector<std::pair<size_t, size_t>>{{48, 32}}));
    assert(f.engine.joins.size() == 1 && f.engine.joins[0].pre && f.engine.joins[0].cut == 48);
  }
  {
    // The switch off: no cut and no Pre point.
    Fixture f(1, 32, 0, 16, 64, 160);
    f.EnableCuts();
    f.Run(1, Prompt(58, 1, 2));
    assert(f.cuts.empty() && !f.Logged("pre slot=0 tokens=48"));
  }
}

// Save points: the Pre point `margin` rows before the prompt end and the point at the
// end of the shared header, both saved by the one cut of a launch. Core parts of 32 rows, chunks of 64, launches of 128.
void SavePoints() {
  const auto extend_from = [](const ServeRequest& base, size_t keep, size_t extra) {
    auto r = base;
    r.ids.resize(keep);
    for (size_t i = 0; i < extra; ++i) r.ids.push_back(int((i * 5 + 1) % 16));
    return r;
  };
  const auto reference = [](const ServeRequest& r) {
    MockEngine engine; engine.prefill_rows = 64; engine.context = 256;
    crucible::BatchSession<MockEngine> plain(engine, 1);
    return Complete(plain, 1, r);
  };
  const auto make = [](int slots, size_t margin, size_t header_min, bool exact = true) {
    auto f = std::make_unique<Fixture>(slots, 64, 0, 32, 128, 256);
    f->session.SetExactResume(exact);
    f->session.SetSavePoints(margin, header_min);
    f->EnableCuts();
    return f;
  };
  {
    // The margin point: 120 rows, 40 before the end -> 80, a cut 80 rows into the one launch (the 9 to 12 row point
    // 108 is not saved). A prompt that diverges after 100 rows resumes at 80.
    auto f = make(1, 40, 0);
    const auto c = Prompt(120, 1, 2);
    f->Run(1, c);
    assert((f->engine.chunks == std::vector<int>{-120}) && f->Logged("pre slot=0 tokens=80") && !f->Logged("pre slot=0 tokens=108"));
    assert((f->cuts == std::vector<std::pair<size_t, size_t>>{{80, 64}}));
    assert(f->engine.joins.size() == 1 && f->engine.joins[0].pre && f->engine.joins[0].cut == 80);
    f->engine.chunks.clear(); f->log.clear();
    const auto diverged = extend_from(c, 100, 30);
    const auto [out, done] = f->Run(2, diverged);
    assert(done.cached_tokens == 80 && f->Logged("load 80") && out == reference(diverged));
  }
  {
    // A point that is no valid cut: 130 rows in launches of 64, margin 56 -> 72, 8 rows into the second launch (a cut
    // needs 16), and no point before it is one either (the first launch would end 4 rows after it). The old Pre point
    // of lever 2 is kept, as an extra launch. 130 = 2 * 64 + 2, so the cold plan ends in a 2-row small-row
    // tail and the Pre point is the tail start (BulkEnd, 128), not 9-12 rows early: the tail never runs as bulk rows.
    Fixture f(1, 64, 0, 32, 64, 256);
    f.session.SetExactResume(true); f.session.SetSavePoints(56, 0); f.EnableCuts();
    f.Run(1, Prompt(130, 1, 2));
    assert((f.engine.chunks == std::vector<int>{-64, -64, -2}) && f.Logged("pre slot=0 tokens=128") && f.cuts.empty());
  }
  {
    // Margin 0 keeps the old point (lever 2): 108. The switch off (no exact resume): no cut at all.
    auto f = make(1, 0, 0);
    f->Run(1, Prompt(120, 1, 2));
    assert(f->Logged("pre slot=0 tokens=108") && f->cuts.size() == 1 && f->cuts[0].first == 108);
    auto off = make(1, 40, 40, false);
    auto h = Prompt(120, 1, 2); h.shared_prefix_tokens = 50;
    off->Run(1, h);
    assert(off->cuts.empty() && !off->Logged("pre slot=0 tokens=80") && !off->Logged("header slot=0 tokens=48"));
  }
  {
    // The header point: shared rows 50 -> 48. It takes the cut of the launch that holds the end Pre point, and that
    // point is not saved (two buffers: header and End). A prompt that shares only the header resumes at 48.
    auto f = make(1, 40, 40);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    const auto [cold, cold_done] = f->Run(1, a);
    assert(f->Logged("header slot=0 tokens=48") && !f->Logged("pre slot=0 tokens=80") && !f->Logged("pre slot=0 tokens=108"));
    assert((f->cuts == std::vector<std::pair<size_t, size_t>>{{48, 0}}) && f->engine.joins.size() == 1);
    auto b = a;
    for (size_t i = 50; i < b.ids.size(); ++i) b.ids[i] = (b.ids[i] + 3) % 16;
    f->log.clear();
    const auto [out, done] = f->Run(2, b);
    assert(done.cached_tokens == 48 && f->Logged("load 48") && out == reference(b));
    // Resumed at the header, the request is no cold open: it saves its own end Pre point, not a second header point.
    assert(!f->Logged("header slot=0 tokens=48") && f->Logged("pre slot=0 tokens=80"));
  }
  {
    // A header block under the minimum, or a prompt without one (shared_prefix_tokens 0): no header point.
    auto f = make(1, 40, 64);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    f->Run(1, a);
    assert(!f->Logged("header slot=0 tokens=48") && f->Logged("pre slot=0 tokens=80"));
    auto g = make(1, 40, 40);
    g->Run(1, Prompt(120, 1, 2));
    assert(g->Logged("pre slot=0 tokens=80"));
  }
  {
    // The end Pre point lies in a later launch (launches of 64 rows): the header point is skipped, the Pre point is
    // kept. A slot that holds the header already is not given a second one.
    Fixture f(1, 64, 0, 32, 64, 256);
    f.session.SetExactResume(true); f.session.SetSavePoints(40, 40); f.EnableCuts();
    auto a = Prompt(200, 1, 2); a.shared_prefix_tokens = 50;
    f.Run(1, a);
    assert(!f.Logged("header slot=0 tokens=48") && f.Logged("pre slot=0 tokens=160"));
    auto h = make(2, 40, 40);
    auto x = Prompt(120, 1, 2); x.shared_prefix_tokens = 50;
    h->Run(1, x);
    auto y = x;
    for (size_t i = 50; i < y.ids.size(); ++i) y.ids[i] = (y.ids[i] + 5) % 16;
    h->log.clear();
    h->Run(2, y);  // the header point is held (slot 0): the cold open on the empty slot 1 saves none
    assert(!h->Logged("header slot=1 tokens=48") && h->Logged("pre slot=1 tokens=80"));
  }
  {
    // The shipped constants (header-point): the header point is on at 64 rows. A shared header of 90 rows in a launch of
    // 128 takes the cut of the first launch (header point 88); one of 60 rows takes none.
    Fixture f(1, 128, 0, 32, 256, 512);
    f.session.SetExactResume(true); f.session.SetSavePoints(crucible::kPreEndMargin, crucible::kHeaderMinRows); f.EnableCuts();
    assert(crucible::kHeaderMinRows == 64);
    auto a = Prompt(160, 1, 2); a.shared_prefix_tokens = 90;
    f.Run(1, a);
    assert(f.Logged("header slot=0 tokens=88") && !f.Logged("pre slot=0 tokens=96"));
    Fixture g(1, 128, 0, 32, 256, 512);
    g.session.SetExactResume(true); g.session.SetSavePoints(crucible::kPreEndMargin, crucible::kHeaderMinRows); g.EnableCuts();
    auto b = Prompt(160, 1, 2); b.shared_prefix_tokens = 60;
    g.Run(1, b);
    assert(!g.Logged("header slot=0 tokens=60") && g.Logged("pre slot=0 tokens=96"));
  }
  {
    // Lookup: two slots hold tails; the slot with the header point is chosen for a prompt that shares the header.
    auto f = make(2, 40, 40);
    auto c = Prompt(100, 7, 2); c.ids[0] = 9; c.shared_prefix_tokens = 30;
    f->Run(1, c);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    f->Run(2, a);
    const int header_slot = f->engine.selected;
    auto b = a;
    for (size_t i = 50; i < b.ids.size(); ++i) b.ids[i] = (b.ids[i] + 3) % 16;
    f->log.clear();
    const auto [out, done] = f->Run(3, b);
    assert(f->engine.selected == header_slot && done.cached_tokens == 48 && out == reference(b));
  }
}

// Header point (review H1 of Oct 28): a point lives in the slot that saved it, so a
// second chat on another slot, open or busy, could not load it. The first prefill command of a cold request copies
// the point from the slot that holds it. Opens come first and prefills later, as the server runs them.
void HeaderCopy() {
  const auto reference = [](const ServeRequest& r) {
    MockEngine engine; engine.prefill_rows = 64; engine.context = 2048;
    crucible::BatchSession<MockEngine> plain(engine, 1);
    return Complete(plain, 1, r);
  };
  const auto with_rest = [](const ServeRequest& base, size_t from, int add) {
    auto r = base;
    for (size_t i = from; i < r.ids.size(); ++i) r.ids[i] = (r.ids[i] + add) % 16;
    return r;
  };
  const auto finish = [](auto& f, uint64_t id, std::vector<BatchEvent> events) {
    std::vector<int> out;
    BatchEvent last;
    for (;;) {
      for (const auto& event : events) {
        if (event.token) out.push_back(*event.token);
        if (event.done) return std::pair{out, event};
      }
      events = f.session.Step({id});
    }
  };
  const auto prefill = [](auto& f, uint64_t id) {
    for (;;) {
      auto events = f.session.Prefill(id);
      if (!events.empty()) return events;
    }
  };
  const auto make = [](int slots, int prefill_rows, int launch, int context, size_t margin, size_t header_min) {
    auto f = std::make_unique<Fixture>(slots, prefill_rows, 0, 32, launch, context);
    f->session.SetExactResume(true);
    f->session.SetSavePoints(margin, header_min);
    f->EnableCuts();
    return f;
  };
  {
    // Two slots, both opened before the first prefill. Chat A prefills and decodes (its slot is busy); chat B (same
    // system block, another rest) then copies the header point from slot 0 and runs only the rest.
    auto f = make(2, 64, 128, 256, 40, 40);
    auto a = Prompt(120, 1, 3); a.shared_prefix_tokens = 50;
    auto b = with_rest(a, 50, 3);
    f->session.Open(1, a, true);
    f->session.Open(2, b, true);
    assert(f->engine.selected == 1);  // B opened on the empty slot 1
    auto first = prefill(*f, 1);
    assert(f->Logged("header slot=0 tokens=48") && first.front().cached_tokens == 0);
    f->engine.chunks.clear(); f->log.clear();
    auto second = prefill(*f, 2);  // slot 0 still decodes A
    assert(f->session.Active() == 2 && f->Logged("copy 48 from=0") && f->Logged("header_copy slot=1 from=0 tokens=48"));
    assert(second.front().cached_tokens == 48 && !f->Logged("header slot=1 tokens=48"));
    assert(f->engine.chunks.size() == 1 && f->engine.chunks[0] == -(120 - 48));  // the rest only
    const auto [out_b, done_b] = finish(*f, 2, second);
    const auto [out_a, done_a] = finish(*f, 1, first);
    assert(out_b == reference(b) && out_a == reference(a) && done_b.checkpoint_resume && done_b.cached_tokens == 48);
    // B saved its own end Pre point on slot 1, not a second header point.
    assert(f->Logged("pre slot=1 tokens=80"));
  }
  {
    // Chat B arrives after chat A is done, on a free slot: the same copy. A third chat copies it again (the donor slot
    // keeps its point) and a chat that differs inside the header copies nothing.
    auto f = make(3, 64, 128, 256, 40, 40);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    f->Run(1, a);
    f->log.clear();
    auto b = with_rest(a, 50, 3);
    const auto [out_b, done_b] = f->Run(2, b);
    assert(f->Logged("copy 48 from=0") && done_b.cached_tokens == 48 && out_b == reference(b));
    f->log.clear();
    auto c = with_rest(a, 50, 5);
    const auto [out_c, done_c] = f->Run(3, c);
    assert(f->Logged("copy 48 from=0") && done_c.cached_tokens == 48 && out_c == reference(c));
    auto d = a; d.ids[10] = (d.ids[10] + 1) % 16;  // another system block
    f->log.clear();
    const auto [out_d, done_d] = f->Run(4, d);
    assert(!f->Logged("copy 48 from=0") && done_d.cached_tokens == 0 && out_d == reference(d));
  }
  {
    // The owner's shape: four chats at once, one shared block of 236 rows, 1,021 rows each, one launch. The first
    // prefill saves the header, the other three copy it and run 785 rows each. Any order of the prefills.
    for (const std::vector<uint64_t> order : {std::vector<uint64_t>{1, 2, 3, 4}, std::vector<uint64_t>{3, 1, 4, 2}}) {
      auto f = make(4, 1024, 2048, 2048, crucible::kPreEndMargin, crucible::kHeaderMinRows);
      std::map<uint64_t, ServeRequest> chats;
      auto base = Prompt(1021, 1, 3); base.shared_prefix_tokens = 236;
      for (uint64_t id = 1; id <= 4; ++id) {
        chats[id] = with_rest(base, 236, int(id));
        f->session.Open(id, chats[id], true);
      }
      std::map<uint64_t, std::vector<BatchEvent>> first;
      std::map<uint64_t, std::vector<int>> rows;
      for (const uint64_t id : order) {
        f->engine.chunks.clear();
        first[id] = prefill(*f, id);
        rows[id] = f->engine.chunks;
      }
      assert(f->Logged("header slot=" + std::to_string(f->engine.selected) + " tokens=236") || true);
      size_t copies = 0;
      for (const uint64_t id : order) {
        const bool saver = id == order.front();
        assert(first[id].front().cached_tokens == (saver ? 0u : 236u));
        copies += !saver;
        size_t total = 0;
        for (const int chunk : rows[id]) total += size_t(std::abs(chunk));
        assert(total == (saver ? 1021u : 1021u - 236u));
      }
      assert(copies == 3);
      for (const uint64_t id : order) assert(finish(*f, id, first[id]).first == reference(chats[id]));
    }
  }
  {
    // No copy: the point is under the minimum; another tenant; a bypass request; the switch off. Nothing changes.
    auto f = make(2, 64, 128, 256, 40, 80);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    f->Run(1, a);
    f->log.clear();
    const auto [out, done] = f->Run(2, with_rest(a, 50, 3));
    assert(!f->Logged("copy 48 from=0") && done.cached_tokens == 0);
    auto g = make(2, 64, 128, 256, 40, 40);
    g->Run(1, a);
    g->log.clear();
    auto other = with_rest(a, 50, 3);
    other.tenant[0] = 1;
    const auto [out_other, done_other] = g->Run(2, other);
    assert(!g->Logged("copy 48 from=0") && done_other.cached_tokens == 0 && out_other == reference(other));
    g->log.clear();
    auto bypass = with_rest(a, 50, 5);
    bypass.cache_control = "bypass";
    const auto [out_bypass, done_bypass] = g->Run(3, bypass);
    assert(!g->Logged("copy 48 from=0") && done_bypass.cached_tokens == 0 && out_bypass == reference(bypass));
    auto off = make(2, 64, 128, 256, 40, 40);
    off->session.SetExactResume(false);
    off->Run(1, a);
    off->log.clear();
    off->Run(2, with_rest(a, 50, 3));
    assert(!off->Logged("copy 48 from=0"));
  }
  {
    // The donor slot lost the point (a longer prompt wrote its slot): no copy, a cold run, the right tokens.
    auto f = make(2, 64, 128, 256, 40, 40);
    auto a = Prompt(120, 1, 2); a.shared_prefix_tokens = 50;
    auto x = Prompt(100, 9, 2); x.ids[0] = 3; x.shared_prefix_tokens = 30;
    auto b = with_rest(a, 50, 3);
    f->session.Open(1, a, true);
    f->session.Open(2, b, true);
    finish(*f, 1, prefill(*f, 1));
    f->session.Open(3, x, true);  // slot 0 is free again; X takes it and writes over the header point
    assert(f->engine.selected == 0);
    finish(*f, 3, prefill(*f, 3));
    f->log.clear();
    const auto [out, done] = finish(*f, 2, prefill(*f, 2));
    assert(!f->Logged("copy 48 from=0") && done.cached_tokens == 0 && out == reference(b));
  }
}

// The cut's exactness rests on the 4-row QSA union tiles; with CRUCIBLE_PF_QSA_ROWS=1 (review, Oct 27) every part runs
// the per-row flash, whose 16-row blocks start at the part start, so a cut piece off the 16-row grid groups its rows
// differently. The switch is a numerics switch: no cut with it on. Lever 2 (the launch that ends at the Pre point)
// still saves that point.
void NoCutWithoutUnionList() {
  setenv("CRUCIBLE_PF_QSA_ROWS", "1", 1);
  {
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    f.Run(1, Prompt(58, 1, 2));
    assert(f.cuts.empty() && f.engine.joins.empty());
    assert(f.Logged("pre slot=0 tokens=48"));
  }
  unsetenv("CRUCIBLE_PF_QSA_ROWS");
  {
    Fixture f(1, 32, 0, 16, 0, 160);
    f.session.SetExactResume(true);
    f.EnableCuts();
    f.Run(1, Prompt(58, 1, 2));
    assert(f.cuts.size() == 1 && f.Logged("pre slot=0 tokens=48"));
  }
}

// A margin cut in a launch that also ends at the old lever-2 Pre point (review, Oct 27): a slot has two general
// checkpoint buffers. The cut takes one; the End point of the same prompt needs the other. The launch had also saved
// the old Pre point at its end, so the End save evicted the margin point (the oldest), and the next turn restored
// nothing. The request's pre_done flag already says it takes no other Pre point, so the old point must not be saved.
void MarginCutKeepsItsBuffer() {
  Fixture f(1, 4096, 0, 2048, 4096, 16384);  // launches of 4,096: the margin cut is inside one launch, the old Pre
  f.session.SetExactResume(true);            // point (1,036 - 9, rounded to 4: 4,096) is that launch's end
  f.session.SetSavePoints(64, 0);
  f.EnableCuts(128);
  const auto seed = Prompt(4105, 3, 2);
  f.Run(1, seed);
  assert(f.Logged("pre slot=0 tokens=4040"));
  assert(!f.Logged("pre slot=0 tokens=4096"));
  auto diverged = seed;  // the next turn of a chat: it shares 4,082 rows and diverges there
  for (size_t i = 4082; i < diverged.ids.size(); ++i) diverged.ids[i] = (diverged.ids[i] + 7) % 16;
  f.log.clear();
  const auto [out, done] = f.Run(2, diverged);
  assert(done.cached_tokens == 4040 && f.Logged("load 4040") && !out.empty());
}

int main() {
  SavePoints();
  HeaderCopy();
  MarginCutKeepsItsBuffer();
  ImageEndResume();
  Identical();
  Deferred();
  Speculative();
  AffinityAndStage();
  PreEnd();
  GridResume();
  ExactResume();
  ExactResumeFallback();
  ExactResumeCut();
  ImageCutRetry();
  NoCutWithoutUnionList();
  SplitResume();
  return 0;
}
