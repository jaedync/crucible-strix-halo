// Re-send fix A and B, on the fake engine.
// A: an identical prompt resumes at the End point of the run that saved it, with no prompt row, and gives that run's
//    first token (greedy ID or keyed draw from the saved logits row).
// B: a long prompt ends a chunk at its final chunk's core cut and keeps a Pre point there; a prompt that extends the
//    old one past that cut resumes there.
#include "batch_mock_engine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
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
// buffers that hold the slot's rows.
struct Fixture {
  MockEngine engine;
  crucible::BatchSession<MockEngine> session;
  std::map<std::pair<int, int>, std::vector<int>> buffers;
  std::vector<std::string> log;
  std::vector<size_t> captures, floors;
  size_t host_tail = 0;  // > 0: the host restores a session tail of this length (a replayed turn)
  explicit Fixture(int slots = 1, int prefill = 4, size_t pre_min = 0, int core = 16,
                   int launch = 0)
      : session((engine.prefill_rows = prefill, engine.launch_rows = launch, engine), slots) {
    if (pre_min) session.SetPreEnd(pre_min, core);
    crucible::BatchCacheHooks hooks;
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
    hooks.bytes = [] { return size_t(0); };
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
    // As an End resume of a longer prompt: one small-row head first, then the rest.
    assert(f.engine.chunks.front() > 0 && f.engine.chunks.front() <= crucible::kMaxRows);
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

int main() {
  ImageEndResume();
  Identical();
  Deferred();
  Speculative();
  AffinityAndStage();
  PreEnd();
  return 0;
}
