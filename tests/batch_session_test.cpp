#include "batch_mock_engine.h"
#include "../engine/serve/batch_conversation_cache.h"
#include "../engine/serve/batch_mtp_history.h"
#include <cassert>
#include <unistd.h>
#include <map>
#include <tuple>
#include <numeric>

// Train 13: an engine that accepts images. BindImages records what each
// open binds; the mock's rows do not depend on images, so only the cache key can tell two images apart.
struct VisionMock : MockEngine {
  std::vector<std::string> bound;
  bool HasVision() const { return true; }
  struct Sequence {};
  double BindImages(Sequence&, const std::vector<crucible::ImageInput>& images, size_t cached, bool = false) {
    bound.push_back((images.empty() ? std::string("-") : images[0].digest.substr(0, 1)) + "@" + std::to_string(cached));
    return 0;
  }
  int VisionSlicesLeft(const Sequence&) const { return 0; }  // the encoder work is not modeled here
  Sequence& ActiveSequence() { return state; }
  Sequence state;
};
// The open checks that each image file exists with height * width * 3 bytes.
const std::string& PixelFile() {
  static const std::string path = [] {
    char pattern[] = "/tmp/crucible-bs-XXXXXX";
    const int fd = mkstemp(pattern);
    assert(fd >= 0);
    const std::vector<char> zeros(64 * 64 * 3, 0);
    assert(write(fd, zeros.data(), zeros.size()) == ssize_t(zeros.size()));
    close(fd);
    return std::string(pattern);
  }();
  return path;
}
crucible::ServeRequest ImageRequest(char digest) {
  crucible::ServeRequest r = Request(1, 2);
  crucible::ImageInput image;
  image.offset = 2; image.height = image.width = 64;  // 2 x 2 merged cells: four pad tokens
  image.digest = std::string(64, digest); image.path = PixelFile();
  r.ids = {1, 2, crucible::kImagePad, crucible::kImagePad, crucible::kImagePad, crucible::kImagePad, 3, 4};
  r.images = {image};
  return r;
}
template <class Engine>
std::vector<int> Finish(crucible::BatchSession<Engine>& session, uint64_t id, const crucible::ServeRequest& r,
                        bool* checkpoint = nullptr) {
  std::vector<int> out;
  auto events = session.Open(id, r);
  for (;;) {
    for (const auto& event : events) {
      if (event.token) out.push_back(*event.token);
      if (event.done) {
        if (checkpoint) *checkpoint = event.checkpoint_resume;
        return out;
      }
    }
    events = session.Step({id});
  }
}

// x4-tail: grid 16, launch 32, join 12, a 74-row speculative prompt with window 6 (warm-up
// from 68) and its own frontier at 64. Without joins: 32, 32, 10 and a capture at 64 after the second launch. With
// joins: 32, 42; the second launch passes 64 (its cut), and the frontier at 64 is saved from the cut buffer.
void JoinTests() {
  for (int mode = 0; mode < 4; ++mode) {
    // 0: no join rows (today); 1: joins with a cut buffer, host frontier; 2: joins, no cut buffer; 3: joins with a
    // cut buffer and slot frontiers.
    MockEngine engine; engine.prefill_rows = 16; engine.launch_rows = 32; engine.context = 128;
    engine.join_rows = mode ? 12 : 0;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<size_t> captured, cut_captured;
    std::vector<std::pair<size_t, size_t>> prepared;
    std::vector<std::pair<int, int>> saved, cut_saved;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [](const crucible::ServeRequest&) { return crucible::BatchRestore{}; };
    hooks.bytes = [] { return size_t(0); };
    hooks.capture = [&](const crucible::ServeRequest&, size_t cursor) { captured.push_back(cursor); };
    hooks.next_capture = [](const crucible::ServeRequest& r, size_t cursor) {
      const size_t own = r.ids.size() / 16 * 16;
      return cursor < own ? own : SIZE_MAX;
    };
    static float state[4], target[4];
    if (mode == 1 || mode == 3) {
      hooks.cut_ready = [](const crucible::ServeRequest&) { return true; };
      hooks.cut_prepare = [&](size_t position, size_t window) {
        prepared.emplace_back(position, window);
        return std::vector<crucible::PrefillCutCopy>{{state, target, sizeof state}};
      };
      hooks.cut_capture = [&](const crucible::ServeRequest&, size_t position) { cut_captured.push_back(position); };
    }
    if (mode == 3) {
      hooks.checkpoint_save = [&](int slot, int index) { saved.emplace_back(slot, index); };
      hooks.checkpoint_load = [](int, int, const crucible::SlotCheckpoint&) {};
      hooks.frontier = [](const crucible::ServeRequest&, size_t,
                          const std::function<bool(const crucible::SlotFrontier&)>& keep) { return keep({}); };
      hooks.frontier_fits = [](const crucible::ServeRequest&, const crucible::SlotCheckpoint&) { return false; };
      hooks.handover = [](int, int, const crucible::SlotCheckpoint&) {};
      hooks.cut_save = [&](int slot, int index) { cut_saved.emplace_back(slot, index); };
    }
    session.SetCacheHooks(hooks);
    std::vector<std::string> chunks;
    crucible::BatchSpecHooks spec;
    spec.check = [](const crucible::ServeRequest&) { return std::string(); };
    spec.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    spec.chunk = [&](uint64_t, size_t start, int rows, bool) { chunks.push_back(std::to_string(start) + "+" + std::to_string(rows)); };
    spec.first = [](uint64_t, int) {};
    spec.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last + 1}; };
    spec.close = [](uint64_t) {};
    session.SetSpecHooks(spec);
    auto request = Request(1, 4); request.spec = 7; request.mtp_prefix = 6;
    request.ids.resize(74); std::iota(request.ids.begin(), request.ids.end(), 1);
    session.Open(100, request, true);
    while (session.Prefill(100).empty()) {}
    session.Close(100);
    assert(std::equal(request.ids.begin(), request.ids.end(), engine.histories[0].begin()));
    if (mode == 0) {
      assert((engine.chunks == std::vector<int>{-32, -32, -10}) && engine.joins.empty());
      assert((captured == std::vector<size_t>{32, 64}) && cut_captured.empty());
      continue;
    }
    if (mode == 2) {
      // No cut buffer: the launch stops at the frontier, as without joins; the 10-row tail cannot join a launch that
      // starts at 64 (it is the launch's first chunk). The engine still gets each launch's ledger.
      assert((engine.chunks == std::vector<int>{-32, -32, -10}));
      assert(engine.joins.size() == 3);
      for (const auto& join : engine.joins) assert(join.ledger == 0 && join.cut == 0 && join.copies.empty());
      assert((captured == std::vector<size_t>{32, 64}));
      continue;
    }
    assert((engine.chunks == std::vector<int>{-32, -42}));
    assert(engine.joins.size() == 2);
    assert(engine.joins[0].ledger == 0 && engine.joins[0].cut == 0 && engine.joins[0].copies.empty());
    // The plan without joins runs [32, 64) and [64, 74) in the second launch: ledger 32, the cut at row 32 of it.
    assert(engine.joins[1].ledger == 32 && engine.joins[1].cut == 32 && engine.joins[1].copies.size() == 1);
    assert(engine.joins[1].copies[0].source == state && engine.joins[1].copies[0].target == target);
    assert((prepared == std::vector<std::pair<size_t, size_t>>{{64, 32}}));
    // The MTP warms the joined launch in one call; the serve hook (BatchSpecDecoder::Chunk) splits it per grid chunk.
    assert((chunks == std::vector<std::string>{"0+32", "32+42"}));
    // Mode 3 keeps the frontier at 32 in the slot (no host capture).
    assert((captured == (mode == 1 ? std::vector<size_t>{32} : std::vector<size_t>{})));
    if (mode == 1) assert((cut_captured == std::vector<size_t>{64}));
    if (mode == 3) {
      // The slot frontier at 64 comes from the cut buffer, not from the slot (which holds 74 rows by then).
      assert(cut_captured.empty() && (cut_saved.size() == 1));
      assert(!saved.empty());  // the frontier at 32 after the first launch (and the End checkpoint), from the slot
    }
  }
  // A bad spec never reaches a launch: an engine without join rows takes none (MockEngine asserts it).
}

int main() {
  JoinTests();
  // Boards 364 and 426: the done event carries the cache diagnosis and the per-tier counters as fixed words and
  // numbers only (the server copies them into its turn line).
  {
    crucible::BatchEvent event;
    event.done = true;
    event.cache_reported = true;
    event.cache_miss = "evicted";
    event.closest_outcome = "reply_differs";
    event.closest_tokens = 32768;
    event.closest_age_s = 612.5;
    event.counters_reported = true;
    event.counters.host.lookups = 5;
    event.counters.host.hits = 4;
    event.counters.host.tokens = 49152;
    const auto fields = crucible::BatchCacheFields(event);
    assert(fields.find(",\"miss_reason\":\"evicted\"") != std::string::npos);
    assert(fields.find(",\"closest_shared_tokens\":32768") != std::string::npos);
    assert(fields.find(",\"closest_age_s\":612.500,\"closest_outcome\":\"reply_differs\"") != std::string::npos);
    assert(fields.find(",\"cache_host_lookups\":5,\"cache_host_hits\":4,\"cache_host_tokens\":49152") !=
           std::string::npos);
    assert(fields.find(",\"cache_slot_lookups\":0,\"cache_slot_hits\":0,\"cache_slot_tokens\":0") !=
           std::string::npos);
    // A word outside the fixed set never reaches the line (it could carry request text).
    event.cache_miss = "the prompt was too long";
    assert(crucible::BatchCacheFields(event).find("miss_reason") == std::string::npos);
    event.cache_miss = nullptr;
    event.closest_outcome = nullptr;
    assert(crucible::BatchCacheFields(event).find("closest_shared_tokens") == std::string::npos);
  }
  MockEngine e;
  crucible::BatchSession<MockEngine> s(e, 8);
  assert(Complete(s, 1, Request(2)) == std::vector<int>({3,4,5,6,7}));
  {  // The scoring position: same token with and without the option; the values are a log-softmax.
    auto scored = Request(2, 1); scored.top_logprobs = 3;
    const auto events = s.Open(60, scored);
    assert(events.size() == 2 && events[0].token == 3 && events[0].logprobs.size() == 3);
    double sum = 0;
    for (int id = 0; id < 16; ++id) sum += std::exp(-std::abs(3 - id));
    const int want[3] = {3, 2, 4};
    for (int i = 0; i < 3; ++i) {
      assert(events[0].logprobs[i].id == want[i]);
      assert(std::fabs(events[0].logprobs[i].logprob - (-std::abs(3 - want[i]) - std::log(sum))) < 1e-5);
    }
    assert(events[1].done && events[1].logprobs.empty());
    const auto plain = s.Open(61, Request(2, 1));
    assert(plain[0].token == 3 && plain[0].logprobs.empty());
  }
  const auto sampled = Complete(s, 2, Request(3, 8, .8, 77));
  auto left = s.Open(3, Request(3, 8, .8, 77));
  auto right = s.Open(4, Request(3, 8, .8, 77));
  std::map<uint64_t, std::vector<int>> outputs;
  outputs[3].push_back(*left.front().token); outputs[4].push_back(*right.front().token);
  for (int step = 1; step < 8; ++step) {
    const auto events = s.Step(step % 2 ? std::vector<uint64_t>{4,3} : std::vector<uint64_t>{3,4});
    for (const auto& event : events) if (event.token) outputs[event.request_id].push_back(*event.token);
  }
  assert(outputs[3] == sampled && outputs[4] == sampled);
  assert(s.Active() == 0);
  s.Open(5, Request(1)); s.Open(6, Request(8));
  const auto before = e.calls;
  for (const auto& invalid : {std::vector<uint64_t>{5,5}, std::vector<uint64_t>{5,99}, std::vector<uint64_t>{}}) {
    try { s.Step(invalid); assert(false); } catch (const std::invalid_argument&) {}
    assert(e.calls == before);
  }
  s.Close(5);
  const auto peer = s.Step({6}); assert(peer[0].token == 10);
  assert(Complete(s, 7, Request(4, 1)) == std::vector<int>{5});
  auto stop = Request(1); stop.stop_ids = {2};
  const auto stopped = s.Open(8, stop); assert(stopped.size() == 2 && stopped.back().stopped);
  auto bad = Request(1); bad.spec = 2;
  const auto calls = e.calls;
  try { s.Open(9, bad); assert(false); } catch (const std::invalid_argument&) {}
  assert(e.calls == calls);
  try { s.Open(6, Request(1)); assert(false); } catch (const std::invalid_argument&) {}
  for (uint64_t id = 10; id < 17; ++id) s.Open(id, Request(1));
  try { s.Open(17, Request(1)); assert(false); } catch (const std::invalid_argument&) {}
  s.Close(6); s.Open(17, Request(2));
  assert(s.Active() == 8);
  MockEngine bulk; bulk.prefill_rows = 4;
  crucible::BatchSession<MockEngine> multi(bulk, 2);
  auto long_request = Request(1, 1, .8, 9); long_request.ids = {1,2,3,4,5,6,7,8,9};
  const auto serial_bulk = Complete(multi, 20, long_request);
  const auto lease = multi.Open(21, long_request, true);
  assert(!lease[0].token && lease[0].slot_generation == 2);
  multi.Open(22, Request(7));
  const auto initial_calls = bulk.calls;
  try { multi.Step({22,21}); assert(false); } catch (const std::invalid_argument&) {}
  assert(bulk.calls == initial_calls);
  assert(multi.Prefill(21).empty());
  assert(*multi.Step({22})[0].token == 9);
  assert(multi.Prefill(21).empty());
  const auto final = multi.Prefill(21);
  assert(*final[0].token == serial_bulk[0] && final.back().done);
  // A plain request reads no final trunk stacks except the head row, which the engine keeps itself.
  assert((bulk.keeps == std::vector<int>{4, 4, 1, 1, 4, 4, 1}));
  multi.Close(22);
  for (bool wrong_width : {false, true}) {
    MockEngine failed;
    crucible::BatchSession<MockEngine> poisoned(failed, 2);
    poisoned.Open(30, Request(1, 8, .8, 9)); poisoned.Open(31, Request(2, 8, .8, 10));
    failed.nan_row = wrong_width ? -1 : 1;
    failed.short_output = wrong_width;
    try { poisoned.Step({30,31}); assert(false); } catch (const std::exception&) {}
    assert(poisoned.Poisoned());
    const auto after_failure = failed.calls;
    try { poisoned.Step({30,31}); assert(false); } catch (const std::runtime_error&) {}
    try { poisoned.Open(32, Request(2)); assert(false); } catch (const std::runtime_error&) {}
    try { poisoned.Close(30); assert(false); } catch (const std::runtime_error&) {}
    assert(failed.calls == after_failure);
  }
  for (int invalid : {-1, crucible::kVocab}) {
    MockEngine failed;
    crucible::BatchSession<MockEngine> poisoned(failed, 1);
    poisoned.Open(35, Request(1));
    failed.forced_greedy = invalid;
    try { poisoned.Step({35}); assert(false); } catch (const std::runtime_error&) {}
    assert(poisoned.Poisoned());
    MockEngine prefill;
    crucible::BatchSession<MockEngine> first(prefill, 1);
    first.Open(36, Request(1), true);
    prefill.forced_greedy = invalid;
    try { first.Prefill(36); assert(false); } catch (const std::runtime_error&) {}
    assert(first.Poisoned());
  }
  MockEngine failed_prefill;
  crucible::BatchSession<MockEngine> poisoned_prefill(failed_prefill, 1);
  poisoned_prefill.Open(40, Request(2, 1, .8, 1), true);
  failed_prefill.nan_row = 0;
  try { poisoned_prefill.Prefill(40); assert(false); } catch (const std::exception&) {}
  assert(poisoned_prefill.Poisoned());

  // A restored prefix continues on the same fixed grid and gives the cache-off output.
  MockEngine cached_engine; cached_engine.prefill_rows = 4;
  crucible::BatchSession<MockEngine> cached(cached_engine, 2);
  std::vector<std::pair<uint64_t, size_t>> captures;
  crucible::BatchCacheHooks hooks;
  hooks.restore = [&](const crucible::ServeRequest& request) -> crucible::BatchRestore {
    if (request.ids.front() != 1) return 0;
    cached_engine.histories[cached_engine.selected].assign(request.ids.begin(), request.ids.begin() + 4);
    return {4, true};  // reported as a shared system-and-tools hit
  };
  hooks.capture = [&](const crucible::ServeRequest& request, size_t cursor) {
    captures.emplace_back(request.ids.size(), cursor);
  };
  hooks.bytes = [] { return size_t(123); };
  cached.SetCacheHooks(hooks);
  const auto before_hit = cached_engine.calls;
  const auto opened_hit = cached.Open(50, long_request, true);
  // The opened event names the restored tokens: a request cancelled before done still reports its cache hit.
  assert(opened_hit.size() == 1 && opened_hit[0].cached_tokens == 4);
  assert(cached.Prefill(50).empty());
  // The cursor counts the restored rows, so the frontend ranks this turn by the 1 row that remains.
  assert(cached.PrefillCursor() == 8);
  const auto hit = cached.Prefill(50);
  assert(cached.PrefillCursor() == 9);
  assert(cached_engine.calls == before_hit + 2);
  assert(*hit[0].token == serial_bulk[0] && hit.back().done);
  assert(hit.back().cached_tokens == 4 && hit.back().cache_bytes == 123 && hit.back().restore_seconds >= 0 && hit.back().cache_shared);
  assert(cached_engine.histories[0] == long_request.ids);
  // Captures run after non-final chunks only: cursor 8 of 9 here.
  assert((captures == std::vector<std::pair<uint64_t, size_t>>{{9, 8}}));
  auto miss = long_request; miss.ids.front() = 2;
  cached.Open(51, miss, true);
  for (int chunk = 0; chunk < 2; ++chunk) assert(cached.Prefill(51).empty());
  const auto missed = cached.Prefill(51).back();
  assert(missed.cached_tokens == 0 && !missed.cache_shared);
  assert(captures.size() == 3 && captures[1].second == 4 && captures[2].second == 8);
  // A restore of the whole prompt would leave no row to predict the first token.
  hooks.restore = [](const crucible::ServeRequest& request) { return request.ids.size(); };
  cached.SetCacheHooks(hooks);
  try { cached.Open(52, long_request, true); assert(false); } catch (const std::logic_error&) {}
  assert(cached.Poisoned());

  // Merged bulk launches (prefill-moe): whole grid chunks merge up to the launch rows, a partial chunk ends a
  // launch, and a launch never crosses a capture point. A capture hook without next_capture keeps the grid.
  for (int mode = 0; mode < 4; ++mode) {
    MockEngine engine; engine.prefill_rows = 4; engine.launch_rows = 16;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<size_t> saved;
    crucible::BatchCacheHooks launch_hooks;
    launch_hooks.restore = [](const crucible::ServeRequest&) { return crucible::BatchRestore{}; };
    launch_hooks.bytes = [] { return size_t(0); };
    if (mode >= 1) launch_hooks.capture = [&](const crucible::ServeRequest&, size_t cursor) { saved.push_back(cursor); };
    if (mode == 2) launch_hooks.next_capture = [](const crucible::ServeRequest&, size_t cursor) {
      return cursor < 12 ? size_t(12) : SIZE_MAX;
    };
    if (mode == 3) launch_hooks.next_capture = [](const crucible::ServeRequest&, size_t) { return SIZE_MAX; };
    session.SetCacheHooks(launch_hooks);
    auto request = Request(1, 1); request.ids.resize(21); std::iota(request.ids.begin(), request.ids.end(), 1);
    session.Open(90, request, true);
    std::vector<crucible::BatchEvent> done;
    while ((done = session.Prefill(90)).empty()) {}
    assert(done.back().done && engine.histories[0] == request.ids);
    const std::vector<int> expected = mode == 1 ? std::vector<int>{-4, -4, -4, -4, -4, -1}
                                     : mode == 2 ? std::vector<int>{-12, -9} : std::vector<int>{-16, -5};
    assert(engine.chunks == expected);
    assert(saved == (mode == 1 ? std::vector<size_t>{4, 8, 12, 16, 20} : mode == 2 ? std::vector<size_t>{12}
                     : mode == 3 ? std::vector<size_t>{16} : std::vector<size_t>{}));
  }

  // A turn restored from a session-replay tail keeps the grid after its small-row head chunk (it saves slot
  // checkpoints at grid ends), even with merged launches allowed.
  {
    MockEngine engine; engine.prefill_rows = 4; engine.launch_rows = 16;
    crucible::BatchSession<MockEngine> session(engine, 1);
    crucible::BatchCacheHooks hooks;
    hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + 3);
      return {3, false, 0, true};
    };
    hooks.bytes = [] { return size_t(0); };
    session.SetCacheHooks(hooks);
    auto request = Request(1, 1); request.ids.resize(21); std::iota(request.ids.begin(), request.ids.end(), 1);
    session.Open(95, request, true);
    std::vector<crucible::BatchEvent> done;
    while ((done = session.Prefill(95)).empty()) {}
    assert(done.back().done && engine.histories[0] == request.ids);
    assert((engine.chunks == std::vector<int>{4, -4, -4, -4, -2}));
  }
  // a resume from a Grid checkpoint (a chunk end of a replayed turn) continues that turn's chunk schedule. A
  // second small-row head would shift every later chunk, so the same prompt would get other logits (sampled rows
  // differed at token 40 in paths256). With no head, the resumed rows run the same calls as the turn that took the point.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::map<int, std::vector<int>> buffers;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + 3);
      return {3, false, 0, true};
    };
    hooks.checkpoint_save = [&](int slot, int index) { buffers[index] = engine.histories[slot]; };
    hooks.checkpoint_load = [&](int slot, int index, const crucible::SlotCheckpoint& point) {
      assert(buffers.at(index) == point.hist);
      engine.histories[slot] = point.hist;
    };
    hooks.bytes = [] { return size_t(0); };
    session.SetCacheHooks(hooks);
    auto request = Request(1, 1); request.ids.resize(21); std::iota(request.ids.begin(), request.ids.end(), 1);
    const auto run = [&](uint64_t id, const crucible::ServeRequest& sent) {
      auto events = session.Open(id, sent, true);
      while (events.empty() || !events.back().done) events = session.Prefill(id);
      assert(engine.histories[0] == sent.ids);
      return events.back();
    };
    const auto tail = run(96, request);
    assert(tail.cached_tokens == 3 && !tail.checkpoint_resume);
    assert((engine.chunks == std::vector<int>{4, -4, -4, -4, -2}));
    engine.chunks.clear();
    // An identical re-send resumes at its End point (fix A); a prompt that differs in its last token takes the Grid point.
    auto changed = request; changed.ids.back() += 1000;
    const auto resumed = run(97, changed);
    assert(resumed.cached_tokens == 19 && resumed.checkpoint_resume);
    assert((engine.chunks == std::vector<int>{-2}));
  }
  // A speculative request merges only up to its MTP warm-up start: 21 rows, warm-up from 15.
  {
    MockEngine engine; engine.prefill_rows = 4; engine.launch_rows = 16;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<std::string> chunks;
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
    hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    hooks.chunk = [&](uint64_t, size_t start, int rows, bool) { chunks.push_back(std::to_string(start) + "+" + std::to_string(rows)); };
    hooks.first = [](uint64_t, int) {};
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last + 1}; };
    hooks.close = [](uint64_t) {};
    session.SetSpecHooks(hooks);
    auto request = Request(1, 4); request.spec = 7; request.mtp_prefix = 6;
    request.ids.resize(21); std::iota(request.ids.begin(), request.ids.end(), 1);
    session.Open(96, request, true);
    while (session.Prefill(96).empty()) {}
    assert((engine.chunks == std::vector<int>{-12, -4, -4, -1}));
    // The final chunk's warm-up follows the first token; the next command runs it.
    assert((chunks == std::vector<std::string>{"0+12", "12+4", "16+4"}));
    session.Close(96);
    assert((chunks == std::vector<std::string>{"0+12", "12+4", "16+4", "20+1"}));
  }
  // an identical re-send whose prompt frontier lies inside its MTP window (len - window < f < len). The hooks
  // mirror BatchPrefixCache (Find with BatchRestoreLimit, capture on the frontiers of BatchCacheFrontiers). 23 rows,
  // boundary 21, grid 4, window 6 (warm-up from 17): the cold send saves 20; the re-send restores 20 and runs one
  // bulk Prefill(20, 3, keep 0), the same call as the cold send's last chunk.
  {
    struct Snap { std::vector<int> hist; crucible::ConversationResume resume; };
    MockEngine engine; engine.prefill_rows = 4; engine.launch_rows = 16;
    crucible::BatchSession<MockEngine> session(engine, 1);
    crucible::BatchConversationCache<Snap> cache(1 << 20, {}, 1, 0);
    crucible::ConversationPolicy policy;
    policy.prefill_rows = 4;
    crucible::BatchCacheHooks cache_hooks;
    cache_hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      const auto entry = cache.Find(r.ids, policy, size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r), 4);
      if (!entry) return 0;
      engine.histories[engine.selected] = entry->snapshot->hist;
      return {entry->snapshot->hist.size()};
    };
    cache_hooks.capture = [&](const crucible::ServeRequest& r, size_t cursor) {
      for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), 4))
        if (frontier.position == cursor)
          assert(cache.Capture(r.ids, policy, frontier, 64, uint64_t(9) << 30, [&] {
            return std::make_shared<const Snap>(Snap{std::vector<int>(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(cursor)),
                                                     crucible::BatchFrontierResume(cursor, size_t(r.cache_prefix_tokens))});
          }));
    };
    cache_hooks.next_capture = [](const crucible::ServeRequest& r, size_t cursor) {
      size_t next = SIZE_MAX;
      for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), 4))
        if (frontier.position > cursor) next = std::min(next, frontier.position);
      return next;
    };
    cache_hooks.bytes = [&] { return cache.Bytes(); };
    session.SetCacheHooks(cache_hooks);
    std::vector<size_t> opened;
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
    hooks.open = [&](uint64_t, int, const crucible::ServeRequest&, size_t cached) { opened.push_back(cached); };
    hooks.chunk = [](uint64_t, size_t, int, bool) {};
    hooks.stacks = [](uint64_t, size_t start, int rows) { return crucible::DraftWarmupRow(start, rows, 17); };
    hooks.first = [](uint64_t, int) {};
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last + 1}; };
    hooks.close = [](uint64_t) {};
    session.SetSpecHooks(hooks);
    auto request = Request(1, 4); request.spec = 7; request.mtp_prefix = 6; request.cache_prefix_tokens = 21;
    request.ids.resize(23); std::iota(request.ids.begin(), request.ids.end(), 1);
    const auto send = [&](uint64_t id) {
      engine.chunks.clear(); engine.keeps.clear();
      auto events = session.Open(id, request, true);
      while (events.empty() || !events.back().token) events = session.Prefill(id);
      assert(engine.histories[0] == request.ids);
      const int token = *events.back().token;
      session.Close(id);
      return std::make_tuple(token, engine.chunks, engine.keeps);
    };
    const auto [cold_token, cold_chunks, cold_keeps] = send(98);
    assert((cold_chunks == std::vector<int>{-16, -4, -3}) && (cold_keeps == std::vector<int>{16, 1, 0}));
    const auto [token, chunks, keeps] = send(99);
    assert((opened == std::vector<size_t>{0, 20}));
    assert((chunks == std::vector<int>{cold_chunks.back()}) && (keeps == std::vector<int>{cold_keeps.back()}));
    assert(token == cold_token);
  }

  // Slot classes: a request takes the smallest free slot that holds it; the full slot stays for long turns.
  MockEngine classed_engine;
  crucible::BatchSession<MockEngine> classed(classed_engine, 3, 0, 2, 16);
  assert((classed.SlotContexts() == std::vector<int>{64, 16, 16}));
  auto opened = classed.Open(60, Request(1, 10), true);
  assert(opened.size() == 1 && opened[0].slot_context == 16 && classed_engine.selected == 1);
  assert(classed.Open(61, Request(1, 16), true)[0].slot_context == 16 && classed_engine.selected == 2);
  // Prompt 1 plus 16 outputs needs 17 tokens: only the full slot holds it.
  assert(classed.Open(62, Request(1, 17), true)[0].slot_context == 64 && classed_engine.selected == 0);
  classed.Close(61);
  // A long request finds no free slot that holds it; the open fails before any state change.
  try { classed.Open(63, Request(1, 30), true); assert(false); } catch (const std::invalid_argument&) {}
  assert(!classed.Poisoned() && classed.Active() == 2);
  assert(classed.Open(64, Request(1, 5), true)[0].slot_context == 16);
  try { crucible::BatchSession<MockEngine> bad(classed_engine, 2, 0, 2, 16); (void)bad; assert(false); }
  catch (const std::exception&) {}

  // Session replay (G15): a finished request saves its tail on its own stream after the command that published
  // its last token. A later prompt that repeats the tail restores it with its pending replay rows, replays them in
  // one small-row Forward, gives the uncached output, and captures no prompt frontiers.
  {
    MockEngine replay_engine; replay_engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> replayed(replay_engine, 2);
    struct Tail { int slot; std::vector<int> hist; int replay; std::vector<int> stream; };
    std::vector<Tail> tails;
    std::vector<size_t> frontiers;
    crucible::BatchCacheHooks replay_hooks;
    replay_hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      for (const auto& t : tails)
        if (t.hist.size() < r.ids.size() && std::equal(t.hist.begin(), t.hist.end(), r.ids.begin())) {
          replay_engine.histories[replay_engine.selected] = t.stream;
          return {t.hist.size(), false, 2, true};
        }
      return 0;
    };
    replay_hooks.capture = [&](const crucible::ServeRequest&, size_t cursor) { frontiers.push_back(cursor); };
    replay_hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int replay,
                            crucible::BatchTailSource)
        -> std::optional<crucible::BatchResident> {
      std::vector<int> hist(r.ids);
      hist.insert(hist.end(), generated.begin(), generated.end() - 1);
      tails.push_back({replay_engine.selected, hist, replay, replay_engine.histories[replay_engine.selected]});
      return std::nullopt;
    };
    // each open names the tail that its request may save (prompt plus output rows), so the host pins that
    // memory while the batch decodes; a close clears it, and so does the tail save, before its copy (a review:
    // the helper must not pin a new reserve for the finished slot during the copy). Bypass requests expect nothing.
    std::vector<std::tuple<int, size_t, size_t>> expected;  // slot, rows, tails saved at the call
    replay_hooks.expect = [&](const crucible::ServeRequest&, int slot, size_t rows) {
      expected.emplace_back(slot, rows, tails.size());
    };
    replayed.SetCacheHooks(replay_hooks);
    auto turn = Request(1, 3); turn.ids = {1, 2, 3, 4, 5, 6};
    auto other = Request(9, 6); other.ids = {9, 9, 9, 9, 9};
    replayed.Open(91, other);
    auto events = replayed.Open(90, turn);
    std::vector<int> first_out{*events.front().token};
    while (first_out.size() < 3) {
      events = replayed.Step({91, 90});
      for (const auto& event : events) if (event.token && event.request_id == 90) first_out.push_back(*event.token);
    }
    assert(tails.size() == 1 && tails[0].slot == 1 && tails[0].replay == 0);
    assert((expected == std::vector<std::tuple<int, size_t, size_t>>{{0, 5 + 6, 0}, {1, 6 + 3, 0}, {1, 0, 0}}));
    std::vector<int> hist = turn.ids;
    hist.insert(hist.end(), first_out.begin(), first_out.end() - 1);
    // The stream holds every forwarded token: the prompt and all outputs but the last.
    assert(tails[0].hist == hist && tails[0].stream == hist);
    replayed.Close(91);
    assert(expected.size() == 4 && expected.back() == std::make_tuple(0, size_t(0), size_t(1)));
    const size_t frontier_count = frontiers.size();
    auto next = Request(1, 3); next.ids = hist;
    next.ids.insert(next.ids.end(), {first_out.back(), 7, 8, 9, 10, 11});
    replay_engine.replays.clear();
    std::vector<int> next_out;
    events = replayed.Open(92, next, true);
    while (events.empty() || !events.back().token) events = replayed.Prefill(92);
    assert(events.back().cache_tail && events.back().cached_tokens == hist.size());
    assert(replay_engine.replays.front() == 2 && frontiers.size() == frontier_count);
    // The first chunk after a tail is a small-row Forward; later chunks resume the bulk path.
    assert(replay_engine.replays.size() == 2 && replay_engine.replays[1] == 0);
    assert(replay_engine.histories[replay_engine.selected] == next.ids);
    next_out.push_back(*events.back().token);
    MockEngine fresh_engine; fresh_engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> fresh(fresh_engine, 1);
    assert(Complete(fresh, 1, next).front() == next_out.front());
    replayed.Close(92);
    // Bypass requests save no tail.
    auto bypass = turn; bypass.cache_control = "bypass";
    const size_t expects = expected.size();
    Complete(replayed, 93, bypass);
    assert(tails.size() == 1 && expected.size() == expects);
  }

  // Slot-resident resume: a turn that repeats the tail still on its slot returns to
  // that slot, resumes from it without a reset or a host restore, and gives the host-restore output.
  {
    struct Held { std::vector<int> hist; int replay; std::vector<int> stream; bool guarded; int slot; };
    const auto run = [](bool slot_resume, std::vector<std::vector<int>>& outputs, std::vector<std::string>& log) {
      MockEngine engine; engine.prefill_rows = 4;
      crucible::BatchSession<MockEngine> session(engine, 2);
      std::vector<Held> host;
      crucible::BatchCacheHooks hooks;
      hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
        log.push_back("restore");
        for (const auto& t : host)
          if (t.hist.size() < r.ids.size() && std::equal(t.hist.begin(), t.hist.end(), r.ids.begin())) {
            engine.histories[engine.selected] = t.stream;
            return {t.hist.size(), false, t.replay, true, false, t.guarded};
          }
        return 0;
      };
      hooks.capture = [](const crucible::ServeRequest&, size_t) {};
      hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                       crucible::BatchTailSource source) -> std::optional<crucible::BatchResident> {
        std::vector<int> hist(r.ids);
        hist.insert(hist.end(), generated.begin(), generated.end() - 1);
        log.push_back("tail slot=" + std::to_string(engine.selected) + " replayed=" + std::to_string(source.replayed) +
                      " guarded=" + std::to_string(source.guarded));
        const bool guarded = r.cache_tail == "guard";
        host.push_back({hist, 2, engine.histories[engine.selected], guarded, engine.selected});
        return crucible::BatchResident{hist.size(), 2, guarded};
      };
      if (slot_resume) {
        hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
          log.push_back("resume slot=" + std::to_string(engine.selected));
          return {resident.length, false, resident.replay, true, true, resident.guarded};
        };
        // Runs before the slot is written: the stream still holds the finished tail.
        hooks.release = [&](int slot, bool resumed) {
          const auto held = std::find_if(host.rbegin(), host.rend(), [&](const Held& t) { return t.slot == slot; });
          assert(slot == engine.selected && held != host.rend() && engine.histories[slot] == held->stream);
          log.push_back("release slot=" + std::to_string(slot) + " resumed=" + std::to_string(resumed));
        };
      }
      session.SetCacheHooks(hooks);
      const auto finish = [&](uint64_t id, const crucible::ServeRequest& r) {
        auto events = session.Open(id, r, true);
        std::vector<int> out;
        while (events.empty() || !events.back().token) events = session.Prefill(id);
        for (;;) {
          for (const auto& event : events) {
            if (event.token) out.push_back(*event.token);
            if (event.done) {
              log.push_back("done slot_resume=" + std::to_string(event.slot_resume) + " cached=" +
                            std::to_string(event.cached_tokens));
              return out;
            }
          }
          events = session.Step({id});
        }
      };
      auto a = Request(1, 3); a.ids = {1, 2, 3, 4, 5, 6}; a.cache_tail = "guard";  // a thinking turn
      auto b = Request(9, 3); b.ids = {9, 9, 9, 9, 9, 1};
      const auto first = finish(1, a);       // slot 0
      finish(2, b);                          // an empty slot, so a's tail stays on slot 0
      auto next = a; next.ids.insert(next.ids.end(), first.begin(), first.end());
      next.ids.insert(next.ids.end(), {7, 8, 9, 10, 11});
      engine.replays.clear();
      outputs.push_back(finish(3, next));
      outputs.push_back(engine.replays);
      outputs.push_back(engine.histories[engine.selected]);
      outputs.push_back({engine.selected});
      // A prompt that drops the output (reasoning) cannot use the slot: reset, then the host path.
      auto edited = a; edited.ids.insert(edited.ids.end(), {12, 13, 14});
      outputs.push_back(finish(4, edited));
    };
    std::vector<std::vector<int>> slot, host;
    std::vector<std::string> slot_log, host_log;
    run(true, slot, slot_log);
    run(false, host, host_log);
    assert(slot == host);
    assert(slot[3] == std::vector<int>{0});  // a's next turn returned to slot 0
    assert((slot_log == std::vector<std::string>{
        "restore", "tail slot=0 replayed=0 guarded=0", "done slot_resume=0 cached=0",
        "restore", "tail slot=1 replayed=0 guarded=0", "done slot_resume=0 cached=0",
        "release slot=0 resumed=1", "resume slot=0", "tail slot=0 replayed=1 guarded=1", "done slot_resume=1 cached=8",
        "release slot=0 resumed=0", "restore", "tail slot=0 replayed=0 guarded=0", "done slot_resume=0 cached=0"}));
    // The host path reports the same source: the restored tail was guarded. It has no release hook.
    assert(host_log[7] == "tail slot=0 replayed=1 guarded=1");
    assert(host_log[6] == "restore" && host_log[8] == "done slot_resume=0 cached=8");
  }

  // In-slot checkpoints: a next prompt that leaves the previous output (a token split) resumes at the
  // previous prompt end on the slot and prefills only the rest. A reset or a host restore drops the slot's points.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 2);
    std::map<std::pair<int, int>, std::vector<int>> buffers;
    std::vector<std::string> log;
    std::vector<std::vector<int>> host;  // tails
    crucible::BatchCacheHooks hooks;
    const auto longest = [&](const crucible::ServeRequest& r) -> size_t {
      size_t best = 0;
      for (const auto& t : host)
        if (t.size() < r.ids.size() && t.size() > best && std::equal(t.begin(), t.end(), r.ids.begin())) best = t.size();
      return best;
    };
    hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
      const size_t best = longest(r);
      log.push_back("restore " + std::to_string(best));
      if (!best) return 0;
      engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(best));
      return {best, false, 0, true};
    };
    hooks.peek = longest;
    std::vector<std::vector<size_t>> left;
    hooks.diverged = [&](size_t prompt, size_t restored, size_t tail, size_t common) {
      left.push_back({prompt, restored, tail, common});
    };
    hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
      return {resident.length, false, resident.replay, true, true, resident.guarded};
    };
    hooks.capture = [](const crucible::ServeRequest&, size_t) {};
    hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                     crucible::BatchTailSource source) -> std::optional<crucible::BatchResident> {
      std::vector<int> hist(r.ids);
      hist.insert(hist.end(), generated.begin(), generated.end() - 1);
      engine.histories[engine.selected] = hist;
      host.push_back(hist);
      log.push_back("tail " + std::to_string(hist.size()) + " replayed=" + std::to_string(source.replayed));
      return crucible::BatchResident{hist.size(), 0, false};
    };
    hooks.checkpoint_save = [&](int slot, int index) {
      assert(slot == engine.selected);
      buffers[{slot, index}] = engine.histories[slot];
      log.push_back("save slot=" + std::to_string(slot) + " at " + std::to_string(engine.histories[slot].size()));
    };
    hooks.checkpoint_load = [&](int slot, int index, const crucible::SlotCheckpoint& point) {
      const auto& stream = engine.histories[slot];
      // The slot still holds every row below the checkpoint, and the buffer holds the state at it.
      assert(buffers.at({slot, index}) == point.hist && stream.size() >= point.hist.size() &&
             std::equal(point.hist.begin(), point.hist.end(), stream.begin()));
      engine.histories[slot] = point.hist;
      log.push_back("load slot=" + std::to_string(slot) + " at " + std::to_string(point.hist.size()));
    };
    session.SetCacheHooks(hooks);
    const auto finish = [&](uint64_t id, const crucible::ServeRequest& r, size_t* cached = nullptr,
                            bool* resumed = nullptr) {
      auto events = session.Open(id, r);
      std::vector<int> out;
      for (;;) {
        for (const auto& event : events) {
          if (event.token) out.push_back(*event.token);
          if (event.done) {
            if (cached) *cached = event.cached_tokens;
            if (resumed) *resumed = event.checkpoint_resume;
            return out;
          }
        }
        events = session.Step({id});
      }
    };
    auto a = Request(1, 3); a.ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto first = finish(1, a);  // slot 0: saves the prompt end (10), then the tail (12)
    auto split = a;
    split.ids.insert(split.ids.end(), first.begin(), first.end());
    split.ids[11] = 15;  // the client's render splits the last output token in another way
    split.ids.insert(split.ids.end(), {3, 4, 5, 6, 7});
    size_t cached = 0;
    bool resumed = false;
    engine.chunks.clear();
    finish(2, split, &cached, &resumed);
    assert(cached == 10 && resumed && engine.selected == 0);
    assert(std::equal(split.ids.begin(), split.ids.end(), engine.histories[0].begin()));
    // The prompt left the slot's previous tail (12 tokens) at the split token (11).
    assert((left == std::vector<std::vector<size_t>>{{18, 10, 12, 11}}));
    // A tail-like resume: one small-row forward first.
    assert(engine.chunks.front() > 0 && engine.chunks.front() <= crucible::kMaxRows);
    const auto split_tail = host.back();
    // A prompt that drops the output (reasoning) of request 1 branches back to request 1's prompt end (10). With
    // two general points a slot (the third buffer holds only frontiers) request 2's grid point
    // (14) and prompt end (18) have replaced it, so no point serves this branch: it restores from the host cache. Three points resumed it at 10. This fake's
    // host cache holds only full tails, none a prefix of this prompt, so the branch prefills from row 0.
    auto drop = a; drop.ids.insert(drop.ids.end(), {12, 12, 12});
    finish(3, drop, &cached, &resumed);
    assert(!resumed && cached == 0 && std::count(log.begin(), log.end(), "load slot=0 at 10") == 1);
    // A turn that repeats the slot's tail resumes from the slot and keeps the points below it.
    auto again = drop; again.ids = host.back(); again.ids.push_back(9);
    finish(4, again, &cached, &resumed);
    assert(!resumed && cached == again.ids.size() - 1);
    // A longer host state wins over a checkpoint (split's prompt end is 17, its tail is longer).
    auto back = split; back.ids = split_tail; back.ids.push_back(9);
    finish(5, back, &cached, &resumed);
    assert(!resumed && cached == split_tail.size());
    // A host restore wrote the slot, so its older points are gone.
    finish(6, drop, &cached, &resumed);
    assert(!resumed);
    // The closed-think pattern: a cancelled first job leaves its prompt end on the slot; the continuation repeats it.
    auto think = Request(1, 8); think.ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22};
    auto opened = session.Open(7, think);
    while (std::none_of(opened.begin(), opened.end(), [](const auto& e) { return e.token.has_value(); }))
      opened = session.Prefill(7);
    const int think_slot = engine.selected;
    session.Close(7);  // cancelled at the cap: no done event, no tail
    auto continuation = think; continuation.ids.insert(continuation.ids.end(), {30, 31, 32});
    finish(8, continuation, &cached, &resumed);
    assert(resumed && cached == think.ids.size() && engine.selected == think_slot);
    // Another session takes the other slot and leaves this slot's points alone; a reset drops them.
    auto other = Request(2, 3); other.ids = {11, 11, 11, 11, 11, 11, 11};
    finish(9, other);
    assert(engine.selected != think_slot);
    auto retry = think; retry.ids.push_back(40);
    finish(10, retry, &cached, &resumed);
    assert(resumed && cached == think.ids.size());
    auto cold = Request(5, 2); cold.ids = {5, 5, 5, 5, 5};
    finish(11, cold); finish(12, cold);
    finish(13, cold);  // both slots now hold other states: one of them resets
    auto late = think; late.ids.push_back(41);
    finish(14, late, &cached, &resumed);
    assert(!resumed || cached == think.ids.size());
    // The engine-side and host logs agree on what each request did.
    assert(std::count(log.begin(), log.end(), "load slot=" + std::to_string(think_slot) + " at 13") >= 2);
  }

  // The End checkpoint follows the publication of the first token: SettlePrompt saves it.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<std::string> log;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore { log.push_back("restore"); return 0; };
    hooks.peek = [](const crucible::ServeRequest&) { return size_t(0); };
    hooks.checkpoint_save = [&](int, int) { log.push_back("save " + std::to_string(engine.histories[0].size())); };
    hooks.checkpoint_load = [](int, int, const crucible::SlotCheckpoint&) {};
    session.SetCacheHooks(hooks);
    auto a = Request(1, 8); a.ids = {1, 2, 3, 4, 5, 6};
    auto events = session.Open(1, a);
    while (std::none_of(events.begin(), events.end(), [](const auto& e) { return e.token.has_value(); }))
      events = session.Prefill(1);
    assert((log == std::vector<std::string>{"restore"}));
    session.SettlePrompt();
    assert((log == std::vector<std::string>{"restore", "save 6"}));
    session.Close(1);
  }

  // In-slot checkpoints compare cache keys (the vision merge): a point taken under one image is not resumed by a
  // request with the same pad token IDs and another image. Synthetic keys: the seed stands in for the image.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<std::string> log;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore { log.push_back("restore"); return 0; };
    hooks.peek = [](const crucible::ServeRequest&) { return size_t(0); };
    hooks.capture = [](const crucible::ServeRequest&, size_t) {};
    hooks.checkpoint_save = [&](int, int) { log.push_back("save " + std::to_string(engine.histories[0].size())); };
    hooks.checkpoint_load = [&](int, int, const crucible::SlotCheckpoint& point) {
      engine.histories[0] = point.hist;
      log.push_back("load " + std::to_string(point.hist.size()));
    };
    hooks.key = [](const crucible::ServeRequest& r, std::span<const int> tokens) {
      std::vector<int> key(tokens.begin(), tokens.end());
      for (auto& k : key) k += int(r.sampling.seed) << 20;
      return key;
    };
    session.SetCacheHooks(hooks);
    const auto cancel = [&](uint64_t id, const crucible::ServeRequest& r) {  // leaves the prompt-end point
      auto events = session.Open(id, r);
      while (std::none_of(events.begin(), events.end(), [](const auto& e) { return e.token.has_value(); }))
        events = session.Prefill(id);
      session.Close(id);
    };
    auto a = Request(1, 8, 0, 1); a.ids = {1, 2, 3, 4, 5, 6};
    cancel(1, a);
    auto other = a; other.sampling.seed = 2;  // repeats a's tokens with another image: no resume from a's point
    other.ids.insert(other.ids.end(), {7, 8});
    cancel(2, other);
    auto again = other; again.ids.push_back(9);  // repeats other's key: resumes
    cancel(3, again);
    assert((log == std::vector<std::string>{"restore", "save 6", "restore", "save 8", "load 8", "save 9"}));
  }

  // Train 13 integration (the HIGH hazard of the vision merge): a slot, a slot checkpoint or a held tail with image
  // A never resumes a request with image B and the same pad token IDs; image A again resumes. No key hook is set,
  // so BatchSession uses RequestCacheKey, the key that BatchPrefixCache gives the host cache too.
  {
    assert(crucible::RequestCacheKey(Request(1), std::vector<int>{5, 6}) == (std::vector<int>{5, 6}));
    assert(crucible::RequestCacheKey(ImageRequest('a'), ImageRequest('a').ids) !=
           crucible::RequestCacheKey(ImageRequest('b'), ImageRequest('b').ids));
    const auto slot_run = [](char second) {
      VisionMock engine; engine.prefill_rows = 4;
      crucible::BatchSession<VisionMock> session(engine, 1);
      std::vector<std::string> log;
      crucible::BatchCacheHooks hooks;
      hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore { log.push_back("restore"); return 0; };
      hooks.capture = [](const crucible::ServeRequest&, size_t) {};
      hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                       crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
        return crucible::BatchResident{r.ids.size() + generated.size() - 1, 0, false};
      };
      hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
        log.push_back("resume");
        return {resident.length, false, resident.replay, true, true};
      };
      hooks.release = [&](int, bool resumed) { log.push_back("release resumed=" + std::to_string(resumed)); };
      session.SetCacheHooks(hooks);
      const auto a = ImageRequest('a');
      const auto first = Finish(session, 1, a);
      auto next = ImageRequest(second);
      next.ids.insert(next.ids.end(), first.begin(), first.end());
      next.ids.push_back(7);
      Finish(session, 2, next);
      return std::make_pair(log, engine.bound);
    };
    const auto [other, other_bound] = slot_run('b');
    assert((other == std::vector<std::string>{"restore", "release resumed=0", "restore"}));
    assert((other_bound == std::vector<std::string>{"a@0", "b@0"}));
    const auto [same, same_bound] = slot_run('a');
    assert((same == std::vector<std::string>{"restore", "release resumed=1", "resume"}));
    assert((same_bound == std::vector<std::string>{"a@0", "a@9"}));  // the image ends below the resumed tail

    const auto checkpoint_run = [](char second) {
      VisionMock engine; engine.prefill_rows = 4;
      crucible::BatchSession<VisionMock> session(engine, 1);
      std::vector<std::string> log;
      crucible::BatchCacheHooks hooks;
      hooks.restore = [](const crucible::ServeRequest&) -> crucible::BatchRestore { return 0; };
      hooks.peek = [](const crucible::ServeRequest&) -> size_t { return 0; };
      hooks.capture = [](const crucible::ServeRequest&, size_t) {};
      hooks.checkpoint_save = [&](int slot, int) { log.push_back("save " + std::to_string(engine.histories[slot].size())); };
      hooks.checkpoint_load = [&](int slot, int, const crucible::SlotCheckpoint& point) {
        engine.histories[slot] = point.hist;
        log.push_back("load " + std::to_string(point.hist.size()));
      };
      session.SetCacheHooks(hooks);
      bool resumed = false;
      Finish(session, 1, ImageRequest('a'));
      auto edit = ImageRequest(second); edit.ids.insert(edit.ids.end(), {12, 12});  // the client drops the output
      Finish(session, 2, edit, &resumed);
      return std::make_pair(log, resumed);
    };
    const auto [other_points, other_resumed] = checkpoint_run('b');
    assert(!other_resumed && (other_points == std::vector<std::string>{"save 8", "save 10"}));
    const auto [same_points, same_resumed] = checkpoint_run('a');
    assert(same_resumed && (same_points == std::vector<std::string>{"save 8", "load 8", "save 10"}));
  }

  // a sibling with a short common prefix takes an empty slot, not another session's slot. A returning
  // session whose slot another session took restores from the host and gives the host-path output.
  {
    const auto run = [](bool slot_resume, std::vector<std::vector<int>>& outputs, std::vector<std::string>& log) {
      MockEngine engine; engine.prefill_rows = 4;
      crucible::BatchSession<MockEngine> session(engine, 2);
      struct Held { std::vector<int> hist; std::vector<int> stream; };
      std::vector<Held> host;
      crucible::BatchCacheHooks hooks;
      hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
        for (auto t = host.rbegin(); t != host.rend(); ++t)
          if (t->hist.size() < r.ids.size() && std::equal(t->hist.begin(), t->hist.end(), r.ids.begin())) {
            log.push_back("restore " + std::to_string(t->hist.size()));
            engine.histories[engine.selected] = t->stream;
            return {t->hist.size(), false, 0, true};
          }
        log.push_back("restore 0");
        return 0;
      };
      hooks.capture = [](const crucible::ServeRequest&, size_t) {};
      hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                       crucible::BatchTailSource source) -> std::optional<crucible::BatchResident> {
        assert(source.slot == engine.selected);
        std::vector<int> hist(r.ids);
        hist.insert(hist.end(), generated.begin(), generated.end() - 1);
        engine.histories[engine.selected].resize(hist.size());
        host.push_back({hist, engine.histories[engine.selected]});
        log.push_back("tail slot=" + std::to_string(source.slot));
        return crucible::BatchResident{hist.size(), 0, false};
      };
      if (slot_resume) {
        hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
          return {resident.length, false, resident.replay, true, true};
        };
        hooks.release = [&](int slot, bool resumed) {
          log.push_back("release slot=" + std::to_string(slot) + " resumed=" + std::to_string(resumed));
        };
      }
      session.SetCacheHooks(hooks);
      uint64_t id = 0;
      const auto finish = [&](const crucible::ServeRequest& r) {
        auto out = Complete(session, ++id, r);
        outputs.push_back(out);
        return out;
      };
      auto a = Request(1, 3); a.ids = {1, 2, 3, 4, 5, 6};
      auto sibling = Request(1, 3); sibling.ids = {1, 2, 3, 20, 21, 22};
      auto c = Request(9, 3); c.ids = {9, 9, 9, 9, 9, 9};
      const auto first = finish(a);   // slot 0
      finish(sibling);                // shares 3 of a's 8 tail tokens: the empty slot 1
      finish(c);                      // no empty slot: the slot left longest ago (a's slot 0)
      auto next = a; next.ids.insert(next.ids.end(), first.begin(), first.end());
      next.ids.insert(next.ids.end(), {7, 8});
      finish(next);                   // a's slot was taken: slot 1 (the older), then a's tail from the host
      outputs.push_back(engine.histories[engine.selected]);
    };
    std::vector<std::vector<int>> slot, host;
    std::vector<std::string> slot_log, host_log;
    run(true, slot, slot_log);
    run(false, host, host_log);
    assert(slot == host);
    assert((slot_log == std::vector<std::string>{
        "restore 0", "tail slot=0", "restore 0", "tail slot=1", "release slot=0 resumed=0", "restore 0", "tail slot=0",
        "release slot=1 resumed=0", "restore 8", "tail slot=1"}));
    // A host tail goes first only when it is longer, or the same length with other pending rows.
    assert(crucible::HostTailFirst(10, 0, {8, 0}) && crucible::HostTailFirst(8, 1, {8, 0}));
    assert(!crucible::HostTailFirst(8, 0, {8, 0}) && !crucible::HostTailFirst(6, 0, {8, 0}));
  }

  // Cache keys (the vision merge): two images have the same pad token IDs, so a slot resumes only when the
  // request's cache key repeats the held tail's key. Synthetic keys: the seed stands in for the image identity.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    std::vector<std::string> log;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore { log.push_back("restore"); return 0; };
    hooks.capture = [](const crucible::ServeRequest&, size_t) {};
    hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                     crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
      return crucible::BatchResident{r.ids.size() + generated.size() - 1, 0, false};
    };
    hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
      log.push_back("resume");
      return {resident.length, false, resident.replay, true, true};
    };
    hooks.release = [&](int, bool resumed) { log.push_back("release resumed=" + std::to_string(resumed)); };
    hooks.key = [](const crucible::ServeRequest& r, std::span<const int> tokens) {
      std::vector<int> key(tokens.begin(), tokens.end());
      for (auto& k : key) k += int(r.sampling.seed) << 20;
      return key;
    };
    session.SetCacheHooks(hooks);
    auto a = Request(1, 3, 0, 1); a.ids = {1, 2, 3, 4, 5, 6};
    const auto first = Complete(session, 1, a);
    auto other = a; other.sampling.seed = 2;  // same tokens, another image
    other.ids.insert(other.ids.end(), first.begin(), first.end());
    other.ids.push_back(7);
    const auto second = Complete(session, 2, other);  // no slot resume: the key differs
    auto next = other;  // repeats `other`'s key and tail: resumes
    next.ids.insert(next.ids.end(), second.begin(), second.end());
    next.ids.push_back(8);
    Complete(session, 3, next);
    assert((log == std::vector<std::string>{"restore", "release resumed=0", "restore", "release resumed=1", "resume"}));
  }

  // Cache visibility (Oct 5 incident: a full cache refused every large frontier and no log showed it). The done
  // event names this request's largest saved frontier, its saved tail, its last refusal, and the cache totals after
  // the tail save.
  {
    MockEngine seen_engine; seen_engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> seen(seen_engine, 1);
    crucible::BatchCacheStatus status;
    crucible::BatchCacheHooks seen_hooks;
    seen_hooks.capture = [&](const crucible::ServeRequest&, size_t cursor) {
      ++status.attempts;
      status.last = {cursor, cursor == 4, cursor == 4 ? "" : "host"};
      status.entries += cursor == 4;
    };
    seen_hooks.tail = [&](const crucible::ServeRequest&, std::span<const int>, int, crucible::BatchTailSource)
        -> std::optional<crucible::BatchResident> {
      ++status.attempts;
      status.last = {11, true, ""};
      ++status.entries; ++status.evictions; status.bytes = 77; status.host_available = uint64_t(3) << 30;
      status.pinned_bytes = uint64_t(5) << 20;  // frontier-copy: the pool's pinned chunks (soak memory rule)
      return std::nullopt;
    };
    seen_hooks.status = [&] { return status; };
    seen.SetCacheHooks(seen_hooks);
    auto turn = Request(1, 2); turn.ids.resize(10); std::iota(turn.ids.begin(), turn.ids.end(), 1);
    auto events = seen.Open(1, turn);
    while (!events.back().done) events = seen.Step({1});
    const auto& done = events.back();
    assert(done.cache_saved_tokens == 4 && done.cache_save_refused == "host" && done.cache_tail_tokens == 11);
    assert(done.cache_entries == 2 && done.cache_evictions == 1 && done.cache_bytes == 77);
    assert(done.host_available == uint64_t(3) << 30 && done.slot_context == 64 && done.cache_reported);
    assert(done.cache_pinned_bytes == uint64_t(5) << 20 && !done.frontier_copy);
    assert(crucible::BatchCacheFields(done) == ",\"cache_saved_tokens\":4,\"cache_tail_tokens\":11,\"cache_entries\":2,"
           "\"cache_evictions\":1,\"cache_save_refused\":\"host\",\"host_free_mib\":3072,\"cache_pinned_bytes\":5242880");
    auto copied = done;
    copied.frontier_copy = true;
    assert(crucible::BatchCacheFields(copied).ends_with(",\"cache_pinned_bytes\":5242880,\"frontier_copy\":true"));
    assert(crucible::BatchCacheFields(events.front()).empty());
    // A request without speculation prints no draft counts: its done line stays byte-identical.
    assert(!done.draft && crucible::BatchDraftFields(done).empty());
  }

  // An earlier change: the open asks the slot tier first, and the restore hook counts the host and disk tiers. The done event
  // carries the counters of the three tiers, and a miss without a diagnosis says "unknown" rather than "none".
  {
    MockEngine tier_engine; tier_engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> tiered(tier_engine, 1);
    std::vector<size_t> slots;
    crucible::BatchCacheCounters counters;
    crucible::BatchCacheHooks tier_hooks;
    tier_hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore {
      counters.Count(crucible::BatchCacheTier::Host, 0);  // a host lookup that served nothing
      return {};
    };
    tier_hooks.slot_served = [&](size_t tokens) { counters.Count(crucible::BatchCacheTier::Slot, tokens); };
    tier_hooks.status = [&] {
      crucible::BatchCacheStatus status;
      status.counters = counters;
      return status;
    };
    tiered.SetCacheHooks(tier_hooks);
    auto events = tiered.Open(1, Request(1, 2));
    while (!events.back().done) events = tiered.Step({1});
    const auto& done = events.back();
    assert(done.counters_reported);
    assert(done.counters.slot.lookups == 1 && done.counters.slot.hits == 0 && done.counters.slot.tokens == 0);
    assert(done.counters.host.lookups == 1 && done.counters.host.hits == 0);
    assert(done.counters.disk.lookups == 0);
    assert(done.cache_miss && std::string_view(done.cache_miss) == "unknown");
  }

  // Session replay (G15): a tail without pending GDN rows still starts with one small-row decode-path Forward, as
  // serial RunPrompt does. The pending count depends on the spec and packing schedule, not on the history.
  {
    MockEngine tail_engine; tail_engine.prefill_rows = 16;
    crucible::BatchSession<MockEngine> tailed(tail_engine, 1);
    std::vector<int> hist(21); std::iota(hist.begin(), hist.end(), 1);
    crucible::BatchCacheHooks tail_hooks;
    tail_hooks.restore = [&](const crucible::ServeRequest&) -> crucible::BatchRestore {
      tail_engine.histories[tail_engine.selected] = hist;
      return {hist.size(), false, 0, true};
    };
    tail_hooks.capture = [](const crucible::ServeRequest&, size_t) { assert(false); };
    tailed.SetCacheHooks(tail_hooks);
    auto next = Request(1, 2); next.ids.resize(60); std::iota(next.ids.begin(), next.ids.end(), 1);
    auto events = tailed.Open(1, next, true);
    events = tailed.Prefill(1);
    assert(tailed.PrefillCursor() == 29);  // the 8-row head chunk after a 21-token tail
    while (events.empty() || !events.back().token) events = tailed.Prefill(1);
    assert(tailed.PrefillCursor() == 60);
    assert(tail_engine.chunks == (std::vector<int>{8, -16, -15}));
    assert(tail_engine.histories[tail_engine.selected] == next.ids);
    tailed.Close(1);
  }

  // Speculative requests: hooks see every chunk, the first token, and each step; publication trims the window.
  MockEngine spec_engine; spec_engine.prefill_rows = 4;
  crucible::BatchSession<MockEngine> speculative(spec_engine, 2);
  auto spec_request = Request(3, 6); spec_request.spec = 7; spec_request.ids = {1, 2, 3, 4, 5, 6};
  try { speculative.Open(70, spec_request, true); assert(false); } catch (const std::invalid_argument&) {}
  std::vector<std::string> hook_calls;
  crucible::BatchSpecHooks spec_hooks;
  spec_hooks.check = [](const crucible::ServeRequest& r) {
    return r.sampling.temperature > 0 ? std::string("sampled speculation is not supported") : std::string();
  };
  spec_hooks.open = [&](uint64_t id, int slot, const crucible::ServeRequest&, size_t cached) {
    hook_calls.push_back("open " + std::to_string(id) + " " + std::to_string(slot) + " " + std::to_string(cached));
  };
  spec_hooks.chunk = [&](uint64_t, size_t start, int rows, bool bulk) {
    hook_calls.push_back("chunk " + std::to_string(start) + " " + std::to_string(rows) + (bulk ? " bulk" : ""));
  };
  spec_hooks.first = [&](uint64_t, int token) { hook_calls.push_back("first " + std::to_string(token)); };
  spec_hooks.step = [&](uint64_t, int last, size_t, uint64_t) {
    hook_calls.push_back("step " + std::to_string(last));
    return std::vector<int>{last + 1, last + 2, last + 3};
  };
  spec_hooks.close = [&](uint64_t id) { hook_calls.push_back("close " + std::to_string(id)); };
  speculative.SetSpecHooks(spec_hooks);
  // Without a stacks hook, a speculative request keeps every final trunk stack row.
  speculative.Open(69, spec_request, true);
  while (speculative.Prefill(69).empty()) {}
  speculative.Close(69);
  assert((spec_engine.keeps == std::vector<int>{0, 0}));
  spec_engine.keeps.clear(); hook_calls.clear();
  spec_hooks.stacks = [](uint64_t, size_t start, int rows) { return start == 0 ? rows : 1; };
  // The decoder's per-request draft counters, read before its close erases them.
  spec_hooks.stats = [&](uint64_t id) {
    hook_calls.push_back("stats " + std::to_string(id));
    return crucible::BatchDraftStats{9, 4, 2};
  };
  // The live counters after each step (crucible_decode): a fake request that proposes 3 and accepts 2 per step.
  long counted_steps = 0;
  spec_hooks.counts = [&](uint64_t) {
    ++counted_steps;
    return crucible::BatchDraftCounts{3 * counted_steps, 2 * counted_steps, counted_steps};
  };
  speculative.SetSpecHooks(spec_hooks);
  auto sampled_spec = spec_request; sampled_spec.sampling.temperature = .5;
  try { speculative.Open(71, sampled_spec, true); assert(false); } catch (const std::invalid_argument&) {}
  assert(!speculative.Poisoned());
  speculative.Open(72, spec_request, true);
  assert(speculative.Prefill(72).empty());
  const auto first = speculative.Prefill(72);
  assert(first.size() == 1 && *first[0].token == 7);
  // The first token is out before the final chunk's MTP warm-up and its End checkpoint. SettlePrompt (the
  // serve loop calls it after each command) or the next command runs them, in the old order.
  assert(hook_calls.back() == "chunk 0 4 bulk");
  speculative.SettlePrompt();
  assert(hook_calls.back() == "first 7");
  speculative.SettlePrompt();
  assert(hook_calls.back() == "first 7");
  try { speculative.Step({72}); assert(false); } catch (const std::invalid_argument&) {}
  // The first token never carries step counters: it comes from the prompt, not from a step.
  assert(!first[0].step_draft && crucible::BatchStepDraftFields(first[0]).empty());
  auto window = speculative.SpecStep(72);
  assert(window.size() == 3 && *window[2].token == 10 && !window[2].done);
  // Only the step's first token event carries the counts after the step; the others keep their bytes.
  assert(window[0].step_draft && window[0].step_draft->proposed == 3 && window[0].step_draft->steps == 1);
  assert(crucible::BatchStepDraftFields(window[0]) == ",\"draft_proposed\":3,\"draft_accepted\":2,\"spec_steps\":1");
  assert(!window[1].step_draft && !window[2].step_draft && crucible::BatchStepDraftFields(window[1]).empty());
  window = speculative.SpecStep(72);  // six tokens requested: the window is cut after two
  assert(window.size() == 3 && *window[1].token == 12 && window[2].done && window[2].completion_tokens == 6);
  // A step that ends the request still reports its counts: they are read before the close erases them.
  assert(window[0].step_draft && window[0].step_draft->steps == 2 && !window[2].step_draft);
  assert(crucible::BatchStepDraftFields(window[2]).empty());  // a done event has no step counters
  assert(window[2].draft && window[2].draft->proposed == 9 && window[2].draft->accepted == 4 && window[2].draft->steps == 2);
  assert(crucible::BatchDraftFields(window[2]) == ",\"draft_proposed\":9,\"draft_accepted\":4,\"spec_steps\":2");
  assert(crucible::BatchDraftFields(window[1]).empty());
  {  // The position and token counters follow the three scalars when the decoder reports them.
    crucible::BatchEvent counted = window[2];
    counted.draft->tally.emplace();
    counted.draft->tally->Round(std::vector<int>{1, 2}, 1, 3, false);
    counted.draft->tally->Pass(0.5);
    const std::string fields = crucible::BatchDraftFields(counted);
    assert(fields.rfind(",\"draft_proposed\":9,\"draft_accepted\":4,\"spec_steps\":2,\"draft_pos_proposed\":[1,1,0", 0) == 0);
    assert(fields.find("\"draft_ms\":0.500") != std::string::npos);
  }
  assert((hook_calls == std::vector<std::string>{"open 72 0 0", "chunk 0 4 bulk", "chunk 4 2 bulk", "first 7",
                                            "step 7", "step 10", "stats 72", "close 72"}));
  assert((spec_engine.keeps == std::vector<int>{4, 1}));
  try { speculative.SpecStep(72); assert(false); } catch (const std::invalid_argument&) {}
  // A single requested token never speculates, as in serial generation.
  auto one = spec_request; one.max_tokens = 1;
  hook_calls.clear();
  const auto single = speculative.Open(73, one);
  assert(single.back().done && hook_calls.empty());
  speculative.Open(74, spec_request, true);
  speculative.Close(74);
  assert(hook_calls.back() == "close 74");
  // Without SettlePrompt the next command settles first.
  hook_calls.clear();
  speculative.Open(75, spec_request, true);
  while (speculative.Prefill(75).empty()) {}
  speculative.SpecStep(75);
  speculative.Close(75);
  assert((hook_calls == std::vector<std::string>{"open 75 0 0", "chunk 0 4 bulk", "chunk 4 2 bulk", "first 7",
                                            "step 7", "close 75"}));
  // A failed prompt end: the serve loop gets its text for an error event, and the session is poisoned.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    crucible::BatchSpecHooks hooks = spec_hooks;
    hooks.chunk = [](uint64_t, size_t start, int, bool) { if (start > 0) throw std::runtime_error("warm-up fault"); };
    session.SetSpecHooks(hooks);
    session.Open(1, spec_request, true);
    while (session.Prefill(1).empty()) {}
    assert(!session.Poisoned());
    const auto failed = session.TrySettlePrompt();
    assert(failed && *failed == "warm-up fault" && session.Poisoned());
    assert(!session.TrySettlePrompt());
  }
  // A first token that ends its request keeps the old order: its events follow the warm-up.
  auto stops = spec_request; stops.stop_ids = {7};
  hook_calls.clear();
  speculative.Open(76, stops, true);
  while (speculative.Prefill(76).empty()) {}
  assert((hook_calls == std::vector<std::string>{"open 76 0 0", "chunk 0 4 bulk", "chunk 4 2 bulk", "first 7",
                                            "stats 76", "close 76"}));

  // Load-adaptive decode: a speculative request joins a packed step. A pending replay commits with one solo row
  // first; otherwise the request takes a row of the pass, and the draft hook sees that row with its token.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 2);
    std::vector<std::string> calls;
    bool pending_replay = false;
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
    hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    hooks.chunk = [](uint64_t, size_t, int, bool) {};
    hooks.first = [](uint64_t, int) {};
    hooks.step = [&](uint64_t, int last, size_t, uint64_t) {
      pending_replay = true;
      return std::vector<int>{(last + 1) % 16, (last + 2) % 16};
    };
    hooks.close = [&](uint64_t id) { calls.push_back("close " + std::to_string(id)); };
    hooks.commit = [&](uint64_t id, int last) -> std::optional<int> {
      calls.push_back("commit " + std::to_string(id) + " " + std::to_string(last) + " slot " + std::to_string(engine.selected));
      if (!pending_replay) return std::nullopt;
      pending_replay = false;
      return (last + 1) % 16;
    };
    hooks.packed = [&](const std::vector<uint64_t>& ids, const std::vector<int>& rows, const std::vector<int>& tokens) {
      for (size_t i = 0; i < ids.size(); ++i)
        calls.push_back("packed " + std::to_string(ids[i]) + " row " + std::to_string(rows[i]) + " " + std::to_string(tokens[i]));
    };
    session.SetSpecHooks(hooks);
    auto plain = Request(9, 8);
    auto spec = Request(3, 8); spec.spec = 7; spec.ids = {1, 2, 3};
    const auto plain_first = session.Open(91, plain);
    const auto spec_first = session.Open(92, spec);
    assert(*plain_first[0].token == 10 && *spec_first[0].token == 4);
    assert(session.SpecStep(92).size() == 2);  // 5, 6: the verify leaves a replay pending
    auto events = session.Step({92, 91});
    // The commit row publishes first; the packed pass then holds only the plain request.
    assert(events.size() == 2 && events[0].request_id == 92 && *events[0].token == 7 && *events[1].token == 11);
    events = session.Step({91, 92});
    assert(events.size() == 2 && *events[0].token == 12 && *events[1].token == 8);
    assert((calls == std::vector<std::string>{"commit 92 6 slot 1", "packed 92 row 0 7", "commit 92 7 slot 1",
                                              "packed 92 row 1 8"}));
    assert(session.SpecStep(92).size() == 2);  // switching back to solo steps keeps the same request
    session.Close(92);
    assert(calls.back() == "close 92" && !session.Poisoned());
  }

  // Row caps for packed verification: one row each, spare rows shared by speculative requests, rotating remainder.
  assert((crucible::DraftCaps({true, true, true, true}, 8, 0) == std::vector<int>{1, 1, 1, 1}));
  assert((crucible::DraftCaps({true, false, true}, 8, 0) == std::vector<int>{3, 0, 2}));
  assert((crucible::DraftCaps({true, false, true}, 8, 1) == std::vector<int>{2, 0, 3}));
  assert((crucible::DraftCaps({true, true, true, true, true, true}, 8, 2) == std::vector<int>{0, 0, 1, 1, 0, 0}));
  assert((crucible::DraftCaps({false, false}, 8, 0) == std::vector<int>{0, 0}));
  assert((crucible::DraftCaps(std::vector<bool>(8, true), 8, 5) == std::vector<int>(8, 0)));
  // Copy rows first: no want keeps DraftCaps; a want takes up to the spare rows; the rest share what is left.
  assert((crucible::DraftCapsFor({true, true, true, true}, {0, 0, 0, 0}, 8, 0) == std::vector<int>{1, 1, 1, 1}));
  assert((crucible::DraftCapsFor({true, true, true, true}, {0, 7, 0, 0}, 8, 0) == std::vector<int>{0, 4, 0, 0}));
  assert((crucible::DraftCapsFor({true, true, true, true}, {0, 2, 0, 0}, 8, 0) == std::vector<int>{1, 2, 1, 0}));
  assert((crucible::DraftCapsFor({true, true, true, true}, {0, 2, 0, 0}, 8, 1) == std::vector<int>{0, 2, 1, 1}));
  assert((crucible::DraftCapsFor({true, true, false}, {3, 5, 0}, 8, 0) == std::vector<int>{0, 5, 0}));
  assert((crucible::DraftCapsFor({true, false, true}, {0, 4, 0}, 8, 0) == crucible::DraftCaps({true, false, true}, 8, 0)));
  assert((crucible::DraftCapsFor(std::vector<bool>(8, true), std::vector<int>(8, 7), 8, 0) == std::vector<int>(8, 0)));
  // 16-row passes (CRUCIBLE_PACKED_ROWS=16): the same shares, but one span holds at most kMaxRows rows, so no cap
  // exceeds kMaxRows - 1 drafts.
  assert((crucible::DraftCaps({true, true, true, true}, 16, 0) == std::vector<int>{3, 3, 3, 3}));
  assert((crucible::DraftCaps({true, true, true}, 16, 1) == std::vector<int>{4, 5, 4}));
  assert((crucible::DraftCaps({true}, 16, 0) == std::vector<int>{7}));
  assert((crucible::DraftCaps({true, true}, 16, 0) == std::vector<int>{7, 7}));
  assert((crucible::DraftCapsFor({true, true, true, true}, {0, 15, 0, 0}, 16, 0) == std::vector<int>{2, 7, 2, 1}));
  assert((crucible::DraftCapsFor({true}, {15}, 16, 0) == std::vector<int>{7}));
  for (int rows = 1; rows <= crucible::kPackedRows; ++rows)
    for (int n = 1; n <= crucible::kMaxRows; ++n)
      for (uint64_t turn = 0; turn < 3; ++turn) {
        const auto caps = crucible::DraftCapsFor(std::vector<bool>(size_t(n), true), std::vector<int>(size_t(n), 0),
                                                 rows, turn);
        int total = n;
        for (const int cap : caps) { assert(cap >= 0 && cap < crucible::kMaxRows); total += cap; }
        assert(total <= std::max(rows, n));
      }
  // Pass rows (CRUCIBLE_PACKED_ROWS): unset or empty is automatic (16 rows from 4 decoders, else 8); a value
  // 8..16 is a fixed cap, as before; 9..16 only when the engine can run such passes.
  using Setting = crucible::PackedRowsSetting;
  assert((crucible::PackedRowsOption(nullptr) == Setting{crucible::kPackedRows, crucible::kWideDecoders}));
  assert((crucible::PackedRowsOption("") == Setting{crucible::kPackedRows, crucible::kWideDecoders}));
  assert((crucible::PackedRowsOption("8") == Setting{8, 0}) && (crucible::PackedRowsOption("16") == Setting{16, 0}) &&
         (crucible::PackedRowsOption("12") == Setting{12, 0}));
  assert(crucible::kWideDecoders == 4);
  // Serve's stderr stays byte-identical at the default (automatic) and at 8 rows; a fixed wider cap prints it.
  assert(crucible::PackedRowsNotice(crucible::PackedRowsOption(nullptr)).empty());
  assert(crucible::PackedRowsNotice(Setting{8, 0}).empty());
  assert(crucible::PackedRowsNotice(Setting{16, 0}) == "serve: packed verify passes of up to 16 rows\n");
  for (const char* bad : {"7", "17", "0", "-16", "16x", "x", " 16"})
    try { crucible::PackedRowsOption(bad); assert(false); } catch (const std::invalid_argument&) {}

  // Packed verification: speculative requests propose rows within their caps, plain requests take one row,
  // each request accepts its own rows of the shared pass, and deferred draft work runs once before publication.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 3);
    std::vector<std::string> calls;
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
    hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    hooks.chunk = [](uint64_t, size_t, int, bool) {};
    hooks.first = [](uint64_t, int) {};
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
    hooks.close = [](uint64_t) {};
    hooks.propose = [&](uint64_t id, int last, size_t remaining, uint64_t position, int cap) {
      calls.push_back("propose " + std::to_string(id) + " cap " + std::to_string(cap) + " remaining " +
                      std::to_string(remaining) + " position " + std::to_string(position));
      // Drafts that the mock target accepts: each row predicts token + 1.
      std::vector<int> rows{last};
      for (int k = 0; k < cap; ++k) rows.push_back((rows.back() + 1) % 16);
      return crucible::VerifySpan{-1, rows, 0, 1};
    };
    hooks.accept = [&](uint64_t id, int first, std::span<const int> greedy) {
      calls.push_back("accept " + std::to_string(id) + " first " + std::to_string(first) + " rows " + std::to_string(greedy.size()));
      return std::vector<int>(greedy.begin(), greedy.end());
    };
    hooks.settle = [&] { calls.push_back("settle"); };
    hooks.counts = [](uint64_t id) { return crucible::BatchDraftCounts{long(id), 1, 1}; };
    session.SetSpecHooks(hooks);
    auto a = Request(3, 6); a.spec = 7;
    auto b = Request(9, 6); b.spec = 7;
    const auto plain = Request(12, 6);
    assert(*session.Open(101, a)[0].token == 4 && *session.Open(102, plain)[0].token == 13 &&
           *session.Open(103, b)[0].token == 10);
    const auto events = session.VerifyStep({101, 102, 103});
    // Spare rows: 8 - 3 = 5 for two speculative requests, so caps 3 and 2 at turn 0.
    assert((calls == std::vector<std::string>{"propose 101 cap 3 remaining 5 position 2",
                                              "propose 103 cap 2 remaining 5 position 2",
                                              "accept 101 first 0 rows 4", "accept 103 first 5 rows 3", "settle"}));
    std::vector<int> first_tokens, plain_tokens;
    for (const auto& event : events) {
      if (event.request_id == 101 && event.token) first_tokens.push_back(*event.token);
      if (event.request_id == 102 && event.token) plain_tokens.push_back(*event.token);
    }
    assert((first_tokens == std::vector<int>{5, 6, 7, 8}) && (plain_tokens == std::vector<int>{14}));
    // Step counters: on the first token event of each speculative request in the pass, none for the plain request.
    std::map<uint64_t, std::vector<long>> counted;
    for (const auto& event : events)
      if (event.token) counted[event.request_id].push_back(event.step_draft ? event.step_draft->proposed : -1);
    assert((counted[101] == std::vector<long>{101, -1, -1, -1}) && (counted[102] == std::vector<long>{-1}) &&
           (counted[103] == std::vector<long>{103, -1, -1}));
    // The next pass trims publication at max_tokens: request 101 has one token left.
    calls.clear();
    const auto last = session.VerifyStep({101, 102, 103});
    assert(calls[0] == "propose 101 cap 2 remaining 1 position 6");
    bool done = false;
    for (const auto& event : last) done |= event.request_id == 101 && event.done && event.completion_tokens == 6;
    assert(done && !session.Poisoned());
    // A proposal beyond its cap poisons the worker, because drafting may already have changed state.
    hooks.propose = [](uint64_t, int last, size_t, uint64_t, int cap) {
      return crucible::VerifySpan{-1, std::vector<int>(size_t(cap) + 2, last), 0, 1};
    };
    session.SetSpecHooks(hooks);
    for (const uint64_t id : {102, 103}) if (session.Active() && !session.Poisoned()) try { session.Close(id); } catch (const std::invalid_argument&) {}
    auto c = Request(5, 20); c.spec = 7;
    session.Open(104, c);
    try { session.VerifyStep({104}); assert(false); }
    catch (const std::invalid_argument&) { assert(false); }
    catch (const std::logic_error&) {}
    assert(session.Poisoned());
  }

  // 16-row packed passes (CRUCIBLE_PACKED_ROWS=16): four speculative requests take four rows each in one pass, and
  // every request publishes the same tokens as with 8-row passes; only the pass count changes. An engine that runs
  // only 8-row passes (Engine::MaxPassRows is kMaxRows) refuses the setting, as does a value out of range.
  {
    const auto run = [](int pass_rows) {
      MockEngine engine; engine.prefill_rows = 4;
      crucible::BatchSession<MockEngine> session(engine, 4);
      crucible::BatchSpecHooks hooks;
      std::vector<int> caps;
      hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
      hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
      hooks.chunk = [](uint64_t, size_t, int, bool) {};
      hooks.first = [](uint64_t, int) {};
      hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
      hooks.close = [](uint64_t) {};
      hooks.propose = [&](uint64_t, int last, size_t, uint64_t, int cap) {
        caps.push_back(cap);
        std::vector<int> rows{last};
        for (int k = 0; k < cap; ++k) rows.push_back((rows.back() + 1) % 16);
        return crucible::VerifySpan{-1, rows, 0, 1};
      };
      hooks.accept = [](uint64_t, int, std::span<const int> greedy) { return std::vector<int>(greedy.begin(), greedy.end()); };
      hooks.settle = [] {};
      session.SetSpecHooks(hooks);
      if (pass_rows != crucible::kMaxRows) session.SetPassRows(pass_rows);
      assert(session.PassRows() == pass_rows);
      std::map<uint64_t, std::vector<int>> tokens;
      std::vector<uint64_t> ids;
      for (uint64_t id = 1; id <= 4; ++id) {
        auto r = Request(int(id) * 3, 25); r.spec = 7;
        for (const auto& event : session.Open(id, r)) if (event.token) tokens[id].push_back(*event.token);
        ids.push_back(id);
      }
      int passes = 0;
      while (!ids.empty()) {
        for (const auto& event : session.VerifyStep(ids)) if (event.token) tokens[event.request_id].push_back(*event.token);
        ++passes;
        std::erase_if(ids, [&](uint64_t id) { return tokens[id].size() >= 25; });
      }
      return std::make_tuple(tokens, passes, engine.pass_widths, caps);
    };
    const auto [eight, eight_passes, eight_widths, eight_caps] = run(8);
    const auto [sixteen, sixteen_passes, sixteen_widths, sixteen_caps] = run(16);
    assert(eight == sixteen);
    for (const auto& [id, list] : eight) assert(list.size() == 25);
    assert(eight_widths.front() == 8 && sixteen_widths.front() == 16);
    assert(std::vector<int>(sixteen_caps.begin(), sixteen_caps.begin() + 4) == std::vector<int>(4, 3));
    assert(sixteen_passes < eight_passes);
    for (const int width : sixteen_widths) assert(width <= crucible::kPackedRows);
    MockEngine small; small.max_pass_rows = crucible::kMaxRows;
    crucible::BatchSession<MockEngine> session(small, 4);
    for (const int rows : {16, 9, 7, 17})
      try { session.SetPassRows(rows); assert(false); } catch (const std::invalid_argument&) {}
    session.SetPassRows(8);
    assert(session.PassRows() == 8);
    // Automatic rows fall back to 8 on such an engine, and on a session with fewer slots than kWideDecoders
    // (there it does not ask the engine, whose first MaxPassRows call builds the 16-row plans).
    session.SetPassRows(crucible::kPackedRows, crucible::kWideDecoders);
    assert(session.PassRows() == 8 && session.WideFrom() == 0);
    MockEngine wide;
    crucible::BatchSession<MockEngine> three(wide, 3);
    three.SetPassRows(crucible::kPackedRows, crucible::kWideDecoders);
    assert(three.PassRows() == 8 && three.WideFrom() == 0 && wide.max_pass_rows_calls == 0);
    crucible::BatchSession<MockEngine> four(wide, 4);
    four.SetPassRows(crucible::kPackedRows, crucible::kWideDecoders);
    assert(four.PassRows() == 16 && four.WideFrom() == 4);
    try { four.SetPassRows(17, crucible::kWideDecoders); assert(false); } catch (const std::invalid_argument&) {}
  }

  // Automatic pass rows: the row cap follows the decoder count of each pass (16 rows from 4 decoders, 8 below),
  // so it changes in the middle of a request when a request joins or ends. Every request publishes the same
  // tokens as with fixed 8-row and fixed 16-row passes.
  {
    const auto run = [](int pass_rows, int wide_from) {
      MockEngine engine; engine.prefill_rows = 4;
      crucible::BatchSession<MockEngine> session(engine, 4);
      crucible::BatchSpecHooks hooks;
      hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
      hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
      hooks.chunk = [](uint64_t, size_t, int, bool) {};
      hooks.first = [](uint64_t, int) {};
      hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
      hooks.close = [](uint64_t) {};
      hooks.propose = [](uint64_t, int last, size_t, uint64_t, int cap) {
        std::vector<int> rows{last};
        for (int k = 0; k < cap; ++k) rows.push_back((rows.back() + 1) % 16);
        return crucible::VerifySpan{-1, rows, 0, 1};
      };
      hooks.accept = [](uint64_t, int, std::span<const int> greedy) { return std::vector<int>(greedy.begin(), greedy.end()); };
      hooks.settle = [] {};
      session.SetSpecHooks(hooks);
      session.SetPassRows(pass_rows, wide_from);
      std::map<uint64_t, std::vector<int>> tokens;
      std::vector<uint64_t> ids;
      const auto open = [&](uint64_t id, int n) {
        auto r = Request(int(id) * 3, n); r.spec = 7;
        for (const auto& event : session.Open(id, r)) if (event.token) tokens[id].push_back(*event.token);
        ids.push_back(id);
      };
      const std::map<uint64_t, int> wanted{{1, 40}, {2, 40}, {3, 40}, {4, 14}};
      for (uint64_t id = 1; id <= 3; ++id) open(id, wanted.at(id));
      std::vector<std::pair<size_t, int>> passes;  // (decoders, rows) of each pass
      for (int pass = 0; !ids.empty(); ++pass) {
        if (pass == 2) open(4, wanted.at(4));  // 3 -> 4 decoders; request 4 ends early: 4 -> 3
        const size_t decoders = ids.size();
        const size_t before = engine.pass_widths.size();
        for (const auto& event : session.VerifyStep(ids)) if (event.token) tokens[event.request_id].push_back(*event.token);
        assert(engine.pass_widths.size() == before + 1);
        passes.emplace_back(decoders, engine.pass_widths.back());
        std::erase_if(ids, [&](uint64_t id) { return tokens[id].size() >= size_t(wanted.at(id)); });
      }
      return std::make_pair(tokens, passes);
    };
    const auto [fixed8, passes8] = run(8, 0);
    const auto [fixed16, passes16] = run(16, 0);
    const auto [automatic, passes_auto] = run(crucible::kPackedRows, crucible::kWideDecoders);
    assert(fixed8 == fixed16 && fixed8 == automatic);
    bool wide = false, narrow_after = false;
    for (const auto& [decoders, rows] : passes_auto) {
      assert(rows <= (decoders >= 4 ? 16 : 8));
      if (decoders >= 4) wide |= rows > 8;
      else if (wide) narrow_after = true;
    }
    assert(passes_auto.front().first == 3 && wide && narrow_after);
  }

  // Packed verification audit (G6d): stop IDs and token limits at every proposal position, and cancellation
  // between passes with slot reuse.
  {
    const auto verify_hooks = [](std::vector<std::string>& calls) {
      crucible::BatchSpecHooks hooks;
      hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
      hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
      hooks.chunk = [](uint64_t, size_t, int, bool) {};
      hooks.first = [](uint64_t, int) {};
      hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
      hooks.close = [&calls](uint64_t id) { calls.push_back("close " + std::to_string(id)); };
      hooks.propose = [](uint64_t, int last, size_t, uint64_t, int cap) {
        std::vector<int> rows{last};
        for (int k = 0; k < cap; ++k) rows.push_back((rows.back() + 1) % 16);
        return crucible::VerifySpan{-1, rows, 0, 1};
      };
      hooks.accept = [](uint64_t, int, std::span<const int> greedy) { return std::vector<int>(greedy.begin(), greedy.end()); };
      hooks.settle = [] {};
      return hooks;
    };
    // A lone speculative request gets 7 drafts, so it proposes 8 rows: tokens 4..11 after last = 3.
    for (int position = 0; position < 8; ++position) {
      for (const bool by_stop : {true, false}) {
        MockEngine engine; engine.prefill_rows = 4;
        crucible::BatchSession<MockEngine> session(engine, 2);
        std::vector<std::string> calls;
        session.SetSpecHooks(verify_hooks(calls));
        auto r = Request(2, by_stop ? 30 : 2 + position); r.spec = 7;
        if (by_stop) r.stop_ids = {4 + position};
        assert(*session.Open(1, r)[0].token == 3);
        const auto events = session.VerifyStep({1});
        std::vector<int> tokens;
        bool done = false, stopped = false;
        for (const auto& event : events) {
          assert(!done);  // nothing follows the terminal event
          if (event.token) tokens.push_back(*event.token);
          if (event.done) { done = true; stopped = event.stopped; }
        }
        assert(done && stopped == by_stop && tokens.size() == size_t(position + 1) && tokens.back() == 4 + position);
        assert(calls == std::vector<std::string>{"close 1"} && session.Active() == 0);
      }
    }
    // Cancellation between passes: the closed request leaves the next pass; its slot takes a new generation.
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 3);
    std::vector<std::string> calls;
    session.SetSpecHooks(verify_hooks(calls));
    auto a = Request(1, 40); a.spec = 7;
    auto b = Request(5, 40); b.spec = 7;
    const auto opened_b = session.Open(2, b);
    session.Open(1, a);
    session.Open(3, Request(9, 40));
    session.VerifyStep({1, 2, 3});
    session.Close(2);
    assert(calls == std::vector<std::string>{"close 2"});
    try { session.VerifyStep({1, 2, 3}); assert(false); } catch (const std::invalid_argument&) {}
    assert(!session.Poisoned());
    const auto events = session.VerifyStep({3, 1});
    assert(!events.empty() && events.front().request_id == 3 && !session.Poisoned());
    const auto reopened = session.Open(4, b);
    assert(reopened.front().slot_generation != opened_b.front().slot_generation);
    assert(!session.VerifyStep({1, 4, 3}).empty() && session.Active() == 3);
  }

  // A sampled first token uses the prompt logits, even though the MTP hook later reuses the logits buffer.
  const auto sampled_first = [](bool clobber) {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 1);
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
    hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    hooks.chunk = [&engine, clobber](uint64_t, size_t, int, bool) { if (clobber) engine.last = {15}; };
    hooks.first = [](uint64_t, int) {};
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
    hooks.close = [](uint64_t) {};
    session.SetSpecHooks(hooks);
    auto request = Request(3, 4, .7, 99); request.spec = 7; request.ids = {1, 2, 3, 4, 5, 6};
    session.Open(80, request, true);
    std::vector<crucible::BatchEvent> events;
    while (events.empty()) events = session.Prefill(80);
    return *events.front().token;
  };
  assert(sampled_first(true) == sampled_first(false));

  // Grouped proposals (propose_all) and the want hook in VerifyStep: plain requests keep one row and stay out of
  // the group; caps follow DraftCapsFor with the wants; a wrong count or an oversized span poisons the worker.
  {
    enum class Mode { good, short_count, oversized };
    const auto run = [](Mode mode) {
      MockEngine engine; engine.prefill_rows = 4;
      crucible::BatchSession<MockEngine> session(engine, 4);
      std::vector<std::vector<uint64_t>> groups;
      std::vector<std::vector<int>> caps_seen;
      crucible::BatchSpecHooks hooks;
      hooks.check = [](const crucible::ServeRequest&) { return std::string(); };
      hooks.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
      hooks.chunk = [](uint64_t, size_t, int, bool) {};
      hooks.first = [](uint64_t, int) {};
      hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
      hooks.close = [](uint64_t) {};
      hooks.propose = [](uint64_t, int, size_t, uint64_t, int) -> crucible::VerifySpan { assert(false); return {}; };
      hooks.accept = [](uint64_t, int, std::span<const int> greedy) { return std::vector<int>(greedy.begin(), greedy.end()); };
      hooks.settle = [] {};
      hooks.want = [](uint64_t id, int) { return id == 3 ? 2 : 0; };
      hooks.propose_all = [&](const std::vector<uint64_t>& ids, const std::vector<int>& last,
                              const std::vector<size_t>&, const std::vector<uint64_t>&, const std::vector<int>& cap) {
        groups.push_back(ids); caps_seen.push_back(cap);
        std::vector<crucible::VerifySpan> spans;
        for (size_t k = 0; k < ids.size(); ++k) {
          std::vector<int> rows{last[k]};
          const int drafts = mode == Mode::oversized && k == 0 ? cap[k] + 1 : cap[k];
          for (int d = 0; d < drafts; ++d) rows.push_back((rows.back() + 1) % 16);
          spans.push_back(crucible::VerifySpan{-1, rows, 0, 1});
        }
        if (mode == Mode::short_count) spans.pop_back();
        return spans;
      };
      session.SetSpecHooks(hooks);
      auto spec = Request(1, 30); spec.spec = 7;
      session.Open(1, spec);
      session.Open(2, Request(5, 30));
      auto other = Request(9, 30); other.spec = 7;
      session.Open(3, other);
      std::map<uint64_t, int> published;
      bool threw = false;
      try {
        for (const auto& event : session.VerifyStep({1, 2, 3})) if (event.token) ++published[event.request_id];
      } catch (const std::logic_error&) { threw = true; }
      return std::make_tuple(groups, caps_seen, published, threw, session.Poisoned());
    };
    const auto [groups, caps, published, threw, poisoned] = run(Mode::good);
    assert(!threw && !poisoned);
    assert((groups == std::vector<std::vector<uint64_t>>{{1, 3}}));
    // 8 rows, 3 requests: 5 spare; request 3 wants 2 and takes them first, request 1 gets the other 3.
    const auto expected = crucible::DraftCapsFor({true, false, true}, {0, 0, 2}, 8, 0);
    assert((expected == std::vector<int>{3, 0, 2}) && (caps[0] == std::vector<int>{3, 2}));
    assert(published.at(1) == 4 && published.at(2) == 1 && published.at(3) == 3);
    for (const Mode mode : {Mode::short_count, Mode::oversized}) {
      const auto [g, c, p, failed, bad] = run(mode);
      assert(failed && bad && p.empty());
    }
  }

  // Interleaved prefill (prefill_interleave.h): verify passes between the layers of a bulk chunk.
  {
    static_assert(crucible::InterleavePasses(8, 48) == 6 && crucible::InterleavePasses(1, 48) == 46);
    static_assert(crucible::InterleavePasses(0, 48) == 0 && crucible::InterleavePasses(48, 48) == 1);
    // every 8: after layers 3, 11, 19, 27, 35, 43 (half an interval from each chunk edge).
    static_assert(crucible::InterleaveAfterLayer(3, 8, 48) && crucible::InterleaveAfterLayer(43, 8, 48) &&
                  !crucible::InterleaveAfterLayer(7, 8, 48) && !crucible::InterleaveAfterLayer(47, 8, 48));
    static_assert(!crucible::InterleaveAfterLayer(0, 1, 48) && crucible::InterleaveAfterLayer(2, 2, 48) &&
                  !crucible::InterleaveAfterLayer(1, 2, 48));
    static_assert(!crucible::InterleaveAfterLayer(47, 1, 48) && crucible::InterleaveAfterLayer(46, 1, 48));
    static_assert(!crucible::InterleaveAllowed(8, 8, true) && crucible::InterleaveAllowed(9, 8, true) &&
                  !crucible::InterleaveAllowed(4096, 8, false));
    const auto greedy = Request(2, 40), sampled = Request(5, 40, .8, 3);
    auto big = Request(1, 3); big.ids = std::vector<int>(30, 0);
    std::iota(big.ids.begin(), big.ids.end(), 0);
    for (int& id : big.ids) id %= 16;
    std::map<uint64_t, std::vector<int>> serial;
    {
      MockEngine engine; engine.prefill_rows = 12;
      crucible::BatchSession<MockEngine> session(engine, 3);
      serial[1] = Complete(session, 1, greedy);
      serial[2] = Complete(session, 2, sampled);
      serial[3] = Complete(session, 3, big);
    }
    struct Run { std::map<uint64_t, std::vector<int>> tokens; int parts = 0, between = 0; std::vector<int> chunks; };
    const auto run = [&](int every, size_t limit, int launch = 0) {
      MockEngine engine; engine.prefill_rows = 12; engine.launch_rows = launch;
      crucible::BatchSession<MockEngine> session(engine, 3);
      Run out;
      const auto collect = [&](const std::vector<crucible::BatchEvent>& events) {
        for (const auto& event : events) if (event.token) out.tokens[event.request_id].push_back(*event.token);
      };
      auto g = greedy, s = sampled; g.max_tokens = s.max_tokens = int(limit);
      collect(session.Open(1, g)); collect(session.Open(2, s));
      session.Open(3, big, true);
      const auto part = [&](std::vector<crucible::BatchEvent> events) { ++out.parts; collect(events); };
      // The prefilling request cannot decode itself; a bad interval or an unknown decoder fails before any work.
      const auto calls = engine.calls;
      for (const auto& bad : {std::vector<uint64_t>{3}, std::vector<uint64_t>{99}, std::vector<uint64_t>{1, 1}})
        try { session.Prefill(3, bad, every, part); assert(false); } catch (const std::invalid_argument&) {}
      try { session.Prefill(3, {1}, 0, part); assert(false); } catch (const std::invalid_argument&) {}
      assert(engine.calls == calls && !session.Poisoned());
      // The server names only decoders that are still active, as for a verify step.
      const auto live = [&] {
        std::vector<uint64_t> ids;
        for (const uint64_t id : {1, 2}) if (out.tokens[id].size() < limit) ids.push_back(id);
        return ids;
      };
      std::vector<crucible::BatchEvent> final;
      while (final.empty())
        final = live().empty() ? session.Prefill(3) : session.Prefill(3, live(), every, part);
      collect(final);
      while (!live().empty()) collect(session.VerifyStep(live()));
      out.between = engine.between_runs;
      out.chunks = engine.chunks;
      return out;
    };
    // Chunks of 12, 12 and 6 rows: the two 12-row chunks run 6 passes each at every 8; the 6-row chunk none.
    const auto eight = run(8, 40);
    assert(eight.parts == 12 && eight.between == 12);
    for (const uint64_t id : {1, 2, 3}) assert(eight.tokens.at(id) == std::vector<int>(serial[id].begin(), serial[id].begin() + eight.tokens.at(id).size()));
    assert(eight.tokens.at(1).size() == 40 && eight.tokens.at(2).size() == 40);
    // Decoders that finish in a pass drop out; the chunk's other 21 passes (every 2 layers) run nothing, and the
    // next chunk names no decoders.
    static_assert(crucible::InterleavePasses(2, 48) == 23);
    const auto short_run = run(2, 3);
    assert(short_run.parts == 2 && short_run.between == 23);
    assert(short_run.tokens.at(1) == std::vector<int>(serial[1].begin(), serial[1].begin() + 3));
    assert(short_run.tokens.at(3).size() == 1 && short_run.tokens.at(3)[0] == serial[3][0]);
    // With merged launches allowed (24 rows), chunks that have decoders keep the grid; the outputs equal the
    // serial runs. Once the decoders finish, the rest merges.
    const auto merged = run(8, 40, 24);
    assert(merged.parts == 12 && merged.between == 12);
    assert(std::count(merged.chunks.begin(), merged.chunks.end(), -24) == 0);
    for (const uint64_t id : {1, 2, 3})
      assert(merged.tokens.at(id) == std::vector<int>(serial[id].begin(), serial[id].begin() + merged.tokens.at(id).size()));
    const auto early = run(2, 3, 24);
    assert(early.parts == 2 && std::count(early.chunks.begin(), early.chunks.end(), -18) == 1);
    assert(early.tokens.at(3).size() == 1 && early.tokens.at(3)[0] == serial[3][0]);
  }
  // Train 13 audit: a decoder that finishes inside a pass saves its tail
  // on its own slot and keeps that slot resident; the chunk's prompt capture and its prompt-end checkpoint then run
  // on the prefilling slot. Every hook that reads the active stream must see the slot that it names.
  {
    MockEngine engine; engine.prefill_rows = 12;
    crucible::BatchSession<MockEngine> session(engine, 3);
    std::vector<std::string> log;
    crucible::BatchCacheHooks hooks;
    hooks.restore = [](const crucible::ServeRequest&) { return crucible::BatchRestore{}; };
    hooks.peek = [](const crucible::ServeRequest&) { return size_t(0); };
    hooks.capture = [&](const crucible::ServeRequest& r, size_t cursor) {
      log.push_back("capture n=" + std::to_string(r.ids.size()) + " at " + std::to_string(cursor) + " slot=" +
                    std::to_string(engine.selected) + " len=" + std::to_string(engine.histories[engine.selected].size()));
    };
    hooks.resume = [](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
      return {resident.length, false, resident.replay, true, true, resident.guarded};
    };
    hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                     crucible::BatchTailSource source) -> std::optional<crucible::BatchResident> {
      assert(source.slot == engine.selected);
      const size_t length = r.ids.size() + generated.size() - 1;
      engine.histories[engine.selected].resize(length);
      log.push_back("tail slot=" + std::to_string(engine.selected) + " len=" + std::to_string(length));
      return crucible::BatchResident{length, 0, false};
    };
    hooks.checkpoint_save = [&](int slot, int) {
      log.push_back("save slot=" + std::to_string(slot) + " selected=" + std::to_string(engine.selected) + " len=" +
                    std::to_string(engine.histories[engine.selected].size()));
    };
    hooks.checkpoint_load = [](int, int, const crucible::SlotCheckpoint&) { assert(false); };
    session.SetCacheHooks(hooks);
    auto decoder = Request(2, 3);
    session.Open(1, decoder);  // slot 0: prompt token, first output token
    auto big = Request(1, 2); big.ids.assign(36, 0);
    std::iota(big.ids.begin(), big.ids.end(), 0);
    for (int& id : big.ids) id %= 16;
    session.Open(2, big, true);  // slot 1
    log.clear();  // the decoder's own prompt-end checkpoint
    std::vector<crucible::BatchEvent> parts;
    const auto part = [&](std::vector<crucible::BatchEvent> events) { parts.insert(parts.end(), events.begin(), events.end()); };
    std::vector<crucible::BatchEvent> final;
    while (final.empty()) {
      std::vector<uint64_t> live;
      if (session.Active() == 2) live = {1};
      final = live.empty() ? session.Prefill(2) : session.Prefill(2, live, 8, part);
    }
    // The decoder (max 3 tokens) finished in the second pass of the first chunk.
    assert(std::count_if(parts.begin(), parts.end(), [](const auto& e) { return e.done; }) == 1);
    const std::vector<std::string> expected{"tail slot=0 len=3", "capture n=36 at 12 slot=1 len=12",
                                            "capture n=36 at 24 slot=1 len=24", "save slot=1 selected=1 len=36"};
    session.SettlePrompt();  // the prompt-end checkpoint follows the first token
    assert(log == expected);
    // The finished decoder's slot keeps its tail: a prompt that repeats it resumes there (no restore, no reset).
    auto next = decoder; next.ids = {2, 3, 4, 9};
    const auto opened = session.Open(3, next, true);
    assert(opened.front().cached_tokens == 3 && engine.selected == 0);
  }
  // The disk tier (state_disk.h): Stage asks the cache to start a disk read before the open. Its floor is the
  // longest state that a slot already holds for the prompt (a resident tail or a checkpoint). It writes no slot.
  {
    MockEngine engine; engine.prefill_rows = 4;
    crucible::BatchSession<MockEngine> session(engine, 2);
    std::vector<std::pair<size_t, size_t>> staged;  // prompt length, floor
    assert(session.Stage(Request(1, 3)).state == "none");  // no hook
    crucible::BatchCacheHooks hooks;
    hooks.restore = [](const crucible::ServeRequest&) -> crucible::BatchRestore { return 0; };
    hooks.capture = [](const crucible::ServeRequest&, size_t) {};
    hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
      return {resident.length, false, resident.replay, true, true, resident.guarded};
    };
    hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                     crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
      std::vector<int> hist(r.ids);
      hist.insert(hist.end(), generated.begin(), generated.end() - 1);
      engine.histories[engine.selected] = hist;
      return crucible::BatchResident{hist.size(), 0, false};
    };
    crucible::BatchStageQueue seen;
    std::vector<bool> slot_prefix;  // an earlier change: in_slot of the held tail's first 5 tokens and of a foreign key
    hooks.stage = [&](const crucible::ServeRequest& r, size_t floor, const crucible::BatchStageQueue& queue) {
      staged.emplace_back(r.ids.size(), floor);
      seen = queue;
      const std::vector<int> head = {1, 2, 3, 4, 5}, other = {9, 9};
      slot_prefix = {queue.in_slot(head), queue.in_slot(other)};
      return crucible::BatchStage{"reading", 7};
    };
    session.SetCacheHooks(hooks);
    auto a = Request(1, 3); a.ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    assert(session.Stage(a).state == "reading" && staged.back() == std::make_pair(size_t(10), size_t(0)));
    auto events = session.Open(1, a);
    std::vector<int> out;
    while (true) {
      bool done = false;
      for (const auto& event : events) { if (event.token) out.push_back(*event.token); done |= event.done; }
      if (done) break;
      events = session.Step({1});
    }
    auto next = a;
    next.ids.insert(next.ids.end(), out.begin(), out.end());
    next.ids.insert(next.ids.end(), {3, 4});
    const int calls = engine.calls;
    const auto status = session.Stage(next, 2, false);
    assert(status.state == "reading" && status.tokens == 7 && session.Active() == 0 && engine.calls == calls);
    // The queue place reaches the hook; the slot's held tail starts with the prompt's tokens.
    assert(seen.position == 2 && !seen.read && slot_prefix == (std::vector<bool>{true, false}) && seen.job == 0);
    session.Stage(next, 0, true, 77);
    assert(seen.job == 77);  // fix 2 (item 10): the server's tag reaches the hook
    assert(staged.back() == std::make_pair(next.ids.size(), size_t(12)));  // slot 0 holds the 12-token tail
    auto bypass = next; bypass.cache_control = "bypass";
    assert(session.Stage(bypass).state == "none" && staged.size() == 3);
  }
}
