// with exact resume on (CRUCIBLE_EXACT_RESUME=1), the rows that run the small-row kernels must be the rows
// that the cold prefill runs there. A prompt of 4,096 k + t rows (k >= 1, t in 1..8) ends its cold plan in a small-row
// launch of t rows (F3); the small-row and the bulk kernels are not bitwise equal (engine
// kMaxRows). Fast gate 3.05 B p4100-g7 (Oct 27): the Pre point 4,088 of the 4,100-row prompt ended a launch there, so
// rows 4,088 to 4,099 ran as one 12-row bulk launch where cold runs 4,096 bulk rows and a 4-row small launch.
// Part A: cold prompts, exact resume on against off, idle and with three ready decoders.
// Part B: a prompt that extends an earlier one resumes from that run's End or Pre point: the small-row rows of the
// restored state and of the resumed launches together must be the cold plan's.
// Part C (switch off, the default path): the same extensions with exact resume off. The rows that the resumed launches
// run with the small-row kernels must be the cold plan's from the restored length on (a point saved after a small-row
// tail holds small-row state that the default path keeps; it is no part of this check).
// The fake engine runs at the real sizes (4,096-row grid, 8,192+128 and 16,384+128 launches, 2,048-row core parts).
#include "batch_mock_engine.h"
#include "../engine/serve/batch_conversation_cache.h"
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <string>

namespace {
constexpr int kStep = 4096, kJoin = 128, kDecoders = 3, kEvery = 10;
int failures = 0;

enum class Cache {
  AutoCut,   // auto, spec 0: capture points, and a cut buffer (lever 2 saves the Pre point inside a launch)
  AutoStop,  // auto with MTP history: capture points are stops, no cut buffer
};
const char* Name(Cache cache) { return cache == Cache::AutoCut ? "auto+cut" : "auto+stop"; }

struct Launch { size_t start; int rows; bool small; };

// One session with the cache hooks of batched_serve.h in miniature: no host states, slot checkpoints that hold the
// slot's rows, and (AutoCut) a cut buffer.
struct Fixture {
  MockEngine engine;
  crucible::BatchSession<MockEngine> session;
  std::map<std::pair<int, int>, std::vector<int>> buffers;
  size_t cut_at = 0;
  Fixture(int slots, int launch_rows, Cache cache, bool exact, int context, int join = kJoin)
      : session((engine.prefill_rows = kStep, engine.launch_rows = launch_rows, engine.join_rows = join,
                 engine.context = context, engine), slots) {
    session.SetPreEnd(crucible::kPreEndMinTokens);
    session.SetExactResume(exact);
    crucible::BatchCacheHooks hooks;
    hooks.restore = [](const crucible::ServeRequest&) { return crucible::BatchRestore{}; };
    hooks.peek = [](const crucible::ServeRequest&) { return size_t(0); };
    hooks.bytes = [] { return size_t(0); };
    hooks.capture = [](const crucible::ServeRequest&, size_t) {};
    hooks.next_capture = [](const crucible::ServeRequest& r, size_t cursor) {
      size_t next = SIZE_MAX;
      for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), kStep))
        if (frontier.position > cursor) next = std::min(next, frontier.position);
      return next;
    };
    hooks.checkpoint_save = [this](int slot, int index) { buffers[{slot, index}] = engine.histories[slot]; };
    hooks.checkpoint_load = [this](int slot, int index, const crucible::SlotCheckpoint& point) {
      assert(buffers.at({slot, index}) == point.hist);
      engine.histories[slot] = point.hist;
    };
    if (cache == Cache::AutoCut) {
      static float state[4], target[4];
      hooks.cut_ready = [](const crucible::ServeRequest& r) { return r.cache_control != "bypass"; };
      hooks.cut_prepare = [this](size_t position, size_t) {
        cut_at = position;
        return std::vector<crucible::PrefillCutCopy>{{state, target, sizeof state}};
      };
      hooks.cut_capture = [](const crucible::ServeRequest&, size_t) {};
      hooks.cut_save = [this](int slot, int index) {
        const auto& stream = engine.histories[slot];
        buffers[{slot, index}].assign(stream.begin(), stream.begin() + std::ptrdiff_t(cut_at));
      };
    }
    session.SetCacheHooks(hooks);
  }
  bool session_close_after = false;
  void Speculative() {
    crucible::BatchSpecHooks spec;
    spec.check = [](const crucible::ServeRequest&) { return std::string(); };
    spec.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    spec.chunk = [](uint64_t, size_t, int, bool) {};
    spec.first = [](uint64_t, int) {};
    spec.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last + 1}; };
    spec.close = [](uint64_t) {};
    session.SetSpecHooks(spec);
  }
  // Prefills `request` (id 100) to its first token and returns its launches and its cached rows. `loaded`: three
  // ready decoders are named on every command. A request with spec 0 then decodes to its end.
  std::vector<Launch> Run(const crucible::ServeRequest& request, bool loaded, size_t* cached = nullptr) {
    const size_t before = engine.chunks.size();
    session.Open(100, request, true);
    std::vector<uint64_t> decoders;
    if (loaded) decoders = {1, 2, 3};
    const auto part = [](std::vector<crucible::BatchEvent>) {};
    std::vector<crucible::BatchEvent> events;
    while ((events = loaded ? session.Prefill(100, decoders, kEvery, part) : session.Prefill(100)).empty()) {}
    session.SettlePrompt();
    const size_t prefilled = engine.chunks.size();
    size_t start = events.front().cached_tokens;
    if (cached) *cached = start;
    // A speculative request (Part A only) stops at its first token: the fake has no spec_step hook.
    for (; request.spec == 0;) {
      bool done = false;
      for (const auto& event : events) done = done || (event.done && event.request_id == 100);
      if (done) break;
      events = session.Step({100});
    }
    std::vector<Launch> launches;
    for (size_t i = before; i < prefilled; ++i) {
      const int rows = std::abs(engine.chunks[i]);
      // A Forward (positive) runs the small-row kernels; so does a Prefill of 8 rows or fewer.
      launches.push_back({start, rows, engine.chunks[i] > 0 || rows <= crucible::kPrefillJoinSmallRows});
      start += size_t(rows);
    }
    // A speculative request stays open at its first token (no spec_step hook in the fake): close it for the next turn.
    if (request.spec != 0 && session_close_after) session.Close(100);
    if (start != request.ids.size()) {
      std::printf("FAIL plan covers %zu of %zu rows\n", start, request.ids.size());
      ++failures;
    }
    return launches;
  }
};

crucible::ServeRequest Prompt(size_t length, uint64_t seed, int window) {
  auto request = Request(1, 2);
  if (window > 0) { request.spec = 7; request.mtp_prefix = window; }
  request.cache_control = "auto";
  request.ids.resize(length);
  std::mt19937_64 random(seed);
  for (int& id : request.ids) id = int(random() % 16);
  return request;
}

std::set<size_t> SmallRows(const std::vector<Launch>& launches, size_t below = SIZE_MAX) {
  std::set<size_t> rows;
  for (const auto& launch : launches)
    if (launch.small)
      for (int row = 0; row < launch.rows; ++row)
        if (launch.start + size_t(row) < below) rows.insert(launch.start + size_t(row));
  return rows;
}
std::string Text(const std::vector<Launch>& launches) {
  std::string text;
  for (const auto& launch : launches)
    text += std::to_string(launch.start) + "+" + std::to_string(launch.rows) + (launch.small ? "s " : " ");
  return text;
}
int ContextFor(size_t length) { return int(length) + 4096; }

// The cold plan of `request` with exact resume off: the reference set of small-row rows.
struct Shape { bool loaded = false; int window = 0; int join = kJoin; };
std::vector<Launch> Cold(const crucible::ServeRequest& request, int launch_rows, const Shape& shape = {}) {
  Fixture f(shape.loaded ? kDecoders + 1 : 1, launch_rows, Cache::AutoStop, false, ContextFor(request.ids.size()),
            shape.join);
  f.session_close_after = true;
  if (shape.window > 0) f.Speculative();
  if (shape.loaded)
    for (int id = 1; id <= kDecoders; ++id) f.session.Open(uint64_t(id), Request(id, 500));
  auto bypass = request;
  bypass.cache_control = "bypass";
  return f.Run(bypass, shape.loaded);
}

// Part A: one cold prompt, exact resume on against off.
void Cold(size_t length, int launch_rows, Cache cache, int window, bool loaded) {
  std::vector<Launch> plans[2];
  for (const bool exact : {false, true}) {
    Fixture f(kDecoders + 1, launch_rows, cache, exact, ContextFor(length));
    if (window > 0) f.Speculative();
    if (loaded)
      for (int id = 1; id <= kDecoders; ++id) f.session.Open(uint64_t(id), Request(id, 500));
    plans[exact] = f.Run(Prompt(length, length, window), loaded);
  }
  if (SmallRows(plans[0]) == SmallRows(plans[1])) return;
  ++failures;
  std::printf("FAIL cold N=%zu launch=%d %s window=%d %s\n  off: %s\n  on:  %s\n", length, launch_rows, Name(cache),
              window, loaded ? "loaded" : "idle", Text(plans[0]).c_str(), Text(plans[1]).c_str());
}

// Part B: a chain of turns on one slot. Each prompt extends the one before it (the same first rows) and resumes from a
// point that the previous run saved. The small-row rows that the restored states and the resumed launches hold together
// must be the cold plan's. `Shape`: three ready decoders (loaded), the spec window (> 0: MTP
// history, no cut buffer) and the join rows of the engine (0: launches of one grid chunk, 4,096 rows).
int resumed = 0;
void Chain(const std::vector<size_t>& turns, int launch_rows, Cache cache, const Shape& shape = {}) {
  const size_t length = turns.back();
  const auto full = Prompt(length, turns.front() * 7919 + length, shape.window);
  Fixture f(shape.loaded ? kDecoders + 1 : 1, launch_rows, cache, true, ContextFor(length), shape.join);
  f.session_close_after = true;
  if (shape.window > 0) f.Speculative();
  if (shape.loaded)
    for (int id = 1; id <= kDecoders; ++id) f.session.Open(uint64_t(id), Request(id, 500));
  std::set<size_t> got;
  std::string text;
  size_t last = 0;
  for (size_t i = 0; i < turns.size(); ++i) {
    auto request = full;
    request.ids.resize(turns[i]);
    size_t cached = 0;
    const auto plan = f.Run(request, shape.loaded, &cached);
    // The rows of the earlier turns count below this turn's restore point; a miss starts over.
    if (!cached) got.clear();
    else {
      for (auto it = got.begin(); it != got.end();) it = *it >= cached ? got.erase(it) : std::next(it);
    }
    for (const size_t row : SmallRows(plan)) got.insert(row);
    last = cached;
    text += "  turn " + std::to_string(turns[i]) + " resumed at " + std::to_string(cached) + ": " + Text(plan) + "\n";
  }
  if (last) ++resumed;
  const auto cold = Cold(full, launch_rows, shape);
  if (got == SmallRows(cold)) return;
  ++failures;
  std::printf("FAIL chain %zu launch=%d %s join=%d%s window=%d\n%s  cold: %s\n", length, launch_rows, Name(cache),
              shape.join, shape.loaded ? " loaded" : "", shape.window, text.c_str(), Text(cold).c_str());
}
void Extend(size_t base, size_t length, int launch_rows, Cache cache, const Shape& shape = {}) {
  Chain({base, length}, launch_rows, cache, shape);
}

// Part C: as Extend, with exact resume off. Only the rows from the restored length on count.
int resumed_off = 0;
void ExtendOff(size_t base, size_t length, int launch_rows, Cache cache) {
  const auto ext = Prompt(length, base * 7919 + length, 0);
  auto first = ext;
  first.ids.resize(base);
  Fixture f(1, launch_rows, cache, false, ContextFor(length));
  f.Run(first, false);
  size_t cached = 0;
  const auto ext_plan = f.Run(ext, false, &cached);
  if (!cached) return;
  // Only a resume that runs as bulk rows from its point (BatchSession::BulkResumes) is a prompt-state resume of the
  // default path; the others keep their small-row head (replay) and are no part of this check.
  if (cached % 2048 && !(cached % 4 == 0 && length > cached + 8)) return;
  ++resumed_off;
  const auto cold = Cold(ext, launch_rows);
  const auto got = SmallRows(ext_plan);
  std::set<size_t> want;
  for (const size_t row : SmallRows(cold)) if (row >= cached) want.insert(row);
  if (got == want) return;
  ++failures;
  std::printf("FAIL extend off %zu -> %zu launch=%d %s: resumed at %zu\n  ext:  %s\n  cold: %s\n", base, length,
              launch_rows, Name(cache), cached, Text(ext_plan).c_str(), Text(cold).c_str());
}

// Part C, third turn (review L3): a third prompt diverges from the second inside its tail (or at its end). The launch
// that ends at 4,096 k now saves a Grid point there, so the third turn can resume at 4,096 k. The rows it runs with the
// small-row kernels from its restore point on must be the cold plan's.
int third_turns = 0, third_grid = 0;
void ThirdTurn(size_t b1, size_t l2, size_t l3, size_t share, int launch_rows, Cache cache) {
  const auto t2 = Prompt(l2, b1 * 31 + 7, 0);
  auto t1 = t2;
  t1.ids.resize(b1);
  auto t3 = Prompt(l3, b1 * 31 + 7, 0);
  const auto other = Prompt(l3, b1 * 31 + 106, 0);
  for (size_t k = share; k < l3; ++k) t3.ids[k] = other.ids[k];
  Fixture f(1, launch_rows, cache, false, ContextFor(l3));
  f.Run(t1, false);
  f.Run(t2, false);
  size_t cached = 0;
  const auto plan = f.Run(t3, false, &cached);
  if (!cached) return;
  // Only a resume that runs as bulk rows from its point counts (as in ExtendOff); a Grid point resumes at a chunk end.
  if (cached % 2048 && !(cached % 4 == 0 && l3 > cached + 8)) return;
  ++third_turns;
  if (cached == share || cached % 4096 == 0) ++third_grid;
  const auto cold = Cold(t3, launch_rows);
  std::set<size_t> want;
  for (const size_t row : SmallRows(cold)) if (row >= cached) want.insert(row);
  if (SmallRows(plan) == want) return;
  ++failures;
  std::printf("FAIL third turn %zu -> %zu -> %zu (shares %zu) launch=%d %s: resumed at %zu\n  t3:   %s\n  cold: %s\n",
              b1, l2, l3, share, launch_rows, Name(cache), cached, Text(plan).c_str(), Text(cold).c_str());
}
}  // namespace

int main() {
  std::vector<size_t> lengths = {9, 12, 4096, 4097, 4100, 4104, 4105, 4108, 8192, 8196, 8200, 8201, 12292, 12296,
                                 32772, 36868};
  for (int k = 1; k <= 4; ++k)
    for (int tail = 1; tail <= 12; ++tail) lengths.push_back(size_t(kStep) * size_t(k) + size_t(tail));
  std::mt19937_64 random(582);
  for (int i = 0; i < 16; ++i) lengths.push_back(size_t(kStep) * (1 + random() % 6) + 1 + random() % 8);
  for (int i = 0; i < 16; ++i) lengths.push_back(9 + random() % 26000);
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  int cold_cases = 0;
  for (const size_t length : lengths)
    for (const int launch : {8192, 16384})
      for (const Cache cache : {Cache::AutoCut, Cache::AutoStop})
        for (const int window : {0, 100})
          for (const bool loaded : {false, true}) {
            // A speculative request has MTP history, so it has no cut buffer (batched_serve.h cut_ready).
            if (window > 0 && cache == Cache::AutoCut) continue;
            Cold(length, launch, cache, window, loaded);
            ++cold_cases;
          }
  // Extensions: the fault's prompt, End points on and near a small tail, core-grid End points, short prompts.
  std::vector<std::pair<size_t, size_t>> pairs = {
      {4100, 4101}, {4100, 4104}, {4100, 4108}, {4100, 4112}, {4100, 4120}, {4100, 8196}, {4100, 12292},
      {4092, 4101}, {4092, 4104}, {4092, 4120}, {4088, 4100}, {4084, 4097}, {6144, 8196}, {6144, 8200},
      {8, 20}, {12, 24}, {58, 70}, {8300, 8900}, {8300, 12292}, {8900, 12296}, {4120, 8193}, {30000, 32772}};
  for (int i = 0; i < 60; ++i) {
    const size_t base = 9 + random() % 14000;
    pairs.emplace_back(base, base + 1 + random() % 24);
    const size_t grid = (base / kStep + 1 + random() % 2) * kStep;
    pairs.emplace_back(base, grid + 1 + random() % 8);
  }
  int ext_cases = 0;
  for (const auto& [base, length] : pairs)
    for (const int launch : {8192, 16384})
      for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
        Extend(base, length, launch, cache);
        ++ext_cases;
      }
  // Part C: switch off. End and Pre points 2,048 off the 4,096 grid (and on it) against prompts of 4,096 k + t rows,
  // t = 1..8 (the fault) and t = 9..10 (control: cold runs those rows as bulk rows).
  int off_cases = 0;
  for (const size_t base : {size_t(2048), size_t(6144), size_t(10240), size_t(4096), size_t(8192), size_t(6000)})
    for (const size_t grid : {size_t(4096), size_t(8192), size_t(12288)})
      for (size_t tail = 1; tail <= 10; ++tail) {
        if (grid + tail <= base + 8) continue;
        for (const int launch : {8192, 16384})
          for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
            ExtendOff(base, grid + tail, launch, cache);
            ++off_cases;
          }
      }
  // End points 4 and 8 rows below a cold small-row tail (review M1): 8 new rows or fewer before the tail would run the
  // small-row kernels where cold runs bulk rows, so BestPoint refuses the point (the exact path does too).
  const std::vector<std::pair<size_t, size_t>> below = {{8184, 8196}, {8188, 8200}, {4088, 4100},    {4092, 4101},
                                                        {4092, 4104}, {12280, 12292}, {32764, 32776}, {131064, 131076}};
  for (const auto& [base, length] : below)
    for (const int launch : {8192, 16384})
      for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
        ExtendOff(base, length, launch, cache);
        ++off_cases;
      }
  for (const auto& [base, length] : pairs)
    for (const int launch : {8192, 16384})
      for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
        ExtendOff(base, length, launch, cache);
        ++off_cases;
      }
  for (const size_t b1 : {size_t(2048), size_t(6144), size_t(4100), size_t(4088), size_t(8184), size_t(10240)})
    for (size_t k = 1; k <= 3; ++k)
      for (size_t t = 1; t <= 8; ++t) {
        const size_t l2 = 4096 * k + t;
        if (l2 <= b1) continue;
        for (const size_t add : {size_t(1), size_t(30), size_t(4100)})
          for (const size_t share : {l2, 4096 * k, 4096 * k + 1})
            for (const int launch : {8192, 16384})
              for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
                ThirdTurn(b1, l2, l2 + add, share, launch, cache);
                ++off_cases;
              }
      }
  // Part B, more shapes: ready decoders, an MTP window, no join rows (launch 4,096), and chains of
  // three turns that resume from a point that a resumed run saved.
  int shape_cases = 0;
  const std::vector<std::pair<size_t, size_t>> few = {{4100, 8196}, {4100, 4108}, {4092, 4120}, {6144, 8196},
                                                      {2048, 4100}, {8300, 12292}, {4120, 8193}, {8, 20}};
  const std::vector<std::vector<size_t>> chains = {{2048, 4100, 8196}, {4100, 8196, 12292}, {4120, 8300, 8196 + 4096},
                                                   {6144, 8300, 12292}, {4096, 6144, 8196}, {5000, 8196, 8200}};
  for (const Cache cache : {Cache::AutoCut, Cache::AutoStop}) {
    for (const Shape shape : {Shape{true, 0, kJoin}, Shape{false, 100, kJoin}, Shape{true, 100, kJoin},
                              Shape{false, 0, 0}, Shape{true, 0, 0}})
      for (const int launch : {kStep, 8192, 16384}) {
        if (shape.join == 0 && launch != kStep) continue;
        if (shape.window > 0 && cache == Cache::AutoCut) continue;
        for (const auto& [base, length] : few) { Extend(base, length, launch, cache, shape); ++shape_cases; }
        for (const auto& chain : chains) { Chain(chain, launch, cache, shape); ++shape_cases; }
      }
    for (const int launch : {8192, 16384})
      for (const auto& chain : chains) { Chain(chain, launch, cache); ++shape_cases; }
  }
  std::printf("exact_resume_tail_test: %d shape cases, %d cold cases, %d extensions (%d resumed), %d switch-off extensions (%d resumed), "
              "%d third turns (%d from a chunk end), %d failures\n", shape_cases, cold_cases, ext_cases, resumed, off_cases, resumed_off, third_turns, third_grid, failures);
  return failures ? 1 : 0;
}
