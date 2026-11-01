// F3: a cold prompt gives the same launch plan whether the server is idle or
// a decoder is ready. A tail of 1 to kPrefillJoinSmallRows rows runs the small-row kernels alone and the bulk kernels
// in a launch, and the two are not bitwise equal, so the set of rows that run alone must not depend on the load. F3
// keeps such a tail alone in both states (trunk merged it into the launch before it on an idle server only).
// This test runs BatchSession on the fake engine at the real sizes (4,096-row grid, 8,192+128 and 16,384+128 launches)
// and compares the launch lists of the idle run and the run with three ready decoders.
#include "batch_mock_engine.h"
#include "../engine/serve/batch_conversation_cache.h"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <string>

namespace {
constexpr int kStep = 4096, kJoin = 128, kDecoders = 3, kEvery = 10;
int failures = 0;

// How the server treats the prompt's capture points (batched_serve.h NextCapture and cut_ready).
enum class Cache {
  Bypass,    // cache_control bypass: no capture point, no cut buffer
  AutoCut,   // auto, spec 0: capture points, and a cut buffer lets a launch pass one
  AutoStop,  // auto with MTP history: capture points are stops, no cut buffer
};
const char* Name(Cache cache) {
  return cache == Cache::Bypass ? "bypass" : cache == Cache::AutoCut ? "auto+cut" : "auto+stop";
}

struct Launch { size_t start; int rows; };

// The prefill launches of a cold prompt of `length` rows. `loaded`: three ready decoders are named on every command.
// `window`: the MTP warm-up window of a speculative request (0: spec 0).
std::vector<Launch> Plan(size_t length, bool loaded, int launch_rows, Cache cache, size_t boundary, int window) {
  MockEngine engine;
  engine.prefill_rows = kStep;
  engine.launch_rows = launch_rows;
  engine.join_rows = kJoin;
  engine.context = int(length) + 4096;
  crucible::BatchSession<MockEngine> session(engine, kDecoders + 1);
  crucible::BatchCacheHooks hooks;
  hooks.restore = [](const crucible::ServeRequest&) { return crucible::BatchRestore{}; };
  hooks.bytes = [] { return size_t(0); };
  hooks.capture = [](const crucible::ServeRequest&, size_t) {};
  hooks.next_capture = [cache](const crucible::ServeRequest& r, size_t cursor) {
    size_t next = SIZE_MAX;
    if (cache == Cache::Bypass) return next;
    for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), kStep))
      if (frontier.position > cursor) next = std::min(next, frontier.position);
    return next;
  };
  static float state[4], target[4];
  if (cache != Cache::AutoStop) {
    hooks.cut_ready = [cache](const crucible::ServeRequest&) { return cache == Cache::AutoCut; };
    hooks.cut_prepare = [](size_t, size_t) {
      return std::vector<crucible::PrefillCutCopy>{{state, target, sizeof state}};
    };
    hooks.cut_capture = [](const crucible::ServeRequest&, size_t) {};
  }
  session.SetCacheHooks(hooks);
  if (window > 0) {
    crucible::BatchSpecHooks spec;
    spec.check = [](const crucible::ServeRequest&) { return std::string(); };
    spec.open = [](uint64_t, int, const crucible::ServeRequest&, size_t) {};
    spec.chunk = [](uint64_t, size_t, int, bool) {};
    spec.first = [](uint64_t, int) {};
    spec.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last + 1}; };
    spec.close = [](uint64_t) {};
    session.SetSpecHooks(spec);
  }
  const auto part = [](std::vector<crucible::BatchEvent>) {};
  if (loaded)
    for (int id = 1; id <= kDecoders; ++id) session.Open(uint64_t(id), Request(id, 500));
  auto request = Request(1, window > 0 ? 2 : 1);
  if (window > 0) { request.spec = 7; request.mtp_prefix = window; }
  request.cache_prefix_tokens = int(boundary);
  request.cache_control = cache == Cache::Bypass ? "bypass" : "auto";
  request.ids.resize(length);
  std::iota(request.ids.begin(), request.ids.end(), 0);
  for (int& id : request.ids) id %= 16;
  const size_t before = engine.chunks.size();
  session.Open(100, request, true);
  std::vector<uint64_t> decoders;
  if (loaded) decoders = {1, 2, 3};
  while ((loaded ? session.Prefill(100, decoders, kEvery, part) : session.Prefill(100)).empty()) {}
  std::vector<Launch> launches;
  size_t start = 0;
  for (size_t i = before; i < engine.chunks.size(); ++i)
    if (engine.chunks[i] < 0) {
      launches.push_back({start, -engine.chunks[i]});
      start += size_t(-engine.chunks[i]);
    }
  if (start != length) { std::printf("FAIL plan covers %zu of %zu rows\n", start, length); ++failures; }
  return launches;
}

// The rows that run alone in a launch of kPrefillJoinSmallRows rows or fewer (the small-row kernels).
std::set<size_t> SmallRows(const std::vector<Launch>& launches) {
  std::set<size_t> rows;
  for (const auto& launch : launches)
    if (launch.rows <= crucible::kPrefillJoinSmallRows)
      for (int row = 0; row < launch.rows; ++row) rows.insert(launch.start + size_t(row));
  return rows;
}
// The grid chunks of a plan: a launch of merged chunks splits at the grid, and a partial last chunk stays with the
// whole chunk before it (a joined tail). Merged launches are bitwise equal to the grid chunks (prefill_launch.h).
std::vector<int> Chunks(const std::vector<Launch>& launches) {
  std::vector<int> chunks;
  for (const auto& launch : launches) {
    int left = launch.rows;
    if (left <= crucible::kPrefillJoinSmallRows) { chunks.push_back(left); continue; }
    for (; left > kStep; left -= kStep) chunks.push_back(kStep);
    chunks.push_back(left);
  }
  return chunks;
}
std::string Text(const std::vector<Launch>& launches) {
  std::string text;
  for (const auto& launch : launches) text += std::to_string(launch.rows) + " ";
  return text;
}

// One case: the idle and the loaded plan must put the same rows in small launches, and a prompt of 4,096 k + t rows
// (k >= 1, t in 1..8) must end with a small-row launch of t rows or fewer in both (F3; a capture point or warm-up start inside the tail splits it, in both). The chunk lists must match.
void Check(size_t length, int launch_rows, Cache cache, size_t boundary, int window) {
  const auto idle = Plan(length, false, launch_rows, cache, boundary, window);
  const auto loaded = Plan(length, true, launch_rows, cache, boundary, window);
  const size_t tail = length > kStep ? (length - 1) % kStep + 1 : 0;
  const bool small_tail = tail >= 1 && tail <= size_t(crucible::kPrefillJoinSmallRows);
  const auto small_idle = SmallRows(idle), small_loaded = SmallRows(loaded);
  bool bad = small_idle != small_loaded || Chunks(idle) != Chunks(loaded);
  if (small_tail && (idle.empty() || loaded.empty() || idle.back().rows > int(tail) || loaded.back().rows > int(tail)))
    bad = true;
  // A tail of 9 rows or more is bulk in both states (the first launch of a prompt of 8 rows or fewer is small-row).
  if (!small_tail && length > size_t(crucible::kPrefillJoinSmallRows) && (!small_idle.empty() || !small_loaded.empty()))
    bad = true;
  if (!bad) return;
  ++failures;
  std::printf("FAIL N=%zu launch=%d %s boundary=%zu window=%d\n  idle:   %s(small rows %zu)\n  loaded: %s(small rows %zu)\n", length,
              launch_rows, Name(cache), boundary, window, Text(idle).c_str(), small_idle.size(), Text(loaded).c_str(),
              small_loaded.size());
}

// The launch function alone: a tail of 1 to 8 rows is never merged into the launch before it, 9 rows or more are.
void LaunchGate() {
  for (int tail = 1; tail <= 12; ++tail) {
    const int rows = crucible::PrefillLaunchRows(0, size_t(kStep + tail), kStep, 16384, SIZE_MAX);
    const int want = tail <= crucible::kPrefillJoinSmallRows ? kStep : kStep + tail;
    if (rows != want) { std::printf("FAIL launch gate: tail %d gives a launch of %d rows\n", tail, rows); ++failures; }
    crucible::PrefillJoinInput in;
    in.at = 0; in.end = size_t(kStep + tail); in.step = kStep; in.launch = kStep; in.join = kJoin;
    const int joined = crucible::PrefillJoinPlan(in).rows;
    if (joined != want) { std::printf("FAIL join gate: tail %d gives a launch of %d rows\n", tail, joined); ++failures; }
  }
}
}  // namespace

int main() {
  LaunchGate();
  std::vector<size_t> lengths = {1, 8, 9, 4096, 4097, 4100, 4104, 4105, 8192, 8196, 8200, 8201, 12292, 12296, 12297};
  for (int k = 1; k <= 6; ++k)
    for (int tail = 0; tail <= 10; ++tail) lengths.push_back(size_t(kStep) * size_t(k) + size_t(tail));
  std::mt19937_64 random(563);
  for (int i = 0; i < 40; ++i) lengths.push_back(1 + random() % 30000);
  // Lengths with a tail of 1 to 8 rows at random grid counts, and a tail of 9 or more.
  for (int i = 0; i < 20; ++i) lengths.push_back(size_t(kStep) * (1 + random() % 7) + 1 + random() % 8);
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  int cases = 0;
  for (const size_t length : lengths)
    for (const int launch : {8192, 16384})
      for (const Cache cache : {Cache::Bypass, Cache::AutoCut, Cache::AutoStop})
        // Cache boundaries: none, early, and the capture frontiers that BatchCacheFrontiers puts near the prompt end
        // (the last row, 4 rows and 8 rows before it, and a whole grid chunk before it).
        for (const size_t boundary : {size_t(0), size_t(100), length - 1, length - 4, length - 9, length - 1 - kStep})
          for (const int window : {0, 100, 6000}) {
            if (boundary >= length) continue;
            // A bypass request has no stable boundary in the plan; one value is enough. A speculative request has
            // MTP history, so it has no cut buffer (batched_serve.h cut_ready): auto+stop only.
            if (cache == Cache::Bypass && boundary != 0) continue;
            if (window > 0 && cache == Cache::AutoCut) continue;
            Check(length, launch, cache, boundary, window);
            ++cases;
          }
  std::printf("tail_launch_test: %d cases, %d failures\n", cases, failures);
  return failures ? 1 : 0;
}
