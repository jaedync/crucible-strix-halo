// An earlier change: the split2048 probe's Pre-cut mode (bench/split2048.hip `pre`) must prefill with the chunk
// schedule that BatchSession::Prefill runs for a cold prompt with fix B off (whole) and on (split). This test runs
// BatchSession on the fake engine at the real sizes (4,096-row grid, 8,192-row launches, 2,048-row core parts,
// kPreEndMinTokens) and compares each launch with bench/split2048_schedule.h.
#include "batch_mock_engine.h"
#include "../bench/split2048_schedule.h"
#include <cassert>
#include <cstdio>
#include <numeric>

namespace {
constexpr int kStep = 4096, kLaunch = 8192;

struct Served { std::vector<int> rows; size_t pre = 0; };

Served Serve(size_t length, bool pre_end) {
  MockEngine engine;
  engine.prefill_rows = kStep;
  engine.launch_rows = kLaunch;
  engine.context = int(length) + 16;
  crucible::BatchSession<MockEngine> session(engine, 1);
  if (pre_end) session.SetPreEnd(crucible::kPreEndMinTokens);
  Served served;
  crucible::BatchCacheHooks hooks;
  hooks.restore = [](const crucible::ServeRequest&) -> crucible::BatchRestore { return 0; };
  hooks.peek = [](const crucible::ServeRequest&) { return size_t(0); };
  hooks.capture = [](const crucible::ServeRequest&, size_t) {};
  hooks.next_capture = [](const crucible::ServeRequest&, size_t) { return SIZE_MAX; };
  hooks.checkpoint_save = [](int, int) {};
  hooks.pre_saved = [&](int, size_t tokens) { served.pre = tokens; };
  hooks.bytes = [] { return size_t(0); };
  session.SetCacheHooks(hooks);
  crucible::ServeRequest request = Request(1, 2);
  request.ids.resize(length);
  std::iota(request.ids.begin(), request.ids.end(), 0);
  for (int& id : request.ids) id %= 16;
  Complete(session, 1, request);
  for (const int rows : engine.chunks)
    if (rows < 0) served.rows.push_back(-rows);  // the bulk launches; decode steps follow the prompt
  return served;
}

size_t Pre(size_t length) {
  return split2048::PrePoint(length, kStep, crucible::kPrefillLaunchCoreRows, crucible::kPreEndMinTokens,
                             crucible::kMaxRows);
}

void Check(size_t length) {
  const auto whole = Serve(length, false), split = Serve(length, true);
  const size_t pre = Pre(length);
  assert(whole.pre == 0 && split.pre == pre);
  assert(whole.rows == split2048::ServeSchedule(length, kStep, kLaunch, 0));
  assert(split.rows == split2048::ServeSchedule(length, kStep, kLaunch, pre));
  if (pre) assert(split2048::CutOnlyAt(whole.rows, split.rows, pre));
  else assert(split.rows == whole.rows);
}
}  // namespace

int main() {
  // The probe's cases (the deleted tools/resend_split_check.sh): second pieces of 9, 9, 1,052, 2,047 and 2,048 rows, after a final
  // chunk that starts on an even (32,768) or an odd (36,864) grid chunk of the 8,192-row launches.
  const std::vector<std::pair<size_t, size_t>> cases{
      {34825, 9}, {38921, 9}, {39964, 1052}, {36863, 2047}, {40960, 2048}};
  for (const auto& [length, piece] : cases) {
    assert(Pre(length) && length - Pre(length) == piece);
    Check(length);
  }
  // The odd case: the unsplit final launch merges the last whole grid chunk; the cut also ends that merge.
  assert((split2048::ServeSchedule(39964, kStep, kLaunch, 0) == std::vector<int>{8192, 8192, 8192, 8192, 7196}));
  assert((split2048::ServeSchedule(39964, kStep, kLaunch, Pre(39964)) ==
          std::vector<int>{8192, 8192, 8192, 8192, 4096, 2048, 1052}));
  // No cut: a second piece of 8 rows or fewer, and a prompt under the minimum.
  assert(Pre(34824) == 0 && Pre(32767) == 0 && Pre(34825) == 34816);
  for (const size_t length : {size_t(32767), size_t(34824), size_t(32768), size_t(33000)}) Check(length);
  // Every length of one odd and one even final chunk.
  for (size_t length = 34810; length <= 34830; ++length) Check(length);
  for (size_t length = 38900; length <= 38930; ++length) Check(length);
  // A cut that does not split one piece in two is refused.
  assert(!split2048::CutOnlyAt({4096, 4096}, {4096, 2048, 2048}, 4096));
  assert(!split2048::CutOnlyAt({4096}, {2048, 2048}, 0));
  std::puts("split2048_schedule_test: ok");
  return 0;
}
