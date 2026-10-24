// x4-tail: the join plan of bulk prefill launches, CRUCIBLE_PF_LAUNCH_ROWS="L+T".
#include "../engine/model/flashnext/prefill_launch.h"
#include "../engine/runtime/draft_warmup.h"
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <vector>

namespace {
using namespace crucible;
constexpr int kGrid = 4096, kLaunch = 8192, kJoin = 128;

struct Launch {
  size_t start = 0;
  int rows = 0;
  size_t cut = 0;
};

// The capture points of a cold prompt on the daily layout: the own frontier floor(L / grid) * grid (BatchCacheFrontiers
// with no boundary), plus `extra` points.
std::vector<size_t> Points(size_t end, std::vector<size_t> extra = {}) {
  std::vector<size_t> points = std::move(extra);
  const size_t own = end / kGrid * kGrid;
  if (own > 0 && own < end) points.push_back(own);
  // BatchCacheFrontiers names each position once (a shared prefix on the own frontier is one point).
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  return points;
}
size_t After(const std::vector<size_t>& points, size_t cursor) {
  size_t next = SIZE_MAX;
  for (const size_t point : points)
    if (point > cursor) next = std::min(next, point);
  return next;
}

// The worker's plan without joins (batch_session.h): stop = min(next capture, warm-up start).
std::vector<Launch> Plain(size_t end, const std::vector<size_t>& points, size_t warm) {
  std::vector<Launch> launches;
  for (size_t at = 0; at < end;) {
    const int rows = PrefillLaunchRows(at, end, kGrid, kLaunch, std::min(After(points, at), warm));
    launches.push_back({at, rows, 0});
    at += size_t(rows);
  }
  return launches;
}
// The plan with joins. `lift`: the warm-up start is not a stop (0 < window <= grid). `cut`: a cut buffer is free.
std::vector<Launch> Joined(size_t end, const std::vector<size_t>& points, size_t warm, bool lift, bool cut,
                           int join = kJoin) {
  std::vector<Launch> launches;
  for (size_t at = 0; at < end;) {
    PrefillJoinInput in;
    in.at = at; in.end = end; in.step = kGrid; in.launch = kLaunch; in.join = join;
    in.first = After(points, at);
    in.second = in.first == SIZE_MAX ? SIZE_MAX : After(points, in.first);
    in.cut = cut;
    in.warm = lift ? SIZE_MAX : warm;
    const auto plan = PrefillJoinPlan(in);
    assert(plan.rows > 0);
    launches.push_back({at, plan.rows, plan.cut});
    at += size_t(plan.rows);
  }
  return launches;
}
std::vector<int> Rows(const std::vector<Launch>& launches) {
  std::vector<int> rows;
  for (const auto& launch : launches) rows.push_back(launch.rows);
  return rows;
}

// Every launch of `joined` starts and ends on the grid (or ends at the prompt end), so its core parts are the grid's;
// it holds at most launch + join rows, joins a partial chunk only at the prompt end with more than
// kPrefillJoinSmallRows rows, and passes at most one capture point, which it names as its cut. Every capture point
// and the prompt end are launch ends, as in the plain plan.
void CheckJoined(size_t end, const std::vector<Launch>& joined, const std::vector<size_t>& points, size_t warm,
                 bool lift, bool cut) {
  size_t covered = 0;
  std::vector<size_t> ends;
  for (const auto& launch : joined) {
    assert(launch.start == covered);
    const size_t stop = launch.start + size_t(launch.rows);
    covered = stop;
    ends.push_back(stop);
    assert(launch.start % kGrid == 0 && (stop % kGrid == 0 || stop == end));
    assert(launch.rows <= kLaunch + kJoin);
    if (launch.rows > kLaunch) {
      assert(stop == end && end % kGrid > size_t(kPrefillJoinSmallRows) && end % kGrid <= size_t(kJoin));
    }
    size_t passed = 0, last = 0;
    for (const size_t point : points)
      if (point > launch.start && point < stop) ++passed, last = point;
    assert(passed <= 1);
    assert(launch.cut == (passed ? last : 0));
    if (passed) assert(cut && (last - launch.start) % kGrid == 0);
    if (!lift && warm < end) assert(!(launch.start < warm && warm < stop) || launch.rows <= kGrid);
  }
  assert(covered == end);
  for (const size_t point : points)
    if (point < end && std::find(ends.begin(), ends.end(), point) == ends.end())
      assert(std::count_if(joined.begin(), joined.end(), [&](const Launch& l) { return l.cut == point; }) == 1);
}
}  // namespace

int main() {
  // Settings: "L+T" sets the join rows; every other value has none, so the default allocates what it did before.
  assert(PrefillLaunchSetting("8192+128", 4096) == 8192);
  assert(PrefillJoinSetting("8192+128", 4096) == 128);
  assert(PrefillJoinSetting("16384+4095", 4096) == 4095);
  assert(PrefillJoinSetting(nullptr, 4096) == 128 && kDefaultPrefillJoinRows == 128);  // the default is "8192+128"
  assert(PrefillJoinSetting(nullptr, 2048) == 0 && PrefillJoinSetting(nullptr, 0) == 0);  // other grids: no join
  assert(PrefillLaunchSetting(nullptr, 4096) + PrefillJoinSetting(nullptr, 4096) == 8320);
  assert(PrefillJoinSetting("8192", 4096) == 0);
  assert(PrefillJoinSetting("4096", 4096) == 0);
  for (const char* bad : {"8192+", "8192+0", "8192+-5", "8192++5", "8192+4096", "4096+128", "32768+1", "8192+12x",
                          "6000+128", "8192 +128", "+128"}) {
    assert(PrefillLaunchSetting(bad, 4096) == -1);
    assert(PrefillJoinSetting(bad, 4096) == 0);
  }
  assert(PrefillLaunchSetting("8192+128", 0) == 0 && PrefillJoinSetting("8192+128", 0) == 0);
  assert(PrefillLaunchSetting("8192+128", 1024) == -1);  // no merged launches on a grid of part core parts
  assert(PrefillRoomRows(8192, 0) == 8192 && PrefillRoomRows(4096, 0) == 4096 && PrefillRoomRows(8192, 128) == 8320);
  assert(kPrefillJoinSmallRows == 8);
  // Every set value without "+T" has join 0, so the row buffers hold the launch rows (the bytes before x4-tail;
  // tests/test_x4_tail_room.py checks that the engine sizes them with PrefillRoomRows only).
  for (const char* value : {"8192", "4096", "16384", "32768"})
    for (const int grid : {0, 2048, 4096}) {
      const int launch = PrefillLaunchSetting(value, grid);
      if (launch < 0) continue;
      assert(PrefillJoinSetting(value, grid) == 0 && PrefillRoomRows(launch, PrefillJoinSetting(value, grid)) == launch);
    }

  // The spec checks of Engine::BeginPrefillJoin.
  {
    float state[64], ring[16], raw[8], target[64];
    const std::vector<std::pair<const void*, size_t>> spans{{state, sizeof state}, {ring, sizeof ring}, {raw, sizeof raw}};
    PrefillJoinSpec ok{8192, 4096, {{state, target, sizeof state}, {ring, target, sizeof ring}}};
    assert(PrefillJoinValid(ok, 8272, kGrid, spans));
    assert(PrefillJoinValid({0, 0, {}}, 80, kGrid, spans));
    assert(PrefillJoinValid({4096, 0, {}}, 8272, kGrid, spans));
    auto bad = ok; bad.ledger = 100; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));       // off the grid
    bad = ok; bad.ledger = 8272; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));           // not inside
    bad = ok; bad.cut = 3000; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));              // not a core part end
    bad = ok; bad.cut = 8272; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));              // not inside
    bad = ok; bad.copies.clear(); assert(!PrefillJoinValid(bad, 8272, kGrid, spans));          // a cut copies
    bad = ok; bad.cut = 0; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));                 // no cut, no copies
    bad = ok; bad.copies[1].bytes = 4; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));     // wrong size
    bad = ok; bad.copies[1].source = target; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));  // not a span
    bad = ok; bad.copies[1] = bad.copies[0]; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));  // twice
    bad = ok; bad.copies[0].target = nullptr; assert(!PrefillJoinValid(bad, 8272, kGrid, spans));
  }

  // The release prompts (4.05 a4v, window 256, own frontier only).
  {
    const size_t end = 32848, warm = DraftWarmupStart(end, 256);
    const auto points = Points(end);
    assert((Rows(Plain(end, points, warm)) == std::vector<int>{8192, 8192, 8192, 4096, 4096, 80}));
    const auto joined = Joined(end, points, warm, true, true);
    assert((Rows(joined) == std::vector<int>{8192, 8192, 8192, 8272}));
    assert(joined.back().cut == 32768);
    // No cut buffer: the launch stops at the frontier, and the 80-row tail runs alone.
    assert((Rows(Joined(end, points, warm, true, false)) == std::vector<int>{8192, 8192, 8192, 8192, 80}));
    // The warm-up stop stays (a window above one grid chunk): today's last three launches.
    assert((Rows(Joined(end, points, warm, false, true)) == std::vector<int>{8192, 8192, 8192, 4096, 4096, 80}));
    // The plain plan's last launch before the frontier starts at 28,672, and the last one at 32,768.
    const auto stop = [&](size_t at) { return std::min(After(points, at), warm); };
    assert(PrefillPlainStart(24576, 32768, end, kGrid, kLaunch, stop) == 28672);
    assert(PrefillPlainStart(24576, end, end, kGrid, kLaunch, stop) == 32768);
  }
  {
    const size_t end = 130081, warm = DraftWarmupStart(end, 256);
    const auto points = Points(end);
    std::vector<int> plain(15, 8192), joined(15, 8192);
    plain.insert(plain.end(), {4096, 3105});
    joined.push_back(7201);
    assert(Rows(Plain(end, points, warm)) == plain);
    const auto launches = Joined(end, points, warm, true, true);
    assert(Rows(launches) == joined);
    assert(launches.back().cut == 126976);
    const auto stop = [&](size_t at) { return std::min(After(points, at), warm); };
    assert(PrefillPlainStart(122880, 126976, end, kGrid, kLaunch, stop) == 122880);
    assert(PrefillPlainStart(122880, end, end, kGrid, kLaunch, stop) == 126976);
  }
  // The MTP pieces of a joined launch are the plain plan's launches that hold warm-up rows.
  assert((PrefillGridPieces(8272, kGrid) == std::vector<std::pair<int, int>>{{0, 4096}, {4096, 4096}, {8192, 80}}));
  assert((PrefillGridPieces(7201, kGrid) == std::vector<std::pair<int, int>>{{0, 4096}, {4096, 3105}}));
  assert((PrefillGridPieces(80, kGrid) == std::vector<std::pair<int, int>>{{0, 80}}));

  // Lengths n * 8192 +/- 1..100, n * 4096 + 1..100, 32,848 and about 130K; windows 0 (full), 256 and 8192; with and
  // without a cut buffer; with a second capture point (a shared prefix at 12,288). Join 0 is the plain plan.
  std::vector<size_t> lengths{32848, 130081, 130076, 131072, 131073, 131172, 126977};
  for (size_t n = 1; n <= 16; ++n)
    for (size_t d = 1; d <= 100; ++d) {
      lengths.push_back(n * 8192 + d);
      lengths.push_back(n * 8192 - d);
      lengths.push_back(n * 4096 + d);
    }
  size_t joins = 0, cuts = 0, saved = 0;
  for (const size_t end : lengths)
    for (const int window : {0, 256, 8192})
      for (const bool shared : {false, true})
        for (const bool cut : {false, true}) {
          const auto points = Points(end, shared && end > 12288 ? std::vector<size_t>{12288} : std::vector<size_t>{});
          const size_t warm = DraftWarmupStart(end, window);
          const bool lift = window > 0 && window <= kGrid;
          const auto plain = Plain(end, points, warm);
          // Join 0 and no cut buffer, with the warm-up stop kept: the plain plan, launch for launch.
          const auto off = Joined(end, points, warm, false, false, 0);
          assert(Rows(off) == Rows(plain));
          for (const auto& launch : off) assert(launch.cut == 0);
          const auto joined = Joined(end, points, warm, lift, cut);
          CheckJoined(end, joined, points, warm, lift, cut);
          assert(joined.size() <= plain.size());
          saved += plain.size() - joined.size();
          for (const auto& launch : joined) {
            joins += launch.rows > kLaunch;
            cuts += launch.cut != 0;
          }
        }
  assert(joins > 0 && cuts > 0 && saved > 0);
  std::printf("prefill_join_test: %zu lengths, %zu joined tails, %zu cuts, %zu launches saved\n", lengths.size(), joins,
              cuts, saved);
  return 0;
}
