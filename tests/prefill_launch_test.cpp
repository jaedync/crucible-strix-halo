#include "../engine/model/flashnext/prefill_launch.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>

int main() {
  using namespace crucible;
  constexpr size_t kNone = SIZE_MAX;
  // Launch rows equal to the grid: the grid schedule, one chunk at a time.
  assert(PrefillLaunchRows(0, 32768, 4096, 4096, kNone) == 4096);
  assert(PrefillLaunchRows(0, 1000, 4096, 4096, kNone) == 1000);
  // Whole grid chunks merge up to the launch rows.
  assert(PrefillLaunchRows(0, 32768, 4096, 16384, kNone) == 16384);
  assert(PrefillLaunchRows(16384, 32768, 4096, 16384, kNone) == 16384);
  assert(PrefillLaunchRows(0, 32768, 4096, 32768, kNone) == 32768);
  // A partial last chunk merges into the launch before it.
  assert(PrefillLaunchRows(0, 10000, 4096, 16384, kNone) == 10000);
  assert(PrefillLaunchRows(8192, 10000, 4096, 16384, kNone) == 1808);
  // The partial chunk does not merge when the launch has no room for it.
  assert(PrefillLaunchRows(0, 18000, 4096, 16384, kNone) == 16384);
  assert(PrefillLaunchRows(16384, 18000, 4096, 16384, kNone) == 1616);
  // A launch row count that is not a grid multiple takes whole chunks only.
  assert(PrefillLaunchRows(0, 32768, 4096, 10000, kNone) == 8192);
  // A stop (snapshot frontier, MTP warm-up start) ends the launch on the grid.
  assert(PrefillLaunchRows(0, 32768, 4096, 32768, 12288) == 12288);
  assert(PrefillLaunchRows(0, 32768, 4096, 32768, 12000) == 8192);
  assert(PrefillLaunchRows(12288, 32768, 4096, 32768, 12288) == 4096);
  // A stop inside the first chunk keeps the single grid chunk.
  assert(PrefillLaunchRows(0, 32768, 4096, 32768, 100) == 4096);
  assert(PrefillLaunchRows(0, 32768, 4096, 32768, 0) == 4096);
  // A start off the grid (a restored frontier) keeps the step-relative schedule.
  assert(PrefillLaunchRows(5000, 30000, 4096, 16384, kNone) == 16384);
  assert(PrefillLaunchRows(21384, 30000, 4096, 16384, kNone) == 8616);
  // The merged schedule visits a subset of the grid chunk ends, and a merged
  // launch never crosses a stop.
  for (size_t end : {1ul, 4095ul, 4096ul, 4097ul, 40000ul, 131072ul, 131117ul})
    for (int launch : {4096, 8192, 16384, 32768})
      for (size_t stop : {kNone, 20480ul, 0ul}) {
        size_t at = 0, grid = 0;
        while (at < end) {
          const int rows = PrefillLaunchRows(at, end, 4096, launch, stop);
          assert(rows > 0 && rows <= std::max(launch, 4096));
          at += size_t(rows);
          while (grid < at) grid += std::min<size_t>(4096, end - grid);
          assert(grid == at);
          if (rows > 4096) assert(!(at - rows < stop && stop < at));
        }
      }
  // Invalid inputs fail closed to one grid chunk.
  assert(PrefillLaunchRows(0, 32768, 4096, 0, kNone) == 4096);
  assert(PrefillLaunchRows(0, 32768, 4096, -1, kNone) == 4096);
  // Launch row settings.
  assert(PrefillLaunchSetting(nullptr, 4096) == kDefaultPrefillLaunchRows);  // prefill-rows: unset merges four grid chunks
  assert(kDefaultPrefillLaunchRows == 16384);
  assert(PrefillLaunchSetting("16384", 4096) == 16384);
  assert(PrefillLaunchSetting("32768", 4096) == 32768);
  assert(PrefillLaunchSetting("4096", 4096) == 4096);  // the off value: one grid chunk a launch
  assert(PrefillLaunchSetting("8192", 2048) == 8192);
  assert(PrefillLaunchSetting("6000", 4096) == -1);   // not a grid multiple
  assert(PrefillLaunchSetting("2048", 4096) == -1);   // below the grid
  assert(PrefillLaunchSetting("65536", 4096) == -1);  // above kMaxPrefillLaunchRows
  assert(PrefillLaunchSetting("abc", 4096) == -1);
  assert(PrefillLaunchSetting("", 4096) == -1);
  assert(PrefillLaunchSetting("16384x", 4096) == -1);
  assert(PrefillLaunchSetting("16384", 0) == 0);      // small-row path: no bulk launches
  // A merged launch needs a grid of whole GDN/QSA core parts: the core scratch holds
  // min(grid, kPrefillCoreRows) rows, and parts cut at 2048 from the launch start.
  assert(PrefillLaunchSetting("8192", 1024) == -1);
  assert(PrefillLaunchSetting("6144", 3072) == -1);
  assert(PrefillLaunchSetting("4096", 2048) == 4096);
  assert(PrefillLaunchSetting("1024", 1024) == 1024);  // no merge: valid for any grid
  assert(PrefillLaunchSetting(nullptr, 0) == 0);
  // The default merges only the 4,096-row grid. Every other grid keeps one chunk a launch,
  // also a 2048-row grid, which a set value may still merge.
  assert(kDefaultPrefillLaunchGrid == 4096);
  assert(PrefillLaunchSetting(nullptr, 2048) == 2048);
  assert(PrefillLaunchSetting("8192", 2048) == 8192);
  assert(PrefillLaunchSetting(nullptr, 8192) == 8192);
  assert(PrefillLaunchSetting(nullptr, 16384) == 16384);
  assert(PrefillLaunchSetting(nullptr, 1024) == 1024);
  assert(PrefillLaunchSetting(nullptr, 3072) == 3072);
  assert(PrefillLaunchSetting(nullptr, 1000) == 1000);
  assert(PrefillLaunchSetting(nullptr, 6144) == 6144);
  return 0;
}
