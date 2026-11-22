// The pinned PLE host stage holds a whole prefill launch for every row kind (engine/model/flashnext/ple_stage.h):
// the host gather of a launch then starts at its row upload and overlaps layer 0.
// Before, the BF16 rows (CRUCIBLE_PLE_BF16) kept a 4,096-row stage, so a 16384+128 launch gathered 5 pieces while
// the device waited (103 ms of idle GPU per cold 32K prefill on the test machine, Nov 18).
#include "../engine/model/flashnext/ple_stage.h"
#include <cassert>
#include <cstdio>

using namespace crucible;

int main() {
  static_assert(PleStageRowsFor(true) == kMaxPrefillLaunchRows);
  static_assert(PleStageRowsFor(false) == kMaxPrefillLaunchRows);
  // The daily launch (16384 rows and the 128-row join) fits one stage.
  static_assert(PleStageRowsFor(true) >= 16384 + 128);
  // The cost: 16 BF16 rows of 160 words per token, 160 MiB pinned.
  static_assert(PleStageBytes(true) == (size_t)kMaxPrefillLaunchRows * 16 * 160 * 2);
  static_assert(PleStageBytes(true) == (size_t)160 << 20);
  std::printf("ple_stage_rows_test: PASS\n");
  return 0;
}
