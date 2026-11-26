// CPU checks of CRUCIBLE_HC_MIX_HI (engine/layers/hc_mix_hi.h): a landed option-B numerics
// default. Unset, it is on where the WMMA HC read exists and off with the VALU read; 0 turns it off, 1 turns it on,
// option A (CRUCIBLE_EXACT=1) keeps it off, and other text stops the start. An explicit 1 with the VALU HC read (the
// only HC read that it cannot change) is refused. The device bit rules (the high-half mix equals the exact kernels on
// a zero tlo, 1 row = R rows) are in tests/hc_mix_hi_gpu_test.hip (GPU).
#include "../engine/layers/hc_mix_hi.h"
#include "../engine/layers/numerics.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace crucible;

int main() {
  unsetenv("CRUCIBLE_EXACT");
  unsetenv("CRUCIBLE_HC_MIX_HI");
  // Default on with the WMMA HC read (CRUCIBLE_HC_WMMA unset, 1 or 2); default off with the VALU read (no split of t).
  assert(HcMixHiFromEnv(true));
  assert(!HcMixHiFromEnv(false));
  // 0 is the rollback; 1 is on, and the engine refuses it with the VALU read (HcMixHiConflict below).
  setenv("CRUCIBLE_HC_MIX_HI", "0", 1);
  assert(!HcMixHiFromEnv(true));
  assert(!HcMixHiFromEnv(false));
  setenv("CRUCIBLE_HC_MIX_HI", "1", 1);
  assert(HcMixHiFromEnv(true));
  assert(HcMixHiFromEnv(false));
  // Option A ignores the switch and keeps the baseline arithmetic (both halves of t), set or unset.
  setenv("CRUCIBLE_EXACT", "1", 1);
  assert(!HcMixHiFromEnv(true));
  unsetenv("CRUCIBLE_HC_MIX_HI");
  assert(!HcMixHiFromEnv(true));
  unsetenv("CRUCIBLE_EXACT");
  // A typo (also an empty value) stops the start: it must not leave the default silently on or off.
  for (const char* bad : {"", "yes", "true", "2", "-1", "on", "off"}) {
    setenv("CRUCIBLE_HC_MIX_HI", bad, 1);
    bool rejected = false;
    try { (void)HcMixHiFromEnv(true); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  unsetenv("CRUCIBLE_HC_MIX_HI");
  // Every WMMA HC read (bulk prefill, chunked prefill, verify, decode, the MTP last row) takes the switch; the VALU
  // read (CRUCIBLE_HC_WMMA=0, also the persistent GDN path) does not, so the engine refuses that pair.
  assert(HcMixHiConflict(false, false) == nullptr);
  assert(HcMixHiConflict(false, true) == nullptr);
  assert(HcMixHiConflict(true, true) == nullptr);
  assert(HcMixHiConflict(true, false) != nullptr);
  assert(std::strstr(HcMixHiConflict(true, false), "CRUCIBLE_HC_WMMA") != nullptr);
  // The bulk kernel: RT 16-row tiles per wave, 8 waves, so 128 * RT rows per block; every row is covered once.
  static_assert(kHcMixHiRowTiles >= 1 && kHcMixHiRowTiles <= 2);
  for (int R : {9, 16, 32, 97, 128, 255, 256, 257, 4096, 16384, 16512}) {
    const int blocks = HcMixHiRowBlocks(R);
    assert(blocks * kHcMixHiBlockRows >= R && (blocks - 1) * kHcMixHiBlockRows < R);
  }
  std::puts("hc_mix_hi_test: PASS");
  return 0;
}
