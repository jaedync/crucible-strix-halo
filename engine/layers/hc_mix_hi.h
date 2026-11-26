// CRUCIBLE_HC_MIX_HI (a landed option-B numerics default): the HC mix of bulk prefill and
// of the WMMA HC read uses the fp16 high half of t only. One VALU read keeps both halves in the default path: the final
// HC of the last row of a predicting bulk prefill chunk (HcReadR in runtime/prefill_engine.inc). Unset, the switch is
// on with the WMMA HC read; CRUCIBLE_HC_MIX_HI=0 is the rollback, and option A (CRUCIBLE_EXACT=1) keeps both halves.
// The low half (t - fp16(t), scaled) is the second WMMA chain of the exact split; without it the mix of x changes by
// about 9e-5 relative L2.
//
// The engine has two HC arithmetic classes, and the switch changes both in the same way:
// - bulk prefill (R > 8, prefill_hc.h): PrefillHcMixHiKernel (prefill_hc_hi.h) instead of PrefillHcMixWideKernel,
//   and the mid pack writes no low half;
// - the WMMA read (hc_wmma2.h: decode, verify, chunked prefill, the MTP last row): the mid stores a zero low half
//   (HcWScratch::hi_only), so the low rows of each mix tile add +0. Each row stays bitwise equal for any R.
// The VALU read (HcReadR) has no split of t, so the switch does not change it. With CRUCIBLE_HC_WMMA=0 (also the
// persistent GDN path that needs it) the default is off, and the engine refuses an explicit CRUCIBLE_HC_MIX_HI=1.
// This header holds no device code (tests/hc_mix_hi_test.cpp includes it).
#pragma once
#include "numerics.h"

namespace crucible {

// 16-row tiles per wave of PrefillHcMixHiKernel: each up fragment serves this many WMMAs (prefill-hc 2.3: two
// tiles fit in registers once the low chain is gone; 2,393 against 3,728 us at R = 8,192 in the Oct 24 bench).
constexpr int kHcMixHiRowTiles = 2;
constexpr int kHcMixHiBlockRows = 8 * 16 * kHcMixHiRowTiles;  // 8 waves

constexpr int HcMixHiRowBlocks(int R) { return (R + kHcMixHiBlockRows - 1) / kHcMixHiBlockRows; }

// Read once per engine. Unset: on with the WMMA HC read (hc_wmma), off with the VALU read. 0: off. 1: on. Option A
// (CRUCIBLE_EXACT=1): off without a read. Other text (also empty) throws std::invalid_argument.
inline bool HcMixHiFromEnv(bool hc_wmma) {
  return NumericsDefaultInt("CRUCIBLE_HC_MIX_HI", hc_wmma ? 1 : 0, 0, 0, 1) != 0;
}

// Null when the pair can run; else the startup error.
constexpr const char* HcMixHiConflict(bool mix_hi, bool hc_wmma) {
  return mix_hi && !hc_wmma ? "CRUCIBLE_HC_MIX_HI=1 needs the WMMA HC read (CRUCIBLE_HC_WMMA unset, 1 or 2)" : nullptr;
}

}  // namespace crucible
