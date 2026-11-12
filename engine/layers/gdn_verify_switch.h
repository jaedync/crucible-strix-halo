// Resolution of CRUCIBLE_GDN_VERIFY, the smallest row count that runs GdnVerifyKernel: 0 (default) keeps GdnKernel,
// 1..8 run the verify core for forwards and packed passes of that many rows or more (gdn_verify.h).
// A pure function so a CPU test can check it (tests/gdn_verify_switch_test.cpp).
#pragma once
#include <cstring>

namespace crucible {
inline constexpr int kGdnVerifyOff = 0;
inline constexpr int kGdnVerifyMaxRows = 8;  // gdn_verify.h static_asserts that this is kMaxRows

// nullptr (unset) and "" give off; "0".."8" give their value; any other text gives -1 (the caller stops).
inline int ResolveGdnVerify(const char* v) {
  if (!v || !*v) return kGdnVerifyOff;
  if (std::strlen(v) == 1 && v[0] >= '0' && v[0] <= '0' + kGdnVerifyMaxRows) return v[0] - '0';
  return -1;
}

// min_rows is the resolved value. A forward of `rows` rows runs GdnVerifyKernel when it reaches the threshold.
inline constexpr bool GdnVerifyUses(int min_rows, int rows) { return min_rows > 0 && rows >= min_rows; }
}  // namespace crucible
