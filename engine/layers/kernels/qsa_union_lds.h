// The LDS budget of the QSA union flash. One part's block mask and its fixed tables must fit in 64 KiB;
// a part that does not fit runs the per-row flash instead. The choice depends only on the part's end (pos0 + R), so a
// Pre cut piece must end on the same side of the budget as the whole part (PrefillPreCutValid).
#pragma once

namespace crucible::prefill {
constexpr int kUnionStaticLds = (16 * 256 + 2 * 16) * (int)sizeof(int), kUnionMaxLds = 65536;
// True when a part of `R` rows from `pos0` runs the union flash; false runs the per-row flash.
inline bool QsaUnionFits(int pos0, int R) {
  return (pos0 + R - 1) / 4 + 1 + kUnionStaticLds <= kUnionMaxLds;
}
}  // namespace crucible::prefill
