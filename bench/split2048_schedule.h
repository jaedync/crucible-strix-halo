// The chunk schedule of a cold batch-serve prompt, for the split2048 probe's Pre-cut mode.
// It is BatchSession::Prefill (engine/serve/batch_session.h) for a request with no capture point and no MTP warm-up
// start before its end: PrefillLaunchRows with stop = the Pre point, then the clamp at the Pre point. Fix B off gives
// `pre` 0. tests/split2048_schedule_test.cpp checks both against BatchSession on the fake engine. Host code only.
#pragma once
#include "../engine/model/flashnext/prefill_launch.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace split2048 {
// BatchSession::PreEndPoint for a cold prompt on the default schedule (`step`-row grid of `core`-row parts).
inline size_t PrePoint(size_t length, int step, int core, size_t min_tokens, int max_rows) {
  if (!min_tokens || length < min_tokens || step <= core || step % core) return 0;
  const size_t point = (length - 1) / size_t(step) * size_t(step) + size_t(core);
  return point < length && length - point > size_t(max_rows) ? point : 0;
}

// The rows of each bulk launch of a `length`-row prompt.
inline std::vector<int> ServeSchedule(size_t length, int step, int launch, size_t pre) {
  std::vector<int> rows;
  for (size_t at = 0; at < length;) {
    int count = int(std::min<size_t>(size_t(step), length - at));
    if (launch > step) count = crucible::PrefillLaunchRows(at, length, step, launch, pre > at ? pre : SIZE_MAX);
    if (pre > at && pre < at + size_t(count)) count = int(pre - at);
    rows.push_back(count);
    at += size_t(count);
  }
  return rows;
}

// Whether `split` is `whole` with launch ends added at grid chunk ends and exactly one new end at `pre`: the probe
// then compares one schedule that differs from the unsplit one only by fix B's cut (and the merges it stops).
inline bool CutOnlyAt(const std::vector<int>& whole, const std::vector<int>& split, size_t pre, int step = 4096) {
  if (!pre) return false;
  std::vector<size_t> a, b;
  size_t at = 0;
  for (const int rows : whole) a.push_back(at += size_t(rows));
  at = 0;
  for (const int rows : split) b.push_back(at += size_t(rows));
  if (a.empty() || b.empty() || a.back() != b.back() || !std::binary_search(b.begin(), b.end(), pre) ||
      std::binary_search(a.begin(), a.end(), pre))
    return false;
  // Every unsplit end stays; every new end other than `pre` is a grid chunk end.
  for (const size_t end : a)
    if (!std::binary_search(b.begin(), b.end(), end)) return false;
  for (const size_t end : b)
    if (end != pre && !std::binary_search(a.begin(), a.end(), end) && end % size_t(step)) return false;
  return true;
}
}  // namespace split2048
