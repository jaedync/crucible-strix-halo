// A bounded auxiliary prefix changes proposals, not the target-model context.
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace crucible {
inline size_t DraftWarmupStart(size_t prompt, int limit) {
  return limit > 0 && prompt > (size_t)limit ? prompt - limit : 0;
}
inline int DraftWarmupRow(size_t offset, int rows, size_t first) {
  return first > offset ? (int)std::min((size_t)rows, first - offset) : 0;
}
// the positions of a batch slot's MTP cache. A bounded window starts the MTP at position 0 on each request
// (window rows of warm-up), so it needs the window, one client output and draft slack: the clients' largest
// max_tokens is 32,768 (Pi), and 256 rows cover the catch-up and draft rows of the last steps. Window 0 (the whole
// prompt, earlier changes) keeps MTP positions equal to the trunk rows and needs the context. A request that passes
// the bound restarts its MTP (bounds.h MtpRestart): drafts change, the output does not.
inline constexpr int kMtpSlotOutputRows = 32768 + 256;
inline int MtpSlotRows(int context, int window) {
  // Y2: a YaRN slot holds up to 1,048,576 positions (rope_mode.h kRopeMaxPositions); the window stays at 262,144.
  if (context < 1 || context > 1048576 || window < 0 || window > 262144)
    throw std::invalid_argument("MTP slot rows need a context in [1,1048576] and a window in [0,262144]");
  return window == 0 ? context : std::min(context, window + std::min(kMtpSlotOutputRows, 262144 - window));
}
}  // namespace crucible
