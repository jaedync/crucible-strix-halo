// Draft rows that cannot commit a token. Verification emits the accepted drafts and one more token, and a request
// ends at its first stop ID or at its token limit, so a draft after a stop ID, or a draft that would need a token past
// the limit, only costs a verify row. These helpers remove such rows. They never add a row and never change a draft
// that stays, so the committed tokens do not change.
#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <span>
#include <vector>

namespace crucible {

// The most drafts a step may propose when `remaining` tokens are still allowed: the step commits up to
// drafts + 1 tokens. Zero when one token or none remains.
inline int DraftRoom(size_t remaining) {
  return remaining > 1 ? int(std::min<size_t>(remaining - 1, size_t(INT_MAX))) : 0;
}

inline bool IsStopToken(std::span<const int> stops, int token) {
  return std::find(stops.begin(), stops.end(), token) != stops.end();
}

// Cuts `drafts` after its first stop ID (the stop ID stays: it may be accepted and ends the request) and to `room`
// tokens.
inline void TrimDrafts(std::vector<int>& drafts, std::span<const int> stops, int room) {
  size_t keep = size_t(std::max(room, 0));
  for (size_t i = 0; i < drafts.size() && i < keep; ++i)
    if (IsStopToken(stops, drafts[i])) { keep = i + 1; break; }
  if (drafts.size() > keep) drafts.resize(keep);
}

// Whether a chain that ends in `last` may draft one more token.
inline bool ChainMayGrow(std::span<const int> drafts, std::span<const int> stops, int room) {
  return int(drafts.size()) < room && !(drafts.size() && IsStopToken(stops, drafts.back()));
}

}  // namespace crucible
