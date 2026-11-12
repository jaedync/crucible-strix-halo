// Partial last-layer pruning (idea from Gufo, gufo-org/gufo, MIT). CPU-visible, so the rule has a table test.
#pragma once
#include <algorithm>

namespace crucible {
// The first row of an R-row chunk whose last-layer output a reader needs: the head reads R-1 when predict is
// set, and the MTP warm-up reads rows from keep. It rounds down to a multiple of align; 0 (no pruning) when
// fewer than 9 rows would remain, since the bulk kernels need more than 8 rows.
constexpr int PruneFirstRow(int R, bool predict, int keep, int align) {
  const int needed = std::min(std::max(keep, 0), predict ? R - 1 : R);
  const int first = needed / align * align;
  return R - first > 8 ? first : 0;
}

// Rows of one piece of a pruned chunk that get only the cache kernels: the rows below `first`, rounded down to a
// multiple of 16. A chunk is cut into pieces at a Pre cut (PrefillPieces), and a cut is a multiple of the 4-row QSA
// tile rows, not of 16, so `first - offset` can be 4, 8 or 12 above a multiple of 16. The QSA prefill launches take a
// skip that is a multiple of 16 (prefill_qsa.h, prefill_qsa_union.h). The rows below `first` are dead, so the extra
// dead rows [skip, first) change no result.
// The default union path groups rows in 4-row tiles from the part start, and a piece start is a multiple of 4. So a
// 16-aligned skip keeps every kept row in its cold 4-row tile, and the arithmetic is the same. The per-row flash does
// not hold that: when the part ends at or before row 2,048 it groups rows in 16-row dense blocks, and above that it
// runs one row per block (prefill_qsa.h LaunchQsaFlashSplit). A piece that starts off the 16-row grid can then group
// its rows differently from the cold run (CRUCIBLE_PF_QSA_ROWS=1), so PreCutReady keeps the cut off with that switch
// on. The union/per-row choice is part of PrefillPreCutValid, which rejects a cut whose first piece would run the
// union flash while the whole part runs the per-row flash. skip == rows: no output row of the piece is read (cache
// kernels only).
constexpr int PrunePieceSkip(int first, int offset, int rows) {
  if (first >= offset + rows) return rows;  // the whole piece is below first
  const int skip = first > offset ? first - offset : 0;
  return skip / 16 * 16;
}
}  // namespace crucible
