// The first last-layer row that a pruned prefill chunk must compute (PruneFirstRow), and the skip that one piece of
// the chunk gets (the QSA prefill kernels).
#include "../engine/runtime/prefill_prune.h"
#include "../engine/model/flashnext/prefill_launch.h"
#include <algorithm>
#include <cassert>
#include <cstdio>

using crucible::PruneFirstRow;

// The skip of one piece of a pruned part (Engine::PrefillBlock, prefill_engine.inc): rows [0, skip) of the piece get
// only the cache kernels, and skip == rows means that the engine calls the cache kernels only (no skip goes to a QSA
// prefill launch). Before the fix the engine computed std::clamp(first - offset, 0, rows) here, and a cut that was a
// multiple of 4 but not of 16 gave a skip of 4, 8 or 12 past a multiple of 16.
int EngineSkip(int first, int offset, int rows) { return crucible::PrunePieceSkip(first, offset, rows); }

// The invariant that prefill_qsa.h LaunchQsaPrefill and prefill_qsa_union.h LaunchQsaPrefillUnion check, and that keeps
// a kept row's tile, block and arithmetic: every skip that a launch gets is a multiple of 16, below R.
void CheckPieces(int rows, int cut, bool predict, int keep) {
  const int first = PruneFirstRow(rows, predict, keep, 128);
  for (const auto& piece : crucible::PrefillPieces(rows, cut)) {
    const int skip = EngineSkip(first, piece.offset, piece.rows);
    if (skip == piece.rows) continue;  // the cache-only branch: it passes no skip to a QSA prefill launch
    if (skip % 16 || skip < 0 || skip >= piece.rows)
      std::fprintf(stderr, "rows %d cut %d: piece offset %d rows %d, first %d: skip %d is not a multiple of 16 below R\n",
                   rows, cut, piece.offset, piece.rows, first, skip);
    assert(skip % 16 == 0 && skip >= 0 && skip < piece.rows);
    // A live row is never skipped: the kept rows start at the first row that a reader needs (or the piece runs whole).
    assert(skip == 0 || piece.offset + skip <= first);
  }
}

void SkipOfEveryPiece() {
  // The test machine night of Oct 26 21:04: a 1,021-row chat prompt (no spec, so keep = rows), a header point at 236 rows (a
  // multiple of 4, not of 16), cut inside the one launch. first = 896 and the piece after the cut got skip 660.
  assert(PruneFirstRow(1021, false, 1021, 128) == 896);
  const auto pieces = crucible::PrefillPieces(1021, 236);
  assert(pieces.size() == 2 && pieces[0].offset == 0 && pieces[1].offset == 236 && pieces[1].rows == 785);
  CheckPieces(1021, 236, false, 1021);
  // The margin point 64 rows before the end of a 1,036-row prompt: first 1,024 (its head row), the cut at 972, so the
  // second piece (64 rows) got skip 52 with the old rule (52 % 16 = 4). Trunk's lever-2 point cannot do this: it is at
  // most 12 rows before the launch end (tests/prefill_precut_test.cpp).
  assert(std::clamp(1024 - 972, 0, 64) == 52 && 52 % 16 == 4);
  assert(crucible::PrunePieceSkip(1024, 972, 1036) == 48);
  CheckPieces(1036, 972, false, 1036);
  // A skip that is already a multiple of 16 is untouched: every launch without a cut has piece offsets that are
  // multiples of the core rows, so the fix cannot change the bits or the work of the default path.
  for (int first = 0; first <= 4096; first += 128)
    for (const auto& piece : crucible::PrefillPieces(8192, 0))
      assert(crucible::PrunePieceSkip(first, piece.offset, piece.rows) == std::clamp(first - piece.offset, 0, piece.rows));
  // Every 4-aligned cut of a pruned chunk: a Pre cut (PrefillPreCutValid) is a multiple of 4, not of 16, and it can
  // start the second piece at an off-16 row.
  for (const int rows : {9, 45, 137, 1021, 2049, 4096})
    for (const bool predict : {false, true})
      for (const int keep : {0, 1, 128, rows / 2, rows - 1, rows})
        for (int cut = 0; cut <= rows; cut += 4) CheckPieces(rows, cut, predict, keep);
}

int main() {
  struct Case { int R; bool predict; int keep; int align; int first; };
  constexpr Case cases[] = {
    {4096, false, -5, 128, 0},       // keep < 0: every row is read
    {4096, false, 5000, 128, 0},     // keep > R without predict: no row is read, but fewer than 9 rows remain
    {4096, true, 5000, 128, 3968},   // keep > R with predict: the head row R-1 rounds down to 3968
    {1, true, 1, 128, 0},            // predict with R = 1
    {136, true, 136, 128, 0},        // R - first = 8: no pruning
    {137, true, 137, 128, 128},      // R - first = 9: prune from 128
    {4096, false, 3840, 128, 3840},  // an exact multiple of 128
    {4096, false, 3841, 128, 3840},  // one past it rounds down, never up
    {4096, false, 3839, 128, 3712},
    {4000, true, 3744, 128, 3712},   // the MTP warm-up of the 4,000-token proof
    {2500, true, 2500, 128, 2432},   // spec 0: only the head reads
  };
  for (const Case& c : cases) assert(PruneFirstRow(c.R, c.predict, c.keep, c.align) == c.first);
  static_assert(PruneFirstRow(4096, true, 3840, 128) == 3840);
  SkipOfEveryPiece();
}
