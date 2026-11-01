// Trunk's lever-2 Pre cut with CRUCIBLE_EXACT_RESUME=1 against the QSA prefill skip rule. The rule (prefill_qsa.h
// LaunchQsaPrefill, prefill_qsa_union.h LaunchQsaPrefillUnion): a skip is a multiple of 16, or the piece runs the
// cache kernels only (skip == its rows). This test uses the skip rule that the engine and trunk had before the
// sp-align fix (std::clamp(first - offset, 0, rows)), so it answers for trunk, and it must stay true on trunk.
//
// Why the lever-2 cut is safe: the point is at most 12 rows before the prompt end, so the launch that holds it ends at
// most 12 rows after the cut row, and first (a multiple of 128, and at most R - 9) is at most cut + 3. skip =
// first - cut is a multiple of the 4-row QSA tile (first is a multiple of 4, the cut is a multiple of the tile rows) and
// at most 3, so skip = 0. The sweep also covers a launch that ends before the cut's chunk (a capture or MTP stop),
// every keep value, both predict values, and 4- and 2,048-aligned launch starts.
#include "../engine/runtime/prefill_prune.h"
#include "../engine/layers/kernels/qsa_union_lds.h"
#include "../engine/model/flashnext/prefill_launch.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

namespace {
constexpr size_t kStep = 4096;  // Engine::PrefillRows of the bulk path (the night's --prefill 4096)
constexpr int kTile = crucible::kQsaUnionTileRows;
constexpr int kAlign = 128;     // kPrefillPruneAlign
constexpr size_t kBlockSteps[] = {512, 8192, 32768, 65536};  // PrefillPreCutValid

// BatchSession::PreEndPoint, the exact_resume_ branch (engine/serve/batch_session.h). Keep in sync with it.
size_t PreEndPoint(size_t length, size_t cursor) {
  constexpr size_t kMaxRows = 8;
  if (length <= kMaxRows) return 0;
  size_t point = (length - kMaxRows - 1) / kTile * kTile;
  if (point <= cursor) return 0;
  const size_t chunk = cursor + (point - cursor) / kStep * kStep;
  if (point - chunk <= kMaxRows) point = chunk;
  return point > cursor + kMaxRows ? point : 0;
}

// crucible::PrefillPreCutValid (engine/model/flashnext/prefill_launch.h), with its engine values.
bool PreCutValid(size_t start, int rows, int cut) {
  if (cut < 16 || cut >= rows || cut % kTile || start % kTile) return false;
  const int part = cut / crucible::kPrefillLaunchCoreRows * crucible::kPrefillLaunchCoreRows;
  if (cut == part) return true;
  const int end = std::min(rows, part + crucible::kPrefillLaunchCoreRows);
  if (cut - part <= crucible::kPrefillJoinSmallRows || end - cut <= crucible::kPrefillJoinSmallRows) return false;
  for (const size_t blocks : kBlockSteps)
    if ((start + size_t(cut)) / 4 <= blocks && (start + size_t(end)) / 4 > blocks) return false;
  return true;
}

// The skip rule of the engine before the sp-align fix. Trunk has this rule today: do not replace it with
// crucible::PrunePieceSkip here, or this test stops answering for trunk.
int PieceSkip(int first, int offset, int rows) { return std::clamp(first - offset, 0, rows); }

struct Launch { size_t length, cursor, end; int keep; bool predict; };
size_t cuts = 0, skips = 0, failures = 0;
size_t cap_changed = 0;  // cuts at or below 132,050 that the union rule changed (must stay 0)

void Check(const Launch& l) {
  const size_t point = PreEndPoint(l.length, l.cursor);
  if (point == 0 || point >= l.end) return;  // no cut in this launch
  const int rows = int(l.end - l.cursor), cut = int(point - l.cursor);
  // Review Oct 27 round 3: the union LDS rule must not change a cut at or below the night's longest prompt.
  if (l.end <= 132050 && crucible::PrefillPreCutValid(l.cursor, rows, cut) != PreCutValid(l.cursor, rows, cut))
    ++cap_changed;
  if (!PreCutValid(l.cursor, rows, cut)) return;  // not a cut: no skip
  ++cuts;
  const int first = crucible::PruneFirstRow(rows, l.predict, l.keep, kAlign);
  for (const auto& piece : crucible::PrefillPieces(rows, cut)) {
    const int skip = PieceSkip(first, piece.offset, piece.rows);
    if (skip == piece.rows) continue;  // the cache-only branch: no skip goes to a QSA prefill launch
    ++skips;
    if (skip < 0 || skip >= piece.rows || skip % 16) {
      if (++failures <= 5)
        std::fprintf(stderr, "UNSAFE length %zu cursor %zu end %zu keep %d predict %d: point %zu first %d -> "
                             "piece offset %d rows %d skip %d\n", l.length, l.cursor, l.end, l.keep,
                     int(l.predict), point, first, piece.offset, piece.rows, skip);
    }
  }
}

// The union LDS budget (review, Oct 27 round 3): a part that ends past the budget runs the per-row flash, but a piece
// of it that ends below the budget would run the union flash, which is another arithmetic. PrefillPreCutValid rejects
// that cut; the whole part and piece 2 end together, so they always agree.
void CapCase() {
  // 196,000 + 80 fits the union; the whole 128-row part ends at 196,128 and does not.
  assert(crucible::prefill::QsaUnionFits(196000, 80) && !crucible::prefill::QsaUnionFits(196000, 128));
  assert(!crucible::PrefillPreCutValid(196000, 128, 80));
  // Both pieces and the part past the budget: the per-row flash for all, so the cut stays.
  assert(!crucible::prefill::QsaUnionFits(196052, 80) && !crucible::prefill::QsaUnionFits(196052, 128));
  assert(crucible::PrefillPreCutValid(196052, 128, 80));
  // Both below the budget: the union flash for all, so the cut stays.
  assert(crucible::prefill::QsaUnionFits(195900, 80) && crucible::prefill::QsaUnionFits(195900, 128));
  assert(crucible::PrefillPreCutValid(195900, 128, 80));
  // Across the budget, 4 rows at a time: the rule rejects a cut exactly when its first piece fits and the part does not.
  size_t rejects = 0, checks = 0;
  for (int start = 195900; start <= 196300; start += kTile) {
    for (int cut = 16; cut <= 512; cut += kTile) {
      if (cut >= 1024 || start % kTile) continue;
      const bool piece_fits = crucible::prefill::QsaUnionFits(start, cut);
      const bool part_fits = crucible::prefill::QsaUnionFits(start, 1024);
      const bool cut_valid = crucible::PrefillPreCutValid(size_t(start), 1024, cut);
      ++checks;
      if (piece_fits && !part_fits) ++rejects;
      if (piece_fits && !part_fits) assert(!cut_valid);
      else if (!piece_fits && !part_fits) assert(cut_valid);
    }
  }
  assert(checks > 1000 && rejects > 100);
}

std::vector<int> Keeps(int rows) {
  std::vector<int> keeps{0, 9, 127, 128, 129, rows / 2, rows - 9, rows - 1, rows};
  for (int& k : keeps) k = std::clamp(k, 0, rows);
  std::sort(keeps.begin(), keeps.end());
  keeps.erase(std::unique(keeps.begin(), keeps.end()), keeps.end());
  return keeps;
}

// The launches of one prompt: the natural launch ends (the cut's chunk end and the prompt end) and the stops between
// the cut and that chunk end (a capture or the MTP warm-up can end a launch there).
void SweepLength(size_t length, const std::vector<size_t>& starts) {
  std::vector<size_t> ends;
  for (const size_t cursor : {size_t{0}, size_t{4}, size_t{2048}, size_t{4096}}) {
    const size_t point = PreEndPoint(length, cursor);
    if (!point) continue;
    const size_t end = std::min(length, cursor + (point - cursor + kStep - 1) / kStep * kStep);
    if (end > point) ends.push_back(end);
    ends.push_back(length);
    for (size_t e = 1; e <= 12 && point + e <= length; ++e) ends.push_back(point + e);
  }
  std::sort(ends.begin(), ends.end());
  ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
  for (const size_t end : ends) {
    // The starts of the launch: the fixed ones, and the 4- and 2,048-aligned rows near the launch end (the cut must be
    // at least 16 rows into the launch and the pieces of its part need more than 8 rows).
    std::vector<size_t> all = starts;
    for (size_t k = 1; k <= 32; ++k) all.push_back(end - size_t(k) * kTile);
    for (size_t k = 1; k <= 8; ++k) all.push_back(end - k * 2048);
    for (const size_t start : all) {
      if (start >= end) continue;
      for (const int keep : Keeps(int(end - start)))
        for (const bool predict : {false, true}) Check({length, start, end, keep, predict});
    }
  }
}
}  // namespace

int main() {
  CapCase();
  // Every residue of length mod 4,096 (the chunk grid) and mod 128 (the prune align), on the 4,096-row grid: 4- and
  // 2,048-aligned launch starts, and every launch end a stop can make.
  std::vector<size_t> starts{0};
  for (size_t k = 1; k <= 32; ++k) starts.push_back(size_t(k) * kTile);
  for (size_t k = 1; k <= 16; ++k) starts.push_back(k * 2048);
  for (size_t length = 9; length <= 4100; ++length) SweepLength(length, starts);
  const size_t small_cuts = cuts, small_skips = skips;
  if (failures) std::fprintf(stderr, "small sweep: %zu failures\n", failures);
  assert(failures == 0 && small_cuts > 1000000);
  // Long prompts: the night's 8,192+128 launch, the P1c and P2c shapes, and the block-step crossings of
  // PrefillPreCutValid. A 2,048 multiple and a 4-grid start, and start positions near each block step.
  cuts = skips = failures = 0;
  for (const size_t length : {size_t{8900}, size_t{34816}, size_t{132050}, size_t{262144}}) {
    std::vector<size_t> long_starts{0, 4, 2048, 4096, 34816, 129936};
    for (const size_t blocks : kBlockSteps) long_starts.push_back(std::min(length - 16, blocks * 4 - 4));
    SweepLength(length, long_starts);
  }
  const size_t long_cuts = cuts;
  if (failures) std::fprintf(stderr, "long sweep: %zu failures\n", failures);
  assert(failures == 0 && long_cuts > 100);
  // The union LDS rule changes no cut at or below 132,050 (the night's longest prompt).
  assert(cap_changed == 0);
  std::printf("trunk lever-2 Pre cut: %zu cuts and %zu skips in the 4,096-row sweep, %zu and %zu on long prompts; "
              "every skip is a multiple of 16 or cache-only; union LDS rule: 0 changes at or below 132,050\n",
              small_cuts, small_skips, long_cuts, skips);
  return 0;
}
