// CPU checks of the single-session verify span (spec_span.h): the CRUCIBLE_SPEC_SPAN
// resolution, the GDN pending-row stride, the request depth map and the side rows of the PLE ring.
#include "../engine/runtime/spec_span.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace crucible;
int main() {
  static_assert(kMaxRows == 8 && kSpecSpanMax == 13 && kSpecSpanMax <= kPackedRows);
  // Unset, empty and "8" keep the trunk path; 9..13 turn the span on; any other text is an error.
  assert(ResolveSpecSpan(nullptr) == kMaxRows);
  assert(ResolveSpecSpan("") == kMaxRows);
  assert(ResolveSpecSpan("8") == kMaxRows);
  for (int n = 9; n <= 13; ++n) assert(ResolveSpecSpan(std::to_string(n).c_str()) == n);
  for (const char* bad : {"0", "1", "7", "14", "16", "17", "-9", "+9", " 9", "9 ", "09", "9x", "on", "1.0"})
    assert(ResolveSpecSpan(bad) == -1);
  // The GDN pending rows of one head: 8 on the trunk path (the default stride), 16 with a span.
  assert(SpecPendRows(8) == 8);
  for (int n = 9; n <= kSpecSpanMax; ++n) assert(SpecPendRows(n) == 16 && SpecPendRows(n) >= n - 1);
  // A request at the API maximum (7) drafts up to span - 1; a smaller spec stays as it is; span 8 changes nothing.
  for (int spec = 0; spec <= 7; ++spec) assert(SpanDraftDepth(spec, 8) == spec);
  assert(SpanDraftDepth(7, 13) == 12 && SpanDraftDepth(7, 9) == 8);
  assert(SpanDraftDepth(6, 13) == 6 && SpanDraftDepth(0, 13) == 0 && SpanDraftDepth(1, 13) == 1);
  // A span of span rows fits; one row more does not. Several spans keep the packed limit of kMaxRows each.
  assert(SpanRowsFit(1, 13, 1) && SpanRowsFit(13, 13, 1) && !SpanRowsFit(14, 13, 1));
  assert(SpanRowsFit(8, 13, 2) && !SpanRowsFit(9, 13, 2) && !SpanRowsFit(9, 8, 1));
  // PLE side rows: rows 0..7 of a span write the 16-slot ring; rows from 8 on go to the side buffer until the
  // host keeps them, so a rejected row never overwrites one of the nine old rows the next forward reads.
  assert(!PleSideRow(0) && !PleSideRow(7) && PleSideRow(8) && PleSideRow(12));
  // Rows to copy from the side buffer into the ring when the sequence keeps `kept` rows of a span of `rows`.
  assert(PleSideKeep(13, 1) == 0 && PleSideKeep(13, 8) == 0 && PleSideKeep(13, 9) == 1);
  assert(PleSideKeep(13, 13) == 5 && PleSideKeep(9, 9) == 1 && PleSideKeep(8, 8) == 0);
  // Pending MTP pairs past `keep` go to the MTP in passes of at most kMaxRows rows, oldest first (BatchSpecDecoder::Trim).
  assert(SpanTrimPasses(8, 8).empty() && SpanTrimPasses(1, 8).empty() && SpanTrimPasses(13, 13).empty());
  assert(SpanTrimPasses(9, 1) == std::vector<int>{8});    // the trunk Flush: nine pairs, eight flushed
  assert(SpanTrimPasses(13, 8) == std::vector<int>{5});   // a 13-row span: five pairs before drafting
  assert(SpanTrimPasses(14, 6) == std::vector<int>{8});   // Flush after a packed row: 14 pairs, 6 kept
  assert((SpanTrimPasses(17, 1) == std::vector<int>{8, 8}));
  assert(SpanTrimPasses(3, 0).empty());                   // keep 0 is not a valid request
  // A span longer than 13 rows would write a GDN or QSA ring slot that the next forward still reads.
  static_assert(kSpecSpanMax + 3 <= 16, "the 16-slot GDN qkv and QSA raw rings keep 3 old rows");
  std::cout << "spec_span_test: ok\n";
  return 0;
}
