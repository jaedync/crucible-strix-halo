// Single-session verify span. CRUCIBLE_SPEC_SPAN=N (9..13) lets one session verify up
// to N rows in one pass (draft depth N - 1); unset or 8 keeps the trunk limit of kMaxRows rows. Exact: only the
// verify width changes, and every row's result is the row's own (the packed passes of 9..16 rows prove the trunk
// layers). The pure rules are here so a CPU test checks them (tests/spec_span_test.cpp).
//
// Why 13 and not 16: the GDN qkv ring and the QSA raw-key ring have 16 slots and the next forward reads 3 old rows,
// so a span may write at most 13 new slots. The PLE ring reads 9 old rows, so rows from 8 on of a span go to a side
// buffer and enter the ring only when the host keeps them (Engine::Truncate). Spans of 14..16 rows need the same side
// rows for the GDN and QSA rings (next step in the findings file).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../model/flashnext/geometry.h"
#include "../switches.h"

namespace crucible {
inline constexpr int kSpecSpanMax = 13;
inline constexpr int kPleSideRows = kSpecSpanMax - kMaxRows;  // span rows past the PLE ring's safe 8

// nullptr, "" and "8": kMaxRows (off). "9".."13": that span. Anything else: -1 (the caller stops).
inline int ResolveSpecSpan(const char* v) {
  if (!v || !*v) return kMaxRows;
  const std::string s(v);
  for (int n = kMaxRows; n <= kSpecSpanMax; ++n)
    if (s == std::to_string(n)) return n;
  return -1;
}
// Pending GDN rows per head (the stride of SequenceState::gpend). The trunk stride is kMaxRows.
inline constexpr int SpecPendRows(int span) { return span > kMaxRows ? kPackedRows : kMaxRows; }
// The draft depth of a request: the API maximum (7) means "as deep as the span allows".
inline constexpr int SpanDraftDepth(int spec, int span) { return span > kMaxRows && spec == kMaxRows - 1 ? span - 1 : spec; }
// One verify pass: a lone span may hold `span` rows; spans of a packed pass keep kMaxRows each.
inline constexpr bool SpanRowsFit(int rows, int span, int spans) { return rows <= (spans == 1 ? span : kMaxRows); }
// PLE ring: row r of a span goes to the side buffer when r >= kMaxRows.
inline constexpr bool PleSideRow(int row) { return row >= kMaxRows; }
// Side rows to copy into the ring when `kept` rows of a span of `rows` stay in the history.
inline constexpr int PleSideKeep(int rows, int kept) {
  const int k = kept < rows ? kept : rows;
  return k > kMaxRows ? k - kMaxRows : 0;
}
// MTP passes (rows each, oldest pairs first) that leave at most `keep` of `pending` pairs (keep >= 1). The MTP takes
// kMaxRows rows per call.
inline std::vector<int> SpanTrimPasses(int pending, int keep) {
  std::vector<int> passes;
  if (keep < 1) return passes;
  for (int left = pending; left > keep;) {
    const int n = left - keep < kMaxRows ? left - keep : kMaxRows;
    passes.push_back(n);
    left -= n;
  }
  return passes;
}
// The process setting: read once at the first call (the engine reads it at start), so a bad value stops there.
inline int SpecSpanRows() {
  static const int v = [] {
    const int n = ResolveSpecSpan(SwitchText("CRUCIBLE_SPEC_SPAN"));
    if (n < 0) {
      std::fprintf(stderr, "CRUCIBLE_SPEC_SPAN must be %d..%d\n", kMaxRows, kSpecSpanMax);
      std::exit(2);
    }
    return n;
  }();
  return v;
}
inline bool SpecSpanOn() { return SpecSpanRows() > kMaxRows; }
}  // namespace crucible
