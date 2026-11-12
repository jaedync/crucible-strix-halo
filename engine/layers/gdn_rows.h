// The GDN core launch for a.R = 1..8 rows (decode, verify forwards and short
// prefill chunks). GdnRowsKernel, a multi-row variant behind
// the GDN rows switch, was bitwise equal but gave no measurable gain
// an earlier change deleted it (git history has it).
#pragma once

#include "gdn_verify.h"
#include "ops.h"

namespace crucible {

// GdnKernel with a pending-row stride of 16 rows per head: the sequences of an engine with a single-session verify
// span (CRUCIBLE_SPEC_SPAN, runtime/spec_span.h). The engine refuses CRUCIBLE_GDN_VERIFY with a span.
// Args: GdnSpanArgs (one sequence, grid 48) or GdnSpanPackedArgs (one grid row per span).
template <class Args>
inline void LaunchGdnSpan(const Args& a, dim3 grid, int state_cache, hipStream_t stream = 0) {
  static_assert(kGdnPendRows<Args> == kPackedRows);
  if (state_cache == 6) GdnKernel<4, true, Args><<<grid, 256, 0, stream>>>(a);
  else if (state_cache == 4) GdnKernel<4, false, Args><<<grid, 256, 0, stream>>>(a);
  else GdnKernel<0, false, Args><<<grid, 256, 0, stream>>>(a);
}

// state_cache as LayerGroups. pend_rows: the pending rows per head of the sequence (kMaxRows, or kPackedRows with a
// verify span).
// CRUCIBLE_GDN_VERIFY=N runs GdnVerifyKernel (bitwise equal) at N or more rows.
inline void LaunchGdnCore(const GdnArgs& a, int state_cache, hipStream_t stream = 0, int pend_rows = kMaxRows) {
  if (pend_rows != kMaxRows) LaunchGdnSpan(GdnSpanArgs{a}, dim3(48), state_cache, stream);
  else if (GdnVerifyOn(a.R)) LaunchGdnVerify(a, 1, state_cache, stream);
  else if (state_cache == 6) GdnKernel<4, true><<<48, 256, 0, stream>>>(a);
  else if (state_cache == 4) GdnKernel<4><<<48, 256, 0, stream>>>(a);
  else GdnKernel<0><<<48, 256, 0, stream>>>(a);
}

}  // namespace crucible
