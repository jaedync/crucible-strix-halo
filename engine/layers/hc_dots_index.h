// The bulk prefill HC dots buffer (prefill_hc.h: the dots kernels store it, PrefillHcMidPackKernel reads it).
// Stream-major [kStreams][R][M]: a dots block owns one stream, a run of rows and every column, so it stores one
// contiguous run. The old order [R][M][kStreams] made the four stream blocks of a row group each store 4 bytes of
// every 16; the L2 sent partly written lines to DRAM more than once (2.3 times the buffer).
// The decode HC kernels keep their own [R][M][kStreams] order in the same buffer.
// Host code can include this file (tests/hc_dots_index_test.cpp).
#pragma once
#include <cstddef>

#if defined(__HIPCC__) || defined(__HIP__)
#define HC_DOTS_HOST_DEVICE __host__ __device__
#else
#define HC_DOTS_HOST_DEVICE
#endif

namespace crucible {
constexpr int kHcDotsStreams = 4;  // kStreams (geometry.h)

// Index of dot (stream s, row r, column m) in a buffer of R rows of M columns.
HC_DOTS_HOST_DEVICE constexpr size_t HcDotsAt(int s, int r, int m, int R, int M) {
  return ((size_t)s * R + r) * M + m;
}
}  // namespace crucible
