// Host reorders of the HC weights into the WMMA fragment orders (host only,
// for the loader and the benches). The device copies are HcFragFnKernel and
// HcFragUpKernel in prefill_hc.h.
#pragma once

#include <cstddef>

#include "../model/flashnext/geometry.h"

namespace crucible {

// fn row-major [M][kWide] to the compact fragment order of FB 2: row tile
// nt at nt * 16 * kWide, then [stream][kHidden / 16][TR][16] with TR =
// min(16, M - 16 * nt). Full tiles equal the HcFragFnKernel layout.
template <class T>
inline void HcFnCompactOrder(const T* fn, int M, T* out) {
  for (int m = 0; m < M; ++m) {
    const int nt = m / 16, tr = M - nt * 16 < 16 ? M - nt * 16 : 16;
    for (int c = 0; c < kWide; ++c) {
      const int s = c / kHidden, k = c % kHidden;
      out[(size_t)nt * 16 * kWide + (((size_t)s * (kHidden / 16) + k / 16) * tr + m % 16) * 16 + k % 16] =
          fn[(size_t)m * kWide + c];
    }
  }
}

// Host copies of HcFragFnKernel (fn rows padded to whole tiles with zeros;
// `out` holds (M + 15) / 16 * 16 * kWide halves) and HcFragUpKernel, for the
// loader. The same index maps as the device kernels in prefill_hc.h.
template <class T>
inline void HcFnFragOrder(const T* fn, int M, T* out) {
  const int rows = (M + 15) / 16 * 16;
  for (int row = 0; row < rows; ++row)
    for (int c = 0; c < kWide; ++c) {
      const int s = c / kHidden, k = c % kHidden;
      out[(((size_t)(row >> 4) * kStreams + s) * (kHidden / 16) + k / 16) * 256 + (row & 15) * 16 + (k & 15)] =
          row < M ? fn[(size_t)row * kWide + c] : T(0.f);
    }
}
template <class T>
inline void HcUpFragOrder(const T* up, T* out) {
  for (size_t i = 0; i < (size_t)kWide * kRank; ++i) {
    const int row = (int)(i / kRank), k = (int)(i % kRank);
    out[((size_t)(row >> 4) * (kRank / 16) + k / 16) * 256 + (row & 15) * 16 + (k & 15)] = up[i];
  }
}

}  // namespace crucible
