// Device half of the opt-in abliteration (ablation.h has the host half and
// the CPU reference AblateWriteRef, which this kernel matches bit for bit).
#pragma once

#include "ablation.h"
#include "../gfx1151/common.h"

namespace crucible {

// One block per row: c = dot(y, a) in a fixed order, then the
// hyper-connection write-back H[s] += post[s] * (y - c * b). The order depends
// only on the row, never on R, so ablated runs keep row-count exactness.
// ab: a, then b, each [kHidden]. y: [R][kHidden]. H rows have stride ldh.
__global__ void __launch_bounds__(kAblateThreads) AblateWriteKernel(const float* __restrict__ y,
                                                                    const float* __restrict__ ab,
                                                                    const float* __restrict__ post,
                                                                    float* __restrict__ H, int ldh,
                                                                    uint32_t stock = 0) {
  const int r = blockIdx.x, t = threadIdx.x;
  const float* yr = y + (size_t)r * kHidden;
  // CRUCIBLE_ABLATE_MODE=request: bit r marks a stock row of a mixed step (AblateWriteStockRef). Rows >= 32 are never
  // stock here: a pass of more than 32 rows is one stream's prefill, all stock (no ablated launch) or all ablated.
  if (r < 32 && (stock >> r & 1u)) {
    float g[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) g[s] = post[r * kStreams + s];
    float* hr = H + (size_t)r * ldh;
#pragma unroll
    for (int k = 0; k < kAblatePer; ++k) {
      const int d = t + kAblateThreads * k;
#pragma unroll
      for (int s = 0; s < kStreams; ++s) hr[s * kHidden + d] = __fmaf_rn(g[s], yr[d], hr[s * kHidden + d]);
    }
    return;
  }
  const float* a = ab;
  const float* b = ab + kHidden;
  __shared__ float part[kAblateThreads];
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < kAblatePer; ++k) acc = __fmaf_rn(yr[t + kAblateThreads * k], a[t + kAblateThreads * k], acc);
  part[t] = acc;
  __syncthreads();
  for (int s = kAblateThreads / 2; s > 0; s >>= 1) {
    if (t < s) part[t] = __fadd_rn(part[t], part[t + s]);
    __syncthreads();
  }
  const float c = part[0];
  float g[kStreams];
#pragma unroll
  for (int s = 0; s < kStreams; ++s) g[s] = post[r * kStreams + s];
  float* hr = H + (size_t)r * ldh;
#pragma unroll
  for (int k = 0; k < kAblatePer; ++k) {
    const int d = t + kAblateThreads * k;
    const float v = __fmaf_rn(-c, b[d], yr[d]);
#pragma unroll
    for (int s = 0; s < kStreams; ++s) hr[s * kHidden + d] = __fmaf_rn(g[s], v, hr[s * kHidden + d]);
  }
}

}  // namespace crucible
