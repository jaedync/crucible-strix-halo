// Dense fp16 rows of chosen output blocks of an EXL3 linear (the low-bit draft head of runtime/draft_quant.h
// quantizes them at load). The EXL3 transforms stay in engine/exl3/ (tests/test_format_seam.py).
#pragma once

#include <cstddef>
#include <cstdint>

#include "exl3.h"
#include "front_order.h"

namespace crucible {

// The fp16 rows of `count` 128-column blocks (column block IDs `blocks`) of an EXL3 linear (I inputs, O outputs,
// tiles in engine order or front order front_s), output Hadamard and svh folded in, as DraftDensePackKernel
// (runtime/draft_dense.h) for any integer K: dense[(i * 128 + c) * I + input], i the place in `blocks`. With the
// input transform of x (LinInKernel), dense row j times xh is output column j before rounding. Grid (I / 16, count).
template <int K>
__global__ void __launch_bounds__(256) Exl3DenseRowsKernel(const uint32_t* tiles, const __half* svh, const int* blocks,
                                                           int I, int O, int front_s, __half* dense) {
  constexpr int T = 8 * K;
  __shared__ __half tile[16][128];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int a = blockIdx.x, group = blocks[blockIdx.y], b = group * 8 + wave;
  const LaneWindow<K> win(lane);
  const uint32_t* src = tiles + (front_s ? FrontTile(b, a, I / 16, O / 16, front_s) : (size_t)b * (I / 16) + a) * T;
  __half2 p[4];
  DecodeLane<K, kMulMode>(src[win.d0], src[(win.d0 + 1) % T], src[(win.d0 + 2) % T], win.shift, p);
  const int r = 2 * (lane & 3), c = wave * 16 + lane / 4;
  tile[r][c] = __low2half(p[0]);
  tile[r + 1][c] = __high2half(p[0]);
  tile[r + 8][c] = __low2half(p[1]);
  tile[r + 9][c] = __high2half(p[1]);
  tile[r][c + 8] = __low2half(p[2]);
  tile[r + 1][c + 8] = __high2half(p[2]);
  tile[r + 8][c + 8] = __low2half(p[3]);
  tile[r + 9][c + 8] = __high2half(p[3]);
  __syncthreads();
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = wave + half * 8;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half2float(tile[row][lane * 4 + j]);
    Wht128(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int column = lane * 4 + j;
      dense[((size_t)blockIdx.y * 128 + column) * I + a * 16 + row] =
          __float2half(v[j] * kHadNorm * __half2float(svh[group * 128 + column]));
    }
  }
}

}  // namespace crucible
