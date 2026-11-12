// Exact decode expert bodies with b128 (Q4) and b64 (B64) weight loads and no LDS transpose (moe-b128). The engine runs them behind CRUCIBLE_MOE_B128 (default 12; 0 is the rollback); the bench
// (bench/kernels/moe_b128.h, verify_moe variants 110 to 119) runs the same code.
//
// - Q4 (K4, 32 dwords per tile): lane 8 g + m loads dwords 4 m .. 4 m + 3 of the current tile of column g (one b128
//   per lane: 4 columns x 128 B per wave instruction). For K4 the engine lane t decodes (dword t - 1, dword t) with
//   shift 20 (LaneWindow<4>; w2 is unused). So lane (g, m) is the engine's lanes t = 4 m .. 4 m + 3 of column g:
//   the only foreign dword, 4 m - 1 (mod 32), comes from lane (g, m - 1 mod 8) by one DPP8 move. Those 4 engine
//   lanes are one column pair (t / 4 = m) with q = 0..3, so the lane keeps the 4 per-q Dot2 chains of the engine
//   and sums them as the engine's shuffles do: (a0 + a1) + (a2 + a3). Bitwise equal by construction.
//   A wave covers 4 columns, a block of W waves 4 W columns (item_blocks 4 W).
// - B64 (K4): lane 16 g + m loads dwords 2 m, 2 m + 1 of column g (columns b and b + 8, as the engine's C2): the
//   engine lanes t = 2 m, 2 m + 1. The foreign dword 2 m - 1 comes from lane m - 1 (mod 16) by v_permlane16; the
//   q pairs (0, 1) and (2, 3) are in lanes 2 k and 2 k + 1, summed in the engine order by one shuffle.
//   Same item shape as MoeFusedKernel (8 waves, item_blocks 16).
// - The K6 shared expert keeps the engine body (MoeGemvCols; any CC gives the same per-column arithmetic).
// MODE (timing knockouts, bench only): 1 = x from registers, 2 = no decode, 3 = decode only.
#pragma once
#include "moe_rows.h"

namespace crucible {

typedef unsigned int B128u4 __attribute__((ext_vector_type(4)));
typedef unsigned int B128u2 __attribute__((ext_vector_type(2)));

// DPP8 selector: lane i of each 8-lane group reads lane (i + 7) & 7.
constexpr unsigned kDpp8Prev = 7u | 0u << 3 | 1u << 6 | 2u << 9 | 3u << 12 | 4u << 15 | 5u << 18 | 6u << 21;
// v_permlane16 selectors: lane i of each 16-lane row reads lane (i + 15) & 15.
constexpr unsigned kPerm16PrevLo = 0x6543210Fu, kPerm16PrevHi = 0xEDCBA987u;

__device__ __forceinline__ __half2 B128AsH2(uint32_t v) { return *reinterpret_cast<__half2*>(&v); }

// Q4 body: columns b .. b + nc - 1 (nc <= 4) of one item, K4 tiles, D b128 loads in flight per lane.
template <int R, int D, int MODE, int CACHE>
__device__ __forceinline__ void MoeQ4Cols(const LinJob& job, int s, int b, int nc, const uint32_t* xs) {
  constexpr int T = 32;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 3, m = lane & 7;
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  // One resource over the nc column blocks (column g starts g A tiles after column 0); lanes of a missing
  // column load out of range (zeros, no memory traffic) and store nothing.
  const uint32_t bytes = (uint32_t)(((nc - 1) * A + span) * T * 4);
  const auto rs = StreamRsrc(tiles + ((size_t)b * A + a0) * T, bytes);
  const int vo = g < nc ? (g * A * T + 4 * m) * 4 : (int)bytes;
  B128u4 ring[D];
#pragma unroll
  for (int f = 0; f < D; ++f) {
    ring[f] = __builtin_amdgcn_raw_buffer_load_b128(rs, vo, f * T * 4, CACHE);
    __builtin_amdgcn_sched_barrier(0);
  }
  float acc[4][R][2];
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[q][r][0] = acc[q][r][1] = 0.f;
  B128u4 xreg[2] = {};
  if constexpr (MODE == 1) {
    // Knockout: fixed x in registers (timing only).
    const B128u4* x4 = reinterpret_cast<const B128u4*>(xs);
    xreg[0] = x4[0], xreg[1] = x4[1];
  }
  auto step = [&](int a, int f) {
    const B128u4 w = ring[f];
    __half2 P[4][4];
    if constexpr (MODE == 2) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int k = 0; k < 4; ++k) P[q][k] = B128AsH2(w[(q + k) & 3]);
    } else {
      const uint32_t prev = (uint32_t)__builtin_amdgcn_mov_dpp8((int)w[3], kDpp8Prev);
      DecodeLane<4, kMulMode>(prev, w[0], 0u, 20, P[0]);
      DecodeLane<4, kMulMode>(w[0], w[1], 0u, 20, P[1]);
      DecodeLane<4, kMulMode>(w[1], w[2], 0u, 20, P[2]);
      DecodeLane<4, kMulMode>(w[2], w[3], 0u, 20, P[3]);
    }
    ring[f] = __builtin_amdgcn_raw_buffer_load_b128(rs, vo, (a + f + D) * T * 4, CACHE);
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (MODE == 3) {
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[q][0][k >> 1] += __low2float(P[q][k]);
      return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      B128u4 xa, xb;
      if constexpr (MODE == 1) {
        xa = xreg[0], xb = xreg[1];
      } else {
        const B128u4* xr = reinterpret_cast<const B128u4*>(xs + (r * span + a + f) * 8);
        xa = xr[0], xb = xr[1];
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // The engine lane 4 m + q: ha = x[q], hb = x[4 + q].
        const __half2 ha = B128AsH2(xa[q]), hb = B128AsH2(xb[q]);
        acc[q][r][0] = Dot2(P[q][0], ha, acc[q][r][0]);
        acc[q][r][0] = Dot2(P[q][1], hb, acc[q][r][0]);
        acc[q][r][1] = Dot2(P[q][2], ha, acc[q][r][1]);
        acc[q][r][1] = Dot2(P[q][3], hb, acc[q][r][1]);
      }
    }
  };
  // Full rounds only (span % D == 0 for the 160- and 40-tile spans): each ring slot keeps its registers.
  if (span % D != 0) __builtin_trap();
  for (int a = 0; a < span; a += D) {
#pragma unroll
    for (int f = 0; f < D; ++f) step(a, f);
  }
  if (g < nc) {
    float* out = job.partial + (size_t)s * R * job.O;
    const int col = 16 * (b + g) + m;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      // MoeGemvCols's shuffle sums, lane q = 0: (a0 + a1) + (a2 + a3).
      out[r * job.O + col] = (acc[0][r][0] + acc[1][r][0]) + (acc[2][r][0] + acc[3][r][0]);
      out[r * job.O + col + 8] = (acc[0][r][1] + acc[1][r][1]) + (acc[2][r][1] + acc[3][r][1]);
    }
  }
}

// B64 body: columns b (lanes 0..15) and b + 8 (lanes 16..31, if c2), K4 tiles, D b64 loads in flight per lane.
template <int R, int D, int CACHE>
__device__ __forceinline__ void MoeB64Cols(const LinJob& job, int s, int b, bool c2, const uint32_t* xs) {
  constexpr int T = 32;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 4, m = lane & 15;
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  const uint32_t bytes = (uint32_t)(((c2 ? 8 * A : 0) + span) * T * 4);
  const auto rs = StreamRsrc(tiles + ((size_t)b * A + a0) * T, bytes);
  const bool live = g == 0 || c2;
  const int vo = live ? (g * 8 * A * T + 2 * m) * 4 : (int)bytes;
  const int q0 = 2 * (m & 1);
  B128u2 ring[D];
#pragma unroll
  for (int f = 0; f < D; ++f) {
    ring[f] = __builtin_amdgcn_raw_buffer_load_b64(rs, vo, f * T * 4, CACHE);
    __builtin_amdgcn_sched_barrier(0);
  }
  float acc[2][R][2];
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[k][r][0] = acc[k][r][1] = 0.f;
  auto step = [&](int a, int f) {
    const B128u2 w = ring[f];
    const uint32_t prev = __builtin_amdgcn_permlane16(w[1], w[1], kPerm16PrevLo, kPerm16PrevHi, false, false);
    __half2 P[2][4];
    DecodeLane<4, kMulMode>(prev, w[0], 0u, 20, P[0]);
    DecodeLane<4, kMulMode>(w[0], w[1], 0u, 20, P[1]);
    ring[f] = __builtin_amdgcn_raw_buffer_load_b64(rs, vo, (a + f + D) * T * 4, CACHE);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t* xr = xs + (r * span + a + f) * 8;
      const B128u2 xa = *reinterpret_cast<const B128u2*>(xr + q0);
      const B128u2 xb = *reinterpret_cast<const B128u2*>(xr + 4 + q0);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        // The engine lane 2 m + k: q = q0 + k.
        const __half2 ha = B128AsH2(xa[k]), hb = B128AsH2(xb[k]);
        acc[k][r][0] = Dot2(P[k][0], ha, acc[k][r][0]);
        acc[k][r][0] = Dot2(P[k][1], hb, acc[k][r][0]);
        acc[k][r][1] = Dot2(P[k][2], ha, acc[k][r][1]);
        acc[k][r][1] = Dot2(P[k][3], hb, acc[k][r][1]);
      }
    }
  };
  if (span % D != 0) __builtin_trap();
  for (int a = 0; a < span; a += D) {
#pragma unroll
    for (int f = 0; f < D; ++f) step(a, f);
  }
  float v[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float p = acc[0][r][h] + acc[1][r][h];  // (a0 + a1) in lane 2 k, (a2 + a3) in lane 2 k + 1
      v[r][h] = p + __shfl_xor(p, 1);
    }
  if (live && (m & 1) == 0) {
    float* out = job.partial + (size_t)s * R * job.O;
    const int col = 16 * (b + 8 * g) + m / 2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      out[r * job.O + col] = v[r][0];
      out[r * job.O + col + 8] = v[r][1];
    }
  }
}

// Per-item body. BODY 0: Q4 (W waves x 4 columns); BODY 1: B64 (8 waves, columns w and w + 8, as C2).
// K6 (shared expert) keeps MoeGemvCols with the same columns per wave.
template <int K, int R, int D, int W, int BODY, int MODE, int CACHE, typename Prep>
__device__ __forceinline__ void MoeQ4Core(const LinJob& job, const GemvItem& item, uint32_t* xs, Prep prep) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int nb = __builtin_amdgcn_readfirstlane(item.nb);
  prep();
  if constexpr (BODY == 0) {
    const int c0 = 4 * wave;
    if (c0 < nb) {
      const int nc = min(4, nb - c0), b = item.b0 + c0;
      if constexpr (TileDwords(K) == 32) {
        MoeQ4Cols<R, D, MODE, CACHE>(job, item.s, b, nc, xs);
      } else {
        if (nc != 4) __builtin_trap();
        MoeGemvCols<K, R, 2, 4, CACHE>(job, item.s, {b, b + 1, b + 2, b + 3}, xs);
      }
    }
  } else {
    if (wave < nb) {
      const int b = item.b0 + wave;
      const bool c2 = wave + 8 < nb;
      if constexpr (TileDwords(K) == 32) {
        MoeB64Cols<R, D, CACHE>(job, item.s, b, c2, xs);
      } else {
        if (c2) MoeGemvCols<K, R, 4, 2, CACHE>(job, item.s, {b, b + 8}, xs);
        else MoeGemvCols<K, R, 4, 1, CACHE>(job, item.s, {b}, xs);
      }
    }
  }
  __syncthreads();
}

template <int K, int N, int D, int W, int BODY, int MODE, int CACHE, typename Prep>
__device__ __forceinline__ void MoeQ4Rows(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs,
                                          Prep prep) {
  if constexpr (N >= 1) {
    if (rows == N) MoeQ4Core<K, N, D, W, BODY, MODE, CACHE>(job, item, xs, prep);
    else MoeQ4Rows<K, N - 1, D, W, BODY, MODE, CACHE>(rows, job, item, xs, prep);
  }
}

template <int R, int KR, int KS, int D, int W, int BODY, int MODE, int CACHE, typename Prep>
__device__ __forceinline__ void MoeQ4Any(int K, int rows, const LinJob& job, const GemvItem& item, uint32_t* xs,
                                         Prep prep) {
  if (K == KR) MoeQ4Rows<KR, R, D, W, BODY, MODE, CACHE>(rows, job, item, xs, prep);
  else if (K == KS && rows == R) MoeQ4Core<KS, R, D, W, BODY, MODE, CACHE>(job, item, xs, prep);
  else __builtin_trap();
}

// MoeFusedKernel with the Q4 or B64 body: one block of W waves per item, gate/up items first, each down item
// waits for its job's gate/up items (2 ceil(40 / IB) of them, IB = 4 W for Q4 and 16 for B64) and runs the mid.
template <int R, int D, int W, int BODY, int MODE = 0, int CACHE = 4>
__global__ void __launch_bounds__(32 * W) MoeQ4Kernel(const LinJob* __restrict__ gu,
                                                      const GemvItem* __restrict__ gu_items, int n_gu,
                                                      const LinJob* __restrict__ dn,
                                                      const GemvItem* __restrict__ dn_items, int n_dn,
                                                      unsigned* __restrict__ done) {
  static_assert(BODY == 0 || W == 8, "B64 uses the engine's 8 waves");
  constexpr int IB = BODY == 0 ? 4 * W : 16;
  constexpr unsigned kGuItems = 2 * ((kMoeInter / 16 + IB - 1) / IB);
  extern __shared__ uint32_t xs[];
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) return;
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    MoeQ4Any<R, 4, 6, D, W, BODY, MODE, CACHE>(K, rows, job, item, xs, [&] { MoeFillX(job, item, rows, xs); });
    if (threadIdx.x == 0)
      __hip_atomic_fetch_add(done + (item.job >> 1), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    return;
  }
  const GemvItem item = dn_items[blockIdx.x - n_gu];
  const int u = item.job;
  const LinJob& d = dn[u];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K == 0) return;
  const int rows = __builtin_amdgcn_readfirstlane(d.R);
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < kGuItems)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += W) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float sv[4], v[4];
      __half sh[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sv[i] = Silu(gv[i]) * uv[i];
      LinScaleLane(d.suh, c0, sh);
      LinInLane(sv, sh, v);  // RoundH(RoundH(sv) * suh), Wht128: MoeFusedKernel's mid
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
  };
  MoeQ4Any<R, 4, 6, D, W, BODY, MODE, CACHE>(K, rows, d, item, xs, prep);
}

}  // namespace crucible
