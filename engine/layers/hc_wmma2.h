// HcW2Kernel: the decode HC read of HcWmmaKernel (hc_wmma.h) with a new grid
// and the same arithmetic.
//
// - GT fn column tiles per dots block (HcWmmaKernel: 7). kHcWTiles / GT
//   column groups, so KS * kStreams * kHcWTiles / GT dots blocks (GT 3: 112,
//   GT 1: 336) for the same 336 dots waves. Each wave still owns one column
//   tile of one stream and K chunk and runs the same 40-step WMMA chain.
// - The mid phase runs in the last dots block of each column group, for the
//   columns of that group only. Every group sums the squares of its own
//   (stream, K chunk) slice into its own pss slot with the same code, so
//   every group gets the same bits of 1 / rms. The last group to finish
//   publishes `epoch`; the mix blocks wait for it as before.
// - FB 1: fn and up are the fragment-order copies (HcFragFnKernel and
//   HcFragUpKernel in prefill_hc.h, as bulk prefill makes them). A B
//   fragment load is then one 512-byte run instead of 16 row-strided 32-byte
//   pieces. FB 2: fn is the compact fragment order (HcFnCompactOrder): the
//   last row tile holds only its M - 16 * (M / 16) live rows, so fn keeps
//   M x kWide halves (the padded copy reads 12 zero rows per step of the
//   inject tail tile, +245 KB per site). up is the FB 1 copy (no tail). The
//   int8 blocks (Q8) are in fragment order already.
//
// KS, the K chunk sums, the WMMA chains, the high/low split and the epilogue
// do not change, and a WMMA output element depends only on its own A row and
// B column. So x and post are bitwise equal to HcWmmaKernel<KS, PF, Q8> for
// every R (bench/hc8.hip checks it).
#pragma once

#include "hc_frag.h"
#include "hc_wmma.h"

namespace crucible {

template <int GT>
constexpr int HcW2Groups() {
  static_assert(kHcWTiles % GT == 0);
  return kHcWTiles / GT;
}

template <int KS, int GT>
constexpr int HcW2DotsBlocks() { return KS * kStreams * HcW2Groups<GT>(); }

// pss floats: one [KS][kStreams][16] slot per column group.
template <int KS, int GT>
constexpr size_t HcW2PssFloats() { return (size_t)HcW2Groups<GT>() * KS * kStreams * kHcWRows; }

// count: one counter per column group, then the finished-groups counter.
template <int GT>
constexpr int HcW2Counters() { return HcW2Groups<GT>() + 1; }

// HcWDots with GT tiles per block, the pss slot of the block's group and
// the fragment-order read (FB). The staging, the square sums and the WMMA
// loop are the same text.
template <int KS, int PF, bool Q8, int GT, int FB>
__device__ __forceinline__ void HcW2Dots(int db, const __half* __restrict__ fn, int M,
                                         const float* __restrict__ H, int R, const HcWScratch& sc,
                                         unsigned char* smem, const uint8_t* __restrict__ fq) {
  constexpr int KC = kHidden / KS, STEPS = KC / 16, LD = HcWStride<KS>(), V = KC / 4;
  constexpr int NV = kHcWRows * V / 256;
  constexpr int NGR = HcW2Groups<GT>();
  static_assert(STEPS % PF == 0 && kHcWRows * V % 256 == 0 && V % 16 == 0 && GT <= 8);
  _Float16* ahi = reinterpret_cast<_Float16*>(smem);
  _Float16* alo = ahi + R * LD;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, row = lane & 15;
  const int ntg = db % NGR, s = (db / NGR) % kStreams, kc = db / (NGR * kStreams);
  const int nt = ntg * GT + wave;
  const bool work = wave < GT && nt * 16 < M;
  // Row-major: columns past M read column M - 1. Fragment order: the copy
  // has zero rows up to the tile edge. Neither is ever stored past M.
  const int m = min(nt * 16 + row, M - 1);
  // FB: tile nt, stream s, K tiles from kc * STEPS; one step is TR x 16
  // halves (TR = 16, or the live rows of the compact tail tile), a lane
  // reads row min(lane & 15, TR - 1) of it.
  const int TR = FB == 2 ? min(16, M - nt * 16) : 16;
  const int BS = FB ? TR : 1;
  const auto* bp =
      FB ? reinterpret_cast<const prefill::H16*>(
               fn + (size_t)nt * 16 * kWide +
               (((size_t)s * (kHidden / 16) + kc * STEPS) * TR + min(row, TR - 1)) * 16)
         : reinterpret_cast<const prefill::H16*>(fn + (size_t)m * kWide + s * kHidden + kc * KC);
  constexpr int G = kHc8FnGroup, NG = Q8 ? KC / G : 1;
  static_assert(!Q8 || (KC % G == 0 && G % 16 == 0));
  const int j0 = (s * kHidden + kc * KC) / 16;
  const uint4* qp = reinterpret_cast<const uint4*>(fq) + ((size_t)nt * (kWide / 16) + j0) * 16 + row;
  using Ring = std::conditional_t<Q8, uint4, prefill::H16>;
  Ring ring[PF];
  _Float16 sv[NG];
  if (work) {
    if constexpr (Q8) {
      const _Float16* sp = reinterpret_cast<const _Float16*>(fq + Hc8ScaleOffset(M, kWide)) +
                           (size_t)(nt * 16 + row) * (kWide / G) + j0 * 16 / G;
#pragma unroll
      for (int f = 0; f < PF; ++f) ring[f] = qp[f * 16];
#pragma unroll
      for (int g = 0; g < NG; ++g) sv[g] = sp[g];
    } else {
#pragma unroll
      for (int f = 0; f < PF; ++f) ring[f] = bp[f * BS];
    }
  }
  const float* hs = H + s * kHidden + kc * KC;
  {
    float4 v[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int i = t + 256 * j, r = i / V, k4 = i - r * V;
      if (r < R) v[j] = *reinterpret_cast<const float4*>(hs + (size_t)r * kWide + k4 * 4);
    }
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int i = t + 256 * j, r = i / V, k4 = i - r * V;
      if (r >= R) continue;
      const float f[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
      typedef _Float16 H4 __attribute__((ext_vector_type(4)));
      H4 h4, l4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        h4[e] = (_Float16)f[e];
        l4[e] = (_Float16)((f[e] - (float)h4[e]) * prefill::kResidualScale);
      }
      *reinterpret_cast<H4*>(ahi + r * LD + k4 * 4) = h4;
      *reinterpret_cast<H4*>(alo + r * LD + k4 * 4) = l4;
    }
  }
  // The square sums of HcWDots (same order), into the group's own slot.
  {
    const int r = t >> 4, l = t & 15;
    float4 v[V / 16];
    if (r < R) {
#pragma unroll
      for (int j = 0; j < V / 16; ++j)
        v[j] = *reinterpret_cast<const float4*>(hs + (size_t)r * kWide + (l + 16 * j) * 4);
    }
    float sq = 0.f;
    if (r < R) {
#pragma unroll
      for (int j = 0; j < V / 16; ++j) {
        sq += v[j].x * v[j].x;
        sq += v[j].y * v[j].y;
        sq += v[j].z * v[j].z;
        sq += v[j].w * v[j].w;
      }
    }
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) sq += __shfl_xor(sq, o);
    if (l == 0 && r < R) sc.pss[((ntg * KS + kc) * kStreams + s) * kHcWRows + r] = sq;
  }
  __syncthreads();
  if (!work) return;
  const int NT = (R + 7) >> 3, i7 = row & 7;
  const _Float16* ar = (row < 8 ? ahi : alo) + i7 * LD;
  const bool live0 = i7 < R, live1 = 8 + i7 < R;
  prefill::F8 acc[2] = {};
  if constexpr (Q8) {
#pragma unroll
    for (int st0 = 0; st0 < STEPS; st0 += PF) {
#pragma unroll
      for (int f = 0; f < PF; ++f) {
        const int st = st0 + f;
        const prefill::H16 b = Hc8Dequant(ring[f], sv[st * 16 / G]);
        prefill::H16 a0 = {};
        if (live0) a0 = *reinterpret_cast<const prefill::H16*>(ar + st * 16);
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b, acc[0]);
        if (NT > 1) {
          prefill::H16 a1 = {};
          if (live1) a1 = *reinterpret_cast<const prefill::H16*>(ar + 8 * LD + st * 16);
          acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b, acc[1]);
        }
        if (st + PF < STEPS) ring[f] = qp[(st + PF) * 16];
        __builtin_amdgcn_sched_barrier(0);
      }
    }
  } else
  for (int st0 = 0; st0 < STEPS; st0 += PF) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      const int st = st0 + f;
      prefill::H16 a0 = {};
      if (live0) a0 = *reinterpret_cast<const prefill::H16*>(ar + st * 16);
      acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, ring[f], acc[0]);
      if (NT > 1) {
        prefill::H16 a1 = {};
        if (live1) a1 = *reinterpret_cast<const prefill::H16*>(ar + 8 * LD + st * 16);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, ring[f], acc[1]);
      }
      ring[f] = bp[min(st + PF, STEPS - 1) * BS];
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  float* out = sc.part + ((size_t)(kc * kStreams + s) * kHcWRows) * kHcWPad + nt * 16 + row;
#pragma unroll
  for (int t2 = 0; t2 < 2; ++t2) {
#pragma unroll
    for (int z = 0; z < 4; ++z) {
      const int r = 8 * t2 + prefill::FragmentRow(z);
      if (r < R) out[(size_t)r * kHcWPad] = acc[t2][z] + acc[t2][z + 4] * (1.f / prefill::kResidualScale);
    }
  }
}

// HcWMid for the columns [g * GT * 16, min(M, (g + 1) * GT * 16)) of group g,
// with 1 / rms from the group's own pss slot. Group 0 writes sc.inv for the
// mix (every group computes the same bits).
template <int KS, int GT>
__device__ __forceinline__ void HcW2Mid(int g, int M, int R, float* __restrict__ post, const HcWScratch& sc) {
  __shared__ float inv[kHcWRows][kStreams];
  const int t = threadIdx.x;
  const float* pss = sc.pss + (size_t)g * KS * kStreams * kHcWRows;
  if (t < R * kStreams) {
    const int r = t / kStreams, s = t % kStreams;
    float a = 0.f;
#pragma unroll
    for (int kc = 0; kc < KS; ++kc) a += pss[(kc * kStreams + s) * kHcWRows + r];
    inv[r][s] = rsqrtf(a / kHidden + kEps);
    if (g == 0) sc.inv[r * kStreams + s] = inv[r][s];
  }
  __syncthreads();
  const int m0 = g * GT * 16, MG = min(M, m0 + GT * 16) - m0;
  if (MG <= 0) return;
  constexpr int B = 4;
  for (int i0 = t; i0 < R * MG; i0 += B * blockDim.x) {
    float d[B][kStreams][KS];
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int i = i0 + b * blockDim.x;
      if (i >= R * MG) continue;
      const int r = i / MG, m = m0 + i - r * MG;
#pragma unroll
      for (int s = 0; s < kStreams; ++s)
#pragma unroll
        for (int kc = 0; kc < KS; ++kc)
          d[b][s][kc] = sc.part[((size_t)(kc * kStreams + s) * kHcWRows + r) * kHcWPad + m];
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const int i = i0 + b * blockDim.x;
      if (i >= R * MG) continue;
      const int r = i / MG, m = m0 + i - r * MG;
      float a = 0.f;
#pragma unroll
      for (int s = 0; s < kStreams; ++s) {
        float sum = 0.f;
#pragma unroll
        for (int kc = 0; kc < KS; ++kc) sum += d[b][s][kc];
        a += inv[r][s] * sum;
      }
      if (m < kRank) {
        const float v = Silu(a * .25f);
        const __half h = __float2half(v);
        sc.thi[r * kRank + m] = h;
        sc.tlo[r * kRank + m] = __float2half((v - __half2float(h)) * prefill::kResidualScale);
      } else if (post) {
        post[r * kStreams + m - kRank] = 2.f * Sigmoid(a * .25f);
      }
    }
  }
}

// HcWMix with the fragment-order up (FB) and MT column tiles of 16 hidden
// columns per block (HcWMix: 1). Each tile keeps its own WMMA chains and
// epilogue; the block loads the weights of all MT tiles before the wait and
// stages t once. Int8 keeps the codes in registers and rebuilds W' per tile.
template <bool Q8, int FB, int MT>
__device__ __forceinline__ void HcW2Mix(int b, const __half* __restrict__ up, const float* __restrict__ w1,
                                        const float* __restrict__ H, int R, float* __restrict__ x,
                                        const HcWScratch& sc, unsigned epoch, unsigned char* smem,
                                        const uint8_t* __restrict__ uq) {
  constexpr int STEPS = kHcWMixSteps, KT = kRank / 16;
  auto* ts = reinterpret_cast<prefill::H16*>(smem);
  float* g = reinterpret_cast<float*>(ts + 2 * R * KT);
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, row = lane & 15;
  const int s = wave & 3, kh = wave >> 2;
  constexpr int BS = FB ? 16 : 1;  // up has no tail tile
  constexpr int G = kHc8UpGroup, NG = STEPS * 16 / G;
  static_assert(STEPS * 16 % G == 0 && G % 16 == 0);
  prefill::H16 w[Q8 ? 1 : MT][STEPS];
  uint4 q[Q8 ? MT : 1][STEPS];
  _Float16 sv[MT][NG];
#pragma unroll
  for (int u = 0; u < MT; ++u) {
    const int d0 = (b * MT + u) * 16;
    if constexpr (Q8) {
      const size_t urow = (size_t)s * kHidden + d0;
      const uint4* qp = reinterpret_cast<const uint4*>(uq) + ((urow / 16) * (kRank / 16) + kh * STEPS) * 16 + row;
      const _Float16* sp = reinterpret_cast<const _Float16*>(uq + Hc8ScaleOffset(kWide, kRank)) +
                           (urow + row) * (kRank / G) + kh * NG;
#pragma unroll
      for (int st = 0; st < STEPS; ++st) q[u][st] = qp[st * 16];
#pragma unroll
      for (int gg = 0; gg < NG; ++gg) sv[u][gg] = sp[gg];
    } else {
      const auto* bp =
          FB ? reinterpret_cast<const prefill::H16*>(
                   up + ((((size_t)s * kHidden + d0) / 16 * (kRank / 16) + kh * STEPS) * 256 + row * 16))
             : reinterpret_cast<const prefill::H16*>(up + ((size_t)s * kHidden + d0 + row) * kRank) + kh * STEPS;
#pragma unroll
      for (int st = 0; st < STEPS; ++st) w[u][st] = bp[st * BS];
    }
  }
  // Int8 with MT 1 rebuilds W' before the wait, as HcWMix does.
  prefill::H16 w8[Q8 && MT == 1 ? STEPS : 1];
  if constexpr (Q8 && MT == 1) {
#pragma unroll
    for (int st = 0; st < STEPS; ++st) w8[st] = Hc8Dequant(q[0][st], sv[0][st * 16 / G]);
  }
  const int er = t >> 4, ec = t & 15;
  float hv[MT][kStreams], wv[MT][kStreams];
#pragma unroll
  for (int u = 0; u < MT; ++u) {
    const int d = (b * MT + u) * 16 + ec;
#pragma unroll
    for (int st = 0; st < kStreams; ++st) {
      wv[u][st] = w1[st * kHidden + d];
      hv[u][st] = er < R ? H[((size_t)er * kStreams + st) * kHidden + d] : 0.f;
    }
  }
  if (t == 0)
    while (__hip_atomic_load(sc.flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != epoch)
      __builtin_amdgcn_s_sleep(1);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  float iv[kStreams];
#pragma unroll
  for (int st = 0; st < kStreams; ++st) iv[st] = er < R ? sc.inv[er * kStreams + st] : 0.f;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int i = t + 256 * j;
    if (i >= 2 * R * KT) break;
    const int hl = i / (R * KT), rem = i - hl * R * KT;
    ts[i] = *reinterpret_cast<const prefill::H16*>((hl ? sc.tlo : sc.thi) + rem * 16);
  }
  __syncthreads();
  const int NT = (R + 7) >> 3, i7 = row & 7;
  const prefill::H16* ar = ts + ((row < 8 ? 0 : R) + i7) * KT + kh * STEPS;
  const bool live0 = i7 < R, live1 = 8 + i7 < R;
#pragma unroll
  for (int u = 0; u < MT; ++u) {
    prefill::F8 acc[2] = {};
#pragma unroll
    for (int st = 0; st < STEPS; ++st) {
      prefill::H16 bw;
      if constexpr (!Q8) bw = w[u][st];
      else if constexpr (MT == 1) bw = w8[st];
      else bw = Hc8Dequant(q[u][st], sv[u][st * 16 / G]);
      prefill::H16 a0 = {};
      if (live0) a0 = ar[st];
      acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, bw, acc[0]);
      if (NT > 1) {
        prefill::H16 a1 = {};
        if (live1) a1 = ar[8 * KT + st];
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, bw, acc[1]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (u > 0) __syncthreads();  // the epilogue of tile u - 1 has read g
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2) {
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        const int r = 8 * t2 + prefill::FragmentRow(z);
        if (r < R)
          g[((kh * kStreams + s) * R + r) * 17 + row] = acc[t2][z] + acc[t2][z + 4] * (1.f / prefill::kResidualScale);
      }
    }
    __syncthreads();
    if (er < R) {
      float v[kStreams];
#pragma unroll
      for (int st = 0; st < kStreams; ++st) {
        const float gs = g[(st * R + er) * 17 + ec] + g[((kStreams + st) * R + er) * 17 + ec];
        v[st] = Sigmoid(gs) * hv[u][st] * iv[st] * wv[u][st];
      }
      x[(size_t)er * kHidden + (b * MT + u) * 16 + ec] = RoundH(((v[0] + v[1]) + (v[2] + v[3])) * .25f);
    }
  }
}

// Launch with HcW2DotsBlocks<KS, GT>() + kHcWMixBlocks / MT blocks of 256 threads
// and HcWLds<KS>(R) bytes of dynamic LDS. sc.pss holds HcW2PssFloats<KS, GT>()
// floats and sc.count HcW2Counters<GT>() zeroed counters. FB: fn and up are
// the fragment-order copies (fn with its rows padded to kHcWPad).
template <int KS, int PF, bool Q8, int GT, int FB, int MT = 1>
__global__ void __launch_bounds__(256) HcW2Kernel(const __half* __restrict__ fn, int M,
                                                  const __half* __restrict__ up,
                                                  const float* __restrict__ w1,
                                                  const float* __restrict__ H, int R,
                                                  float* __restrict__ x, float* __restrict__ post,
                                                  HcWScratch sc, unsigned epoch, int first = 0,
                                                  HcW8 q8 = {}) {
  static_assert(!(Q8 && FB), "the int8 blocks are in fragment order already");
  static_assert(kHcWMixBlocks % MT == 0);
  constexpr int ND = HcW2DotsBlocks<KS, GT>(), NGR = HcW2Groups<GT>(), PER = KS * kStreams;
  extern __shared__ __attribute__((aligned(16))) unsigned char smem[];
  const int blk = blockIdx.x + first;
  if (blk >= ND) {
    HcW2Mix<Q8, FB, MT>(blk - ND, up, w1, H, R, x, sc, epoch, smem, q8.up);
    return;
  }
  __shared__ bool last;
  if constexpr (Q8) PublishWeights(q8.fn);
  else PublishWeights(fn);
  HcW2Dots<KS, PF, Q8, GT, FB>(blk, fn, M, H, R, sc, smem, q8.fn);
  __syncthreads();
  if constexpr (Q8) PublishWeights(q8.up);
  else PublishWeights(up);
  const int g = blk % NGR;
  if (threadIdx.x == 0) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    last = __hip_atomic_fetch_add(sc.count + g, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == PER - 1;
  }
  __syncthreads();
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  HcW2Mid<KS, GT>(g, M, R, post, sc);
  __syncthreads();
  if (threadIdx.x == 0) {
    sc.count[g] = 0;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    if (__hip_atomic_fetch_add(sc.count + NGR, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == NGR - 1) {
      sc.count[NGR] = 0;
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      __hip_atomic_store(sc.flag, epoch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// Engine shape. The cold bench (bench/hc8.hip section 5, 2026-10-24) has fp16 FB 1 GT 1
// fastest at R = 1 to 8 (73.4 us against 82.4 at R = 1) and GT 3 at R = 16. In the engine the
// lookahead puts the HC weights in the MALL first, and a sweep with a knob in the engine (4.05,
// code prompt, 257 tokens) gave: R = 1 (serial, spec 0)
// GT 1 34.33 tok/s, GT 3 33.96, GT 7 33.60 (baseline 33.82); R = 2..8 (spec 7 verify) GT 1 92.33,
// GT 3 94.10, GT 7 94.17 (baseline 93.17). R = 9..16 keeps GT 3 from the bench (not swept).
// int8 GT 7 (GT 1 and GT 3 are slower than the old kernel). PF 10 and MT 2 are slower everywhere.
// The scratch fits the largest group count (GT 1).
constexpr int kHcW2Pf = 20, kHcW2Mt = 1, kHcW2GtMax = 1;
constexpr int kHcW2GtR1 = 1, kHcW2GtR8 = 7, kHcW2GtR16 = 3, kHcW2GtSplitRows = 8, kHcW2Gt8 = 7;

inline HcWScratch HcW2Alloc() {
  HcWScratch sc;
  sc.part = Scratch<float>(HcWPartFloats<kHcWKs>());
  sc.pss = Scratch<float>(HcW2PssFloats<kHcWKs, kHcW2GtMax>());
  sc.inv = Scratch<float>(kHcWRows * kStreams);
  sc.thi = Scratch<__half>((size_t)kHcWRows * kRank);
  sc.tlo = Scratch<__half>((size_t)kHcWRows * kRank);
  sc.count = Zeros<unsigned>(HcW2Counters<kHcW2GtMax>());
  sc.flag = Zeros<unsigned>(1);
  return sc;
}

template <int KS, int PF, bool Q8, int GT, int FB>
inline void HcW2Run(const __half* fn, int M, const __half* up, const float* w1, const float* H, int R, float* x,
                    float* post, const HcWScratch& sc, unsigned epoch, const HcW8& q8) {
  static_assert(HcW2Groups<GT>() <= HcW2Groups<kHcW2GtMax>(), "HcW2Alloc holds too few group slots");
  constexpr int G = HcW2DotsBlocks<KS, GT>() + kHcWMixBlocks / kHcW2Mt;
  HcW2Kernel<KS, PF, Q8, GT, FB, kHcW2Mt><<<G, 256, HcWLds<KS>(R)>>>(fn, M, up, w1, H, R, x, post, sc, epoch, 0, q8);
}

// HcWLaunch with HcW2Kernel: the same x and post bits. fnf and upf are the
// fragment-order copies (FB 1) or null (the row-major read). Every GT gives
// the same bits, so the choice by R does not change the tokens. The counters
// of a group are zero again at the end of each call, so calls with another
// GT can share the scratch.
inline void HcW2Launch(const __half* fn, const __half* fnf, int M, const __half* up, const __half* upf,
                       const float* w1, const float* H, int R, float* x, float* post, const HcWScratch& sc,
                       unsigned epoch, const HcW8& q8 = {}) {
  if (q8.fn)
    HcW2Run<kHcWKs, kHcW8Pf, true, kHcW2Gt8, 0>(fn, M, up, w1, H, R, x, post, sc, epoch, q8);
  else if (fnf && upf && R == 1)
    HcW2Run<kHcWKs, kHcW2Pf, false, kHcW2GtR1, 1>(fnf, M, upf, w1, H, R, x, post, sc, epoch, {});
  else if (fnf && upf && R <= kHcW2GtSplitRows)
    HcW2Run<kHcWKs, kHcW2Pf, false, kHcW2GtR8, 1>(fnf, M, upf, w1, H, R, x, post, sc, epoch, {});
  else if (fnf && upf)
    HcW2Run<kHcWKs, kHcW2Pf, false, kHcW2GtR16, 1>(fnf, M, upf, w1, H, R, x, post, sc, epoch, {});
  else
    HcW2Run<kHcWKs, kHcW2Pf, false, kHcW2GtR16, 0>(fn, M, up, w1, H, R, x, post, sc, epoch, {});
}

}  // namespace crucible
