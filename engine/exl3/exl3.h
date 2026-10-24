// EXL3 linear layers: input transform, multi-matrix matvec, output transform.
//
// y = (H_O(W_hat^T H_I(x * suh))) * svh, per ExLlamaV3's activation path:
// half(x) * suh rounded to half, fp32 butterflies, one half rounding of the
// transformed input; the matvec accumulates in fp32 and the output transform
// stays fp32.
#pragma once

#include "kernels/exl3_gemv.h"
#include "../gfx1151/common.h"
#include "decode_support.h"
#include "lookahead.h"

namespace crucible {

enum Epilogue { kEpiStore = 0, kEpiResidual = 1 };

// One matrix applied to R activation rows. Device-resident so routing kernels
// can retarget expert jobs (tiles/suh/svh) without a host round trip.
struct LinJob {
  const uint32_t* tiles;  // crucible order
  const __half* suh;      // [I]
  const __half* svh;      // [O]
  const float* src;       // [R][lds] fp32 input, read by the input transform
  __half* xh;             // [R][I] transformed input
  float* partial;         // [S][R][O] fp32 split-K partial sums
  float* dst;             // output: [R][ldd], or the residual stack for kEpiResidual
  const float* aux;       // kEpiResidual: inject gates [R][4]
  unsigned* done;         // [O/128], last split performs the fixed-order output transform
  int cache_policy;       // raw buffer load auxiliary mode: 0 or 4
  int I, O, K, S, R;
  int lds, ldd, epi;
};

struct GemvItem {
  int job, s, b0, nb;  // k split s of column blocks [b0, b0 + nb), one wave each
};

struct XfItem {
  int job, r, blk;  // one 128-wide block of one row
};

constexpr int kMulMode = kMulTrue16;
constexpr int kPrefetch = 8;

// ---- input transform ---------------------------------------------------

__global__ void __launch_bounds__(256) LinInKernel(const LinJob* __restrict__ jobs,
                                                   const XfItem* __restrict__ items, int n) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= n) return;
  const int lane = threadIdx.x & 31;
  const XfItem it = items[w];
  const LinJob& j = jobs[it.job];
  if (j.K == 0) return;  // unused routed slot this forward
  const int c0 = it.blk * 128 + lane * 4;
  const float* s = j.src + (size_t)it.r * j.lds + c0;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(s[i]) * __half2float(j.suh[c0 + i]));
  Wht128(v);
  __half* d = j.xh + (size_t)it.r * j.I + c0;
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
}

// ---- output transform ----------------------------------------------------

__device__ __forceinline__ void LinOutBlock(const LinJob& j, int r, int blk, float v[4]) {
  const int lane = threadIdx.x & 31;
  const int c0 = blk * 128 + lane * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = 0.f;
  for (int s = 0; s < j.S; ++s) {
    const float* p = j.partial + ((size_t)s * j.R + r) * j.O + c0;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] += p[i];
  }
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * __half2float(j.svh[c0 + i]);
}

// The 4 scale values (suh or svh) of the columns c .. c + 3.
__device__ __forceinline__ void LinScaleLane(const __half* s, int c, __half out[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = s[c + i];
}

// LinInBlock's arithmetic on 4 fp32 values s that the lane already holds:
// RoundH(RoundH(s) * suh), Wht128 (the caller scales by kHadNorm and stores
// fp16). suh: the lane's 4 suh values (LinScaleLane). The M1 epilogue uses it
// for the down input (prefill_moe_pair.h), in PrefillMoeMidKernel's order.
// NO_SUH (bench mutant only) skips the suh product.
template <bool NO_SUH = false>
__device__ __forceinline__ void LinInLane(const float s[4], const __half suh[4], float v[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = NO_SUH ? RoundH(s[i]) : RoundH(RoundH(s[i]) * __half2float(suh[i]));
  Wht128(v);
}

// LinOutBlock's arithmetic (S = 1) on 4 fp32 partial values that the lane
// already holds (for example staged in LDS): 0 + p, Wht128, times kHadNorm
// times svh (the lane's 4 svh values, LinScaleLane).
__device__ __forceinline__ void LinOutLane(const float* p, const __half svh[4], float v[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = 0.f + p[i];
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * __half2float(svh[i]);
}

__device__ __forceinline__ void LinWriteBlock(const LinJob& j, int r, int blk) {
  const int lane = threadIdx.x & 31;
  float v[4];
  LinOutBlock(j, r, blk, v);
  const int c0 = blk * 128 + lane * 4;
  if (j.epi == kEpiStore) {
    float* d = j.dst + (size_t)r * j.ldd + c0;
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = v[i];
  } else {
    // Hyper-connection write-back: every raw stream gets inject[s] * y.
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      const float g = j.aux[r * kStreams + s];
      float* d = j.dst + (size_t)r * (j.ldd ? j.ldd : kWide) + s * kHidden + c0;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] += g * v[i];
    }
  }
}

__global__ void __launch_bounds__(256) LinOutKernel(const LinJob* __restrict__ jobs,
                                                    const XfItem* __restrict__ items, int n) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= n) return;
  const XfItem it = items[w];
  LinWriteBlock(jobs[it.job], it.r, it.blk);
}

// ---- matvec ----------------------------------------------------------------

// Front: tiles in the front order of exl3_ring.h (chunks of 8 columns x 4 tiles).
// Sched: MoeGemvCols' loop (moe_rows.h); the caller guarantees span % kPrefetch == 0.
template <int K, int R, int CACHE = 0, bool Front = false, bool Sched = false>
__device__ __forceinline__ void LinGemvBody(const LinJob& job, const GemvItem& item,
                                            uint32_t* xs, bool in = false) {
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = 8 * K;
  constexpr int PF = kPrefetch;
  const int A = job.I >> 4;
  const int span = A / job.S;
  const int a0 = item.s * span;
  if (in) {
    const int blocks = span / 8;
    for (int i = wave; i < R * blocks; i += 8) {
      const int r = i / blocks, blk = i % blocks;
      const int c0 = a0 * 16 + blk * 128 + lane * 4;
      const float* src = job.src + (size_t)r * job.lds + c0;
      float v[4];
#pragma unroll
      for (int c = 0; c < 4; ++c)
        v[c] = RoundH(RoundH(src[c]) * __half2float(job.suh[c0 + c]));
      Wht128(v);
      __half* dst = reinterpret_cast<__half*>(xs) + r * span * 16 + blk * 128 + lane * 4;
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = __float2half(v[c] * kHadNorm);
    }
  } else {
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(job.xh);
    for (int i = threadIdx.x; i < R * span * 8; i += blockDim.x) {
      const int r = i / (span * 8), o = i - r * (span * 8);
      xs[i] = xg[(size_t)r * A * 8 + a0 * 8 + o];
    }
  }
  __syncthreads();
  if (wave < item.nb) {
    const int b = item.b0 + wave;
    // Front order: a split's tile f of chunk q sits at chunk (q, idx), slot c * 4 + f.
    const int items = job.S * (job.O >> 7), idx = item.s * (job.O >> 7) + (b >> 3);
    const __amdgpu_buffer_rsrc_t rs =
        Front ? StreamRsrc(job.tiles + ((size_t)idx * 32 + (b & 7) * 4) * T,
                           (uint32_t)((size_t)(span / 4 - 1) * items * 32 + 4) * T * 4)
              : StreamRsrc(job.tiles + ((size_t)b * A + a0) * T, (uint32_t)span * T * 4);
    const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
    auto load_tile = [&](int tile, uint32_t* dst) {
      // Front prefetches past the span must not read other chunks.
      const int so = Front ? (tile < span ? ((tile >> 2) * items * 32 + (tile & 3)) * T * 4 : 0x40000000)
                           : tile * T * 4;
      dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, CACHE);
      dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, CACHE);
      dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, CACHE);
    };
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
    auto dots = [&](int a, const __half2 (&P)[4]) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t* xr = xs + (r * span + a) * 8;
        uint32_t xa = xr[q], xb = xr[4 + q];
        __half2 ha = *reinterpret_cast<__half2*>(&xa);
        __half2 hb = *reinterpret_cast<__half2*>(&xb);
        acc[r][0] = Dot2(P[0], ha, acc[r][0]);
        acc[r][0] = Dot2(P[1], hb, acc[r][0]);
        acc[r][1] = Dot2(P[2], ha, acc[r][1]);
        acc[r][1] = Dot2(P[3], hb, acc[r][1]);
      }
    };
    uint32_t ring[PF][3];
#pragma unroll
    for (int f = 0; f < PF; ++f) {
      load_tile(f, ring[f]);
      if constexpr (Sched) __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (Sched) {
      // MoeGemvCols' schedule: no early exit, refill after the decode, and
      // the refill pinned before the dots, so PF tiles stay in flight.
      for (int a = 0; a < span; a += PF) {
#pragma unroll
        for (int f = 0; f < PF; ++f) {
          __half2 P[4];
          DecodeLane<K, kMulMode>(ring[f][0], ring[f][1], ring[f][2], win.shift, P);
          load_tile(a + f + PF, ring[f]);
          __builtin_amdgcn_sched_barrier(0);
          dots(a + f, P);
        }
      }
    } else {
      for (int a = 0; a < span; a += PF) {
#pragma unroll
        for (int f = 0; f < PF; ++f) {
          if (a + f >= span) break;
          const uint32_t w0 = ring[f][0], w1 = ring[f][1], w2 = ring[f][2];
          load_tile(a + f + PF, ring[f]);
          __half2 P[4];
          DecodeLane<K, kMulMode>(w0, w1, w2, win.shift, P);
          dots(a + f, P);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float v = acc[r][c];
        v = v + __shfl_xor(v, 1);
        v = v + __shfl_xor(v, 2);
        acc[r][c] = v;
      }
    }
    if (q == 0) {
      const int col = 16 * b + lane / 4;
      float* out = job.partial + (size_t)item.s * R * job.O;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[r][0];
        out[r * job.O + col + 8] = acc[r][1];
      }
    }
  }
  __syncthreads();
}

// The CPU-visible capability list also instantiates every accepted decode width.
// MaxK 8 (the h6_ng8 packs) adds K7 and K8 (kWideDecodeK) in front of the list; the kernels with
// MaxK <= 6 keep their code.
template <int R, size_t N = 0, int MaxK = 6, bool Front = false, bool Sched = false>
__device__ __forceinline__ void DispatchLinGemv(int K, const LinJob& job,
                                               const GemvItem& item, uint32_t* xs, bool in = false) {
  if constexpr (MaxK > kMaxNarrowDecodeK) {
    static_assert(N == 0 && MaxK == kMaxWideDecodeK);
    if (K == 7) {
      if (job.cache_policy == 4) LinGemvBody<7, R, 4, Front, Sched>(job, item, xs, in);
      else LinGemvBody<7, R, 0, Front, Sched>(job, item, xs, in);
    } else if (K == 8) {
      if (job.cache_policy == 4) LinGemvBody<8, R, 4, Front, Sched>(job, item, xs, in);
      else LinGemvBody<8, R, 0, Front, Sched>(job, item, xs, in);
    } else {
      DispatchLinGemv<R, 0, kMaxNarrowDecodeK, Front, Sched>(K, job, item, xs, in);
    }
  } else if (K == kSupportedDecodeK[N]) {
    if (job.cache_policy == 4) LinGemvBody<kSupportedDecodeK[N], R, 4, Front, Sched>(job, item, xs, in);
    else LinGemvBody<kSupportedDecodeK[N], R, 0, Front, Sched>(job, item, xs, in);
  } else if constexpr (N + 1 < kSupportedDecodeK.size()) {
    if constexpr (kSupportedDecodeK[N + 1] <= MaxK)
      DispatchLinGemv<R, N + 1, MaxK, Front, Sched>(K, job, item, xs, in);
  }
}

// One launch covers many matrices of mixed bit widths; K is uniform per item.
template <int R, int MaxK = 6, bool Front = false, bool Sched = false>
__global__ void __launch_bounds__(256) LinPolicyKernel(const LinJob* __restrict__ jobs,
                                                     const GemvItem* __restrict__ items, int n,
                                                     bool in, bool out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) PublishWeights(jobs[items[0].job].tiles);
  extern __shared__ uint32_t xs[];
  __shared__ bool last;
  for (int it = blockIdx.x; it < n; it += gridDim.x) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    DispatchLinGemv<R, 0, MaxK, Front, Sched>(K, job, item, xs, in);
    if (out && K != 0) {
      // Arrival order selects the writer, never the floating-point reduction order.
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) {
        last = job.S == 1 ||
            __hip_atomic_fetch_add(job.done + item.b0 / 8, 1u, __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_AGENT) == unsigned(job.S - 1);
        if (last && job.S > 1)
          __hip_atomic_store(job.done + item.b0 / 8, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      __syncthreads();
      if (last) {
        const int wave = threadIdx.x >> 5;
        for (int r = wave; r < R; r += 8) LinWriteBlock(job, r, item.b0 / 8);
      }
      __syncthreads();
    }
  }
}

// The disabled policy retains a separate kernel to avoid fusion register overhead.
template <int R, size_t N = 0, int MaxK = 6, bool Front = false>
__device__ __forceinline__ void DispatchLinBaseline(int K, const LinJob& job,
                                                   const GemvItem& item, uint32_t* xs) {
  if constexpr (MaxK > kMaxNarrowDecodeK) {
    static_assert(N == 0 && MaxK == kMaxWideDecodeK);
    if (K == 7) LinGemvBody<7, R, 0, Front>(job, item, xs);
    else if (K == 8) LinGemvBody<8, R, 0, Front>(job, item, xs);
    else DispatchLinBaseline<R, 0, kMaxNarrowDecodeK, Front>(K, job, item, xs);
  } else if (K == kSupportedDecodeK[N]) LinGemvBody<kSupportedDecodeK[N], R, 0, Front>(job, item, xs);
  else if constexpr (N + 1 < kSupportedDecodeK.size()) {
    if constexpr (kSupportedDecodeK[N + 1] <= MaxK)
      DispatchLinBaseline<R, N + 1, MaxK, Front>(K, job, item, xs);
  }
}
template <int R, int MaxK = 6, bool Front = false>
__global__ void __launch_bounds__(256) LinGemvKernel(const LinJob* __restrict__ jobs,
                                                    const GemvItem* __restrict__ items, int n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) PublishWeights(jobs[items[0].job].tiles);
  extern __shared__ uint32_t xs[];
  for (int it = blockIdx.x; it < n; it += gridDim.x) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    DispatchLinBaseline<R, 0, MaxK, Front>(K, job, item, xs);
  }
}

}  // namespace crucible
