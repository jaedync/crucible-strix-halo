// QSA prompt kernels. One flash workgroup shares KV across all twelve GQA
// query heads of one row, including rows with different sparse selections.
#pragma once
#include "../ops.h"
#include "prefill_qsa_select.h"
#include "../../gfx1151/prefill_wmma.h"

namespace crucible::prefill {
__host__ __device__ __forceinline__ size_t PrefillPoolOffset(int block, int dim) {
#ifdef CRUCIBLE_QSA_POOL_TILED
  return QsaPoolOffset(block, dim);
#else
  return (size_t)block * 128 + dim;
#endif
}
inline size_t PrefillPoolFloats(int context) {
#ifdef CRUCIBLE_QSA_POOL_TILED
  return QsaPoolFloats(context);
#else
  return (size_t)(context / 4 > 0 ? context / 4 : 1) * 128;
#endif
}

struct QsaPrefill {
  QsaArgs input;
  float* context;  // [R][24][256], un-gated fp32 context
  // [R][24][256], low part scaled by 4096. q_lo == nullptr selects fp16 q and
  // fp16 P without their low halves (approved prefill numerics).
  __half *q_hi, *q_lo;
  // QsaSelectScratchFloats(R) floats; required when (pos0 + R) / 4 > kQsaCandidates.
  float* select_scratch = nullptr;
};

// Completion interval includes a block that started in an earlier chunk.
__global__ void __launch_bounds__(256) QsaPrefillPoolKernel(QsaArgs a) {
  __shared__ float x[256], red[8];
  const int b = a.pos0 / 4 + blockIdx.x, j = threadIdx.x;
  if (b >= (a.pos0 + a.R) / 4) return;
  if (j < 128) {
    float m = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) m += QsaRawValue(a, b * 4 + t, j);
    x[j] = RoundH(m * .25f);
  }
  __syncthreads();
  CNormLds(x, 128, a.ik_norm1, red);
  RopeAt(x, a.rope, a.mrope, b * 4);
  if (j < 128) a.c.ik_pool[PrefillPoolOffset(b, j)] = x[j];
}

__device__ __forceinline__ int QsaPosition(QsaArgs a, int r, int index) {
  const int ns = a.nsel[r], nb = (a.pos0 + r + 1) / 4;
  if (ns < 0) return index;
  return index < ns * 4 ? a.sel[r * 512 + index / 4] * 4 + (index & 3)
      : nb * 4 + index - ns * 4;
}

__device__ __forceinline__ H16 HalfFragment(const __half* x) {
  return *reinterpret_cast<const H16*>(x);
}

// Packed operands replace hundreds of scalar conversions/loads per WMMA.
// V is transposed once per token tile, with a padded stride for b128 LDS
// reads and reduced bank conflicts. No half conversion of the fp32 softmax.
// LO adds the low halves of q and P (two more WMMAs per product).
template <bool Dense = false, bool LO = true>
__global__ void __launch_bounds__(256) QsaPrefillFlashKernel(QsaPrefill a) {
  __shared__ float score[16 * 64], maxima[16], sums[16], scale[16];
  __shared__ __half probability[16][64], residual[LO ? 16 : 1][64], values[256][24];
  const QsaArgs q = a.input;
  const int r = Dense ? blockIdx.x * 16 : blockIdx.x;
  const int group = Dense ? blockIdx.y / 12 : blockIdx.y;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int ns = Dense ? -1 : q.nsel[r];
  const int end = Dense ? q.pos0 + min(r + 16, q.R) : q.pos0 + r + 1;
  const int n = ns < 0 ? end : ns * 4 + end % 4;
  if (threadIdx.x < 16) { maxima[threadIdx.x] = -INFINITY; sums[threadIdx.x] = 0.f; }
  F8 output[2] = {};
  __syncthreads();
  for (int start = 0; start < n; start += 64) {
    if (wave < 4) {
      const int head = lane & 15, col = wave * 16 + (lane & 15), index = start + col;
      const size_t qi = Dense ? ((size_t)(r + head) * 24 + blockIdx.y) * 256
          : ((size_t)r * 24 + group * 12 + head) * 256;
      const bool valid = Dense ? r + head < q.R : head < 12;
      const int pos = index < n ? (Dense ? index : QsaPosition(q, r, index)) : 0;
      F8 high = {}, low = {};
#pragma clang loop unroll(disable)
      for (int k = 0; k < 256; k += 16) {
        const H16 ah = valid ? HalfFragment(a.q_hi + qi + k) : H16{};
        const H16 b = index < n ? HalfFragment(q.c.k + ((size_t)pos * 2 + group) * 256 + k) : H16{};
        high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high);
        if constexpr (LO) {
          const H16 al = valid ? HalfFragment(a.q_lo + qi + k) : H16{};
          low = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, low);
        }
      }
      F8 dots = high;
      if constexpr (LO) dots = high + low * (1.f / kResidualScale);
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = FragmentRow(z);
        const bool visible = index < n && (!Dense || r + m >= q.R || index <= q.pos0 + r + m);
        score[m * 64 + col] = visible ? dots[z] * .0625f : -INFINITY;
      }
    }
    __syncthreads();
    const int head = threadIdx.x / 16, li = threadIdx.x & 15;
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; ++i) maximum = fmaxf(maximum, score[head * 64 + li + i * 16]);
#pragma unroll
    for (int mask = 1; mask < 16; mask <<= 1) maximum = fmaxf(maximum, __shfl_xor(maximum, mask));
    const float new_max = fmaxf(maxima[head], maximum), old_scale = __expf(maxima[head] - new_max);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = li + i * 16;
      const float p = __expf(score[head * 64 + col] - new_max);
      probability[head][col] = __float2half(p);
      if constexpr (LO)
        residual[head][col] = __float2half((p - __half2float(probability[head][col])) * kResidualScale);
      sum += p;
    }
#pragma unroll
    for (int mask = 1; mask < 16; mask <<= 1) sum += __shfl_xor(sum, mask);
    if (li == 0) {
      maxima[head] = new_max; scale[head] = old_scale; sums[head] = sums[head] * old_scale + sum;
    }
    __syncthreads();
    F8 high[2] = {}, low[2] = {};
#pragma clang loop unroll(disable)
    for (int k = 0; k < 64; k += 16) {
#pragma clang loop unroll(disable)
      for (int kk = 0; kk < 16; ++kk) {
        const int index = start + k + kk, d = threadIdx.x;
        const int pos = index < n ? (Dense ? index : QsaPosition(q, r, index)) : 0;
        values[d][kk] = index < n ? q.c.v[((size_t)pos * 2 + group) * 256 + d] : __float2half(0.f);
      }
      __syncthreads();
      const H16 ah = HalfFragment(probability[lane & 15] + k);
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        const int d = wave * 16 + part * 128 + (lane & 15);
        const H16 b = HalfFragment(values[d]);
        high[part] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[part]);
        if constexpr (LO) {
          const H16 al = HalfFragment(residual[lane & 15] + k);
          low[part] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, low[part]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      F8 pv = high[part];
      if constexpr (LO) pv = high[part] + low[part] * (1.f / kResidualScale);
#pragma unroll
      for (int z = 0; z < 8; ++z) output[part][z] = output[part][z] * scale[FragmentRow(z)] + pv[z];
    }
    __syncthreads();
  }
#pragma unroll
  for (int part = 0; part < 2; ++part) {
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int head = FragmentRow(z), d = wave * 16 + part * 128 + (lane & 15);
      if constexpr (Dense) {
        if (r + head < q.R) a.context[((size_t)(r + head) * 24 + blockIdx.y) * 256 + d] = output[part][z] / sums[head];
      } else {
        if (head < 12) a.context[((size_t)r * 24 + group * 12 + head) * 256 + d] = output[part][z] / sums[head];
      }
    }
  }
}

template <bool LO>
inline void LaunchQsaFlashSplit(QsaPrefill a, hipStream_t stream) {
  if (a.input.pos0 + a.input.R <= 2048)
    QsaPrefillFlashKernel<true, LO><<<dim3((a.input.R + 15) / 16, 24), 256, 0, stream>>>(a);
  else QsaPrefillFlashKernel<false, LO><<<dim3(a.input.R, 2), 256, 0, stream>>>(a);
}
inline void LaunchQsaFlash(QsaPrefill a, hipStream_t stream = nullptr) {
  if (a.q_lo) LaunchQsaFlashSplit<true>(a, stream);
  else LaunchQsaFlashSplit<false>(a, stream);
}

__global__ void __launch_bounds__(256) QsaPrefillGateKernel(QsaPrefill a) {
  __shared__ float output[256];
  const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x, wave = t >> 5, lane = t & 31;
  const QsaArgs q = a.input;
  const float gate = Sigmoid(q.q[((size_t)r * 24 + h) * 512 + 256 + t]);
  const float x = a.context[((size_t)r * 24 + h) * 256 + t] * gate;
  output[t] = RoundH(RoundH(x) * __half2float(q.suh_o[h * 256 + t]));
  __syncthreads();
  if (wave < 2) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = output[wave * 128 + lane * 4 + i];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q.xh_o[OutProjIndex(r, h * 256 + wave * 128 + lane * 4 + i, q.pack_rows, q.pack_off)] =
          __float2half(v[i] * kHadNorm);
  }
}

// The only prefill query path since train 20: QsaPrefillQueryKernel writes
// the fp16 query halves of the output rows that the older path (QsaPrepKernel's
// query units, then QsaPrefillPackKernel, now in bench/kernels/prefill_qsa_pack.h)
// computed, and the normalized fp32 query is not stored (no later prefill kernel
// reads it; the gate reads the other half of each 512-float head row). Same q_hi
// and q_lo bits, -13 ms per 4,096 rows in the engine.

// Blocks (head group, output row): the query-head units of QsaPrepKernel (the
// same norm and RoPE arithmetic per head), then QsaPrefillPackKernel's
// conversion of the value that QsaPrepKernel would store. Rows below `skip`
// have no output. HPB heads per block, the next head's value loaded early
// (bench/qsa_prep_half.hip: HPB 2 is the fastest, 0.42 of the default path).
template <int HPB = 1>
__global__ void __launch_bounds__(256) QsaPrefillQueryKernel(QsaPrefill f, int skip) {
  static_assert(kQHeads % HPB == 0);
  const QsaArgs a = f.input;
  PublishWeights(a.q_norm1);
  __shared__ float x[256], red[8];
  const int o = blockIdx.y, r = skip + o, j = threadIdx.x, p = a.pos0 + r;
  const float* src = a.q + ((size_t)r * kQHeads + blockIdx.x * HPB) * 512 + j;
  float next = src[0];
#pragma unroll
  for (int k = 0; k < HPB; ++k) {
    const int u = blockIdx.x * HPB + k;
    if (k) __syncthreads();
    x[j] = next;
    if (k + 1 < HPB) next = src[(k + 1) * 512];
    __syncthreads();
    CNormLds(x, 256, a.q_norm1, red);
    RopeAt(x, a.rope, a.mrope, p);
    const size_t i = ((size_t)o * kQHeads + u) * 256 + j;
    const __half h = __float2half(x[j]);
    f.q_hi[i] = h;
    if (f.q_lo) f.q_lo[i] = __float2half((x[j] - __half2float(h)) * kResidualScale);
  }
}

// The cache half of a QSA prefill part: K/V rows, indexer pool blocks and raw indexer keys. Later chunks and
// decode read only these; the select and flash stages produce this part's output rows.
// cache_units: only units 24..32 of QsaPrepKernel (the query heads go to QsaPrefillQueryKernel).
inline void LaunchQsaPrefillCache(const QsaArgs& q, hipStream_t stream = nullptr, bool cache_units = false) {
  // The cache-only path (last-layer pruning) has no select guard, so it checks its own bounds.
  if (q.R < 1 || q.R > 2048 || q.pos0 < 0 || q.sstride < (q.pos0 + q.R) / 4) {
    std::fprintf(stderr, "QSA prefill cache rows require 1..2048 rows and sufficient score stride\n");
    std::abort();
  }
  if (cache_units) QsaPrepKernel<QsaArgs, 24><<<dim3(9, q.R), 256, 0, stream>>>(q);
  else QsaPrepKernel<<<dim3(33, q.R), 256, 0, stream>>>(q);
  const int completed = (q.pos0 + q.R) / 4 - q.pos0 / 4;
  if (completed) QsaPrefillPoolKernel<<<completed, 256, 0, stream>>>(q);
  LaunchQsaRetainRaw(q, stream);
}

// The output rows [skip, R) of a part (last-layer pruning): rows below skip have only cache work. Union tiles
// and dense flash blocks start at multiples of 4 and 16 rows, so a skip that is a multiple of 16 keeps every
// kept row's tile, block and arithmetic.
inline QsaArgs QsaOutputRows(QsaArgs q, int skip) {
  q.pos0 += skip; q.R -= skip;
  q.q += (size_t)skip * kQsaQWidth; q.idx += (size_t)skip * kQsaIdxWidth;
  if (q.pack_rows) q.pack_off += skip;
  else q.xh_o += (size_t)skip * kQsaOutWidth;
  return q;
}

// mark(name) runs after each stage when profiling; the default does nothing.
// skip: rows [0, skip) of the part get only the cache kernels (a multiple of 16, below R).
template <class Mark>
inline void LaunchQsaPrefill(QsaPrefill a, Mark mark, hipStream_t stream = nullptr, int skip = 0) {
  const auto q = a.input;
  if (q.R < 1 || q.R > 2048 || q.pos0 < 0 || q.sstride < (q.pos0 + q.R) / 4
      || !a.context || !a.q_hi
      || ((q.pos0 + q.R) / 4 > kQsaCandidates && !a.select_scratch)) {
    std::fprintf(stderr, "QSA prefill requires 1..2048 rows and sufficient score storage\n");
    std::abort();
  }
  if (skip < 0 || skip >= q.R || skip % 16) { std::fprintf(stderr, "QSA prefill skip must be a multiple of 16 below R\n"); std::abort(); }
  LaunchQsaPrefillCache(q, stream, true);
  QsaPrefillQueryKernel<2><<<dim3(12, q.R - skip), 256, 0, stream>>>(a, skip);
  mark("qsa_prep");
  a.input = QsaOutputRows(q, skip);
  const auto out = a.input;
  // Context is dead until the flash stage writes it, so select may keep its
  // query tiles there: QsaQueryTileFloats(R) <= R * 6144 floats.
  LaunchQsaPrefillSelect(out, a.select_scratch, a.context, stream);
  mark("qsa_select");
  LaunchQsaFlash(a, stream);
  mark("qsa_flash");
  QsaPrefillGateKernel<<<dim3(24, out.R), 256, 0, stream>>>(a);
}
inline void LaunchQsaPrefill(QsaPrefill a, hipStream_t stream = nullptr) {
  LaunchQsaPrefill(a, [](const char*) {}, stream);
}
}  // namespace crucible::prefill
