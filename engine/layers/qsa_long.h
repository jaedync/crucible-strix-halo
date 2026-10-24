// Exact fp32 indexer and split-K sparse attention for long-context decoding.
#pragma once
#include "ops.h"

namespace crucible {

// Preserve serial dimension-order FMAs. Parallel reduction or fp16 WMMA can
// change a marginal top-512 decision even when their dot error is tiny.
template <class Args = QsaArgs>
__global__ void __launch_bounds__(256) QsaScoreKernel(Args args) {
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.y >= a.R) return;
  const int r = blockIdx.y, b = blockIdx.x * 256 + threadIdx.x;
  const int nb = (a.pos0 + r + 1) / 4;
  if (nb <= kSelBlocks) return;
  __shared__ float iq[512];
  for (int i = threadIdx.x; i < 512; i += 256) iq[i] = a.idx[(size_t)r*kQsaIdxWidth+i];
  __syncthreads();
  if (b >= nb) return;
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#ifndef CRUCIBLE_QSA_PREFETCH
#define CRUCIBLE_QSA_PREFETCH 64
#endif
  constexpr int PF = CRUCIBLE_QSA_PREFETCH;
  static_assert(PF == 16 || PF == 32 || PF == 64 || PF == 128);
  for (int tile = 0; tile < 128; tile += PF) {
    float keys[PF];
#pragma unroll
    for (int j = 0; j < PF; ++j) keys[j] = a.c.ik_pool[QsaPoolOffset(b,tile+j)];
#pragma unroll
    for (int j = 0; j < PF; ++j) {
      const int i = tile+j; const float k = keys[j];
      d0 = fmaf(iq[i], k, d0); d1 = fmaf(iq[128+i], k, d1);
      d2 = fmaf(iq[256+i], k, d2); d3 = fmaf(iq[384+i], k, d3);
    }
  }
#if CRUCIBLE_DEBUG_NONFINITE & 1
  if (Nonfinite(d0) || Nonfinite(d1) || Nonfinite(d2) || Nonfinite(d3)) FlagNonfinite(kNfQsaScore);
#endif
  float s = fmaxf(d0,0.f);
  s += fmaxf(d1,0.f); s += fmaxf(d2,0.f); s += fmaxf(d3,0.f);
  a.scores[(size_t)r*a.sstride+b] = s * 0.08838834764831845f;
}

// One thread scores block b for all R rows of a verify window, so each pooled
// key is read once per forward, not once per row. Key loads of dimension tile
// t+1 are issued before the FMAs of tile t. Each (row, head) keeps the serial
// dimension-order FMA chain of QsaScoreKernel, so the scores are bitwise equal
// (bench/decode_long/score_rows.hip).
template<int R>
__global__ void __launch_bounds__(256) QsaScoreRowsKernel(QsaArgs a) {
  constexpr int PF = 32;
  const int b = blockIdx.x * 256 + threadIdx.x;
  const int nb_max = (a.pos0 + R) / 4;
  if (nb_max <= kSelBlocks) return;
  __shared__ float iq[R][512];
  for (int i = threadIdx.x; i < R*512; i += 256) iq[i/512][i%512] = a.idx[(size_t)(i/512)*kQsaIdxWidth+i%512];
  __syncthreads();
  if (b >= nb_max) return;
  float d[R][4] = {};
  float cur[PF], nxt[PF];
#pragma unroll
  for (int j = 0; j < PF; ++j) cur[j] = a.c.ik_pool[QsaPoolOffset(b,j)];
#pragma unroll 1
  for (int tile = 0; tile < 128; tile += PF) {
    if (tile + PF < 128) {
#pragma unroll
      for (int j = 0; j < PF; ++j) nxt[j] = a.c.ik_pool[QsaPoolOffset(b,tile+PF+j)];
    }
#pragma unroll
    for (int j = 0; j < PF; ++j) {
      const int i = tile+j; const float k = cur[j];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        d[r][0] = fmaf(iq[r][i], k, d[r][0]); d[r][1] = fmaf(iq[r][128+i], k, d[r][1]);
        d[r][2] = fmaf(iq[r][256+i], k, d[r][2]); d[r][3] = fmaf(iq[r][384+i], k, d[r][3]);
      }
    }
#pragma unroll
    for (int j = 0; j < PF; ++j) cur[j] = nxt[j];
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int nb = (a.pos0 + r + 1) / 4;
    if (nb <= kSelBlocks || b >= nb) continue;
#if CRUCIBLE_DEBUG_NONFINITE & 1
    if (Nonfinite(d[r][0]) || Nonfinite(d[r][1]) || Nonfinite(d[r][2]) || Nonfinite(d[r][3])) FlagNonfinite(kNfQsaScore);
#endif
    float s = fmaxf(d[r][0],0.f);
    s += fmaxf(d[r][1],0.f); s += fmaxf(d[r][2],0.f); s += fmaxf(d[r][3],0.f);
    a.scores[(size_t)r*a.sstride+b] = s * 0.08838834764831845f;
  }
}

inline void LaunchQsaScore(QsaArgs a, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4, grid = (nb+255)/256;
  switch (a.R) {
    case 1: QsaScoreRowsKernel<1><<<grid,256,0,stream>>>(a); break;
    case 2: QsaScoreRowsKernel<2><<<grid,256,0,stream>>>(a); break;
    case 3: QsaScoreRowsKernel<3><<<grid,256,0,stream>>>(a); break;
    case 4: QsaScoreRowsKernel<4><<<grid,256,0,stream>>>(a); break;
    case 5: QsaScoreRowsKernel<5><<<grid,256,0,stream>>>(a); break;
    case 6: QsaScoreRowsKernel<6><<<grid,256,0,stream>>>(a); break;
    case 7: QsaScoreRowsKernel<7><<<grid,256,0,stream>>>(a); break;
    case 8: QsaScoreRowsKernel<8><<<grid,256,0,stream>>>(a); break;
    default: QsaScoreKernel<<<dim3(grid,a.R),256,0,stream>>>(a);
  }
}

// Small groups of waves share histograms to bound contention.
template<int HistWaves, int Threads = 1024>
__device__ __forceinline__ void QsaRadixPass(const float* sc, int nb, int shift, unsigned mask,
                                            unsigned hist[HistWaves][256], unsigned* prefix,
                                            unsigned& threshold, unsigned& remaining) {
  const int j = threadIdx.x, wave = (j/32)%HistWaves;
  static_assert(Threads >= 256 && Threads % 32 == 0);
  for (int i = j; i < HistWaves*256; i += Threads) hist[i/256][i%256] = 0;
  __syncthreads();
  unsigned last = 0, count = 0;
  for (int b = j; b < nb; b += Threads) {
    const unsigned u = __float_as_uint(sc[b]);
    if ((u & mask) != threshold) continue;
    const unsigned bin = (u >> shift)&255;
    if (count && bin != last) { atomicAdd(&hist[wave][last],count); count = 0; }
    last = bin; ++count;
  }
  if (count) atomicAdd(&hist[wave][last],count);
  __syncthreads();
  unsigned count_bin = 0, inclusive = 0;
  const unsigned target = remaining;
  if (j < 256) {
    for (int w = 0; w < HistWaves; ++w) count_bin += hist[w][255-j];
    inclusive = count_bin;
#pragma unroll
    for (int step = 1; step < 32; step *= 2) {
      const unsigned v = __shfl_up(inclusive,step);
      if ((j&31) >= step) inclusive += v;
    }
    if ((j&31) == 31) prefix[j/32] = inclusive;
  }
  __syncthreads();
  if (j < 256) {
    for (int w = 0; w < j/32; ++w) inclusive += prefix[w];
    if (inclusive-count_bin < target && inclusive >= target) {
      threshold |= (unsigned)(255-j) << shift;
      remaining = target-(inclusive-count_bin);
    }
  }
  __syncthreads();
}

// Aligned wave spans avoid the strided reads of per-thread contiguous spans.
template<int Threads = 1024>
__device__ __forceinline__ void QsaEmitSelected(const float* sc, int nb, unsigned threshold,
                                               int* selected, unsigned* prefix_gt,
                                               unsigned* prefix_eq, const int* source_ids = nullptr,
                                               int base = 0, float* selected_scores = nullptr) {
  constexpr int Waves = Threads/32;
  static_assert(Waves >= 1 && Waves <= 32 && Threads % 32 == 0);
  const int j = threadIdx.x, wave = j/32, lane = j&31;
  const int span = ((nb+Threads-1)/Threads)*32, begin = wave*span, end = min(begin+span,nb);
  unsigned gt = 0, eq = 0;
  for (int b = begin+lane; b < end; b += 32) {
    const unsigned u = __float_as_uint(sc[b]);
    gt += u > threshold; eq += u == threshold;
  }
#pragma unroll
  for (int step = 1; step < 32; step *= 2) {
    gt += __shfl_xor(gt,step); eq += __shfl_xor(eq,step);
  }
  if (lane == 0) { prefix_gt[wave] = gt; prefix_eq[wave] = eq; }
  __syncthreads();
  if (j < 32) {
    unsigned g = j < Waves ? prefix_gt[j] : 0, e = j < Waves ? prefix_eq[j] : 0;
#pragma unroll
    for (int step = 1; step < 32; step *= 2) {
      const unsigned pg = __shfl_up(g,step), pe = __shfl_up(e,step);
      if (j >= step) { g += pg; e += pe; }
    }
    if (j < Waves) { prefix_gt[j] = g; prefix_eq[j] = e; }
  }
  __syncthreads();
  const unsigned need = kSelBlocks-prefix_gt[Waves-1];
  unsigned tie = prefix_eq[wave]-eq;
  unsigned out = prefix_gt[wave]-gt + min(tie,need);
  const unsigned lower = (1u<<lane)-1;
  for (int b = begin; b < end; b += 32) {
    const bool valid = b+lane < end;
    const unsigned u = valid ? __float_as_uint(sc[b+lane]) : 0;
    const unsigned eqmask = (unsigned)__ballot(valid && u == threshold);
    const bool keep = valid && (u > threshold || (u == threshold && tie+__popc(eqmask&lower) < need));
    const unsigned chosen = (unsigned)__ballot(keep);
    if (keep) {
      const unsigned slot = out+__popc(chosen&lower);
      selected[slot] = source_ids ? source_ids[b+lane] : base+b+lane;
      if (selected_scores) selected_scores[slot] = sc[b+lane];
    }
    tie += __popc(eqmask); out += __popc(chosen);
  }
}

// Four byte-radix scans replace 32 bitwise scans. All counts are integers.
template <class Args = QsaArgs>
__global__ void __launch_bounds__(1024) QsaTopKernel(Args args) {
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.x >= a.R) return;
  __shared__ unsigned hist[32][256], prefix_gt[32], prefix_eq[32];
  __shared__ unsigned threshold, remaining;
  const int r = blockIdx.x, nb = (a.pos0+r+1)/4;
  if (nb <= kSelBlocks) { if (threadIdx.x == 0) a.nsel[r] = -1; return; }
  const float* sc = a.scores + (size_t)r*a.sstride;
  if (threadIdx.x == 0) { threshold = 0; remaining = kSelBlocks; }
  __syncthreads();
  unsigned mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    QsaRadixPass<32>(sc,nb,shift,mask,hist,prefix_gt,threshold,remaining);
    mask |= 255u << shift;
  }
  QsaEmitSelected(sc,nb,threshold,a.sel+r*kSelBlocks,prefix_gt,prefix_eq);
  if (threadIdx.x == 0) a.nsel[r] = kSelBlocks;
}

constexpr int kQsaTopMaxBlocks = 262144/4, kQsaTopTile = 4096;
constexpr int kQsaCandidates = kQsaTopMaxBlocks/kQsaTopTile*kSelBlocks;
static_assert(kQsaTopTile >= kSelBlocks && kQsaTopMaxBlocks%kQsaTopTile == 0);
constexpr size_t QsaSelectScratchFloats(int rows) { return (size_t)rows*kQsaCandidates*2; }

// A value outside a tile's top-512 cannot enter the global top-512. Tile
// outputs stay in ID order, so their concatenation preserves the tie rule.
// Each pass loads its score set once. All radix scans then use LDS.
// Threads 1024 is the train-10 shape and the only shape the engine launches;
// only that shape is proven. Other block sizes are bench-only and not exact:
// bench/qsa_top_threads at 256 threads gives other IDs (bitwise=DIFFER) at
// pos0 163,840 and 229,376 (train 14).
template<int Mode, class Args = QsaArgs, int Threads = 1024>  // 0: one cached pass, 1: tile candidates, 2: candidate merge
__global__ void __launch_bounds__(Threads) QsaTopPassKernel(Args args, float* scores, int* ids) {
  static_assert(Mode == 0 || !kPackedQsa<Args>, "only the scratch-free pass is packed");
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.x >= a.R) return;
  constexpr bool Final = Mode != 1, Merge = Mode == 2;
  constexpr int Capacity = Final ? kQsaCandidates : kQsaTopTile;
  __shared__ float cached[Capacity];
  constexpr int Hist = Threads/32 < 16 ? Threads/32 : 16;
  __shared__ unsigned hist[Hist][256], prefix_gt[32], prefix_eq[32], threshold, remaining;
  const int r = Final ? blockIdx.x : blockIdx.y, j = threadIdx.x;
  const int nb = (a.pos0+r+1)/4, part = Final ? 0 : blockIdx.x;
  if (nb <= kSelBlocks) { if (Final && j == 0) a.nsel[r] = -1; return; }
  const int base = part*kQsaTopTile;
  const int count = Merge ? nb/kQsaTopTile*kSelBlocks+min(nb%kQsaTopTile,kSelBlocks)
                          : (Final ? nb : min(kQsaTopTile,max(0,nb-base)));
  const size_t row = (size_t)r*kQsaCandidates;
  const float* source = Merge ? scores+row : a.scores+(size_t)r*a.sstride+base;
  int* selected = Final ? a.sel+r*kSelBlocks : ids+row+part*kSelBlocks;
  float* selected_scores = Final ? nullptr : scores+row+part*kSelBlocks;
  if (!Final && count <= kSelBlocks) {
    if (j < count) { selected[j] = base+j; selected_scores[j] = source[j]; }
    return;
  }
  for (int i = j; i < count; i += Threads) cached[i] = source[i];
  if (j == 0) { threshold = 0; remaining = kSelBlocks; }
  __syncthreads();
  unsigned mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    QsaRadixPass<Hist,Threads>(cached,count,shift,mask,hist,prefix_gt,threshold,remaining);
    mask |= 255u << shift;
  }
  QsaEmitSelected<Threads>(cached,count,threshold,selected,prefix_gt,prefix_eq,
                  Merge ? ids+row : nullptr,base,selected_scores);
  if (Final && j == 0) a.nsel[r] = kSelBlocks;
}

// One block per row keeps the row's scores in VGPRs: wave w, lane l, slot i
// holds ID w*span+l+32i, the emit layout of QsaEmitSelected. This replaces the
// tile pass, the candidate merge and their LDS reloads with one global read.
// The radix counts are integers and the emit runs in ID order, so sel equals
// the tile pass and merge bit for bit (bench/qsa_top_threads.hip).
template <int Threads, int PT>
__global__ void __launch_bounds__(Threads) QsaTopRegKernel(QsaArgs a) {
  constexpr int Waves = Threads / 32, Hist = Waves < 16 ? Waves : 16;
  static_assert(Waves <= 32 && Threads >= 256);
  __shared__ unsigned hist[Hist][256], prefix_gt[32], prefix_eq[32], threshold, remaining;
  const int r = blockIdx.x, j = threadIdx.x, wave = j / 32, lane = j & 31;
  const int nb = (a.pos0 + r + 1) / 4;
  if (nb <= kSelBlocks) { if (j == 0) a.nsel[r] = -1; return; }
  const int per = (nb + Threads - 1) / Threads;  // launcher guarantees per <= PT
  const int span = per * 32, begin = wave * span, end = min(begin + span, nb);
  const int cnt = end > begin + lane ? (end - begin - lane + 31) / 32 : 0;
  const float* sc = a.scores + (size_t)r * a.sstride;
  unsigned u[PT];
#pragma unroll
  for (int i = 0; i < PT; ++i) u[i] = i < cnt ? __float_as_uint(sc[begin + lane + 32 * i]) : 0u;
  if (j == 0) { threshold = 0; remaining = kSelBlocks; }
  unsigned mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = j; i < Hist * 256; i += Threads) hist[i / 256][i % 256] = 0;
    __syncthreads();
    const unsigned thr = threshold;
    unsigned last = 0, count = 0;
#pragma unroll
    for (int i = 0; i < PT; ++i) {
      if (i >= cnt || (u[i] & mask) != thr) continue;
      const unsigned bin = (u[i] >> shift) & 255;
      if (count && bin != last) { atomicAdd(&hist[wave % Hist][last], count); count = 0; }
      last = bin; ++count;
    }
    if (count) atomicAdd(&hist[wave % Hist][last], count);
    __syncthreads();
    unsigned count_bin = 0, inclusive = 0;
    const unsigned target = remaining;
    if (j < 256) {
      for (int w = 0; w < Hist; ++w) count_bin += hist[w][255 - j];
      inclusive = count_bin;
#pragma unroll
      for (int step = 1; step < 32; step *= 2) {
        const unsigned v = __shfl_up(inclusive, step);
        if ((j & 31) >= step) inclusive += v;
      }
      if ((j & 31) == 31) prefix_gt[j / 32] = inclusive;
    }
    __syncthreads();
    if (j < 256) {
      for (int w = 0; w < j / 32; ++w) inclusive += prefix_gt[w];
      if (inclusive - count_bin < target && inclusive >= target) {
        threshold |= (unsigned)(255 - j) << shift;
        remaining = target - (inclusive - count_bin);
      }
    }
    __syncthreads();
    mask |= 255u << shift;
  }
  const unsigned thr = threshold;
  unsigned gt = 0, eq = 0;
#pragma unroll
  for (int i = 0; i < PT; ++i) if (i < cnt) { gt += u[i] > thr; eq += u[i] == thr; }
#pragma unroll
  for (int step = 1; step < 32; step *= 2) { gt += __shfl_xor(gt, step); eq += __shfl_xor(eq, step); }
  if (lane == 0) { prefix_gt[wave] = gt; prefix_eq[wave] = eq; }
  __syncthreads();
  if (j < 32) {
    unsigned g = j < Waves ? prefix_gt[j] : 0, e = j < Waves ? prefix_eq[j] : 0;
#pragma unroll
    for (int step = 1; step < 32; step *= 2) {
      const unsigned pg = __shfl_up(g, step), pe = __shfl_up(e, step);
      if (j >= step) { g += pg; e += pe; }
    }
    if (j < Waves) { prefix_gt[j] = g; prefix_eq[j] = e; }
  }
  __syncthreads();
  const unsigned need = kSelBlocks - prefix_gt[Waves - 1];
  unsigned tie = prefix_eq[wave] - eq;
  unsigned out = prefix_gt[wave] - gt + min(tie, need);
  const unsigned lower = (1u << lane) - 1;
  int* selected = a.sel + r * kSelBlocks;
#pragma unroll
  for (int i = 0; i < PT; ++i) {
    if (begin + 32 * i >= end) break;  // wave uniform
    const bool valid = i < cnt;
    const unsigned eqmask = (unsigned)__ballot(valid && u[i] == thr);
    const bool keep = valid && (u[i] > thr || (u[i] == thr && tie + __popc(eqmask & lower) < need));
    const unsigned chosen = (unsigned)__ballot(keep);
    if (keep) selected[out + __popc(chosen & lower)] = begin + 32 * i + lane;
    tie += __popc(eqmask); out += __popc(chosen);
  }
  if (j == 0) a.nsel[r] = kSelBlocks;
}

inline void LaunchQsaTop(QsaArgs a, float* scratch = nullptr, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4;
  if (nb > kSelBlocks && nb <= kQsaCandidates) {
    QsaTopPassKernel<0><<<a.R,1024,0,stream>>>(a,nullptr,nullptr);
  } else if (a.R > 8 && nb > kQsaCandidates && nb <= kQsaTopMaxBlocks) {
    // Prefill rows only; decode keeps the parallel tile pass for its few rows.
    if (nb <= 1024*32) QsaTopRegKernel<1024,32><<<a.R,1024,0,stream>>>(a);
    else QsaTopRegKernel<1024,64><<<a.R,1024,0,stream>>>(a);
  } else if (scratch && nb > kQsaCandidates && nb <= kQsaTopMaxBlocks) {
    int* ids = reinterpret_cast<int*>(scratch+(size_t)a.R*kQsaCandidates);
    QsaTopPassKernel<1><<<dim3((nb+kQsaTopTile-1)/kQsaTopTile,a.R),1024,0,stream>>>(a,scratch,ids);
    QsaTopPassKernel<2><<<a.R,1024,0,stream>>>(a,scratch,ids);
  } else QsaTopKernel<<<a.R,1024,0,stream>>>(a);
}

// scratch: QsaSelectScratchFloats(a.R) floats, needed when (pos0+R)/4 > kQsaCandidates.
inline void LaunchQsaSelect(QsaArgs a, float* scratch = nullptr, hipStream_t stream = nullptr) {
  const int nb = (a.pos0+a.R)/4;
  if (nb > kSelBlocks) LaunchQsaScore(a,stream);
  LaunchQsaTop(a,scratch,stream);
}

#ifndef CRUCIBLE_QSA_SPLITS
#define CRUCIBLE_QSA_SPLITS 32
#endif
constexpr int kQsaSplits = CRUCIBLE_QSA_SPLITS;
static_assert(kQsaSplits == 16 || kQsaSplits == 32 || kQsaSplits == 64);
constexpr int kQsaSplitTokens = 2048/kQsaSplits;
constexpr int kQsaPart = 258;  // max, denominator, 256 weighted values
constexpr size_t QsaScratchFloats(int rows) { return (size_t)rows*24*kQsaSplits*kQsaPart; }
static_assert(QsaScratchFloats(1) >= QsaSelectScratchFloats(1));

__device__ __forceinline__ int QsaPosition(QsaArgs a, int r, int t, int ns, int nb) {
  return ns < 0 ? t : (t < ns*4 ? a.sel[r*kSelBlocks+t/4]*4+(t&3) : nb*4+t-ns*4);
}

// Twelve query waves share a tiled fp16 KV load. Staging 16 positions at a
// time amortizes barriers without rounding Q or softmax probabilities.
// Split `split` of KV group g for row r; partial is the row span's base.
__device__ __forceinline__ void QsaGqaSplitBody(const QsaArgs& a, float* partial, int g, int r, int split,
                                                uint4 (&keys)[16][32], uint4 (&values)[16][32]) {
  const int lane = threadIdx.x&31, h = g*12+threadIdx.x/32;
  const int ns = a.nsel[r], nb = (a.pos0+r+1)/4;
  const int n = ns < 0 ? a.pos0+r+1 : ns*4+(a.pos0+r+1)%4;
  const int begin = split*kQsaSplitTokens;
  const int end = split == kQsaSplits-1 ? n : min(begin+kQsaSplitTokens,n);
  const float* q = a.q+((size_t)r*24+h)*512;
  float qr[8], acc[8] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) qr[i] = q[lane*8+i];
  float m = -INFINITY, l = 0;
  for (int tile = begin; tile < end; tile += 16) {
    const int count = min(16,end-tile);
    for (int i = threadIdx.x; i < count*32; i += 384) {
      const int p = QsaPosition(a,r,tile+i/32,ns,nb);
      const size_t off = ((size_t)p*2+g)*256+(i%32)*8;
      keys[i/32][i%32] = *reinterpret_cast<const uint4*>(a.c.k+off);
      values[i/32][i%32] = *reinterpret_cast<const uint4*>(a.c.v+off);
    }
    __syncthreads();
    for (int t = 0; t < count; ++t) {
      const uint4 kr = keys[t][lane], vr = values[t][lane];
      const __half* kh = reinterpret_cast<const __half*>(&kr);
      const __half* vh = reinterpret_cast<const __half*>(&vr);
      float d = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) d += qr[i]*__half2float(kh[i]);
      d = WaveSum(d)*0.0625f;
      const float mn = fmaxf(m,d), old = __expf(m-mn), e = __expf(d-mn);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = acc[i]*old+e*__half2float(vh[i]);
      l = l*old+e; m = mn;
    }
    __syncthreads();
  }
  float* dst = partial+(((size_t)r*24+h)*kQsaSplits+split)*kQsaPart;
  if (lane == 0) { dst[0] = m; dst[1] = l; }
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[2+lane*8+i] = acc[i];
}

template <class Args = QsaArgs>
__global__ void __launch_bounds__(384) QsaGqaSplitKernel(Args args, float* partial) {
  const unsigned span = blockIdx.z/kQsaSplits;
  const QsaArgs a = QsaSpanArgs(args, span);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.y >= a.R) return;
  partial += (size_t)QsaSpanBase(args, span)*24*kQsaSplits*kQsaPart;
  __shared__ uint4 keys[16][32], values[16][32];
  QsaGqaSplitBody(a, partial, blockIdx.x, blockIdx.y, blockIdx.z%kQsaSplits, keys, values);
}

// Head h of row r, output channel d = threadIdx.x (threads 0..255). Every
// thread of the block calls it (one barrier); ob holds 256 floats.
__device__ __forceinline__ void QsaMergeBody(const QsaArgs& a, const float* partial, int h, int r, float* ob) {
  const int d = threadIdx.x;
  if (d < 256) {
    const float* src = partial+((size_t)r*24+h)*kQsaSplits*kQsaPart;
    float m = -INFINITY;
    for (int s = 0; s < kQsaSplits; ++s) m = fmaxf(m,src[s*kQsaPart]);
    float l = 0, o = 0;
    for (int s = 0; s < kQsaSplits; ++s) {
      const float* p = src+s*kQsaPart;
      if (p[1] == 0) continue;
      const float f = __expf(p[0]-m);
      l += p[1]*f; o += p[2+d]*f;
    }
    o /= l;
    if (a.context) a.context[((size_t)r*24+h)*256+d] = o;
    const float* q = a.q+((size_t)r*24+h)*512;
    o *= Sigmoid(q[256+d]);
    ob[d] = RoundH(RoundH(o)*__half2float(a.suh_o[h*256+d]));
  }
  __syncthreads();
  if (d < 64) {
    const int half = d/32, lane = d&31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = ob[half*128+lane*4+i];
    Wht128(v);
    __half* dst = a.xh_o+(size_t)r*kQsaOutWidth+h*256+half*128+lane*4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2half(v[i]*kHadNorm);
  }
}

template <class Args = QsaArgs>
__global__ void __launch_bounds__(256) QsaMergeKernel(Args args, const float* partial) {
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.y >= a.R) return;
  partial += (size_t)QsaSpanBase(args, blockIdx.z)*24*kQsaSplits*kQsaPart;
  __shared__ float ob[256];
  QsaMergeBody(a, partial, blockIdx.x, blockIdx.y, ob);
}

// QsaGqaSplitKernel and QsaMergeKernel in one launch for one row
// (the R = 1 tail). Each block counts its split in its group's word of
// sync; the blocks of the last 12 splits of a group then wait for all 32
// splits and merge one head each. They are the last blocks to dispatch, so
// every block they wait for is already resident. sync: [2] split counts and
// [2] finished merges, zero between launches (the last merger of a group
// clears its two words).
__global__ void __launch_bounds__(384) QsaAttnTailKernel(QsaArgs a, float* partial, unsigned* sync) {
  __shared__ uint4 keys[16][32], values[16][32];
  const int g = blockIdx.x, split = blockIdx.z;
  QsaGqaSplitBody(a, partial, g, 0, split, keys, values);
  __syncthreads();  // every wave's partial stores before the release
  if (threadIdx.x == 0) __hip_atomic_fetch_add(sync + g, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  constexpr int kFirstMerger = kQsaSplits - 12;
  if (split < kFirstMerger) return;
  if (threadIdx.x == 0)
    while (__hip_atomic_load(sync + g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kQsaSplits)
      __builtin_amdgcn_s_sleep(1);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  QsaMergeBody(a, partial, g*12+split-kFirstMerger, 0, reinterpret_cast<float*>(&keys[0][0]));
  // Every split of the group has counted itself (the wait above), so its words are free after 12 merges.
  if (threadIdx.x == 0 &&
      __hip_atomic_fetch_add(sync + 2 + g, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 11u) {
    sync[g] = 0;
    sync[2 + g] = 0;
  }
}
static_assert(kQsaSplits >= 12, "twelve merging splits per group");

inline void LaunchQsaAttn(QsaArgs a, float* scratch) {
  QsaGqaSplitKernel<<<dim3(2,a.R,kQsaSplits),384>>>(a,scratch);
  QsaMergeKernel<<<dim3(24,a.R),256>>>(a,scratch);
}

}  // namespace crucible
