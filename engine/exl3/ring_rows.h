// R > 1 decode ring for front-order dense groups (exl3_ring.h holds R = 1).
// A block-wide activation buffer for R rows and spans up to 160 tiles needs
// 40 KB of LDS and would cut the grid to one block per CU. Here the loader
// waves also stage each chunk's activations (R rows x 4 tiles x 8 words)
// behind its weights in the ring slot, so the LDS per block stays at 21.5 KB.
// The per-row arithmetic is LinGemvBody's: tiles in order, the same dot2
// pairs into two accumulators, the same shuffles and partial layout.
#pragma once
#include "exl3_ring.h"

namespace crucible {
namespace ring_rows {

using lin_ring::kChunk;
using lin_ring::kConsumers;
using lin_ring::kLoaders;
using lin_ring::kSlots;
using lin_ring::kSlotWords;
constexpr int kRowWords = kChunk * 8;  // one row's activations for one chunk
// Rows staged per slot: 8 up to R=8 (the LDS of the R <= 8 kernels stays
// 21.5 KB), 16 for a packed pass. A wide copy span (copy-e, R 17..kRingRowsMax) stages R rounded up to 8 rows: the
// engine does not launch it yet; bench/copy_rows.hip measures it. R <= 16 keeps its code.
constexpr int kRingRowsMax = 32;
constexpr int StageRows(int R) { return R <= kMaxRows ? kMaxRows : R <= kPackedRows ? kPackedRows : (R + 7) / 8 * 8; }
constexpr int SlotWordsRows(int R) { return kSlotWords + StageRows(R) * kRowWords; }
// Activation vectors (uint4) per loader lane: 8 per row, 64 loader lanes.
constexpr int XVecsPerLane(int R) { return StageRows(R) * 8 / (32 * kLoaders); }

// Front: front-order tiles; otherwise engine order (lin_ring::Load's two
// address forms). The slot layout is the same for both.
template <int K, int R, bool Front = true>
__device__ __forceinline__ void Load(const LinJob& job, const GemvItem& item, uint32_t* payload,
                                     lin_ring::Control& control, uint32_t base_seq) {
  constexpr int T = 8 * K, ColVectors = kChunk * T / 4;
  constexpr int VectorsPerLane = kConsumers * ColVectors / (32 * kLoaders);
  constexpr int SW = SlotWordsRows(R), XV = XVecsPerLane(R);
  static_assert(VectorsPerLane <= 6 && XV * 32 * kLoaders * 4 == StageRows(R) * kRowWords && XV <= 4);
  const int A = job.I / 16, span = A / job.S, a0 = item.s * span;
  const int count = span / kChunk;  // FrontFits or Group::aligned_chunk: whole chunks
  const int lane = threadIdx.x & 31, loader = threadIdx.x >> 5;
  const uint32_t bytes = (uint32_t)(size_t)A * (job.O / 16) * T * 4;
  const auto rs = StreamRsrc(job.tiles, bytes);
  const int items = job.S * (job.O / 128), idx = item.s * (job.O / 128) + item.b0 / 8;
  // Activation vector v: row v / 8, words (v % 8) * 4 .. + 3 of the chunk.
  // Lane vector k is v = xv + 64 k.
  const int xv = loader * 32 + lane, xo = (xv & 7) * 4;
  const uint32_t xbytes = (uint32_t)R * A * 8 * 4;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(job.xh), xbytes);
  auto ld = [&](int seq, int v) -> uint4 {
    const int vi = v * 64 + loader * 32 + lane;
    int vo, so;
    if constexpr (Front) {
      vo = vi < item.nb * ColVectors ? idx * lin_ring::ChunkBytes(K) + ((item.b0 & 7) * ColVectors + vi) * 16
                                     : (int)bytes;
      so = seq * items * lin_ring::ChunkBytes(K);
    } else {
      const int col = vi / ColVectors, off = vi % ColVectors;
      vo = col < item.nb ? ((item.b0 + col) * A + a0) * T * 4 + off * 16 : (int)bytes;
      so = seq * kChunk * T * 4;
    }
    const auto raw = __builtin_amdgcn_raw_buffer_load_b128(rs, vo, so, 0);
    return *reinterpret_cast<const uint4*>(&raw);
  };
  auto ldx = [&](int seq, int k) -> uint4 {
    const int xr = (xv + 64 * k) >> 3;
    const int vo = xr < R ? (xr * A * 8 + (a0 + seq * kChunk) * 8 + xo) * 4 : (int)xbytes;
    const auto raw = __builtin_amdgcn_raw_buffer_load_b128(xrs, vo, 0, 0);
    return *reinterpret_cast<const uint4*>(&raw);
  };
#define RR_LD(R_, S) \
  R_##0 = ld(S, 0); R_##1 = ld(S, 1); R_##2 = ld(S, 2); \
  if constexpr (VectorsPerLane > 3) R_##3 = ld(S, 3); \
  if constexpr (VectorsPerLane > 4) R_##4 = ld(S, 4); \
  if constexpr (VectorsPerLane > 5) R_##5 = ld(S, 5); \
  R_##X = ldx(S, 0); \
  if constexpr (XV > 1) R_##Y = ldx(S, 1); \
  if constexpr (XV > 2) R_##Z = ldx(S, 2); \
  if constexpr (XV > 3) R_##W = ldx(S, 3)
#define RR_PUT(R_, S) \
  if ((S) < count) { \
    const uint32_t global_seq = base_seq + (S); \
    control.WaitWritable(global_seq); \
    uint32_t* slot = payload + (global_seq % kSlots) * SW; \
    uint4* d = reinterpret_cast<uint4*>(slot) + loader * 32 + lane; \
    d[0] = R_##0; d[64] = R_##1; d[128] = R_##2; \
    if constexpr (VectorsPerLane > 3) d[192] = R_##3; \
    if constexpr (VectorsPerLane > 4) d[256] = R_##4; \
    if constexpr (VectorsPerLane > 5) d[320] = R_##5; \
    reinterpret_cast<uint4*>(slot + kSlotWords)[xv] = R_##X; \
    if constexpr (XV > 1) reinterpret_cast<uint4*>(slot + kSlotWords)[xv + 64] = R_##Y; \
    if constexpr (XV > 2) reinterpret_cast<uint4*>(slot + kSlotWords)[xv + 128] = R_##Z; \
    if constexpr (XV > 3) reinterpret_cast<uint4*>(slot + kSlotWords)[xv + 192] = R_##W; \
    control.Publish(global_seq, lane); \
  }
  uint4 A0, A1, A2, A3, A4, A5, AX, AY, AZ, AW, B0, B1, B2, B3, B4, B5, BX, BY, BZ, BW;
  RR_LD(A, 0);
  int seq = 0;
  for (; seq + 2 < count; seq += 2) {
    RR_LD(B, seq + 1);
    RR_PUT(A, seq);
    RR_LD(A, seq + 2);
    RR_PUT(B, seq + 1);
  }
  if (seq + 1 < count) { RR_LD(B, seq + 1); }
  RR_PUT(A, seq);
  RR_PUT(B, seq + 1);
#undef RR_LD
#undef RR_PUT
}

template <int K, int R>
__device__ __forceinline__ void Compute(const LinJob& job, const GemvItem& item, const uint32_t* payload,
                                        lin_ring::Control& control, uint32_t base_seq) {
  constexpr int T = 8 * K;
  const int lane = threadIdx.x & 31, cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - kLoaders);
  const int span = job.I / 16 / job.S, q = lane & 3;
  const LaneWindow<K> win(lane);
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.f;
  for (int seq = 0; seq * kChunk < span; ++seq) {
    const uint32_t global_seq = base_seq + seq;
    control.WaitReady(global_seq);
    if (cw < item.nb) {
      const uint32_t* slot = payload + (global_seq % kSlots) * SlotWordsRows(R);
      const uint32_t* column = slot + cw * kChunk * T;
#pragma unroll
      for (int f = 0; f < kChunk; ++f) {
        const uint32_t* tile = column + f * T;
        __half2 P[4];
        DecodeLane<K, kMulMode>(tile[win.d0], tile[(win.d0 + 1) % T], tile[(win.d0 + 2) % T], win.shift, P);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const uint32_t* xr = slot + kSlotWords + r * kRowWords + f * 8;
          const uint32_t xa = xr[q], xb = xr[4 + q];
          const __half2 ha = *reinterpret_cast<const __half2*>(&xa);
          const __half2 hb = *reinterpret_cast<const __half2*>(&xb);
          acc[r][0] = Dot2(P[0], ha, acc[r][0]);
          acc[r][0] = Dot2(P[1], hb, acc[r][0]);
          acc[r][1] = Dot2(P[2], ha, acc[r][1]);
          acc[r][1] = Dot2(P[3], hb, acc[r][1]);
        }
      }
    }
    control.Release(global_seq, lane);
  }
  if (cw < item.nb) {
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
      const int col = 16 * (item.b0 + cw) + lane / 4;
      float* out = job.partial + (size_t)item.s * R * job.O;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[r][0];
        out[r * job.O + col + 8] = acc[r][1];
      }
    }
  }
}

template <int R, bool Front, size_t N = 0>
__device__ __forceinline__ void LoadAny(int K, const LinJob& job, const GemvItem& item, uint32_t* payload,
                                       lin_ring::Control& control, uint32_t seq) {
  if (K == kSupportedDecodeK[N]) Load<kSupportedDecodeK[N], R, Front>(job, item, payload, control, seq);
  else if constexpr (N + 1 < kSupportedDecodeK.size()) LoadAny<R, Front, N + 1>(K, job, item, payload, control, seq);
}

template <int R, size_t N = 0>
__device__ __forceinline__ void ComputeAny(int K, const LinJob& job, const GemvItem& item,
                                          const uint32_t* payload, lin_ring::Control& control, uint32_t seq) {
  if (K == kSupportedDecodeK[N]) Compute<kSupportedDecodeK[N], R>(job, item, payload, control, seq);
  else if constexpr (N + 1 < kSupportedDecodeK.size()) ComputeAny<R, N + 1>(K, job, item, payload, control, seq);
}

}  // namespace ring_rows

// R rows, front order, items strided over the grid, one continuous ring.
// The caller runs the input transform (LinInKernel) first. out: the last
// split of each column group writes rows 0..R-1 (compute wave w writes rows
// w, w + 8), as in LinPolicyKernel.
// Front = false reads engine-order tiles (packed passes of 4.05 groups that
// stay in engine order); every span must be a multiple of kChunk.
template <int R, bool Front = true>
__global__ void __launch_bounds__(320) LinRingRowsKernel(const LinJob* __restrict__ jobs,
                                                        const GemvItem* __restrict__ items, int n, bool out) {
  static_assert(R >= 1 && R <= ring_rows::kRingRowsMax);
  __shared__ lin_ring::Control control;
  __shared__ uint32_t payload[ring_rows::kSlots * ring_rows::SlotWordsRows(R)];
  __shared__ uint32_t arrived;
  __shared__ int last;
  if (blockIdx.x == 0 && threadIdx.x == 0) PublishWeights(jobs[items[0].job].tiles);
  const int lane = threadIdx.x & 31;
  control.Initialize(threadIdx.x);
  if (threadIdx.x == 0) arrived = 0;
  __syncthreads();
  uint32_t seq = 0;
  if (threadIdx.x < 32 * lin_ring::kLoaders) {
    for (int it = blockIdx.x; it < n; it += gridDim.x) {
      const GemvItem item = items[it];
      const LinJob& job = jobs[item.job];
      ring_rows::LoadAny<R, Front>(__builtin_amdgcn_readfirstlane(job.K), job, item, payload, control, seq);
      seq += lin_ring::Chunks(job);
    }
    return;
  }
  const int cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - lin_ring::kLoaders);
  uint32_t round = 0;
  for (int it = blockIdx.x; it < n; it += gridDim.x) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    ring_rows::ComputeAny<R>(__builtin_amdgcn_readfirstlane(job.K), job, item, payload, control, seq);
    seq += lin_ring::Chunks(job);
    if (!out) continue;
    // Arrival order selects the writer, never the reduction order.
    __threadfence();
    lin_ring::ConsumerBarrier(&arrived, ++round, lane);
    if (cw == 0 && lane == 0) {
      bool l = job.S == 1;
      if (!l) {
        l = __hip_atomic_fetch_add(job.done + item.b0 / 8, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==
            unsigned(job.S - 1);
        if (l) __hip_atomic_store(job.done + item.b0 / 8, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      last = l;
    }
    // The flag is written after barrier 2k - 1 and read before barrier 2k + 1.
    lin_ring::ConsumerBarrier(&arrived, ++round, lane);
    if (last && cw < R) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      for (int r = cw; r < R; r += lin_ring::kConsumers) LinWriteBlock(job, r, item.b0 / 8);
    }
  }
}

}  // namespace crucible
