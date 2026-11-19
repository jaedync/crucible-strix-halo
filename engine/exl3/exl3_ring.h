// Opt-in dense EXL3 schedule. The reference contraction stays in exl3.h.
#pragma once
#include "exl3.h"
#include "../gfx1151/lds_stream_ring.h"
#include "front_order.h"

namespace crucible {

namespace lin_ring {
constexpr int kSlots = 3, kLoaders = 2, kConsumers = 8, kChunk = 4;
constexpr int kMaxK = 6;
// MaxK 8 (the h6_ng8 packs): a wider slot in separate instantiations, so the K <= 6 kernels keep their code.
template <int MaxK>
constexpr int SlotWords = kConsumers * kChunk * 8 * MaxK;
constexpr int kSlotWords = SlotWords<kMaxK>;
using Control = LdsStreamRing<kSlots, kLoaders, kConsumers>;

// Front order (front_order.h) stores chunk q of every item of a job side
// by side. Concurrent blocks then read one contiguous region, where engine
// order gives eight 128K-byte runs per chunk at a column stride.
constexpr int ChunkBytes(int K) { return kConsumers * kChunk * 8 * K * 4; }

// A named register pair keeps next-chunk VMEM live while this chunk reaches LDS.
// Front-order items must stay inside one eight-column group. A narrower item
// reads the contiguous part of each chunk that holds its columns.
template <int K, bool Front = false, int MaxK = kMaxK>
__device__ __forceinline__ void Load(const LinJob& job, const GemvItem& item,
                                    uint32_t* payload, Control& control, uint32_t base_seq = 0) {
  constexpr int T = 8 * K, ColVectors = kChunk * T / 4;
  constexpr int VectorsPerLane = kConsumers * ColVectors / (32 * kLoaders);
  static_assert(K <= MaxK && VectorsPerLane <= 8);
  const int A = job.I / 16, span = A / job.S, a0 = item.s * span;
  const int count = (span + kChunk - 1) / kChunk;
  const int lane = threadIdx.x & 31, loader = threadIdx.x >> 5;
  const uint32_t bytes = (uint32_t)(size_t)A * (job.O / 16) * T * 4;
  const auto rs = StreamRsrc(job.tiles, bytes);
  const int items = job.S * (job.O / 128), idx = item.s * (job.O / 128) + item.b0 / 8;
  auto ld = [&](int seq, int v) -> uint4 {
    const int vi = v * 64 + loader * 32 + lane;
    int vo, so;
    if constexpr (Front) {
      vo = vi < item.nb * ColVectors ? idx * ChunkBytes(K) + ((item.b0 & 7) * ColVectors + vi) * 16 : (int)bytes;
      so = seq * items * ChunkBytes(K);
    } else {
      const int col = vi / ColVectors, off = vi % ColVectors;
      // Columns past a narrow item load out of range: zero, and no DRAM traffic.
      vo = col < item.nb ? ((item.b0 + col) * A + a0) * T * 4 + off * 16 : (int)bytes;
      so = seq * kChunk * T * 4;
    }
    const auto raw = __builtin_amdgcn_raw_buffer_load_b128(rs, vo, so, 0);
    return *reinterpret_cast<const uint4*>(&raw);
  };
#define LR_LD(R, S) \
  R##0 = ld(S, 0); R##1 = ld(S, 1); R##2 = ld(S, 2); \
  if constexpr (VectorsPerLane > 3) R##3 = ld(S, 3); \
  if constexpr (VectorsPerLane > 4) R##4 = ld(S, 4); \
  if constexpr (VectorsPerLane > 5) R##5 = ld(S, 5); \
  if constexpr (VectorsPerLane > 6) R##6 = ld(S, 6); \
  if constexpr (VectorsPerLane > 7) R##7 = ld(S, 7)
#define LR_PUT(R, S) \
  if ((S) < count) { \
    const uint32_t global_seq = base_seq + (S); \
    control.WaitWritable(global_seq); \
    uint4* d = reinterpret_cast<uint4*>(payload + (global_seq % kSlots) * SlotWords<MaxK>) + loader * 32 + lane; \
    d[0] = R##0; d[64] = R##1; d[128] = R##2; \
    if constexpr (VectorsPerLane > 3) d[192] = R##3; \
    if constexpr (VectorsPerLane > 4) d[256] = R##4; \
    if constexpr (VectorsPerLane > 5) d[320] = R##5; \
    if constexpr (VectorsPerLane > 6) d[384] = R##6; \
    if constexpr (VectorsPerLane > 7) d[448] = R##7; \
    control.Publish(global_seq, lane); \
  }
  uint4 A0, A1, A2, A3, A4, A5, B0, B1, B2, B3, B4, B5;
  [[maybe_unused]] uint4 A6, A7, B6, B7;
  // Chunks past the item end would cost DRAM reads that nobody consumes.
  LR_LD(A, 0);
  // Full pairs with unconditional loads let each store wait only for its own
  // set (vmcnt 9..5 at K5); a conditional load before the stores made the
  // compiler drain the younger set too (isa-audit, vmcnt 4..0).
  int seq = 0;
  for (; seq + 2 < count; seq += 2) {
    LR_LD(B, seq + 1);
    LR_PUT(A, seq);
    LR_LD(A, seq + 2);
    LR_PUT(B, seq + 1);
  }
  if (seq + 1 < count) { LR_LD(B, seq + 1); }
  LR_PUT(A, seq);
  LR_PUT(B, seq + 1);
#undef LR_LD
#undef LR_PUT
}

template <int K, int MaxK = kMaxK>
__device__ __forceinline__ void Compute(const LinJob& job, const GemvItem& item,
                                       const uint32_t* payload, Control& control,
                                       const uint32_t* xs, uint32_t base_seq = 0) {
  constexpr int T = 8 * K;
  const int lane = threadIdx.x & 31, cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - kLoaders);
  const int span = job.I / 16 / job.S, q = lane & 3;
  const LaneWindow<K> win(lane);
  float acc0 = 0.f, acc1 = 0.f;
  for (int seq = 0, a = 0; a < span; ++seq, a += kChunk) {
    const uint32_t global_seq = base_seq + seq;
    control.WaitReady(global_seq);
    if (cw < item.nb) {
      const uint32_t* column = payload + (global_seq % kSlots) * SlotWords<MaxK> + cw * kChunk * T;
#pragma unroll
      for (int f = 0; f < kChunk; ++f) {
        if (a + f >= span) break;
        const uint32_t* tile = column + f * T;
        __half2 P[4];
        DecodeLane<K, kMulMode>(tile[win.d0], tile[(win.d0 + 1) % T], tile[(win.d0 + 2) % T], win.shift, P);
        const uint32_t xa = xs[(a + f) * 8 + q], xb = xs[(a + f) * 8 + 4 + q];
        const __half2 ha = *reinterpret_cast<const __half2*>(&xa), hb = *reinterpret_cast<const __half2*>(&xb);
        acc0 = Dot2(P[0], ha, acc0);
        acc0 = Dot2(P[1], hb, acc0);
        acc1 = Dot2(P[2], ha, acc1);
        acc1 = Dot2(P[3], hb, acc1);
      }
    }
    control.Release(global_seq, lane);
  }
  if (cw < item.nb) {
    acc0 = acc0 + __shfl_xor(acc0, 1);
    acc0 = acc0 + __shfl_xor(acc0, 2);
    acc1 = acc1 + __shfl_xor(acc1, 1);
    acc1 = acc1 + __shfl_xor(acc1, 2);
    if (q == 0) {
      const int col = 16 * (item.b0 + cw) + lane / 4;
      float* out = job.partial + (size_t)item.s * job.O;
      out[col] = acc0; out[col + 8] = acc1;
    }
  }
}

// Fills one item's activations (all compute waves). in: the input
// transform from job.src, as in LinGemvBody; otherwise a copy of job.xh.
__device__ __forceinline__ void FillActivations(const LinJob& job, const GemvItem& item, bool in,
                                                uint32_t* xs) {
  const int lane = threadIdx.x & 31, cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - kLoaders);
  const int A = job.I / 16, span = A / job.S, a0 = item.s * span;
  if (in) {
    for (int blk = cw; blk < span / 8; blk += kConsumers) {
      const int c0 = a0 * 16 + blk * 128 + lane * 4;
      float v[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) v[c] = RoundH(RoundH(job.src[c0 + c]) * __half2float(job.suh[c0 + c]));
      Wht128(v);
      __half* dst = reinterpret_cast<__half*>(xs) + blk * 128 + lane * 4;
#pragma unroll
      for (int c = 0; c < 4; ++c) dst[c] = __float2half(v[c] * kHadNorm);
    }
  } else {
    const uint32_t* x = reinterpret_cast<const uint32_t*>(job.xh) + a0 * 8;
    for (int i = threadIdx.x - 32 * kLoaders; i < span * 8; i += 32 * kConsumers) xs[i] = x[i];
  }
}

// Compute-wave barrier on an LDS counter; the loader waves never join it.
__device__ __forceinline__ void ConsumerBarrier(uint32_t* arrived, uint32_t round, int lane) {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  if (lane == 0) __hip_atomic_fetch_add(arrived, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  while (__hip_atomic_load(arrived, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) < kConsumers * round)
    __builtin_amdgcn_s_sleep(1);
  asm volatile("" ::: "memory");
}

// MaxK 8 walks K7 and K8 first, then the narrow list (as DispatchLinGemv).
template <bool Front, size_t N = 0, int MaxK = kMaxK>
__device__ __forceinline__ void LoadAny(int K, const LinJob& job, const GemvItem& item, uint32_t* payload,
                                       Control& control, uint32_t seq) {
  if constexpr (MaxK > kMaxNarrowDecodeK && N == 0) {
    static_assert(MaxK == kMaxWideDecodeK);
    if (K == 7) { Load<7, Front, MaxK>(job, item, payload, control, seq); return; }
    if (K == 8) { Load<8, Front, MaxK>(job, item, payload, control, seq); return; }
  }
  if (K == kSupportedDecodeK[N]) Load<kSupportedDecodeK[N], Front, MaxK>(job, item, payload, control, seq);
  else if constexpr (N + 1 < kSupportedDecodeK.size())
    LoadAny<Front, N + 1, MaxK>(K, job, item, payload, control, seq);
}

template <size_t N = 0, int MaxK = kMaxK>
__device__ __forceinline__ void ComputeAny(int K, const LinJob& job, const GemvItem& item,
                                          const uint32_t* payload, Control& control, const uint32_t* xs,
                                          uint32_t seq) {
  if constexpr (MaxK > kMaxNarrowDecodeK && N == 0) {
    if (K == 7) { Compute<7, MaxK>(job, item, payload, control, xs, seq); return; }
    if (K == 8) { Compute<8, MaxK>(job, item, payload, control, xs, seq); return; }
  }
  if (K == kSupportedDecodeK[N]) Compute<kSupportedDecodeK[N], MaxK>(job, item, payload, control, xs, seq);
  else if constexpr (N + 1 < kSupportedDecodeK.size())
    ComputeAny<N + 1, MaxK>(K, job, item, payload, control, xs, seq);
}

__device__ __forceinline__ uint32_t Chunks(const LinJob& job) {
  return (job.I / 16 / job.S + kChunk - 1) / kChunk;
}

}  // namespace lin_ring

// R=1 ring without a block barrier between items: the loaders stream the
// next item's chunks while the compute waves finish the current one. Items
// are strided over the grid (the front order needs adjacent chunks in
// flight). in: the input transform runs in the item fill (no LinInKernel).
// out: the last split of each column group writes the output transform, as
// in LinPolicyKernel (no LinOutKernel). Dynamic LDS: two activation buffers.
// MaxK 8: the wide groups of the h6_ng8 packs (K7, K8, and any narrower job of the group) in 8 KB slots.
template <bool Front = false, int MaxK = lin_ring::kMaxK>
__global__ void __launch_bounds__(320) LinRingFlowKernel(const LinJob* __restrict__ jobs,
                                                        const GemvItem* __restrict__ items, int n,
                                                        int xs_words, bool in, bool out) {
  __shared__ lin_ring::Control control;
  __shared__ uint32_t payload[lin_ring::kSlots * lin_ring::SlotWords<MaxK>];
  __shared__ uint32_t arrived;
  extern __shared__ uint32_t xs[];
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
      const int K = __builtin_amdgcn_readfirstlane(job.K);
      lin_ring::LoadAny<Front, 0, MaxK>(K, job, item, payload, control, seq);
      seq += lin_ring::Chunks(job);
    }
    return;
  }
  const int cw = __builtin_amdgcn_readfirstlane((threadIdx.x >> 5) - lin_ring::kLoaders);
  uint32_t round = 0;
  for (int it = blockIdx.x, k = 0; it < n; it += gridDim.x, ++k) {
    const GemvItem item = items[it];
    const LinJob& job = jobs[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    // A wave writes buffer k & 1 only after the barrier of item k - 1, which
    // every wave reaches after it finished item k - 2.
    uint32_t* x = xs + (k & 1) * xs_words;
    lin_ring::FillActivations(job, item, in, x);
    lin_ring::ConsumerBarrier(&arrived, ++round, lane);
    lin_ring::ComputeAny<0, MaxK>(K, job, item, payload, control, x, seq);
    seq += lin_ring::Chunks(job);
    if (out) {
      // Arrival order selects the writer, never the reduction order.
      __threadfence();
      lin_ring::ConsumerBarrier(&arrived, ++round, lane);
      if (cw == 0) {
        bool last = job.S == 1;
        if (!last && lane == 0) {
          last = __hip_atomic_fetch_add(job.done + item.b0 / 8, 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT) == unsigned(job.S - 1);
          if (last) __hip_atomic_store(job.done + item.b0 / 8, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (__shfl(last, 0)) {
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
          LinWriteBlock(job, 0, item.b0 / 8);
        }
      }
    }
  }
}

}  // namespace crucible
