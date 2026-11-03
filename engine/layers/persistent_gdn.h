// Persistent R=1 attention-side prototype. MoE remains a separate opaque stage.
#pragma once
// Serial decode only: R=1, no replay, commit 1. The launch checks the contract.
#ifndef CRUCIBLE_PERSISTENT_GDN_SERIAL
#define CRUCIBLE_PERSISTENT_GDN_SERIAL 1
#endif
#include "../model/flashnext/model.h"
#include "../exl3/plans.h"
#include "../exl3/exl3_ring.h"
#include "persistent_ops.h"
#include "../exl3/balanced_items.h"
#include "persistent_hc.h"

namespace crucible {

struct PersistentLinear {
  const LinJob* jobs;
  const GemvItem* items;
  const XfItem *in, *out;
  int nitems, nin, nout;
  const int* offsets;  // per-block item ranges; null: strided items
};
// Front order: strided eight-column items of the plan, or balanced items
// built with front = true.
inline PersistentLinear PersistentFrontGroup(const Group& g, const BalancedItems* b = nullptr) {
  if (b) return {g.d_front, b->items, g.d_in, g.d_out, b->n, g.nin, g.nout, b->offsets};
  return {g.d_front, g.d_items, g.d_in, g.d_out, g.nitems, g.nin, g.nout, nullptr};
}
inline PersistentLinear PersistentGroup(const Group& g, const BalancedItems* b = nullptr) {
  if (b) return {g.d_jobs, b->items, g.d_in, g.d_out, b->n, g.nin, g.nout, b->offsets};
  return {g.d_jobs, g.d_items, g.d_in, g.d_out, g.nitems, g.nin, g.nout, nullptr};
}
struct PersistentGdnArgs {
  Hc hc;
  PersistentLinear in, out;
  GdnArgs gdn;
  float *H, *dots, *ss, *x, *post;
  uint32_t *bar, *error;  // bar: [barrier count, exit count], both zero between launches
  uint64_t* clocks;  // optional: block 0 phase end times
};

namespace persistent_gdn {
__device__ __forceinline__ void StampClock(uint64_t* clocks, const ComputeContext& ct, int k) {
  if (clocks && blockIdx.x == 0 && ct.t == 0) clocks[k] = wall_clock64();
}
template <bool Loader, bool Front, size_t N = 0>
__device__ __forceinline__ void RingDispatch(int K, const LinJob& job, const GemvItem& item,
                                            uint32_t* payload, lin_ring::Control& ring,
                                            uint32_t* xs, uint32_t seq) {
  if (K == kSupportedDecodeK[N]) {
    if constexpr (Loader) lin_ring::Load<kSupportedDecodeK[N], Front>(job, item, payload, ring, seq);
    else lin_ring::Compute<kSupportedDecodeK[N]>(job, item, payload, ring, xs, seq);
  } else if constexpr (N + 1 < kSupportedDecodeK.size()) {
    RingDispatch<Loader, Front, N + 1>(K, job, item, payload, ring, xs, seq);
  }
}

template <bool Loader, bool Front>
__device__ __forceinline__ void RingPhase(PersistentLinear p, uint32_t* payload,
                                         lin_ring::Control& ring, uint32_t* xs,
                                         uint32_t& seq, ComputeContext& ct) {
  const int first = p.offsets ? StreamConstant(p.offsets + blockIdx.x) : (int)blockIdx.x;
  const int last = p.offsets ? StreamConstant(p.offsets + blockIdx.x + 1) : p.nitems;
  const int step = p.offsets ? 1 : (int)gridDim.x;
  for (int it = first; it < last; it += step) {
    const GemvItem item = StreamConstant(p.items + it);
    const LinJob job = StreamConstant(p.jobs + item.job);
    const int span = job.I / 16 / job.S;
    if constexpr (!Loader) {
      const uint32_t* x = reinterpret_cast<const uint32_t*>(job.xh);
      for (int i = ct.t; i < span * 8; i += 256) xs[i] = x[item.s * span * 8 + i];
      ComputeBarrier(ct);
    }
    RingDispatch<Loader, Front>(job.K, job, item, payload, ring, xs, seq);
    seq += (span + lin_ring::kChunk - 1) / lin_ring::kChunk;
    if constexpr (!Loader) ComputeBarrier(ct);
  }
}

// Loader waves touch one dword per 128-byte line, so later ring loads and the
// recurrence's state loads hit L2 or MALL. The values only feed a sink.
__device__ __forceinline__ uint32_t PrefetchSpan(const void* base, uint32_t bytes, uint32_t sink) {
  const int lane = threadIdx.x;  // 0..63: both loader waves
  const auto rs = StreamRsrc(reinterpret_cast<const uint32_t*>(base), bytes);
  const uint32_t lines = (bytes + 127) / 128;
  for (uint32_t l0 = 0; l0 < lines; l0 += 64 * 16) {
    uint32_t v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const uint32_t l = l0 + k * 64 + lane;
      v[k] = l < lines ? __builtin_amdgcn_raw_buffer_load_b32(rs, l * 128, 0, 0) : 0u;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) sink ^= v[k];
  }
  return sink;
}

// This block's items in phase p: each column's split span is one run, or in
// front order each chunk of the item.
template <bool Front>
__device__ __forceinline__ uint32_t PrefetchPhase(PersistentLinear p, uint32_t sink) {
  const int first = p.offsets ? StreamConstant(p.offsets + blockIdx.x) : (int)blockIdx.x;
  const int last = p.offsets ? StreamConstant(p.offsets + blockIdx.x + 1) : p.nitems;
  const int step = p.offsets ? 1 : (int)gridDim.x;
  for (int it = first; it < last; it += step) {
    const GemvItem item = StreamConstant(p.items + it);
    const LinJob job = StreamConstant(p.jobs + item.job);
    const int A = job.I / 16, span = A / job.S, tile = 32 * job.K;
    if constexpr (Front) {
      const int items = job.S * (job.O / 128), idx = item.s * (job.O / 128) + item.b0 / 8;
      const int chunk = 32 * tile, part = 4 * tile;  // part: one column of a chunk
      for (int q = 0; q < span / lin_ring::kChunk; ++q)
        sink = PrefetchSpan(reinterpret_cast<const uint8_t*>(job.tiles) + ((size_t)q * items + idx) * chunk +
                                (item.b0 & 7) * part,
                            item.nb * part, sink);
      continue;
    }
    for (int c = 0; c < item.nb; ++c)
      sink = PrefetchSpan(reinterpret_cast<const uint8_t*>(job.tiles) +
                              ((size_t)(item.b0 + c) * A + (size_t)item.s * span) * tile,
                          span * tile, sink);
  }
  return sink;
}
}  // namespace persistent_gdn

#ifndef CRUCIBLE_PERSISTENT_GDN_PREFETCH
#define CRUCIBLE_PERSISTENT_GDN_PREFETCH 1
#endif
// Nine waves per SIMD (168 VGPRs) give three blocks per WGP: 60 blocks.
// Four blocks per WGP need twelve (128 VGPRs).
// HC read: 2 streams its weights through the loader ring, 1 pipelines the
// loads in the compute waves, 0 runs the per-op contraction per task.
#ifndef CRUCIBLE_PERSISTENT_HC_PIPE
#define CRUCIBLE_PERSISTENT_HC_PIPE 2
#endif
// Dynamic LDS: the ring activations, and the staged HC stream before them.
inline size_t PersistentGdnLds(size_t ring_lds) {
  return std::max(ring_lds, (size_t)(CRUCIBLE_PERSISTENT_HC_PIPE ? kHidden * 4 : 0));
}

// Front: projection jobs hold front-order tiles and strided eight-column items.
template <bool Front>
__global__ void __launch_bounds__(320)
__attribute__((amdgpu_waves_per_eu(9)))  // the nine waves above
PersistentGdnKernel(PersistentGdnArgs a) {
  __shared__ lin_ring::Control ring;
  __shared__ PersistentSync sync;
  __shared__ uint32_t payload[lin_ring::kSlots * lin_ring::kSlotWords];
  extern __shared__ uint32_t xs[];
  ring.Initialize(threadIdx.x);
  if (threadIdx.x == 0) sync.compute_arrivals = sync.phase_ok = 0;
  __syncthreads();
  ComputeContext ct{(int)threadIdx.x - 64, 0, 0, &sync, a.bar, a.error};
  uint32_t seq = 0;
  if (threadIdx.x < 64) {
    // These addresses are static. The loaders run across the recurrence and
    // every activation dependency without joining a compute or grid barrier.
#if CRUCIBLE_PERSISTENT_HC_PIPE == 2
    persistent_hc::LoadDots(a.hc.fn, a.hc.M, payload, ring, seq);
    persistent_hc::LoadMix(a.hc.up, payload, ring, seq);
#endif
    persistent_gdn::RingPhase<true, Front>(a.in, payload, ring, xs, seq, ct);
#if CRUCIBLE_PERSISTENT_GDN_PREFETCH
    // The in-proj stream has ended. While compute runs LinOut and the
    // recurrence, warm this block's head state and its out-proj tiles.
    uint32_t sink = 0;
    for (int h = blockIdx.x; h < 48; h += gridDim.x)
      sink = persistent_gdn::PrefetchSpan(a.gdn.state + (size_t)h * 128 * 128, 128 * 128 * 4, sink);
    sink = persistent_gdn::PrefetchPhase<Front>(a.out, sink);
    if (sink == 0x9e3779b9u) a.error[0] = sink;
#endif
    persistent_gdn::RingPhase<true, Front>(a.out, payload, ring, xs, seq, ct);
    return;
  }
  persistent_gdn::StampClock(a.clocks, ct, 0);
#if CRUCIBLE_PERSISTENT_HC_PIPE == 2
  PersistentHcDotsRing(ct, a.hc.M, a.H, a.dots, a.ss, reinterpret_cast<float*>(xs), payload, ring, seq);
#elif CRUCIBLE_PERSISTENT_HC_PIPE
  PersistentHcDotsPipe(ct, a.hc.fn, a.hc.M, a.H, a.dots, a.ss, reinterpret_cast<float*>(xs));
#else
  const int ndots = ((a.hc.M + 1) * kStreams + 7) / 8;
  for (int task = blockIdx.x; task < ndots; task += gridDim.x)
    PersistentHcDots<1>(ct, task, a.hc.fn, a.hc.M, a.H, a.dots, a.ss);
#endif
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 1);
#if CRUCIBLE_PERSISTENT_HC_PIPE == 2
  PersistentHcMixRing(ct, a.hc.w1, a.dots, a.hc.M, a.ss, a.H, a.x, a.post, payload, ring, seq);
#elif CRUCIBLE_PERSISTENT_HC_PIPE
  PersistentHcMixPipe(ct, a.hc.up, a.hc.w1, a.dots, a.hc.M, a.ss, a.H, a.x, a.post);
#else
  for (int task = blockIdx.x; task < kHidden / 16; task += gridDim.x)
    PersistentHcMix<1>(ct, task, a.hc.up, a.hc.w1, a.dots, a.hc.M, a.ss, a.H, a.x, a.post);
#endif
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 2);
  for (int task = blockIdx.x; task < (a.in.nin + 7) / 8; task += gridDim.x)
    PersistentLinIn(ct, task, a.in.jobs, a.in.in, a.in.nin);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 3);
  persistent_gdn::RingPhase<false, Front>(a.in, payload, ring, xs, seq, ct);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 4);
  for (int task = blockIdx.x; task < (a.in.nout + 7) / 8; task += gridDim.x)
    PersistentLinOut(ct, task, a.in.jobs, a.in.out, a.in.nout);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 5);
  for (int task = blockIdx.x; task < 48; task += gridDim.x) PersistentGdn(ct, task, a.gdn);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 6);
  persistent_gdn::RingPhase<false, Front>(a.out, payload, ring, xs, seq, ct);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 7);
  for (int task = blockIdx.x; task < (a.out.nout + 7) / 8; task += gridDim.x)
    PersistentLinOut(ct, task, a.out.jobs, a.out.out, a.out.nout);
  PersistentGridBarrier(ct);
  persistent_gdn::StampClock(a.clocks, ct, 8);
  // Every block has left its last grid wait before it counts its exit, so the
  // last block can zero both words for the next launch without a memset.
  if (ct.t == 0 &&
      __hip_atomic_fetch_add(a.bar + 1, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == gridDim.x - 1) {
    __hip_atomic_store(a.bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(a.bar + 1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// The HIP occupancy query assumes 64 KB of LDS per WGP on gfx1151 and
// reports 40 blocks. The WGP has 128 KB of LDS and 1,536 VGPRs per lane in
// each of its four SIMDs (granule 8, at most 16 waves). Wave placement is not
// even: 80 blocks of 10 waves at 144 VGPRs (10 waves per SIMD) never all
// started, so count three waves per SIMD per block.
inline int PersistentGdnBlocks(size_t dynamic_lds) {
  hipFuncAttributes f, g;
  HIP_OK(hipFuncGetAttributes(&f, reinterpret_cast<const void*>(PersistentGdnKernel<false>)));
  HIP_OK(hipFuncGetAttributes(&g, reinterpret_cast<const void*>(PersistentGdnKernel<true>)));
  f.numRegs = std::max(f.numRegs, g.numRegs);
  f.sharedSizeBytes = std::max(f.sharedSizeBytes, g.sharedSizeBytes);
  const int vgpr = (f.numRegs + 7) / 8 * 8, waves = std::min(16, 1536 / std::max(8, vgpr));
  const int by_vgpr = waves / 3;
  const int by_lds = (128 << 10) / (int)(f.sharedSizeBytes + PersistentGdnLds(dynamic_lds));
  const int resident = 20 * std::min(by_vgpr, by_lds);
  if (resident < 1) {
    std::fprintf(stderr, "persistent GDN needs a resident workgroup\n");
    std::exit(2);
  }
  return std::min(80, resident) / 4 * 4;  // the HC read needs a multiple of four
}

// blocks must come from PersistentGdnBlocks for the same dynamic LDS: a grid
// that is not co-resident never passes its first grid barrier.
inline void LaunchPersistentGdn(PersistentGdnArgs a, size_t dynamic_lds, int blocks, bool front,
                                 hipStream_t stream = nullptr) {
#if CRUCIBLE_PERSISTENT_GDN_SERIAL
  if (a.gdn.R != 1 || a.gdn.replay != 0 || a.gdn.commit != 1) {
    std::fprintf(stderr, "serial persistent GDN needs R=1, replay=0, commit=1\n");
    std::exit(2);
  }
#endif
  const size_t lds = PersistentGdnLds(dynamic_lds);
  if (front) PersistentGdnKernel<true><<<blocks, 320, lds, stream>>>(a);
  else PersistentGdnKernel<false><<<blocks, 320, lds, stream>>>(a);
}

}  // namespace crucible
