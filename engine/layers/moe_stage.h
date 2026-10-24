// The routed-expert stage of a decode forward in one launch (R = 1).
//
// MoeRows (engine.hip) runs five kernels after the router matvec: MoeRouteKernel,
// MoeRowsUniqueKernel, MoeRowsInSlotsKernel, MoeFusedKernel and MoeRowsCombineKernel.
// The four small ones move almost no bytes; at R = 1 they cost 1.49 ms per
// forward plus about 1 ms of kernel boundaries (oracles/roofline.md 4.1).
// MoeStageKernel does their work inside the MoeFusedKernel launch:
// - every block ranks the router logits itself (MoeStageRouteRow: the same
//   winners and the same softmax expressions and sum order as
//   MoeRouteKernel) and builds the unique-job
//   map (MoeRowsUniqueBody's per-slot code) in LDS;
// - a block's own jobs are copies of the planned jobs with the expert
//   pointers that MoeRowsUniqueKernel would write; the global jobs stay as
//   planned;
// - gate/up items run the input transform of MoeRowsInSlotsKernel for their
//   rows straight into LDS (the xh layout MoeFillX stages);
// - each down item counts itself in its 256-column group; the last one of a
//   group runs MoeRowsCombineKernel's code for those columns (slot order
//   unchanged); the last group resets the counters for the next launch.
// Each value has the same expression and order as in the five kernels, so
// results are bitwise equal (Wht128Dpp is Wht128 with DPP moves, common.h).
#pragma once

#include "moe_rows.h"

namespace crucible {

constexpr int kMoeStageGroups = kHidden / 256;  // combine groups: one down item width (16 column blocks)
// sync words: done per job, arrivals per group, finished groups. Zero between launches.
constexpr int kMoeStageDone = 0;
constexpr int kMoeStageGroup = MoeExpertJobs(kMaxRows);
constexpr int kMoeStageFin = kMoeStageGroup + kMoeStageGroups;
constexpr int kMoeStageWords = kMoeStageFin + 1;
static_assert(kHidden % 256 == 0 && kMoeItemBlocks<kMoeCols> == 16, "a down item spans one combine group");

struct MoeStageArgs {
  const LinJob* gu;
  const GemvItem* gu_items;
  int n_gu;
  const LinJob* dn;
  const GemvItem* dn_items;
  int n_dn;
  const float* logits;  // [R][kExperts + 1] router logits, the shared gate last
  const ExpertPtrs* experts;
  const float* post;    // [R][kStreams] inject gates
  float* H;             // residual streams
  unsigned* sync;       // kMoeStageWords
};

template <int R>
struct MoeRouteState {
  int sel[R * kTopK];
  float wts[R * kSlots];
  int occ[R * kTopK];
  int map[R * kTopK], mapj[R * kTopK];
  int list[R * kTopK], count[R * kTopK];
  int rows[R * kTopK * kMaxRows];
  int n;
  float top[R * kTopK];               // the winners' logits by rank
  float cv[R * kExperts];             // route candidates of each row: logits >= T
  int ci[R * kExperts];               // and their expert IDs
  const float* cpart[R * kSlots];     // MoeStageSlots
  const __half* csvh[R * kSlots];
};

// Orders LDS stores of a wave before its later LDS loads (the waves of a block
// are not involved).
__device__ __forceinline__ void MoeWaveSync() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
  __builtin_amdgcn_wave_barrier();
}

// MoeRouteKernel's result for row r (one wave), into LDS. MoeRouteKernel
// ranks each logit against all 512 (logit desc, id asc); a serial wave
// top-10 takes about 28 us on the critical path of every block (stamps).
// Here: T is the tenth largest of the 32 lane
// maxima. Ten lanes hold a logit >= T, so the ten winners are all >= T, and a
// logit below T outranks none of them. Ranking only the logits >= T with
// MoeRouteKernel's predicate gives its ranks, so sel and the weights (same
// expressions, same sum order) are equal for rows of non-NaN logits. A NaN
// logit is never a candidate (MoeRouteKernel gives it rank 0); such a row
// gives non-finite outputs on either path, and sel stays in range.
template <int R>
__device__ __forceinline__ void MoeStageRouteRow(const float* __restrict__ lg, int r, MoeRouteState<R>& st) {
  const int lane = threadIdx.x & 31;
  constexpr int kPer = kExperts / 32;
  float v[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) v[j] = lg[lane + 32 * j];
  if (lane < kTopK) {
    st.sel[r * kTopK + lane] = 0;
    st.top[r * kTopK + lane] = 0.f;
  }
  float mx = v[0];
#pragma unroll
  for (int j = 1; j < kPer; ++j) mx = fmaxf(mx, v[j]);
  int rk = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const float o = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(mx), j));
    rk += (o > mx) || (o == mx && j < lane);
  }
  const unsigned at = (unsigned)__ballot(rk == kTopK - 1);
  const float T = at ? __int_as_float(__builtin_amdgcn_readlane(__float_as_int(mx), __builtin_ctz(at))) : -INFINITY;
  float* cv = st.cv + r * kExperts;
  int* ci = st.ci + r * kExperts;
  const unsigned lower = (1u << lane) - 1;
  int n = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const bool c = v[j] >= T;
    const unsigned m = (unsigned)__ballot(c);
    if (c) {
      const int q = n + __popc(m & lower);
      cv[q] = v[j];
      ci[q] = lane + 32 * j;
    }
    n += __popc(m);
  }
  MoeWaveSync();
  for (int i = lane; i < n; i += 32) {
    const float x = cv[i];
    const int e = ci[i];
    int rank = 0;
    for (int q = 0; q < n; ++q) {
      const float o = cv[q];
      rank += (o > x) || (o == x && ci[q] < e);
    }
    if (rank < kTopK) {
      st.sel[r * kTopK + rank] = e;
      st.top[r * kTopK + rank] = x;
    }
  }
  MoeWaveSync();
  const float* sl = st.top + r * kTopK;
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) sum += __expf(sl[k] - sl[0]);
  if (lane < kTopK) st.wts[r * kSlots + lane] = __expf(sl[lane] - sl[0]) / sum;
  if (lane == 0) st.wts[r * kSlots + kTopK] = Sigmoid(lg[kExperts]);
}

// The route of all R rows and MoeRowsUniqueBody's map, in LDS. Every thread calls it.
template <int R>
__device__ __forceinline__ void MoeStageRoute(const float* __restrict__ logits, MoeRouteState<R>& st) {
  const int t = threadIdx.x, slots = R * kTopK;
  for (int r = t >> 5; r < R; r += blockDim.x >> 5) MoeStageRouteRow<R>(logits + (size_t)r * (kExperts + 1), r, st);
  if (t == 0) st.n = 0;
  __syncthreads();
  if (t < slots) st.occ[t] = MoeSlotOcc(st.sel, t);
  __syncthreads();
  if (t < slots) {
    const int u = MoeSlotJob(st.sel, st.occ, t), j = st.occ[t] % kMaxRows;
    st.map[t] = u;
    st.mapj[t] = j;
    st.rows[u * kMaxRows + j] = t / kTopK;
    if (j == 0) {
      st.list[u] = st.sel[t];
      st.count[u] = MoeSlotCount(st.sel, st.occ, R, t);
      atomicAdd(&st.n, 1);
    }
  }
  __syncthreads();
}

// Planned job slot u with the pointers MoeRowsUniqueBody writes; m: 0 gate,
// 1 up, 2 down. An unused routed slot gets K = 0.
template <int R>
__device__ __forceinline__ LinJob MoeStageJob(const LinJob* jobs, int index, int u, int m,
                                              const ExpertPtrs* __restrict__ experts, const MoeRouteState<R>& st) {
  LinJob j = jobs[index];
  if (u >= R * kTopK) return j;
  if (u >= st.n) {
    j.K = 0;
    return j;
  }
  const ExpertPtrs& ep = experts[st.list[u]];
  j.tiles = ep.tiles[m];
  j.suh = ep.suh[m];
  j.svh = ep.svh[m];
  j.K = ep.K[m];
  j.R = st.count[u];
  return j;
}

// MoeRowsInSlotsKernel's transform of job u's rows into LDS (MoeFillX's layout, S = 1).
template <int R>
__device__ __forceinline__ void MoeStageX(const LinJob& job, int u, int rows, const MoeRouteState<R>& st,
                                          uint32_t* xs) {
  __half* xh = reinterpret_cast<__half*>(xs);
  const int lane = threadIdx.x & 31, nb = job.I / 128;
  for (int t = threadIdx.x >> 5; t < rows * nb; t += blockDim.x >> 5) {
    const int jr = t / nb, blk = t - jr * nb;
    const int row = u >= R * kTopK ? (u - R * kTopK) * kMaxRows + jr : st.rows[u * kMaxRows + jr];
    const int c0 = blk * 128 + lane * 4;
    const float* s = job.src + (size_t)row * job.lds + c0;
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = RoundH(RoundH(s[k]) * __half2float(job.suh[c0 + k]));
    Wht128Dpp(v);
#pragma unroll
    for (int k = 0; k < 4; ++k) xh[jr * job.I + c0 + k] = __float2half(v[k] * kHadNorm);
  }
  __syncthreads();
}

// The combine's slot pointers (row j of the slot's down partials, and its
// output scales), so the last arriver of a group starts with its data loads.
// Threads 0 .. R * kSlots - 1 call it after the route; the body's barriers
// order it before MoeStageCombine.
template <int R>
__device__ __forceinline__ void MoeStageSlots(const MoeStageArgs& a, MoeRouteState<R>& st) {
  const int t = threadIdx.x;
  if (t >= R * kSlots) return;
  const int r = t / kSlots, k = t - r * kSlots;
  const int u = k < kTopK ? st.map[r * kTopK + k] : MoeSharedJob(R, r);
  const int j = k < kTopK ? st.mapj[r * kTopK + k] : MoeSharedRow(r);
  const LinJob& d = a.dn[u];
  st.cpart[t] = d.partial + (size_t)j * d.O;
  st.csvh[t] = k < kTopK ? a.experts[st.list[u]].svh[2] : d.svh;
}

// R = 1 gate/up source: wave w's blocks w, w + 8, w + 16 of row 0, rounded to
// fp16 as MoeStageX does first. Loaded before the route.
constexpr int kMoeStagePre = (kHidden / 128 + 7) / 8;
__device__ __forceinline__ void MoeStagePrefetch(const LinJob& plan, float (&pre)[kMoeStagePre][4]) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nb = plan.I / 128;
  const float* src = plan.src;
#pragma unroll
  for (int it = 0; it < kMoeStagePre; ++it) {
    const int t = w + 8 * it, c0 = t * 128 + lane * 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) pre[it][k] = 0.f;
    if (t < nb) {
#pragma unroll
      for (int k = 0; k < 4; ++k) pre[it][k] = RoundH(src[c0 + k]);
    }
  }
}

// MoeStageX at R = 1 from the prefetched source (row 0; MoeStageJob keeps the plan's src).
__device__ __forceinline__ void MoeStageX1(const LinJob& job, const float (&pre)[kMoeStagePre][4], uint32_t* xs) {
  __half* xh = reinterpret_cast<__half*>(xs);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nb = job.I / 128;
#pragma unroll
  for (int it = 0; it < kMoeStagePre; ++it) {
    const int t = w + 8 * it;
    if (t >= nb) break;  // wave uniform
    const int c0 = t * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = RoundH(pre[it][k] * __half2float(job.suh[c0 + k]));
    Wht128Dpp(v);
#pragma unroll
    for (int k = 0; k < 4; ++k) xh[c0 + k] = __float2half(v[k] * kHadNorm);
  }
  __syncthreads();
}

// MoeRowsCombineKernel for the columns of group g (two 128-wide blocks) of every row.
template <int R>
__device__ __forceinline__ void MoeStageCombine(const MoeStageArgs& a, int g, const MoeRouteState<R>& st) {
  const int lane = threadIdx.x & 31;
  for (int t = threadIdx.x >> 5; t < R * 2; t += blockDim.x >> 5) {
    const int r = t >> 1, blk = 2 * g + (t & 1);
    const int c0 = blk * 128 + lane * 4;
    // The residual streams and gates load with the first slots; the update below is unchanged.
    float gs[kStreams], hv[kStreams][4];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      gs[s] = a.post[r * kStreams + s];
      const float4 h = *reinterpret_cast<const float4*>(a.H + ((size_t)r * kStreams + s) * kHidden + c0);
      hv[s][0] = h.x;
      hv[s][1] = h.y;
      hv[s][2] = h.z;
      hv[s][3] = h.w;
    }
    float y[4] = {0.f, 0.f, 0.f, 0.f};
    // Slots in groups of kG: all loads of a group before its Hadamards. The kernel's register count is the
    // maximum over its paths, so all 11 slots in flight would cut the expert items' occupancy.
    constexpr int kG = 6;
#pragma unroll 1
    for (int k0 = 0; k0 < kSlots; k0 += kG) {
      float raw[kG][4], sv[kG][4], wk[kG];
#pragma unroll
      for (int q = 0; q < kG; ++q) {
        const int k = k0 + q;
        if (k >= kSlots) break;
        const float4 p = *reinterpret_cast<const float4*>(st.cpart[r * kSlots + k] + c0);
        raw[q][0] = p.x;
        raw[q][1] = p.y;
        raw[q][2] = p.z;
        raw[q][3] = p.w;
        const uint2 hs = *reinterpret_cast<const uint2*>(st.csvh[r * kSlots + k] + c0);
        const __half* h = reinterpret_cast<const __half*>(&hs);
#pragma unroll
        for (int i = 0; i < 4; ++i) sv[q][i] = __half2float(h[i]);
        wk[q] = st.wts[r * kSlots + k];
      }
#pragma unroll
      for (int q = 0; q < kG; ++q) {
        if (k0 + q >= kSlots) break;
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = 0.f + raw[q][i];  // LinOutBlock's S=1 sum, signed zeros included
        Wht128Dpp(v);
#pragma unroll
        for (int i = 0; i < 4; ++i) y[i] += wk[q] * (v[i] * (kHadNorm * sv[q][i]));
      }
    }
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      float4 h;
      h.x = hv[s][0] + gs[s] * y[0];
      h.y = hv[s][1] + gs[s] * y[1];
      h.z = hv[s][2] + gs[s] * y[2];
      h.w = hv[s][3] + gs[s] * y[3];
      *reinterpret_cast<float4*>(a.H + ((size_t)r * kStreams + s) * kHidden + c0) = h;
    }
  }
}

// MoeFusedKernel<R, KR, PF, CACHE, C> (XC = 0) with the route, map, input
// transform and combine inside. Launch with grid n_gu + n_dn and
// MoeFusedLds(R, 0, ...) bytes of dynamic LDS; a.sync must be zero (it is
// after the previous MoeStageKernel launch).
// CG, PFG: the gate/up items' column blocks per wave and prefetch depth (the
// items must span kMoeItemBlocks<CG> blocks). CG = 1, PFG = 8 at R = 1 puts
// twice as many waves on the gate/up stream (8 tiles in flight each, as a
// C = 2 wave); the per-row arithmetic is that of C = 2 (MoeGemvCore). 3.05
// ABAB against CG = 2, PFG = 4: serial +0.51%, spec 7 +0.34% (5 pairs).
// The kernel is held to 96 VGPRs (16 waves per SIMD): with 104 VGPRs (12
// waves) a CU holds 3 blocks, and the 110 down items no longer all fit beside
// the 110 gate/up items (stamps: the last down items start only when earlier
// down items end).
template <int R, int KR, int PF = kMoePf, int CACHE = 0, int C = kMoeCols, int CG = 1, int PFG = 8>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16))) MoeStageKernel(MoeStageArgs a) {
  static_assert(R >= 1 && R <= kMaxRows, "one shared chunk, full-span x staging");
  extern __shared__ uint32_t xs[];
  __shared__ MoeRouteState<R> st;
  __shared__ LinJob sj[3];
  __shared__ bool last;
  unsigned* done = a.sync + kMoeStageDone;
  const int bid = blockIdx.x;
  // Shared-expert gate/up items come first and skip the route (they need none).
  const int n_sh = MoeSharedChunks(R) * kMoeGuItems<CG>;
  const bool gate_up = bid < a.n_gu;
  const int gi = gate_up ? (bid + a.n_gu - n_sh) % a.n_gu : 0;
  const GemvItem gitem = gate_up ? a.gu_items[gi] : GemvItem{};
  const bool shared_gu = gate_up && (gitem.job >> 1) >= R * kTopK;
  // R = 1: gate/up items load their source row while the route runs (MoeStageX1).
  float pre[kMoeStagePre][4];
  if constexpr (R == 1)
    if (gate_up) MoeStagePrefetch(a.gu[gitem.job], pre);
  if (!shared_gu) MoeStageRoute<R>(a.logits, st);
  if (gate_up) {
    const GemvItem item = gitem;
    const int u = item.job >> 1;
    if (threadIdx.x == 0) sj[0] = MoeStageJob<R>(a.gu, item.job, u, item.job & 1, a.experts, st);
    __syncthreads();
    const LinJob& job = sj[0];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) return;
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    MoeAnyBody<R, KR, KR + 2, PFG, CG, CACHE>(K, rows, job, item, xs, [&] {
      if constexpr (R == 1) MoeStageX1(job, pre, xs);
      else MoeStageX<R>(job, u, rows, st, xs);
    });
    if (threadIdx.x == 0) __hip_atomic_fetch_add(done + u, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    return;
  }
  const GemvItem item = a.dn_items[bid - a.n_gu];
  const int u = item.job;
  MoeStageSlots<R>(a, st);
  if (threadIdx.x == 0) {
    sj[0] = MoeStageJob<R>(a.gu, u * 2, u, 0, a.experts, st);
    sj[1] = MoeStageJob<R>(a.gu, u * 2 + 1, u, 1, a.experts, st);
    sj[2] = MoeStageJob<R>(a.dn, u, u, 2, a.experts, st);
  }
  __syncthreads();
  const LinJob& d = sj[2];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K != 0) {
    const int rows = __builtin_amdgcn_readfirstlane(d.R);
    // MoeFusedKernel's wait and mid.
    auto prep = [&] {
      if (threadIdx.x == 0)
        while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<CG>)
          __builtin_amdgcn_s_sleep(1);
      __syncthreads();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      __half* xh = reinterpret_cast<__half*>(xs);
      const int lane = threadIdx.x & 31;
      for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
        const int r = t / 5, blk = t % 5;
        float gv[4], uv[4];
        LinOutBlock(sj[0], r, blk, gv);
        LinOutBlock(sj[1], r, blk, uv);
        const int c0 = blk * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(Silu(gv[i]) * uv[i]) * __half2float(d.suh[c0 + i]));
        Wht128Dpp(v);
#pragma unroll
        for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
      }
      __syncthreads();
    };
    MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, d, item, xs, prep);  // ends with a block barrier
  }
  // The body's barrier orders every wave's partial stores before this release.
  const int g = item.b0 / kMoeItemBlocks<C>;
  if (threadIdx.x == 0) {
    if (item.nb != kMoeItemBlocks<C> || item.s != 0) __builtin_trap();  // Build: S = 1, 16-block down items
    const unsigned old = __hip_atomic_fetch_add(a.sync + kMoeStageGroup + g, 1u, __ATOMIC_ACQ_REL,
                                                __HIP_MEMORY_SCOPE_AGENT);
    last = old + 1 == (unsigned)MoeExpertJobs(R);
  }
  __syncthreads();
  if (!last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  MoeStageCombine<R>(a, g, st);
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned f = __hip_atomic_fetch_add(a.sync + kMoeStageFin, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    // Every down item has passed its wait and every group has combined: the
    // counters are free for the next launch (stream order publishes them).
    if (f + 1 == (unsigned)kMoeStageGroups)
      for (int i = 0; i < kMoeStageWords; ++i) a.sync[i] = 0;
  }
}

}  // namespace crucible
