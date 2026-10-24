// Row-sparse routed experts for multi-row verify forwards.
//
// The dense path (MoeUniqueKernel, bench/kernels/moe_rows.h) applies every
// distinct expert to all R
// rows, so a verify forward computes U * R expert rows where only 10 * R are
// routed. Here each distinct expert keeps a compact list of its own rows:
// job.R becomes that count, xh/partial rows are compact, and the combine maps
// (row, slot) to (expert job, compact row). Each kept row runs the same
// LinGemvBody arithmetic as before, so results are bitwise unchanged.
// The dense path and the first compact-row kernels are bench-only now.
#pragma once

#include "ops.h"
#include "moe_chunk_map.h"

namespace crucible {

// rows[u * kMaxRows + j]: activation row of compact row j of unique job u.
// mapj[r * kTopK + k]: compact row of row r inside its k-th expert's job.
// The body needs 512 threads.
__device__ __forceinline__ void MoeRowsUniqueBody(const int* __restrict__ sel, int R,
                                                  const ExpertPtrs* __restrict__ experts, LinJob* gu, LinJob* dn,
                                                  int* __restrict__ map, int* __restrict__ mapj,
                                                  int* __restrict__ rows, unsigned* __restrict__ done) {
  if (done && (int)threadIdx.x < MoeExpertJobs(R)) done[threadIdx.x] = 0;
  __shared__ int ss[kPackedRows * kTopK], occ[kPackedRows * kTopK];
  __shared__ int list[kPackedRows * kTopK];
  __shared__ int count[kPackedRows * kTopK];
  __shared__ int n;
  const int t = threadIdx.x, slots = R * kTopK;
  if (R < 1 || R > kPackedRows) __builtin_trap();
  if (t == 0) n = 0;
  if (t < slots) {
    ss[t] = sel[t];
    if (ss[t] < 0 || ss[t] >= kExperts) __builtin_trap();
  }
  __syncthreads();
  // BuildMoeChunkMap with one thread per slot (moe_chunk_map.h; the CPU test
  // proves the same map): jobs of at most kMaxRows rows, numbered in the
  // order of their first slot. Up to kMaxRows rows this is one job per
  // distinct expert.
  if (t < slots) occ[t] = MoeSlotOcc(ss, t);
  __syncthreads();
  if (t < slots) {
    const int u = MoeSlotJob(ss, occ, t), j = occ[t] % kMaxRows;
    map[t] = u;
    mapj[t] = j;
    rows[u * kMaxRows + j] = t / kTopK;
    if (j == 0) {
      list[u] = ss[t];
      count[u] = MoeSlotCount(ss, occ, R, t);
      atomicAdd(&n, 1);
    }
  }
  __syncthreads();
  const int U = R * kTopK;
  for (int u = t; u < MoeExpertJobs(R); u += blockDim.x) {
    LinJob* g = gu + (size_t)u * 2;
    LinJob* d = dn + u;
    if (u >= U) {
      // Shared expert chunk: rows [first, first + nr), as planned by BuildLayer.
      // The last chunk of a packed pass runs kMaxRows rows; rows from R on
      // are scratch.
      const int first = (u - U) * kMaxRows, nr = min(kMaxRows, R);
      for (int r = 0; r < nr; ++r) rows[u * kMaxRows + r] = first + r;
    } else if (u < n) {
      const ExpertPtrs& ep = experts[list[u]];
      for (int k = 0; k < 2; ++k) {
        g[k].tiles = ep.tiles[k];
        g[k].suh = ep.suh[k];
        g[k].svh = ep.svh[k];
        g[k].K = ep.K[k];
        g[k].R = count[u];
      }
      d->tiles = ep.tiles[2];
      d->suh = ep.suh[2];
      d->svh = ep.svh[2];
      d->K = ep.K[2];
      d->R = count[u];
    } else {
      g[0].K = g[1].K = d->K = 0;
    }
  }
}

__global__ void __launch_bounds__(512) MoeRowsUniqueKernel(const int* __restrict__ sel, int R,
                                                           const ExpertPtrs* __restrict__ experts,
                                                           LinJob* gu, LinJob* dn, int* __restrict__ map,
                                                           int* __restrict__ mapj, int* __restrict__ rows,
                                                           unsigned* __restrict__ done = nullptr) {
  MoeRowsUniqueBody(sel, R, experts, gu, dn, map, mapj, rows, done);
}

// MoeRowsInKernel (bench/kernels/moe_rows.h) without dead waves: launch with
// grid MoeInSlotsBlocks(R).
// Wave w serves selection slot i = w / 40 (row i / kTopK; slots from
// R * kTopK on are the shared expert's rows), matrix (w / 20) % 2 (gate,
// up) and 128-wide block w % 20. Each (unique job, compact row) pair has
// exactly one slot (MoeRowsUniqueKernel's map and mapj), so this writes the
// same values as the bench MoeRowsInKernel with 11 R x 40 waves instead of
// (R x kTopK + 1) x 2 x R x 20, of which most exit at once.
constexpr int kMoeInBlocks = kHidden / 128;
constexpr int MoeInSlotsBlocks(int R) { return (R * (kTopK + 1) * 2 * kMoeInBlocks + 7) / 8; }
__global__ void __launch_bounds__(256) MoeRowsInSlotsKernel(const LinJob* __restrict__ jobs,
                                                            const int* __restrict__ map,
                                                            const int* __restrict__ mapj, int R) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= R * (kTopK + 1) * 2 * kMoeInBlocks) return;
  const int lane = threadIdx.x & 31;
  const int i = w / (2 * kMoeInBlocks), m = (w / kMoeInBlocks) & 1, blk = w % kMoeInBlocks;
  const bool shared = i >= R * kTopK;
  const int row = shared ? i - R * kTopK : i / kTopK;
  const int u = shared ? MoeSharedJob(R, row) : map[i];
  const int jr = shared ? MoeSharedRow(row) : mapj[i];
  const LinJob& j = jobs[2 * u + m];
  const int c0 = blk * 128 + lane * 4;
  const float* s = j.src + (size_t)row * j.lds + c0;
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = RoundH(RoundH(s[k]) * __half2float(j.suh[c0 + k]));
  Wht128(v);
  __half* d = j.xh + (size_t)jr * j.I + c0;
#pragma unroll
  for (int k = 0; k < 4; ++k) d[k] = __float2half(v[k] * kHadNorm);
}

// Stages tiles [a0, a0 + span) of the job's first `rows` x rows in LDS.
__device__ __forceinline__ void MoeFillX(const LinJob& job, const GemvItem& item, int rows, uint32_t* xs) {
  const int A = job.I >> 4, span = A / job.S, a0 = item.s * span;
  const uint32_t* xg = reinterpret_cast<const uint32_t*>(job.xh);
  for (int i = threadIdx.x; i < rows * span * 8; i += blockDim.x) {
    const int r = i / (span * 8), o = i - r * (span * 8);
    xs[i] = xg[(size_t)r * A * 8 + a0 * 8 + o];
  }
  __syncthreads();
}

// A wave-uniform copy of a pointer. Down items read their job after an
// acquire fence, so the job loads are vector loads the compiler treats as
// divergent; a buffer resource built from them gets a readfirstlane
// waterfall loop around every tile load.
template <typename T>
__device__ __forceinline__ T* UniformPtr(T* p) {
  const uint64_t v = reinterpret_cast<uint64_t>(p);
  const uint64_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)v);
  const uint64_t hi = (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)(v >> 32));
  return reinterpret_cast<T*>(lo | hi << 32);
}

// The tile loop of one wave over CC column blocks b[0..CC). Each column keeps
// its own ring and accumulators, so its arithmetic is the same for any CC;
// the x values read from LDS serve all CC columns. CACHE is the buffer-load
// cache policy (aux operand) of the weight tiles.
template <int K, int R, int PF, int CC, int CACHE = 0>
__device__ __forceinline__ void MoeGemvCols(const LinJob& job, int s, const int (&b)[CC], const uint32_t* xs) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = TileDwords(K);  // 8 K dwords; half K: lane_window.h
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
  for (int c = 0; c < CC; ++c)
    rs[c] = StreamRsrc(tiles + ((size_t)b[c] * A + a0) * T, (uint32_t)span * T * 4);
  const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
  auto load_tile = [&](int c, int tile, uint32_t* dst) {
    const int so = tile * T * 4;
    dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o0, so, CACHE);
    dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o1, so, CACHE);
    dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o2, so, CACHE);
  };
  uint32_t ring[PF][CC][3];
  // Issue order must match the refill order in the loop (slot by slot) so
  // the loop head needs only vmcnt(3 * CC * (PF - 1)), not vmcnt(0).
#pragma unroll
  for (int f = 0; f < PF; ++f) {
#pragma unroll
    for (int c = 0; c < CC; ++c) load_tile(c, f, ring[f][c]);
    __builtin_amdgcn_sched_barrier(0);
  }
  float acc[CC][R][2];
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[c][r][0] = acc[c][r][1] = 0.f;
  auto step = [&](int a, int f) {
    __half2 P[CC][4];
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      DecodeLane<K, kMulMode>(ring[f][c][0], ring[f][c][1], ring[f][c][2], win.shift, P[c]);
      // Refill the slot only after the decode has consumed it, so the new
      // load can reuse the same registers. Loading first forces a second
      // register set and v_mov copies (with vmcnt waits) at every round.
      load_tile(c, a + f + PF, ring[f][c]);
    }
    // Pins the refills here. Without it the scheduler sinks all PF refills
    // to the end of the round in another order, and the next round starts
    // with vmcnt(0): one exposed memory latency per PF tiles.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t* xr = xs + (r * span + a + f) * 8;
      uint32_t xa = xr[q], xb = xr[4 + q];
      __half2 ha = *reinterpret_cast<__half2*>(&xa);
      __half2 hb = *reinterpret_cast<__half2*>(&xb);
#pragma unroll
      for (int c = 0; c < CC; ++c) {
        acc[c][r][0] = Dot2(P[c][0], ha, acc[c][r][0]);
        acc[c][r][0] = Dot2(P[c][1], hb, acc[c][r][0]);
        acc[c][r][1] = Dot2(P[c][2], ha, acc[c][r][1]);
        acc[c][r][1] = Dot2(P[c][3], hb, acc[c][r][1]);
      }
    }
  };
  if (span % PF == 0) {
    // No early exit inside the unrolled round: each ring slot keeps its
    // registers, so PF tiles stay in flight. With the exit, the compiler
    // rotates the ring through v_mov copies and waits on vmcnt for each,
    // which drains the pipeline to about one tile per round.
    for (int a = 0; a < span; a += PF) {
#pragma unroll
      for (int f = 0; f < PF; ++f) step(a, f);
    }
  } else {
    for (int a = 0; a < span; a += PF) {
#pragma unroll
      for (int f = 0; f < PF; ++f) {
        if (a + f >= span) break;
        step(a, f);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = acc[c][r][h];
        v = v + __shfl_xor(v, 1);
        v = v + __shfl_xor(v, 2);
        acc[c][r][h] = v;
      }
    }
  if (q == 0) {
    float* out = job.partial + (size_t)s * R * job.O;
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      const int col = 16 * b[c] + lane / 4;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[c][r][0];
        out[r * job.O + col + 8] = acc[c][r][1];
      }
    }
  }
}

// LinGemvBody with the prefetch depth as a parameter. The arithmetic per row
// is identical; only the number of tiles in flight per wave changes.
// prep() stages x in LDS and ends with a barrier; every wave calls it.
// With C = 2 an item spans up to 16 column blocks: wave w takes blocks
// b0 + w and b0 + w + 8, which halves the LDS and x staging per column.
template <int K, int R, int PF, int C, int CACHE, typename Prep>
__device__ __forceinline__ void MoeGemvCore(const LinJob& job, const GemvItem& item, uint32_t* xs, Prep prep) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  prep();
  if (wave < item.nb) {
    const int b = item.b0 + wave;
    if (C == 2 && wave + 8 < item.nb) MoeGemvCols<K, R, PF, 2, CACHE>(job, item.s, {b, b + 8}, xs);
    else MoeGemvCols<K, R, PF, 1, CACHE>(job, item.s, {b}, xs);
  }
  __syncthreads();
}

// MoeGemvCols with the activations staged through LDS in chunks of XC tiles,
// double-buffered (2 x R x XC x 32 bytes). A full gate/up span (160 tiles)
// needs R x 5 KB, so the fused launch fits 3 workgroups per WGP at R=8;
// two 40-tile chunks need half of that. Every wave of the block calls this
// (block barriers between chunks); active says whether the wave has columns.
// Each wave loads the next chunk into registers at the start of a chunk and
// stores it to LDS at the end, so the load latency hides behind the chunk.
// The tile order and each row's Dot2 sequence are those of MoeGemvCols, so
// the partials are bitwise equal. The caller guarantees span % XC == 0.
template <int K, int R, int PF, int CC, int CACHE, int XC>
__device__ __forceinline__ void MoeGemvColsChunked(const LinJob& job, int s, const int (&b)[CC], bool active,
                                                   uint32_t* xs) {
  static_assert(XC % PF == 0, "chunks hold whole prefetch rounds");
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = TileDwords(K);
  constexpr int XW = R * XC * 8;           // dwords per chunk buffer
  constexpr int XL = (XW + 255) / 256;     // fill dwords per thread
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  const uint32_t* xg = reinterpret_cast<const uint32_t*>(UniformPtr(job.xh));
  auto fill_load = [&](int c, uint32_t (&v)[XL]) {
#pragma unroll
    for (int k = 0; k < XL; ++k) {
      const int i = threadIdx.x + k * 256;
      const int r = i / (XC * 8), o = i - r * (XC * 8);
      if (i < XW) v[k] = xg[(size_t)r * A * 8 + (a0 + c * XC) * 8 + o];
    }
  };
  auto fill_store = [&](const uint32_t (&v)[XL], uint32_t* dst) {
#pragma unroll
    for (int k = 0; k < XL; ++k) {
      const int i = threadIdx.x + k * 256;
      if (i < XW) dst[i] = v[k];
    }
  };
  {
    uint32_t v[XL];
    fill_load(0, v);
    fill_store(v, xs);
  }
  __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
  for (int c = 0; c < CC; ++c)
    rs[c] = StreamRsrc(tiles + ((size_t)b[c] * A + a0) * T, (uint32_t)span * T * 4);
  const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
  auto load_tile = [&](int c, int tile, uint32_t* dst) {
    const int so = tile * T * 4;
    dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o0, so, CACHE);
    dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o1, so, CACHE);
    dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o2, so, CACHE);
  };
  uint32_t ring[PF][CC][3];
  if (active) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
#pragma unroll
      for (int c = 0; c < CC; ++c) load_tile(c, f, ring[f][c]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  float acc[CC][R][2];
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[c][r][0] = acc[c][r][1] = 0.f;
  // a: tile index in the span; xc: this chunk's buffer; x: tile index in it.
  auto step = [&](int a, int f, const uint32_t* xc, int x) {
    __half2 P[CC][4];
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      DecodeLane<K, kMulMode>(ring[f][c][0], ring[f][c][1], ring[f][c][2], win.shift, P[c]);
      load_tile(c, a + f + PF, ring[f][c]);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t* xr = xc + (r * XC + x + f) * 8;
      uint32_t xa = xr[q], xb = xr[4 + q];
      __half2 ha = *reinterpret_cast<__half2*>(&xa);
      __half2 hb = *reinterpret_cast<__half2*>(&xb);
#pragma unroll
      for (int c = 0; c < CC; ++c) {
        acc[c][r][0] = Dot2(P[c][0], ha, acc[c][r][0]);
        acc[c][r][0] = Dot2(P[c][1], hb, acc[c][r][0]);
        acc[c][r][1] = Dot2(P[c][2], ha, acc[c][r][1]);
        acc[c][r][1] = Dot2(P[c][3], hb, acc[c][r][1]);
      }
    }
  };
  __syncthreads();
  const int chunks = span / XC;
  for (int ch = 0; ch < chunks; ++ch) {
    const bool next = ch + 1 < chunks;
    uint32_t v[XL];
    if (next) fill_load(ch + 1, v);
    const uint32_t* xc = xs + (ch & 1) * XW;
    if (active) {
      for (int x = 0; x < XC; x += PF) {
#pragma unroll
        for (int f = 0; f < PF; ++f) step(ch * XC + x, f, xc, x);
      }
    }
    // Buffer (ch + 1) & 1 held chunk ch - 1, which every wave finished
    // before the previous barrier.
    if (next) fill_store(v, xs + ((ch + 1) & 1) * XW);
    __syncthreads();
  }
  if (!active) return;
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = acc[c][r][h];
        v = v + __shfl_xor(v, 1);
        v = v + __shfl_xor(v, 2);
        acc[c][r][h] = v;
      }
    }
  if (q == 0) {
    float* out = job.partial + (size_t)s * R * job.O;
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      const int col = 16 * b[c] + lane / 4;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[c][r][0];
        out[r * job.O + col + 8] = acc[c][r][1];
      }
    }
  }
}

// MoeGemvCore for gate/up items with chunked x staging (no prep: the body
// stages its own chunks). Ends with a block barrier, as MoeGemvCore.
template <int K, int R, int PF, int C, int CACHE, int XC>
__device__ __forceinline__ void MoeGemvCoreChunked(const LinJob& job, const GemvItem& item, uint32_t* xs) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const bool active = wave < item.nb;
  const int b = item.b0 + (active ? wave : 0);
  // Every wave of an item takes the same branch for the item widths in use
  // (16 or 8 blocks), so all waves run the same number of barriers.
  if (C == 2 && item.nb > 8) {
    if (item.nb != 16) __builtin_trap();  // Build makes items of 16 or 8 blocks
    MoeGemvColsChunked<K, R, PF, 2, CACHE, XC>(job, item.s, {b, b + 8}, true, xs);
  } else {
    MoeGemvColsChunked<K, R, PF, 1, CACHE, XC>(job, item.s, {b}, active, xs);
  }
  __syncthreads();
}

// The chunked bodies (MoeGemvChunked in bench/kernels/moe_rows.h, MoeGemvCoreChunked)
// stage the activations through LDS in chunks of C tiles,
// double-buffered. A full-span stage needs R x 5 KB of LDS per workgroup,
// which limits a CU to 1.5 workgroups at R=8 (60 resident instead of 160+).
// The tile order and every row's Dot2 sequence are unchanged.
constexpr int kMoeChunk = 40;  // tiles per stage: divides the 160 (gate/up) and 40 (down) spans

template <int K, int N, int PF, int C, int CACHE, typename Prep>
__device__ __forceinline__ void MoeRowsBodyP(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs,
                                             Prep prep) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvCore<K, N, PF, C, CACHE>(job, item, xs, prep);
    else MoeRowsBodyP<K, N - 1, PF, C, CACHE>(rows, job, item, xs, prep);
  }
}

// Routed experts (KR=3 at 3.05, 4 at 4.05) run their compact row count; the
// shared expert (KS, KR+2 on those packs) always runs all R rows. KR is a template argument, not
// a runtime branch: one kernel holding the bodies of both checkpoints needs
// more registers and lost 6% of serial decode on 3.05.
// The h6_ng8 packs (MoeFusedKernelWide) have a KS 6, 7 or 8 shared expert; each layer's routed experts
// have one width.
template <int R, int KR, int KS, int PF, int C, int CACHE, typename Prep>
__device__ __forceinline__ void MoeAnyBody(int K, int rows, const LinJob& job, const GemvItem& item,
                                           uint32_t* xs, Prep prep) {
  if (K == KR) MoeRowsBodyP<KR, R, PF, C, CACHE>(rows, job, item, xs, prep);
  // A packed pass pads its last shared chunk to kMaxRows rows (BuildLayer).
  else if (K == KS && rows == R) MoeGemvCore<KS, R, PF, C, CACHE>(job, item, xs, prep);
  else __builtin_trap();  // ValidateMoe admits no other widths
}

template <int K, int N, int PF, int C, int CACHE, int XC>
__device__ __forceinline__ void MoeRowsBodyChunked(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvCoreChunked<K, N, PF, C, CACHE, XC>(job, item, xs);
    else MoeRowsBodyChunked<K, N - 1, PF, C, CACHE, XC>(rows, job, item, xs);
  }
}

// MoeAnyBody for gate/up items with chunked x staging.
template <int R, int KR, int KS, int PF, int C, int CACHE, int XC>
__device__ __forceinline__ void MoeAnyBodyChunked(int K, int rows, const LinJob& job, const GemvItem& item,
                                                  uint32_t* xs) {
  if (K == KR) MoeRowsBodyChunked<KR, R, PF, C, CACHE, XC>(rows, job, item, xs);
  else if (K == KS && rows == R) MoeGemvCoreChunked<KS, R, PF, C, CACHE, XC>(job, item, xs);
  else __builtin_trap();
}

// Dynamic LDS of MoeFusedKernel: with XC > 0 the gate/up items need two XC-tile
// chunks per row and the down items their whole 40-tile span.
constexpr size_t MoeFusedLds(int R, int XC, size_t gu_lds, size_t dn_lds) {
  const size_t gu = XC > 0 ? (size_t)2 * R * XC * 32 : gu_lds;
  return gu > dn_lds ? gu : dn_lds;
}

// Column blocks per item and gate/up items per job for C blocks per wave.
template <int C>
constexpr int kMoeItemBlocks = 8 * C;
template <int C>
constexpr int kMoeGuItems = 2 * ((kMoeInter / 16 + 8 * C - 1) / (8 * C));  // gate and up
// Engine launch shape for every R: two column blocks per wave, prefetch
// depth 4 (same tiles in flight per wave as one block at depth 8). Bench
// (moe_rows mode 13 vs 7, after the uniform-pointer fix): R=1 -2%, R=2 -2%,
// R=3..5 within 1%, R=6 -2%, R=7 -5%, R=8 -4%.
constexpr int kMoeCols = 2, kMoePf = 4;

// Gate/up and down of all routed jobs in one launch. Gate/up items come
// first; each down item waits for its own expert's ten gate/up items, so the
// down work fills the gate/up launch tail instead of starting after a kernel
// boundary. Down items compute SiLU(gate) * up and the down input transform
// straight into LDS (the same code as MoeRowsMidKernel in bench/kernels/), so there is no mid
// launch. Workgroups dispatch in index order, so every gate/up item a down
// item waits for is already resident: the wait cannot deadlock.
// done[u] counts finished gate/up items of unique job u; MoeRowsUniqueKernel
// clears it. Values are bitwise equal to the separate launches.
// The parameter order and defaults keep main's spellings valid:
// MoeFusedKernel<R, KR> is the engine shape and <R, KR, PF, 4> selects the
// cache policy. C must match the item width Build used (kMoeItemBlocks<C>).
// XC > 0: gate/up items stage x in double-buffered chunks of XC tiles
// (MoeGemvColsChunked); launch with MoeFusedLds.
template <int R, int KR, int PF = kMoePf, int CACHE = 0, int C = kMoeCols, int XC = 0>
__global__ void __launch_bounds__(256) MoeFusedKernel(const LinJob* __restrict__ gu,
                                                      const GemvItem* __restrict__ gu_items, int n_gu,
                                                      const LinJob* __restrict__ dn,
                                                      const GemvItem* __restrict__ dn_items, int n_dn,
                                                      unsigned* __restrict__ done) {
  extern __shared__ uint32_t xs[];
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) return;
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    if constexpr (XC > 0) MoeAnyBodyChunked<R, KR, KR + 2, PF, C, CACHE, XC>(K, rows, job, item, xs);
    else MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, job, item, xs, [&] { MoeFillX(job, item, rows, xs); });
    // MoeGemvCore and MoeGemvCoreChunked end with a barrier, so every wave's partial stores precede
    // this release.
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
  // The wait for the gate/up items and the mid, as the body's x staging.
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    // Mid for this job's rows into LDS, in the xh layout the body reads.
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(Silu(gv[i]) * uv[i]) * __half2float(d.suh[c0 + i]));
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
  };
  MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, d, item, xs, prep);
}

// MoeFusedKernel for the h6_ng8 packs: shared expert KS (6, 7 or 8), routed experts KR (KS 6: K2, K2.5
// or K3, the 2.18 and 2.68 packs). A copy of
// MoeFusedKernel's body, not a shared inline function: moving that body into a function changed the
// code of the train 23 kernels (isa_same), so the old kernel stays verbatim.
template <int R, int KR, int KS, int PF = kMoePf, int CACHE = 0, int C = kMoeCols, int XC = 0>
__global__ void __launch_bounds__(256) MoeFusedKernelWide(const LinJob* __restrict__ gu,
                                                          const GemvItem* __restrict__ gu_items, int n_gu,
                                                          const LinJob* __restrict__ dn,
                                                          const GemvItem* __restrict__ dn_items, int n_dn,
                                                          unsigned* __restrict__ done) {
  static_assert(KS == 6 || KS == 7 || KS == 8);
  extern __shared__ uint32_t xs[];
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) return;
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    if constexpr (XC > 0) MoeAnyBodyChunked<R, KR, KS, PF, C, CACHE, XC>(K, rows, job, item, xs);
    else MoeAnyBody<R, KR, KS, PF, C, CACHE>(K, rows, job, item, xs, [&] { MoeFillX(job, item, rows, xs); });
    // MoeGemvCore and MoeGemvCoreChunked end with a barrier, so every wave's partial stores precede
    // this release.
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
  // The wait for the gate/up items and the mid, as the body's x staging.
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    // Mid for this job's rows into LDS, in the xh layout the body reads.
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(Silu(gv[i]) * uv[i]) * __half2float(d.suh[c0 + i]));
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
  };
  MoeAnyBody<R, KR, KS, PF, C, CACHE>(K, rows, d, item, xs, prep);
}

// MoeCombineKernel (bench/kernels/moe_rows.h) with compact expert rows. All slot loads issue before the
// first output Hadamard; the sum keeps the slot order, so values are unchanged.
// Expert jobs are planned with S=1 (fixedS), so each slot has one partial row.
// StoreY: as the bench MoeCombineKernel<true>.
template <bool StoreY = false>
__global__ void __launch_bounds__(256) MoeRowsCombineKernel(const LinJob* __restrict__ dn,
                                                            const int* __restrict__ map,
                                                            const int* __restrict__ mapj,
                                                            const float* __restrict__ wts,
                                                            const float* __restrict__ post,
                                                            float* __restrict__ H, int R) {
  const int wv = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wv >= R * 20) return;
  const int r = wv / 20, blk = wv % 20, lane = threadIdx.x & 31;
  const int c0 = blk * 128 + lane * 4;
  float raw[kSlots][4], sv[kSlots][4], wk[kSlots];
#pragma unroll
  for (int k = 0; k < kSlots; ++k) {
    const int u = k < kTopK ? map[r * kTopK + k] : MoeSharedJob(R, r);
    const int j = k < kTopK ? mapj[r * kTopK + k] : MoeSharedRow(r);
    const LinJob& d = dn[u];
    const float4 p = *reinterpret_cast<const float4*>(d.partial + (size_t)j * d.O + c0);
    raw[k][0] = p.x;
    raw[k][1] = p.y;
    raw[k][2] = p.z;
    raw[k][3] = p.w;
#pragma unroll
    for (int i = 0; i < 4; ++i) sv[k][i] = __half2float(d.svh[c0 + i]);
    wk[k] = wts[r * kSlots + k];
  }
  float y[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kSlots; ++k) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = 0.f + raw[k][i];  // LinOutBlock's S=1 sum, signed zeros included
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] += wk[k] * (v[i] * (kHadNorm * sv[k][i]));
  }
  if constexpr (StoreY) {
#pragma unroll
    for (int i = 0; i < 4; ++i) H[(size_t)r * kHidden + c0 + i] = y[i];
    return;
  }
#pragma unroll
  for (int s = 0; s < kStreams; ++s) {
    const float gs = post[r * kStreams + s];
    float* h = H + ((size_t)r * kStreams + s) * kHidden + c0;
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] += gs * y[i];
  }
}

}  // namespace crucible
