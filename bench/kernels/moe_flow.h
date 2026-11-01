// moe-d1: decode expert stage probes (bench/verify_moe.hip variants 40 and up).
// - MoeFlowKernel: a persistent grid (the resident block count) that claims the items of MoeFusedKernel from an
//   order table with one atomic per item. Item bodies are MoeFusedKernel's (same arithmetic, bitwise equal).
// - MoeStreamKernel: timing bound only (no outputs). Every wave claims one column block of one job at a time
//   (the engine's per-wave unit: 160 gate/up tiles or 40 down tiles) and streams its bytes with D loads in
//   flight. SHAPE 0: b128 loads (512 B per wave instruction); SHAPE 1: the engine's 3 b32 window loads per
//   tile. VALU: dummy FMAs per tile (decode and Dot2 stand-in). It bounds what any schedule of these bytes can do.
#pragma once
#include "verify_moe.h"

namespace crucible {

template <int R, int KR, int PF, int CACHE, int C, bool PROBE>
__global__ void __launch_bounds__(256) MoeFlowKernel(const LinJob* __restrict__ gu,
                                                     const GemvItem* __restrict__ gu_items,
                                                     const LinJob* __restrict__ dn,
                                                     const GemvItem* __restrict__ dn_items,
                                                     unsigned* __restrict__ done, const int* __restrict__ order,
                                                     int n, unsigned* __restrict__ next,
                                                     unsigned long long* __restrict__ st) {
  extern __shared__ uint32_t xs[];
  __shared__ int claim;
  for (;;) {
    if (threadIdx.x == 0) claim = (int)atomicAdd(next, 1u);
    __syncthreads();
    const int b = __builtin_amdgcn_readfirstlane(claim);
    __syncthreads();
    if (b >= n) {
      // The last block out resets the counters for the next launch (no memset in the timed stream).
      if (threadIdx.x == 0 && atomicAdd(next + 1, 1u) == gridDim.x - 1) next[0] = 0, next[1] = 0;
      return;
    }
    unsigned long long t0 = 0, t1 = 0;
    if constexpr (PROBE) t0 = wall_clock64();
    const int o = order[b];
    if (o >= 0) {
      const GemvItem item = gu_items[o];
      const LinJob& job = gu[item.job];
      const int K = __builtin_amdgcn_readfirstlane(job.K);
      if (K == 0) continue;
      const int rows = __builtin_amdgcn_readfirstlane(job.R);
      MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, job, item, xs, [&] {
        MoeFillX(job, item, rows, xs);
        if constexpr (PROBE) t1 = wall_clock64();
      });
      if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(done + (item.job >> 1), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if constexpr (PROBE) st[3 * b] = t0, st[3 * b + 1] = t1, st[3 * b + 2] = wall_clock64();
      }
      continue;
    }
    const GemvItem item = dn_items[-1 - o];
    const int u = item.job;
    const LinJob& d = dn[u];
    const int K = __builtin_amdgcn_readfirstlane(d.K);
    if (K == 0) continue;
    const int rows = __builtin_amdgcn_readfirstlane(d.R);
    auto prep = [&] {
      if (threadIdx.x == 0)
        while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
          __builtin_amdgcn_s_sleep(1);
      __syncthreads();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      __half* xh = reinterpret_cast<__half*>(xs);
      const int lane = threadIdx.x & 31;
      for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
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
        LinInLane(sv, sh, v);
#pragma unroll
        for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
      }
      __syncthreads();
      if constexpr (PROBE) t1 = wall_clock64();
    };
    MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, d, item, xs, prep);
    if (PROBE && threadIdx.x == 0) st[3 * b] = t0, st[3 * b + 1] = t1, st[3 * b + 2] = wall_clock64();
  }
}

// ---- own-dword loads (moe-d1 lease l1: the 3 b32 window loads per lane and tile stream at 78% of the floor at
// R = 4, b128 loads at 89 to 91%; VALU per tile changes nothing). For K4 (T = 32 dwords per tile) lane t loads
// only dword t of each tile: one b32 load per tile instead of three, so one VGPR per tile in flight. Its window
// (dwords t - 1, t, t + 1 mod 32: LaneWindow<4>, d0 = t - 1, shift 20) comes from the neighbours by ds_bpermute.
// The decoded bits, the tile order and each row's Dot2 chain are MoeGemvCols's, so the partials are bitwise equal.
template <int K, int R, int PF, int CC, int CACHE = 0>
__device__ __forceinline__ void MoeGemvColsOwn(const LinJob& job, int s, const int (&b)[CC], const uint32_t* xs) {
  static_assert(TileDwords(K) == 32, "own-dword loads need 32 dwords per tile (K4)");
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = 32;
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
  for (int c = 0; c < CC; ++c)
    rs[c] = StreamRsrc(tiles + ((size_t)b[c] * A + a0) * T, (uint32_t)span * T * 4);
  const int lo = 4 * lane, ip = 4 * ((lane + 31) & 31), in = 4 * ((lane + 1) & 31);
  uint32_t ring[PF][CC];
#pragma unroll
  for (int f = 0; f < PF; ++f) {
#pragma unroll
    for (int c = 0; c < CC; ++c) ring[f][c] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], lo, f * T * 4, CACHE);
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
      const uint32_t w1 = ring[f][c];
      const uint32_t w0 = (uint32_t)__builtin_amdgcn_ds_bpermute(ip, (int)w1);
      const uint32_t w2 = (uint32_t)__builtin_amdgcn_ds_bpermute(in, (int)w1);
      DecodeLane<K, kMulMode>(w0, w1, w2, win.shift, P[c]);
      ring[f][c] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], lo, (a + f + PF) * T * 4, CACHE);
    }
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

// ---- b128 loads with an in-wave LDS transpose (moe-d1 lease l2: one b32 per lane and tile streams at 83 to 85% of
// the floor at R = 4 even with 32 tiles in flight; b128 loads with the same bytes in flight reach 91%). For K4 the
// lane window is dwords t - 1 and t of the tile (DecodeLane reads only the top 12 bits of a1, so w2 is unused).
// Lane L = 8 j + m loads dwords 4 m .. 4 m + 3 of tile j of a 4-tile group (one b128 per lane, 512 B per wave),
// writes them to a wave-private LDS slot (36 dwords per tile: dword 31 copied at 3, dword d at 4 + d), and every
// lane then reads its own (w0, w1) for each of the 4 tiles. The decoded bits, the tile order and each row's Dot2
// chain are MoeGemvCols's, so the partials are bitwise equal. tb: 4 * 36 * CC dwords for this wave, 16-B aligned.
constexpr int kT128Slot = 4 * 36;
template <int K, int R, int PG, int CC, int CACHE = 0>
__device__ __forceinline__ void MoeGemvColsT128(const LinJob& job, int s, const int (&b)[CC], const uint32_t* xs,
                                                uint32_t* tb) {
  static_assert(TileDwords(K) == 32, "the b128 transpose needs 32 dwords per tile (K4)");
  const int lane = threadIdx.x & 31;
  const int q = lane & 3;
  constexpr int T = 32;
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
  for (int c = 0; c < CC; ++c)
    rs[c] = StreamRsrc(tiles + ((size_t)b[c] * A + a0) * T, (uint32_t)span * T * 4);
  const int jl = lane >> 3, ml = lane & 7;
  uint32_t* wr = tb + jl * 36 + 4 + 4 * ml;  // this lane's b128 slot (16-B aligned)
  uint32_t* wl = tb + jl * 36 + 3;           // the copy of dword 31 (lanes with m = 7)
  const uint32_t* rd = tb + 3 + lane;        // (w0, w1) of tile f at rd[f * 36], rd[f * 36 + 1]
  typedef unsigned int u4 __attribute__((ext_vector_type(4)));
  u4 ring[PG][CC];
#pragma unroll
  for (int g = 0; g < PG; ++g) {
#pragma unroll
    for (int c = 0; c < CC; ++c) ring[g][c] = __builtin_amdgcn_raw_buffer_load_b128(rs[c], lane * 16, g * 512, CACHE);
    __builtin_amdgcn_sched_barrier(0);
  }
  float acc[CC][R][2];
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[c][r][0] = acc[c][r][1] = 0.f;
  // One group: 4 tiles from tile a on.
  auto step = [&](int a, int g, bool live) {
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      const u4 v = ring[g][c];
      *reinterpret_cast<u4*>(wr + c * kT128Slot) = v;
      if (ml == 7) wl[c * kT128Slot] = v[3];
      ring[g][c] = __builtin_amdgcn_raw_buffer_load_b128(rs[c], lane * 16, (a / 4 + PG) * 512, CACHE);
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_wave_barrier();
    if (!live) return;
#pragma unroll
    for (int f = 0; f < 4; ++f) {
      __half2 P[CC][4];
#pragma unroll
      for (int c = 0; c < CC; ++c) {
        const uint32_t w0 = rd[c * kT128Slot + f * 36], w1 = rd[c * kT128Slot + f * 36 + 1];
        DecodeLane<K, kMulMode>(w0, w1, 0u, 20, P[c]);
      }
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
    }
    // The next group's LDS writes must follow these reads (LDS ops of one wave run in order).
    __builtin_amdgcn_wave_barrier();
  };
  // Full rounds only: a group past span loads out of bounds (zeros, no memory traffic) and skips the math, so each
  // ring slot keeps its registers (see MoeGemvCols).
  for (int a = 0; a < span; a += 4 * PG) {
#pragma unroll
    for (int g = 0; g < PG; ++g) step(a + 4 * g, g, a + 4 * g < span);
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

// ---- weight prefetch before the x staging (moe-d1 lease l4: the occupancy and the load shape do not set the
// real kernel's pace; the verify-moe stamps show a 60 us ramp, and every item stages x (MoeFillX: one b32 load
// per thread and step, about 5 R dependent rounds) before its first weight load). Exact: only the issue point of
// the first PF tile loads moves.
template <int K, int R, int PF, int CC, int CACHE, typename Prep>
__device__ __forceinline__ void MoeGemvColsPre(const LinJob& job, int s, const int (&b)[CC], uint32_t* xs, bool active,
                                               Prep prep) {
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
  // The first PF tiles do not depend on x: issue them before the x staging (prep), so the weight stream
  // starts while the block stages x (gate/up) or waits for its gate/up items (down).
  if (active) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
#pragma unroll
      for (int c = 0; c < CC; ++c) load_tile(c, f, ring[f][c]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  prep();
  if (!active) return;
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


// MoeFillX with b128 loads, 4 in flight per thread (the copy is the same).
__device__ __forceinline__ void MoeFillXFast(const LinJob& job, const GemvItem& item, int rows, uint32_t* xs) {
  const int A = job.I >> 4, span = A / job.S, a0 = item.s * span;
  typedef unsigned int u4 __attribute__((ext_vector_type(4)));
  const u4* xg = reinterpret_cast<const u4*>(job.xh);
  u4* x4 = reinterpret_cast<u4*>(xs);
  const int n = rows * span * 2;
  for (int i0 = threadIdx.x; i0 < n; i0 += 4 * 256) {
    u4 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = i0 + k * 256;
      if (i < n) {
        const int r = i / (span * 2), o = i - r * (span * 2);
        v[k] = xg[(size_t)r * A * 2 + a0 * 2 + o];
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k)
      if (i0 + k * 256 < n) x4[i0 + k * 256] = v[k];
  }
  __syncthreads();
}

// MoeGemvCore with the own-dword body for K4 (OPF: its prefetch depth); other widths keep MoeGemvCols.
template <int K, int R, int PF, int OPF, int C, int CACHE, int BODY, typename Prep>
__device__ __forceinline__ void MoeGemvCoreOwn(const LinJob& job, const GemvItem& item, uint32_t* xs, int toff,
                                               Prep prep) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  if constexpr (BODY == 3) {
    const int nb = __builtin_amdgcn_readfirstlane(item.nb);
    const int b = item.b0 + wave;
    if (C == 2 && nb == 16) {
      MoeGemvColsPre<K, R, PF, 2, CACHE>(job, item.s, {b, b + 8}, xs, true, prep);
      __syncthreads();
      return;
    }
    if (nb <= 8) {
      MoeGemvColsPre<K, R, PF, 1, CACHE>(job, item.s, {b}, xs, wave < nb, prep);
      __syncthreads();
      return;
    }
  }
  prep();
  if (wave < item.nb) {
    const int b = item.b0 + wave;
    if constexpr (BODY == 2 || BODY == 3) {
      if (C == 2 && wave + 8 < item.nb) MoeGemvCols<K, R, PF, 2, CACHE>(job, item.s, {b, b + 8}, xs);
      else MoeGemvCols<K, R, PF, 1, CACHE>(job, item.s, {b}, xs);
    } else if constexpr (TileDwords(K) == 32 && BODY == 1) {
      uint32_t* tb = xs + toff + wave * 2 * kT128Slot;
      if (C == 2 && wave + 8 < item.nb) MoeGemvColsT128<K, R, OPF, 2, CACHE>(job, item.s, {b, b + 8}, xs, tb);
      else MoeGemvColsT128<K, R, OPF, 1, CACHE>(job, item.s, {b}, xs, tb);
    } else if constexpr (TileDwords(K) == 32) {
      if (C == 2 && wave + 8 < item.nb) MoeGemvColsOwn<K, R, OPF, 2, CACHE>(job, item.s, {b, b + 8}, xs);
      else MoeGemvColsOwn<K, R, OPF, 1, CACHE>(job, item.s, {b}, xs);
    } else {
      if (C == 2 && wave + 8 < item.nb) MoeGemvCols<K, R, PF, 2, CACHE>(job, item.s, {b, b + 8}, xs);
      else MoeGemvCols<K, R, PF, 1, CACHE>(job, item.s, {b}, xs);
    }
  }
  __syncthreads();
}
template <int K, int N, int PF, int OPF, int C, int CACHE, int BODY, typename Prep>
__device__ __forceinline__ void MoeRowsBodyOwn(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs,
                                               int toff, Prep prep) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvCoreOwn<K, N, PF, OPF, C, CACHE, BODY>(job, item, xs, toff, prep);
    else MoeRowsBodyOwn<K, N - 1, PF, OPF, C, CACHE, BODY>(rows, job, item, xs, toff, prep);
  }
}
template <int R, int KR, int KS, int PF, int OPF, int C, int CACHE, int BODY, typename Prep>
__device__ __forceinline__ void MoeAnyBodyOwn(int K, int rows, const LinJob& job, const GemvItem& item,
                                              uint32_t* xs, int toff, Prep prep) {
  if (K == KR) MoeRowsBodyOwn<KR, R, PF, OPF, C, CACHE, BODY>(rows, job, item, xs, toff, prep);
  else if (K == KS && rows == R) MoeGemvCoreOwn<KS, R, PF, OPF, C, CACHE, BODY>(job, item, xs, toff, prep);
  else __builtin_trap();
}

// MoeFusedKernel (engine dispatch: one block per item, gate/up first) with the own-dword K4 body.
// W = 0 (timing diagnostics only): down items skip the wait for their gate/up items. FX = 2: no x staging.
template <int R, int KR, int PF, int OPF, int CACHE, int C, int BODY = 0, int FX = 0, int W = 1>
__global__ void __launch_bounds__(256) MoeFusedOwnKernel(const LinJob* __restrict__ gu,
                                                         const GemvItem* __restrict__ gu_items, int n_gu,
                                                         const LinJob* __restrict__ dn,
                                                         const GemvItem* __restrict__ dn_items, int n_dn,
                                                         unsigned* __restrict__ done, int toff) {
  extern __shared__ uint32_t xs[];
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) return;
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    MoeAnyBodyOwn<R, KR, KR + 2, PF, OPF, C, CACHE, BODY>(K, rows, job, item, xs, toff,
                                                          [&] {
                                                            if constexpr (FX == 2) __syncthreads();
                                                            else if constexpr (FX) MoeFillXFast(job, item, rows, xs);
                                                            else MoeFillX(job, item, rows, xs);
                                                          });
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
    if (W && threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
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
  MoeAnyBodyOwn<R, KR, KR + 2, PF, OPF, C, CACHE, BODY>(K, rows, d, item, xs, toff, prep);
}

// Per-wave units in weight-major order: per job slot u (shared job first), the 40 gate column blocks, the 40 up
// column blocks, then the 160 down column blocks.
constexpr int kFlowUnitsPerJob = 240;

template <int D, int SHAPE, int VALU>
__global__ void __launch_bounds__(256) MoeStreamKernel(const LinJob* __restrict__ gu, const LinJob* __restrict__ dn,
                                                       int jobs, int shared_job, unsigned* __restrict__ next,
                                                       uint32_t* __restrict__ sink) {
  const int lane = threadIdx.x & 31;
  const int n = jobs * kFlowUnitsPerJob;
  uint32_t acc = 0;
  float f[4] = {1.f, 1.f, 1.f, 1.f};
  for (;;) {
    int c = 0;
    if (lane == 0) c = (int)atomicAdd(next, 1u);
    c = __builtin_amdgcn_readfirstlane(__shfl(c, 0));
    if (c >= n) {
      if (lane == 0 && atomicAdd(next + 1, 1u) == gridDim.x * 8 - 1) next[0] = 0, next[1] = 0;
      break;
    }
    int u = c / kFlowUnitsPerJob;
    const int r = c - u * kFlowUnitsPerJob;
    // Slot 0 is the shared job; slots 1.. are routed jobs 0 ...
    u = u == 0 ? shared_job : u - 1;
    const LinJob& j = r < 80 ? gu[2 * u + (r >= 40)] : dn[u];
    const int K = __builtin_amdgcn_readfirstlane(j.K);
    if (K == 0) continue;
    const int cb = r < 80 ? r % 40 : r - 80;
    const int A = __builtin_amdgcn_readfirstlane(j.I >> 4), T = 8 * K;
    const uint32_t* tiles = UniformPtr(j.tiles);
    const uint32_t bytes = (uint32_t)A * T * 4;
    const auto rs = StreamRsrc(tiles + (size_t)cb * A * T, bytes);
    if constexpr (SHAPE == 0) {
      const int steps = (int)(bytes / 512);
      for (int s0 = 0; s0 < steps; s0 += D) {
        uint32_t w[D][4];
#pragma unroll
        for (int k = 0; k < D; ++k) {
          const auto v = __builtin_amdgcn_raw_buffer_load_b128(rs, lane * 16, (s0 + k) * 512, 0);
          w[k][0] = v[0], w[k][1] = v[1], w[k][2] = v[2], w[k][3] = v[3];
        }
#pragma unroll
        for (int k = 0; k < D; ++k) {
          acc ^= w[k][0] ^ w[k][1] ^ w[k][2] ^ w[k][3];
#pragma unroll
          for (int i = 0; i < VALU / 4; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) f[q] = __builtin_fmaf(f[q], 1.0001f, __uint_as_float(w[k][q] & 0x3fffffff));
        }
      }
    } else if constexpr (SHAPE == 2 || SHAPE == 3) {
      // One b32 per lane and tile (dword lane of the tile; lanes past T load the tile's last dword). SHAPE 3 adds
      // the two ds_bpermute of the own-dword window.
      const int lo = 4 * min(lane, T - 1);
      uint32_t ring[D];
#pragma unroll
      for (int k = 0; k < D; ++k) {
        ring[k] = __builtin_amdgcn_raw_buffer_load_b32(rs, lo, k * T * 4, 0);
        __builtin_amdgcn_sched_barrier(0);
      }
      for (int a = 0; a < A; a += D) {
#pragma unroll
        for (int k = 0; k < D; ++k) {
          uint32_t x0 = ring[k];
          if constexpr (SHAPE == 3)
            x0 ^= (uint32_t)__builtin_amdgcn_ds_bpermute(4 * ((lane + 31) & 31), (int)x0) +
                  (uint32_t)__builtin_amdgcn_ds_bpermute(4 * ((lane + 1) & 31), (int)x0);
          acc ^= x0;
#pragma unroll
          for (int i = 0; i < VALU / 4; ++i)
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) f[qq] = __builtin_fmaf(f[qq], 1.0001f, __uint_as_float((x0 >> qq) & 0x3fffffff));
          ring[k] = __builtin_amdgcn_raw_buffer_load_b32(rs, lo, (a + k + D) * T * 4, 0);
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    } else {
      // The engine's loads: tile t of the column, 3 dwords per lane at the lane window (K4 windows).
      const int d0 = (lane * 8 * K / 32) % T;
      const int o0 = 4 * d0, o1 = 4 * ((d0 + 1) % T), o2 = 4 * ((d0 + 2) % T);
      uint32_t ring[D][3];
#pragma unroll
      for (int k = 0; k < D; ++k) {
        ring[k][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, k * T * 4, 0);
        ring[k][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, k * T * 4, 0);
        ring[k][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, k * T * 4, 0);
        __builtin_amdgcn_sched_barrier(0);
      }
      for (int a = 0; a < A; a += D) {
#pragma unroll
        for (int k = 0; k < D; ++k) {
          const uint32_t x0 = ring[k][0], x1 = ring[k][1], x2 = ring[k][2];
          acc ^= x0 ^ x1 ^ x2;
#pragma unroll
          for (int i = 0; i < VALU / 4; ++i) {
            f[0] = __builtin_fmaf(f[0], 1.0001f, __uint_as_float(x0 & 0x3fffffff));
            f[1] = __builtin_fmaf(f[1], 1.0001f, __uint_as_float(x1 & 0x3fffffff));
            f[2] = __builtin_fmaf(f[2], 1.0001f, __uint_as_float(x2 & 0x3fffffff));
            f[3] = __builtin_fmaf(f[3], 1.0001f, __uint_as_float((x0 ^ x2) & 0x3fffffff));
          }
          const int so = (a + k + D) * T * 4;
          ring[k][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, 0);
          ring[k][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, 0);
          ring[k][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, 0);
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
  }
  const float fs = f[0] + f[1] + f[2] + f[3];
  sink[blockIdx.x * 256 + threadIdx.x] = acc ^ __float_as_uint(fs);
}

struct FlowState {
  unsigned* next = nullptr;
  uint32_t* sink = nullptr;
  int* order[17][8] = {};
  int grid[17][8] = {};
};
inline FlowState g_flow;

template <class Kern>
int FlowResident(Kern k, size_t lds) {
  int per = 0, dev = 0;
  HIP_OK(hipGetDevice(&dev));
  hipDeviceProp_t p{};
  HIP_OK(hipGetDeviceProperties(&p, dev));
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, reinterpret_cast<const void*>(k), 256, lds));
  return per * p.multiProcessorCount;
}

// Variants 40 to 47: flow (C, PF, lag); 50 to 59: stream bounds. Returns false for an unknown variant.
template <int R>
bool LaunchFlowR(int v, const Group& gu, const Group& dn, unsigned* done) {
  if (!g_flow.next) {
    HIP_OK(hipMalloc(&g_flow.next, 64));
    HIP_OK(hipMalloc(&g_flow.sink, 4 << 20));
    HIP_OK(hipMemset(g_flow.next, 0, 64));
  }
  auto flow = [&](int slot, auto kern, auto order_fn, int grid_div) {
    const size_t lds = MoeFusedLds(R, 0, gu.lds, dn.lds);
    int*& ord = g_flow.order[R][slot];
    static std::vector<int> sizes(17 * 8, 0);
    if (!ord) {
      const auto o = order_fn();
      sizes[R * 8 + slot] = (int)o.size();
      ord = Upload(o);
      g_flow.grid[R][slot] = FlowResident(kern, lds) / grid_div;
      std::printf("flow v%d R=%d grid %d items %zu lds %zu\n", v, R, g_flow.grid[R][slot], o.size(), lds);
    }
    hipLaunchKernelGGL(kern, dim3(g_flow.grid[R][slot]), dim3(256), lds, 0, gu.d_jobs, gu.d_items, dn.d_jobs,
                       dn.d_items, done, ord, sizes[R * 8 + slot], g_flow.next, g_vm_stamps);
  };
#ifndef VM_LEAN
  switch (v) {
    case 40: flow(0, MoeFlowKernel<R, 4, 4, 4, 2, false>, [&] { return VerifyMoeOrder<2>(gu, dn, -1); }, 1); return true;
    case 41: flow(1, MoeFlowKernel<R, 4, 4, 4, 2, false>, [&] { return VerifyMoeOrder<2>(gu, dn, 2); }, 1); return true;
    case 42: flow(2, MoeFlowKernel<R, 4, 4, 4, 2, false>, [&] { return VerifyMoeOrder<2>(gu, dn, 4); }, 1); return true;
    case 43: flow(3, MoeFlowKernel<R, 4, 8, 4, 1, false>, [&] { return VerifyMoeOrder<1>(gu, dn, 4); }, 1); return true;
    case 44: flow(4, MoeFlowKernel<R, 4, 8, 4, 1, false>, [&] { return VerifyMoeOrder<1>(gu, dn, 8); }, 1); return true;
    default: break;
  }
#else
  (void)flow;
#endif
  auto own = [&](auto kern) {
    const size_t x = (MoeFusedLds(R, 0, gu.lds, dn.lds) + 15) / 16 * 16;
    hipLaunchKernelGGL(kern, dim3(gu.nitems + dn.nitems), dim3(256), x + 8 * 2 * kT128Slot * 4, 0, gu.d_jobs,
                       gu.d_items, gu.nitems, dn.d_jobs, dn.d_items, dn.nitems, done, (int)(x / 4));
  };
  switch (v) {
    case 60: own(MoeFusedOwnKernel<R, 4, 4, 8, 4, 2>); return true;
    case 61: own(MoeFusedOwnKernel<R, 4, 4, 12, 4, 2>); return true;
    case 62: own(MoeFusedOwnKernel<R, 4, 4, 16, 4, 2>); return true;
    case 63: own(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2>); return true;
    case 64: own(MoeFusedOwnKernel<R, 4, 8, 16, 4, 1>); return true;
    case 65: own(MoeFusedOwnKernel<R, 4, 8, 24, 4, 1>); return true;
    case 66: own(MoeFusedOwnKernel<R, 4, 4, 2, 4, 2, 1>); return true;
    case 67: own(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 1>); return true;
    case 68: own(MoeFusedOwnKernel<R, 4, 8, 4, 4, 1, 1>); return true;
    case 69: own(MoeFusedOwnKernel<R, 4, 8, 8, 4, 1, 1>); return true;
    default: break;
  }
  // Occupancy caps (moe-d1 lease l3: more tiles in flight per wave makes the real kernel slower while the
  // streams gain): the same kernels with extra dynamic LDS, so fewer blocks (and DRAM streams) are resident.
  static bool shown[100] = {};
  auto capped = [&](auto kern, size_t cap, bool t128) {
    const size_t x = (MoeFusedLds(R, 0, gu.lds, dn.lds) + 15) / 16 * 16;
    size_t lds = t128 ? x + 8 * 2 * kT128Slot * 4 : x;
    if (lds < cap) lds = cap;
    if (!shown[v]) {
      shown[v] = true;
      std::printf("cap v%d R=%d lds %zu resident blocks %d\n", v, R, lds, FlowResident(kern, lds));
    }
    hipLaunchKernelGGL(kern, dim3(gu.nitems + dn.nitems), dim3(256), lds, 0, gu.d_jobs, gu.d_items, gu.nitems,
                       dn.d_jobs, dn.d_items, dn.nitems, done, (int)(x / 4));
  };
  switch (v) {
    case 80: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2>, 16384, false); return true;
    case 81: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2>, 24576, false); return true;
    case 82: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2>, 32768, false); return true;
    case 83: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2>, 49152, false); return true;
    case 84: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2>, 65536, false); return true;
    case 85: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 1>, 24576, true); return true;
    case 86: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 1>, 32768, true); return true;
    case 87: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 1>, 49152, true); return true;
    case 88: capped(MoeFusedOwnKernel<R, 4, 4, 16, 4, 2, 0>, 32768, false); return true;
    case 89: capped(MoeFusedOwnKernel<R, 4, 4, 16, 4, 2, 0>, 49152, false); return true;
    case 94: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 3, 0>, 0, false); return true;
    case 95: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 1>, 0, false); return true;
    case 96: capped(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 3, 1>, 0, false); return true;
    case 97: capped(MoeFusedOwnKernel<R, 4, 8, 8, 4, 2, 3, 1>, 0, false); return true;
    case 98: capped(MoeFusedOwnKernel<R, 4, 6, 6, 4, 2, 3, 1>, 0, false); return true;
    default: break;
  }
  // Phase diagnostics (timing only): gate/up items alone, down items alone (no wait), all items with no wait,
  // and no wait plus no x staging. The engine body (BODY 2) throughout.
  {
    const size_t x = (MoeFusedLds(R, 0, gu.lds, dn.lds) + 15) / 16 * 16;
    auto part = [&](auto kern, int ngu, int ndn) {
      hipLaunchKernelGGL(kern, dim3(ngu + ndn), dim3(256), x, 0, gu.d_jobs, gu.d_items, ngu, dn.d_jobs, dn.d_items,
                         ndn, done, (int)(x / 4));
    };
    switch (v) {
      case 99: part(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 0, 1>, gu.nitems, 0); return true;
      case 100: part(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 0, 0>, 0, dn.nitems); return true;
      case 101: part(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 0, 0>, gu.nitems, dn.nitems); return true;
      case 102: part(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 2, 0>, gu.nitems, dn.nitems); return true;
      case 103: part(MoeFusedOwnKernel<R, 4, 4, 4, 4, 2, 2, 2, 0>, gu.nitems, 0); return true;
      default: break;
    }
  }
  const int jobs = MoeExpertJobs(R), shared = R * kTopK;
  auto stream = [&](auto kern, int grid) {
    hipLaunchKernelGGL(kern, dim3(grid), dim3(256), 0, 0, gu.d_jobs, dn.d_jobs, jobs, shared, g_flow.next,
                       g_flow.sink);
  };
  switch (v) {
    case 50: stream(MoeStreamKernel<4, 0, 0>, 160); return true;
    case 51: stream(MoeStreamKernel<8, 0, 0>, 160); return true;
    case 52: stream(MoeStreamKernel<8, 0, 0>, 80); return true;
    case 53: stream(MoeStreamKernel<2, 0, 0>, 160); return true;
    case 54: stream(MoeStreamKernel<8, 1, 0>, 160); return true;
    case 55: stream(MoeStreamKernel<8, 1, 0>, 120); return true;
    case 56: stream(MoeStreamKernel<8, 1, 64>, 120); return true;
    case 57: stream(MoeStreamKernel<8, 1, 96>, 120); return true;
    case 58: stream(MoeStreamKernel<16, 1, 0>, 120); return true;
    case 59: stream(MoeStreamKernel<4, 1, 0>, 120); return true;
    case 70: stream(MoeStreamKernel<8, 2, 0>, 120); return true;
    case 71: stream(MoeStreamKernel<16, 2, 0>, 120); return true;
    case 72: stream(MoeStreamKernel<24, 2, 0>, 120); return true;
    case 73: stream(MoeStreamKernel<16, 2, 0>, 80); return true;
    case 74: stream(MoeStreamKernel<16, 2, 0>, 160); return true;
    case 75: stream(MoeStreamKernel<16, 3, 64>, 120); return true;
    case 76: stream(MoeStreamKernel<8, 0, 0>, 120); return true;
    case 77: stream(MoeStreamKernel<32, 2, 0>, 120); return true;
    case 90: stream(MoeStreamKernel<8, 0, 0>, 40); return true;
    case 91: stream(MoeStreamKernel<8, 0, 0>, 20); return true;
    case 92: stream(MoeStreamKernel<16, 1, 0>, 40); return true;
    case 93: stream(MoeStreamKernel<16, 1, 0>, 20); return true;
    default: return false;
  }
}

inline bool LaunchFlow(int v, int R, const Group& gu, const Group& dn, unsigned* done) {
  switch (R) {
    case 1: return LaunchFlowR<1>(v, gu, dn, done);
    case 2: return LaunchFlowR<2>(v, gu, dn, done);
    case 3: return LaunchFlowR<3>(v, gu, dn, done);
    case 4: return LaunchFlowR<4>(v, gu, dn, done);
    case 5: return LaunchFlowR<5>(v, gu, dn, done);
    case 6: return LaunchFlowR<6>(v, gu, dn, done);
    case 7: return LaunchFlowR<7>(v, gu, dn, done);
    default: return LaunchFlowR<8>(v, gu, dn, done);
  }
}
// 99 to 103 only: the moe-b128 variants (110 and up) set their own label (B128TimingOnly), so the exact ones
// get the byte check.
inline bool FlowTimingOnly(int v) { return (v >= 50 && v < 60) || (v >= 70 && v < 80) || (v >= 90 && v < 94) || (v >= 99 && v <= 103); }

}  // namespace crucible
