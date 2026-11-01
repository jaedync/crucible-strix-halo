// Stream-grid decode expert stage (moe-b128). One launch does the gate/up,
// the mid and the down of all expert jobs of a verify step or a packed verify pass (1 to kPackedRows rows, jobs of
// at most kMaxRows rows), bitwise equal to MoeFusedKernel.
//
// Grid: G persistent blocks of 8 waves. Every wave claims one unit at a time from a global counter: a unit is CBG
// (gate/up) or CBD (down) consecutive column blocks (16 output columns each, the whole K span) of one matrix of
// one job. This is the grid of the b128 stream
// bench (MoeStreamKernel SHAPE 0, G120), which streams at 88 to 95% of the byte floor. No block barrier, no
// per-item x stage.
//
// Unit order: all gate/up units first (shared job first, then routed jobs 0..n-1; per job the 5 128-column blocks
// in order, per block 8 gate then 8 up column blocks), then all down units in the same job order. OPT bit 0
// (CRUCIBLE_MOE_B128=12, the default) puts the shared job's down units last.
//
// Weights (K4, 32 dwords per tile): each wave instruction loads 512 contiguous bytes (4 tiles of the column block):
// lane L = 8 g + m holds dwords 4 m .. 4 m + 3 of tile 4 s + g. An in-register 4 x 4 transpose (lane bits 3..4
// against the register index; v_permlane16 / v_permlanex16 and selects, no LDS) gives lane L = 8 e + m the dword
// t = 4 m + e of all 4 tiles, and the same transpose of the dwords t - 1 (one DPP8 move first) gives the window. So
// lane 8 e + m is the engine's lane t = 4 m + e (MoeGemvCols, LaneWindow<4>: d0 = t - 1, shift 20, w2 unused): the
// same decode, the same per-lane Dot2 chain in tile order, and the same shuffle sums ((a0 + a1) + (a2 + a3), lane
// xor 8 then xor 16). Bitwise equal by construction.
// The K6 shared expert keeps the engine's 3 x b32 window loads per tile (MoeGemvCols arithmetic, one column).
//
// x: every wave stages its own x through LDS, 4 tiles at a time (R x 128 bytes, double buffered: 2 R x 128 bytes
// per wave). The x loads travel in the weight ring (x of step s + 1 is issued just before the weights of step s), so
// the vmcnt waits stay static. The LDS layout per row is [q][tile][a, b]: lane 8 q + m reads its 8 x values of the
// 4 tiles with 2 ds_load_b128 at a 4-way broadcast address.
//
// Gate/up to down: per (job, 128-column block) a counter of finished gate and up units (16). The unit that finishes
// the 16th computes the mid of that block for all rows (MoeFusedKernel's arithmetic) into the down job's xh in
// global memory and counts the block in the job's mid counter (5). Down units wait for 5. Every gate/up unit comes
// before every down unit in the claim order and a gate/up unit never waits, so the wait cannot deadlock.
// The last wave to leave resets the control words for the next launch.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "moe_b128.h"
#include "moe_b128_switch.h"

namespace crucible {

// Control words: [0] claim counter, [1] exit counter, then 8 per job: [0..4] gate/up units done per 128-column
// block, [5] mid blocks done.
constexpr int kSgJobWords = 8;
constexpr int kSgCtlWords = 2 + kSgJobWords * (kPackedRows * kTopK + 2);
constexpr int kSgGuUnits = 2 * (kMoeInter / 16);  // 80
constexpr int kSgDnUnits = kHidden / 16;          // 160
// Engine launch shape: K4 steps (512 B per wave) in flight, K6 steps in flight, column blocks per gate/up and
// per down unit, the VGPR cap (waves per SIMD) and the K4 bodies (kSgNp 1: for 1, 2, 4 and R rows). Bench
// variant 176 of bench/moe_sg_engine.hip (6 to 9% under MoeFusedKernel at 4 to 16
// rows; the D5 shape of moe-b128 variant 130 was 4% slower than MoeFusedKernel at R 4, 6 and 7). The long units of
// 140 to 146 were slower.
constexpr int kSgDepth = 2, kSgDepthK6 = 1, kSgCbg = 1, kSgCbd = 1, kSgWaves = 12, kSgNp = 1;
// The VGPR cap of the D5 shape (bench variant 177 runs D2 at it).
constexpr int SgWaves(int R) { return R <= 4 ? 12 : 8; }
constexpr int kSgMidBlocks = kMoeInter / 128;     // 5

// v_permlane16: lane i of each 16-lane row reads lane i ^ 8. v_permlanex16: lane i reads lane i of the other row.
constexpr unsigned kPermX8Lo = 0xFEDCBA98u, kPermX8Hi = 0x76543210u;
constexpr unsigned kPermIdLo = 0x76543210u, kPermIdHi = 0xFEDCBA98u;

__device__ __forceinline__ uint32_t SgXor8(uint32_t v) {
  return __builtin_amdgcn_permlane16(v, v, kPermX8Lo, kPermX8Hi, false, false);
}
__device__ __forceinline__ uint32_t SgXor16(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, kPermIdLo, kPermIdHi, false, false);
}
__device__ __forceinline__ float SgXor8f(float v) { return __uint_as_float(SgXor8(__float_as_uint(v))); }
__device__ __forceinline__ float SgXor16f(float v) { return __uint_as_float(SgXor16(__float_as_uint(v))); }

// 4 x 4 transpose: lane (g = lane bits 3..4, m) element e -> lane (e, m) element g.
__device__ __forceinline__ void SgTranspose(uint32_t (&v)[4], int lane) {
  const bool b3 = lane & 8, b4 = lane & 16;
#pragma unroll
  for (int k = 0; k < 4; k += 2) {
    const uint32_t r = SgXor8(b3 ? v[k] : v[k + 1]);
    v[k] = b3 ? r : v[k];
    v[k + 1] = b3 ? v[k + 1] : r;
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const uint32_t r = SgXor16(b4 ? v[k] : v[k + 2]);
    v[k] = b4 ? r : v[k];
    v[k + 2] = b4 ? v[k + 2] : r;
  }
}

// x staging of one job: per 4-tile step, NX b128 loads per lane (rows 4 i + lane / 8, dwords 4 (lane & 7) of the
// step's 32 dwords per row), stored to the wave's LDS slot as [row][q][tile][a, b].
template <int N>
struct SgX {
  static constexpr int NX = (N + 3) / 4;
  __amdgpu_buffer_rsrc_t rs;
  int vo[NX];
  int so0;  // LDS dword offset of the lane's first store (row 0 of instruction 0)
  // rows <= N: the job's rows; rows past it load zeros (offset past the buffer end).
  __device__ __forceinline__ SgX(const LinJob& j, int A, int lane, int rows = N) {
    const uint32_t* xg = UniformPtr(reinterpret_cast<const uint32_t*>(j.xh));
    const uint32_t bytes = (uint32_t)rows * A * 32;
    rs = StreamRsrc(xg, bytes);
#pragma unroll
    for (int i = 0; i < NX; ++i) {
      const int row = 4 * i + (lane >> 3);
      vo[i] = row < rows ? (row * A * 8 + 4 * (lane & 7)) * 4 : (int)bytes;
    }
    so0 = (lane >> 3) * 32 + ((lane & 7) >> 1) * 2 + (lane & 1);
  }
  __device__ __forceinline__ void Load(int s, B128u4 (&d)[NX]) const {
#pragma unroll
    for (int i = 0; i < NX; ++i) d[i] = __builtin_amdgcn_raw_buffer_load_b128(rs, vo[i], s * 128, 0);
  }
  // Lanes of rows past N store to the wave's spare line (dummy) instead of branching around the store.
  __device__ __forceinline__ void Store(uint32_t* slot, uint32_t* dummy, const B128u4 (&d)[NX], int lane) const {
#pragma unroll
    for (int i = 0; i < NX; ++i) {
      uint32_t* p = 4 * i + (lane >> 3) < N ? slot + 128 * i + so0 : dummy + (lane & 7);
      p[0] = d[i][0], p[8] = d[i][1], p[16] = d[i][2], p[24] = d[i][3];
    }
  }
};

// Late claim (moe-r35b, MoeSgKernel OPT bits 8..15): the unit body issues the wave's next claim at the start of the
// round that leaves at most `lead` steps of the unit (round 0 for a short unit), so the claim order follows the
// unit ends and a wave does not bind its next unit while it still has a long unit to run. Lane 0 gets the value.
struct SgLate {
  unsigned* ctl;
  int lead;
  unsigned* next;
  __device__ __forceinline__ void At(int s0, int total, int d, int lane) const {
    if (s0 == (total > lead ? total - lead : 0) / d * d) {
      unsigned c = 0;
      if (lane == 0) c = atomicAdd(ctl, 1u);
      *next = c;
    }
  }
};

// Deferred unit completion (moe-r35b, MoeSgKernel OPT bit 2): a gate/up unit does not wait at its end for its
// partial stores and its block counter (a release fence and an atomic with return, about 4 us under the stream
// load). The next gate/up unit does them at its first round, after its first D weight loads are issued, so the
// store acknowledgements and the atomic overlap those loads; the mid of a completed block runs after that unit.
// A down unit (it may wait for a mid) and the wave exit do them first. pjc: the counter of the pending unit, or
// nullptr; *old gets the counter value before the increment. late: the late claim (lead 0: none).
struct SgDefer {
  SgLate late;
  unsigned* pjc;
  unsigned* old;
  __device__ __forceinline__ void At(int s0, int total, int d, int lane) const {
    if (late.lead > 0) late.At(s0, total, d, lane);
    if (pjc != nullptr && s0 == 0) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      unsigned o = 0;
      if (lane == 0) o = __hip_atomic_fetch_add(pjc, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      *old = o;
    }
  }
};

// The LDS stores of one step must reach LDS before the same wave's loads of the next step read them; the
// hardware keeps one wave's LDS operations in order, this keeps the compiler from moving them.
__device__ __forceinline__ void SgLdsOrder() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront"); }

// K4 unit: column block cb of job j, N rows, D steps (512 B each) in flight. MUT (bench mutants, must fail the
// byte check): 1 = shuffle sums in the order (a0 + a2) + (a1 + a3), 2 = drops the last tile's hb terms, 3 = swaps
// tiles 0 and 1 of every step. Bench knockouts (timing only): 10 = no transpose, 11 = no x loads in the loop,
// 12 = down units do not wait for the mid (kernel level), 13 = no transpose and no decode.
// CB: consecutive column blocks cb .. cb + CB - 1 in one stream (their tiles are contiguous), so the weight ring
// runs on across the block boundaries and a short down unit does not pay a pipeline start for each block. The x
// steps repeat per block; the sums and the stores of a block happen at its last step.
// rows <= N: the job's rows. Rows past it run on zero x and are not stored; every stored row keeps its own
// arithmetic, so a body for more rows serves a job with fewer (fewer bodies, less code).
template <int N, int D, int CACHE, int MUT, int CB, bool LATE = false, class H = SgLate>
__device__ __forceinline__ void MoeSgUnitK4(const LinJob& j, int cb, uint32_t* xw, int rows = N, H lt = {}) {
  const int lane = threadIdx.x & 31;
  const int e = lane >> 3;
  const int A = __builtin_amdgcn_readfirstlane(j.I >> 4);
  const int steps = A >> 2;
  const uint32_t* tiles = UniformPtr(j.tiles);
  const auto rs = StreamRsrc(tiles + (size_t)cb * A * 32, (uint32_t)CB * A * 128);
  // x step of stream step s (x repeats per column block).
  auto xstep = [&](int s) { return CB == 1 ? s : s % steps; };
  const SgX<N> xs(j, A, lane, rows);
  constexpr int NX = SgX<N>::NX;
  B128u4 wr[D];
  B128u4 xr[D][NX];
  // Issue order: x(0), then pairs (x(k + 1), w(k)); the loop refills in the same order.
  B128u4 x0[NX];
  xs.Load(0, x0);
#pragma unroll
  for (int f = 0; f < D; ++f) {
    xs.Load(xstep(f + 1), xr[(f + 1) % D]);
    wr[f] = __builtin_amdgcn_raw_buffer_load_b128(rs, lane * 16, f * 512, CACHE);
    __builtin_amdgcn_sched_barrier(0);
  }
  xs.Store(xw, xw + 2 * N * 32, x0, lane);
  SgLdsOrder();
  float acc[N][2];
#pragma unroll
  for (int r = 0; r < N; ++r) acc[r][0] = acc[r][1] = 0.f;
  auto step = [&](int s, int f) {
    // x(s + 1) arrived with w(s) (issued just before it): stage it in the other slot.
    if constexpr (MUT != 11) xs.Store(xw + ((s + 1) & 1) * N * 32, xw + 2 * N * 32, xr[(f + 1) % D], lane);
    SgLdsOrder();
    const B128u4 w = wr[f];
    uint32_t v[4] = {w[0], w[1], w[2], w[3]};
    uint32_t p[4] = {(uint32_t)__builtin_amdgcn_mov_dpp8((int)w[3], kDpp8Prev), w[0], w[1], w[2]};
    if constexpr (MUT != 10 && MUT != 13) {
      SgTranspose(v, lane);
      SgTranspose(p, lane);
    }
    __half2 P[4][4];
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      if constexpr (MUT == 13) {
#pragma unroll
        for (int k = 0; k < 4; ++k) P[g][k] = B128AsH2(v[k] ^ p[g]);
      } else {
        DecodeLane<4, kMulMode>(p[g], v[g], 0u, 20, P[g]);
      }
    }
    // Refill after the decode has consumed the slots, so the loads reuse their registers (MoeGemvCols).
    if constexpr (MUT != 11) xs.Load(xstep(s + 1 + D), xr[(f + 1) % D]);
    wr[f] = __builtin_amdgcn_raw_buffer_load_b128(rs, lane * 16, (s + D) * 512, CACHE);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t* xsl = xw + (s & 1) * N * 32 + e * 8;
#pragma unroll
    for (int r = 0; r < N; ++r) {
      const B128u4 xa = *reinterpret_cast<const B128u4*>(xsl + r * 32);
      const B128u4 xb = *reinterpret_cast<const B128u4*>(xsl + r * 32 + 4);
      const uint32_t xx[8] = {xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]};
#pragma unroll
      for (int gi = 0; gi < 4; ++gi) {
        const int g = MUT == 3 ? (gi < 2 ? 1 - gi : gi) : gi;
        const __half2 ha = B128AsH2(xx[2 * g]), hb = B128AsH2(xx[2 * g + 1]);
        acc[r][0] = Dot2(P[g][0], ha, acc[r][0]);
        if (!(MUT == 2 && g == 3 && xstep(s) == steps - 1)) acc[r][0] = Dot2(P[g][1], hb, acc[r][0]);
        acc[r][1] = Dot2(P[g][2], ha, acc[r][1]);
        if (!(MUT == 2 && g == 3 && xstep(s) == steps - 1)) acc[r][1] = Dot2(P[g][3], hb, acc[r][1]);
      }
    }
  };
  // MoeGemvCols's shuffle sums: lane q = 0 holds (a0 + a1) + (a2 + a3); q is lane bits 3..4 here.
  float* out = j.partial;
  auto flush = [&](int b) {
    const int col = 16 * (cb + b) + (lane & 7);
#pragma unroll
    for (int r = 0; r < N; ++r) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = acc[r][h];
        if constexpr (MUT == 1) {
          v = v + SgXor16f(v);
          v = v + SgXor8f(v);
        } else {
          v = v + SgXor8f(v);
          v = v + SgXor16f(v);
        }
        acc[r][h] = 0.f;
        if (lane < 8 && r < rows) out[r * j.O + col + 8 * h] = v;
      }
    }
  };
  // Full rounds only (40 and 10 steps per block): each ring slot keeps its registers, and a block ends at the
  // end of a round.
  if (steps % D != 0) __builtin_trap();
  for (int s0 = 0; s0 < CB * steps; s0 += D) {
    if constexpr (LATE) lt.At(s0, CB * steps, D, lane);
#pragma unroll
    for (int f = 0; f < D; ++f) step(s0 + f, f);
    if constexpr (CB > 1) {
      if ((s0 + D) % steps == 0 && s0 + D < CB * steps) flush((s0 + D) / steps - 1);
    }
  }
  flush(CB - 1);
}

// K6 unit (shared expert): MoeGemvCols's loads and arithmetic for one column block, 4 tiles per step, DS steps in
// flight, x from the same per-wave LDS ring.
template <int N, int DS, int CACHE, bool LATE = false, class H = SgLate>
__device__ __forceinline__ void MoeSgUnitK6(const LinJob& j, int cb, uint32_t* xw, H lt = {}) {
  constexpr int K = 6, T = TileDwords(K);
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  const int A = __builtin_amdgcn_readfirstlane(j.I >> 4);
  const int steps = A >> 2;
  const uint32_t* tiles = UniformPtr(j.tiles);
  const auto rs = StreamRsrc(tiles + (size_t)cb * A * T, (uint32_t)A * T * 4);
  const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
  const SgX<N> xs(j, A, lane);
  constexpr int NX = SgX<N>::NX;
  uint32_t wr[DS][4][3];
  B128u4 xr[DS][NX];
  auto load_step = [&](int s, uint32_t (&dst)[4][3]) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int so = (4 * s + g) * T * 4;
      dst[g][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, so, CACHE);
      dst[g][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, so, CACHE);
      dst[g][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2, so, CACHE);
    }
  };
  B128u4 x0[NX];
  xs.Load(0, x0);
#pragma unroll
  for (int f = 0; f < DS; ++f) {
    xs.Load(f + 1, xr[(f + 1) % DS]);
    load_step(f, wr[f]);
    __builtin_amdgcn_sched_barrier(0);
  }
  xs.Store(xw, xw + 2 * N * 32, x0, lane);
  SgLdsOrder();
  float acc[N][2];
#pragma unroll
  for (int r = 0; r < N; ++r) acc[r][0] = acc[r][1] = 0.f;
  auto step = [&](int s, int f) {
    xs.Store(xw + ((s + 1) & 1) * N * 32, xw + 2 * N * 32, xr[(f + 1) % DS], lane);
    SgLdsOrder();
    __half2 P[4][4];
#pragma unroll
    for (int g = 0; g < 4; ++g) DecodeLane<K, kMulMode>(wr[f][g][0], wr[f][g][1], wr[f][g][2], win.shift, P[g]);
    xs.Load(s + 1 + DS, xr[(f + 1) % DS]);
    load_step(s + DS, wr[f]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t* xsl = xw + (s & 1) * N * 32 + q * 8;
#pragma unroll
    for (int r = 0; r < N; ++r) {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const B128u2 x2 = *reinterpret_cast<const B128u2*>(xsl + r * 32 + 2 * g);
        const __half2 ha = B128AsH2(x2[0]), hb = B128AsH2(x2[1]);
        acc[r][0] = Dot2(P[g][0], ha, acc[r][0]);
        acc[r][0] = Dot2(P[g][1], hb, acc[r][0]);
        acc[r][1] = Dot2(P[g][2], ha, acc[r][1]);
        acc[r][1] = Dot2(P[g][3], hb, acc[r][1]);
      }
    }
  };
  if (steps % DS != 0) __builtin_trap();
  for (int s0 = 0; s0 < steps; s0 += DS) {
    if constexpr (LATE) lt.At(s0, steps, DS, lane);
#pragma unroll
    for (int f = 0; f < DS; ++f) step(s0 + f, f);
  }
#pragma unroll
  for (int r = 0; r < N; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float v = acc[r][h];
      v = v + __shfl_xor(v, 1);
      v = v + __shfl_xor(v, 2);
      acc[r][h] = v;
    }
  if (q == 0) {
    float* out = j.partial;
    const int col = 16 * cb + lane / 4;
#pragma unroll
    for (int r = 0; r < N; ++r) {
      out[r * j.O + col] = acc[r][0];
      out[r * j.O + col + 8] = acc[r][1];
    }
  }
}

// K4 bodies for 1, 2, 4 and R rows only (a job of 3 rows runs the 4-row body): fewer bodies in the kernel.
template <int R, int D, int DS, int CACHE, int MUT, int CB, bool LATE = false, class H = SgLate>
__device__ __forceinline__ void MoeSgRowsP2(int K, int rows, const LinJob& j, int cb, uint32_t* xw, H lt = {}) {
  static_assert(!LATE || CB == 1, "late claim: one column block per K6 unit");
  if (K == 6) {
    if (rows != R) __builtin_trap();
    for (int b = 0; b < CB; ++b) MoeSgUnitK6<R, DS, CACHE, LATE, H>(j, cb + b, xw, lt);
  } else if (K != 4) {
    __builtin_trap();
  } else if (rows <= 1) {
    MoeSgUnitK4<1, D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else if (R >= 2 && rows <= 2) {
    MoeSgUnitK4<(R >= 2 ? 2 : 1), D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else if (R >= 4 && rows <= 4) {
    MoeSgUnitK4<(R >= 4 ? 4 : 1), D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else {
    MoeSgUnitK4<R, D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  }
}

// K4 body sets of the one-call-site kernels (MoeSgKernel NP 2 to 5): 2 = bodies
// for 1, 2, 4 and R rows (MoeSgRowsP2's set without its duplicate bodies at R = 2 and 4), 3 = 1 and R rows, 4 = 1,
// 2 and R rows, 5 = R rows only. A job runs the smallest body of at least its rows (same arithmetic per stored row).
// MUT 23 (mutant, must fail the byte check): a job of 2 rows runs the 1-row body.
template <int R, int NP, int D, int DS, int CACHE, int MUT, int CB, bool LATE = false, class H = SgLate>
__device__ __forceinline__ void MoeSgRowsSet(int K, int rows, const LinJob& j, int cb, uint32_t* xw, H lt = {}) {
  static_assert(NP >= 2 && NP <= 5);
  static_assert(!LATE || CB == 1, "late claim: one column block per K6 unit");
  constexpr bool b1 = NP != 5 && R > 1, b2 = (NP == 2 || NP == 4) && R > 2, b4 = NP == 2 && R > 4;
  if (K == 6) {
    if (rows != R) __builtin_trap();
    for (int b = 0; b < CB; ++b) MoeSgUnitK6<R, DS, CACHE, LATE, H>(j, cb + b, xw, lt);
  } else if (K != 4) {
    __builtin_trap();
  } else if (b1 && rows <= (MUT == 23 ? 2 : 1)) {
    MoeSgUnitK4<1, D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else if (b2 && rows <= 2) {
    MoeSgUnitK4<(b2 ? 2 : 1), D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else if (b4 && rows <= 4) {
    MoeSgUnitK4<(b4 ? 4 : 1), D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  } else {
    MoeSgUnitK4<R, D, CACHE, MUT, CB, LATE, H>(j, cb, xw, rows, lt);
  }
}

template <int N, int D, int DS, int CACHE, int MUT, int CB>
__device__ __forceinline__ void MoeSgRows(int K, int rows, int R, const LinJob& j, int cb, uint32_t* xw) {
  if constexpr (N >= 1) {
    if (rows == N) {
      if (K == 4) {
        MoeSgUnitK4<N, D, CACHE, MUT, CB>(j, cb, xw);
      } else if (K == 6 && rows == R) {
        for (int b = 0; b < CB; ++b) MoeSgUnitK6<N, DS, CACHE>(j, cb + b, xw);
      } else {
        __builtin_trap();
      }
    } else {
      MoeSgRows<N - 1, D, DS, CACHE, MUT, CB>(K, rows, R, j, cb, xw);
    }
  }
}

// The mid of one 128-column block of job u for all its rows: MoeFusedKernel's arithmetic, into the down job's xh.
__device__ __forceinline__ void MoeSgMid(const LinJob* gu, const LinJob& d, int u, int blk, int rows) {
  const int lane = threadIdx.x & 31;
  __half* xh = UniformPtr(d.xh);
  for (int r = 0; r < rows; ++r) {
    float gv[4], uv[4];
    LinOutBlock(gu[u * 2], r, blk, gv);
    LinOutBlock(gu[u * 2 + 1], r, blk, uv);
    const int c0 = blk * 128 + lane * 4;
    float sv[4], v[4];
    __half sh[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) sv[i] = Silu(gv[i]) * uv[i];
    LinScaleLane(d.suh, c0, sh);
    LinInLane(sv, sh, v);  // RoundH(RoundH(sv) * suh), Wht128: MoeFusedKernel's mid
#pragma unroll
    for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
  }
}

// Bench timeline trace (MUT 40, moe-r35b, bench/moe_sg_engine.hip variants 190 to 199): a u64 pointer at ctl +
// kSgCtlWords. Words: [0] total units, [1] gate/up units, [2] jobs; per claim c at kSgTrUnit + 6 c: meta (wave id |
// is_gu << 16 | job << 20 | k << 32), t claim known, t mid wait done, t body done, t mid done; per wave id g at
// kSgTrWave + 2 g: entry and exit. Times: wall_clock64 (100 MHz).
constexpr int kSgTrUnit = 8, kSgTrMaxUnits = (kPackedRows * kTopK + 2) * (kSgGuUnits + kSgDnUnits);
constexpr int kSgTrWave = kSgTrUnit + 6 * kSgTrMaxUnits, kSgTrWords = kSgTrWave + 2 * 8 * 1024;

// MoeSgMid with the loads of all rows first (moe-r35b, MoeSgKernel OPT bit 1): MoeSgMid issues each row's partial
// and scale loads after the previous row's xh stores (the compiler cannot move loads above a store that may alias),
// so the mid costs rows x the load latency, which is several us under the stream load. Per element the arithmetic
// is LinOutBlock's (0 + the S partials in order, Wht128, times kHadNorm times svh) and MoeSgMid's (Silu(g) u,
// LinScaleLane, LinInLane, times kHadNorm, to half). Rows past `rows` load row rows - 1 and are not stored.
template <int N>
__device__ __forceinline__ void MoeSgMidRows(const LinJob* gu, const LinJob& d, int u, int blk, int rows) {
  const int lane = threadIdx.x & 31;
  const int c0 = blk * 128 + lane * 4;
  const LinJob& jg = gu[u * 2];
  const LinJob& ju = gu[u * 2 + 1];
  float gv[N][4], uv[N][4];
#pragma unroll
  for (int r = 0; r < N; ++r)
#pragma unroll
    for (int i = 0; i < 4; ++i) gv[r][i] = uv[r][i] = 0.f;
  for (int s = 0; s < jg.S; ++s) {
#pragma unroll
    for (int r = 0; r < N; ++r) {
      const float* p = jg.partial + ((size_t)s * jg.R + (r < rows ? r : rows - 1)) * jg.O + c0;
#pragma unroll
      for (int i = 0; i < 4; ++i) gv[r][i] += p[i];
    }
  }
  for (int s = 0; s < ju.S; ++s) {
#pragma unroll
    for (int r = 0; r < N; ++r) {
      const float* p = ju.partial + ((size_t)s * ju.R + (r < rows ? r : rows - 1)) * ju.O + c0;
#pragma unroll
      for (int i = 0; i < 4; ++i) uv[r][i] += p[i];
    }
  }
  float fg[4], fu[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fg[i] = kHadNorm * __half2float(jg.svh[c0 + i]);
    fu[i] = kHadNorm * __half2float(ju.svh[c0 + i]);
  }
  __half sh[4];
  LinScaleLane(d.suh, c0, sh);
  __half* xh = UniformPtr(d.xh);
#pragma unroll
  for (int r = 0; r < N; ++r) {
    if (r >= rows) break;
    Wht128(gv[r]);
    Wht128(uv[r]);
    float sv[4], v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gv[r][i] *= fg[i];
      uv[r][i] *= fu[i];
      sv[i] = Silu(gv[r][i]) * uv[r][i];
    }
    LinInLane(sv, sh, v);
#pragma unroll
    for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
  }
}

__device__ __forceinline__ unsigned SgClaim(unsigned* ctl, int lane) {
  unsigned c = 0;
  if (lane == 0) c = atomicAdd(ctl, 1u);
  return c;
}

// pass_rows: rows of the pass (1 to kPackedRows; jobs: MoeExpertJobs(pass_rows), routed slots pass_rows * kTopK,
// then the shared chunks). R = min(pass_rows, kMaxRows): rows per job. D: K4 steps in flight, DS: K6 steps in
// flight. ctl: kSgCtlWords zeroed words (the kernel leaves them zeroed).
// WPE: minimum waves per SIMD the compiler must allow (a VGPR cap: 12 = 128 VGPRs, 3 blocks per CU).
// CBG, CBD: column blocks per gate/up unit and per down unit (MoeSgUnitK4's CB; CBG divides 8, CBD divides 160).
// NP: 1 = K4 bodies for 1, 2, 4 and R rows only (MoeSgRowsP2); 2 to 5 = one call site for gate/up and down units
// (CBG = CBD) with the body sets of MoeSgRowsSet (moe-r35: the D2 kernel at R 4 was 56 KB of code, its hot loops
// about 36 KB, over the WGP instruction cache).
// MUT (engine-path test mutants, bench/moe_sg_engine.hip): 20 = the job layout from R instead of pass_rows (the
// packed-pass defect), 21 = the mid of slot u + 1, 22 = the mid writes one row too many, 23 = a 2-row job runs the
// 1-row body (NP 2 to 5).
// NP bits 4 and up: OPT (moe-r35b; 0 = the code above, unchanged; in NP so that the names and the code of the NP 0
// to 5 kernels stay the same): bit 0 = the shared jobs' down units after the routed jobs' down
// units (the shared K6 gate/up units are the slowest, so their mid is late; first in the down order, the waves that
// claim those down units wait for it); bit 1 = the mid with the loads of all rows first (MoeSgMidRows); bit 2 = the
// deferred unit completion (SgDefer); bit 3 = hold the lookahead helper (LookaheadHold) for the launch; bit 4 = hold
// it only between the ramp and the tail (from claim gridDim.x * 8 to the first empty claim); bits 16..18 = helper
// blocks to hold (0 = all); bits 19..22 = with bit 4, release the hold that many quarter rounds of claims (gridDim.x * 2
// claims each) before the first empty claim; bits 23..26 = with bit 4, set it at claim gridDim.x * 2 times this
// (0 = 4, one round); bits 8..15 =
// late claim lead in steps (SgLate; 0 = the next claim at the unit start).
template <int R, int D, int DS, int CACHE = 4, int MUT = 0, int WPE = 1, int CBG = 1, int CBD = 1, int NPO = 0>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE))) MoeSgKernel(const LinJob* __restrict__ gu, const LinJob* __restrict__ dn,
                                                   unsigned* __restrict__ ctl, int pass_rows) {
  constexpr int NP = NPO & 15, OPT = NPO >> 4;
  __shared__ uint32_t xl[8][2 * R * 32 + 32];  // 2 x slots and a spare line
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  uint32_t* xw = xl[wave];
  constexpr bool kTr = MUT == 40;
  constexpr bool kShLast = (OPT & 1) != 0, kFastMid = (OPT & 2) != 0;
  constexpr int kLead = (OPT >> 8) & 0xFF;
  constexpr bool kLate = kLead > 0, kDefer = (OPT & 4) != 0, kHook = kLate || kDefer, kHold = (OPT & 8) != 0;
  constexpr bool kHoldMid = kHold && (OPT & 16) != 0;
  constexpr unsigned kHoldBlocks = (OPT >> 16) & 7 ? (OPT >> 16) & 7 : 255u;
  constexpr unsigned kRelQ = (OPT >> 19) & 15, kStartQ = (OPT >> 23) & 15 ? (OPT >> 23) & 15 : 4u;
  static_assert(!kHook || NP >= 1, "late claim, deferred completion: NP 1 to 5 bodies");
  uint64_t* tr = nullptr;
  const unsigned gid = blockIdx.x * 8 + wave;
  if constexpr (kTr) {
    tr = *reinterpret_cast<uint64_t* const*>(ctl + kSgCtlWords);
    if (lane == 0) tr[kSgTrWave + 2 * gid] = wall_clock64();
  }
  if constexpr (kHold && !kHoldMid) {
    if (blockIdx.x == 0 && threadIdx.x == 0) LookaheadHold(kHoldBlocks);
  }
  if (MUT != 20 && (pass_rows > kPackedRows || (pass_rows < kMaxRows ? pass_rows : kMaxRows) != R)) __builtin_trap();
  const int prows = MUT == 20 ? R : pass_rows;
  const int U = prows * kTopK, J = MoeExpertJobs(prows), NS = J - U;
  // Live routed jobs are 0..n-1 (MoeRowsUniqueKernel clears K of the others).
  int n = 0;
  for (int b = 0; b < U; b += 32) {
    const int u = b + lane;
    const bool live = u < U && gu[2 * u].K != 0;
    n += __builtin_popcount((unsigned)__ballot(live));
  }
  n = __builtin_amdgcn_readfirstlane(n);
  const int nj = NS + n;
  static_assert(8 % CBG == 0 && kSgDnUnits % CBD == 0);
  constexpr int GU = kSgGuUnits / CBG, DN = kSgDnUnits / CBD, GB = 8 / CBG;  // units per job, per 8 blocks
  const unsigned n_gu = (unsigned)nj * GU, total = n_gu + (unsigned)nj * DN;
  if constexpr (kTr) {
    if (gid == 0 && lane == 0) tr[0] = total, tr[1] = n_gu, tr[2] = nj;
  }
  // OPT bit 2: the pending gate/up unit (its counter, job, block and rows) and the mid after a block's last unit.
  [[maybe_unused]] unsigned* pjc = nullptr;
  [[maybe_unused]] int pu = 0, pblk = 0, pnr = 0;
  [[maybe_unused]] unsigned pold = 0;
  [[maybe_unused]] auto pmid = [&](unsigned old) {
    if (__builtin_amdgcn_readfirstlane(__shfl(old, 0)) == (unsigned)(2 * GB - 1)) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      const int drows = __builtin_amdgcn_readfirstlane(dn[pu].R);
      const int mu = MUT == 21 ? (pu + 1) % J : pu, mrows = MUT == 22 ? pnr + 1 : pnr < drows ? pnr : drows;
      if constexpr (kFastMid) MoeSgMidRows<R>(gu, dn[pu], mu, pblk, mrows);
      else MoeSgMid(gu, dn[pu], mu, pblk, mrows);
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      if (lane == 0)
        __hip_atomic_fetch_add(ctl + 2 + kSgJobWords * pu + 5, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  };
  // Completes the pending unit now (before a down unit's wait and at the wave exit).
  [[maybe_unused]] auto pflush = [&] {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    unsigned old = 0;
    if (lane == 0) old = __hip_atomic_fetch_add(pjc, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    pmid(old);
    pjc = nullptr;
  };
  unsigned c = __builtin_amdgcn_readfirstlane(__shfl(SgClaim(ctl, lane), 0));
  while (c < total) {
    uint64_t tt[4] = {};
    if constexpr (kTr) tt[0] = wall_clock64();
    if constexpr (kHoldMid) {
      if (c == gridDim.x * 2 * kStartQ && lane == 0) LookaheadHold(kHoldBlocks);
      if constexpr (kRelQ > 0) {
        if (c + gridDim.x * 2 * kRelQ == total && lane == 0) LookaheadHold(0u);
      }
    }
    // The next claim is issued first, so it returns before this unit's weight loads (late claim: in the body).
    const unsigned cn_raw = kLate ? 0u : SgClaim(ctl, lane);
    unsigned cn_late = 0;
    const bool is_gu = c < n_gu;
    const unsigned cc = is_gu ? c : c - n_gu;
    const int li = (int)(cc / (is_gu ? GU : DN));
    const int k = (int)(cc - li * (is_gu ? GU : DN));
    const int u = kShLast && !is_gu ? (li < n ? li : U + li - n) : li < NS ? U + li : li - NS;
    SgLate lt{nullptr, 0, nullptr};
    if constexpr (kLate) lt = SgLate{ctl, kLead, &cn_late};
    // Gate/up: per 128-column block, GB gate units then GB up units.
    const int blk = k / (2 * GB);
    const LinJob& j = is_gu ? gu[2 * u + (k / GB) % 2] : dn[u];
    const int cb = is_gu ? 8 * blk + CBG * (k % GB) : CBD * k;
    unsigned* jc = ctl + 2 + kSgJobWords * u;
    if constexpr (kDefer) {
      if (!is_gu && pjc != nullptr) pflush();
    }
    if (!is_gu && MUT != 12) {
      if (lane == 0)
        while (__hip_atomic_load(jc + 5, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kSgMidBlocks)
          __builtin_amdgcn_s_sleep(1);
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    if constexpr (kTr) tt[1] = wall_clock64();
    [[maybe_unused]] const SgDefer hd{lt, pjc, &pold};
    const int K = __builtin_amdgcn_readfirstlane(j.K);
    const int rows = __builtin_amdgcn_readfirstlane(j.R);
    if constexpr (NP >= 2) {
      static_assert(CBG == CBD, "one call site needs one unit length");
      if constexpr (kDefer) MoeSgRowsSet<R, NP, D, DS, CACHE, MUT, CBG, true>(K, rows, j, cb, xw, hd);
      else if constexpr (kLate) MoeSgRowsSet<R, NP, D, DS, CACHE, MUT, CBG, true>(K, rows, j, cb, xw, lt);
      else MoeSgRowsSet<R, NP, D, DS, CACHE, MUT, CBG>(K, rows, j, cb, xw);
    } else if constexpr (NP == 1 && kDefer) {
      if (is_gu) MoeSgRowsP2<R, D, DS, CACHE, MUT, CBG, true>(K, rows, j, cb, xw, hd);
      else MoeSgRowsP2<R, D, DS, CACHE, MUT, CBD, true>(K, rows, j, cb, xw, hd);
    } else if constexpr (NP == 1 && kLate) {
      if (is_gu) MoeSgRowsP2<R, D, DS, CACHE, MUT, CBG, true>(K, rows, j, cb, xw, lt);
      else MoeSgRowsP2<R, D, DS, CACHE, MUT, CBD, true>(K, rows, j, cb, xw, lt);
    } else if constexpr (NP == 1) {
      if (is_gu) MoeSgRowsP2<R, D, DS, CACHE, MUT, CBG>(K, rows, j, cb, xw);
      else MoeSgRowsP2<R, D, DS, CACHE, MUT, CBD>(K, rows, j, cb, xw);
    } else {
      if (is_gu) MoeSgRows<R, D, DS, CACHE, MUT, CBG>(K, rows, R, j, cb, xw);
      else MoeSgRows<R, D, DS, CACHE, MUT, CBD>(K, rows, R, j, cb, xw);
    }
    if constexpr (kTr) tt[2] = tt[3] = wall_clock64();
    if constexpr (kDefer) {
      if (is_gu) {
        // The body did the pending unit's fence and counter at its first round (SgDefer).
        if (pjc != nullptr) {
          pmid(pold);
          if constexpr (kTr) tt[3] = wall_clock64();
        }
        pjc = jc + blk, pu = u, pblk = blk, pnr = rows;
      }
    } else if (is_gu) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      unsigned old = 0;
      if (lane == 0) old = __hip_atomic_fetch_add(jc + blk, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      old = __builtin_amdgcn_readfirstlane(__shfl(old, 0));
      if (old == 2 * GB - 1) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        // The down job has the same rows (MoeRowsUniqueBody); the min keeps the mid inside its xh rows.
        const int drows = __builtin_amdgcn_readfirstlane(dn[u].R);
        const int mu = MUT == 21 ? (u + 1) % J : u, mrows = MUT == 22 ? rows + 1 : rows < drows ? rows : drows;
        if constexpr (kFastMid) MoeSgMidRows<R>(gu, dn[u], mu, blk, mrows);
        else MoeSgMid(gu, dn[u], mu, blk, mrows);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (lane == 0) __hip_atomic_fetch_add(jc + 5, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if constexpr (kTr) tt[3] = wall_clock64();
      }
    }
    if constexpr (kTr) {
      if (lane == 0) {
        uint64_t* p = tr + kSgTrUnit + 6 * (size_t)c;
        p[0] = gid | (uint64_t)is_gu << 16 | (uint64_t)u << 20 | (uint64_t)k << 32;
        p[1] = tt[0], p[2] = tt[1], p[3] = tt[2], p[4] = tt[3];
      }
    }
    c = __builtin_amdgcn_readfirstlane(__shfl(kLate ? cn_late : cn_raw, 0));
  }
  if constexpr (kDefer) {
    if (pjc != nullptr) pflush();
  }
  if constexpr (kHoldMid) {
    if (c == total && lane == 0) LookaheadHold(0u);
  }
  if constexpr (kTr) {
    if (lane == 0) tr[kSgTrWave + 2 * gid + 1] = wall_clock64();
  }
  unsigned last = 0;
  if (lane == 0) last = atomicAdd(ctl + 1, 1u) == gridDim.x * 8 - 1;
  if (__builtin_amdgcn_readfirstlane(__shfl(last, 0)) == 1u) {
    // Last wave out: every claim and counter update of this launch is done.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (int i = lane; i < 2 + kSgJobWords * J; i += 32) ctl[i] = 0;
    if constexpr (kHold) {
      if (lane == 0) LookaheadHold(0u);
    }
  }
}

// Engine launch (CRUCIBLE_MOE_B128=3 and 12 through MoeSgModeLaunch; bench/moe_sg_engine.hip calls the same function): R =
// min(pass_rows, kMaxRows). Grid: kSgBlocks persistent blocks, or fewer if not all are resident. MUT, D, WPE, NP:
// the bench mutants and launch shapes (MoeSgKernel); CRUCIBLE_MOE_B128=3 uses the defaults, 12 the shape of
// MoeSgModeShape(12) (MoeSgModeLaunch). grid_cap > 0 caps the grid at that many blocks.
constexpr int kSgBlocks = 120;

// The grid of the instance for R and the shape arguments: the occupancy of the exact kernel instance that is launched.
// The first call asks the occupancy query; the engine makes that call for every instance of the active mode at start-up
// (MoeSgCheckGrids), so a device with no room for the kernel stops there and not at the first verify of one R (review L1
// of t29). A grid under 1 stops the process (review L2 of moe-sg). The launch applies its grid cap after this grid.
// DS, OPT, CB (moe-r35b): K6 steps in flight, MoeSgKernel's OPT, column blocks per unit (CBG = CBD = CB; bench).
template <int R, int MUT = 0, int D = kSgDepth, int WPE = kSgWaves, int NP = kSgNp, int DS = kSgDepthK6, int OPT = 0,
          int CB = 0>
int MoeSgGrid() {
  static_assert(NP >= 0 && NP < 16);
  static const int grid = [] {
    const auto kernel = MoeSgKernel<R, D, DS, 4, MUT, WPE, CB ? CB : kSgCbg, CB ? CB : kSgCbd, NP | OPT << 4>;
    int per = 0, dev = 0;
    HIP_OK(hipGetDevice(&dev));
    hipDeviceProp_t p{};
    HIP_OK(hipGetDeviceProperties(&p, dev));
    HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, reinterpret_cast<const void*>(kernel), 256, 0));
    const int blocks = MoeSgGridBlocks(kSgBlocks, per, p.multiProcessorCount);
    if (blocks < 1) {
      std::fputs(MoeSgNoBlockMessage(R, per).c_str(), stderr);
      std::exit(1);
    }
    return blocks;
  }();
  return grid;
}

template <int R, int MUT = 0, int D = kSgDepth, int WPE = kSgWaves, int NP = kSgNp, int DS = kSgDepthK6, int OPT = 0,
          int CB = 0>
void MoeSgEngineLaunch(const LinJob* gu, const LinJob* dn, unsigned* ctl, int pass_rows, int grid_cap = 0) {
  const auto kernel = MoeSgKernel<R, D, DS, 4, MUT, WPE, CB ? CB : kSgCbg, CB ? CB : kSgCbd, NP | OPT << 4>;
  const int grid = MoeSgGrid<R, MUT, D, WPE, NP, DS, OPT, CB>();
  hipLaunchKernelGGL(kernel, dim3(grid_cap > 0 ? std::min(grid, grid_cap) : grid), dim3(256), 0, 0, gu, dn, ctl,
                     pass_rows);
}

// The shape of mode 12 is MoeSgModeShape(kMoeB128Hold); any other mode launches the default shape (the template
// defaults of MoeSgGrid and MoeSgEngineLaunch). This ties the defaults to MoeSgModeShape(3).
constexpr MoeSgShape kSgModeShape12 = MoeSgModeShape(kMoeB128Hold);
static_assert(kSgModeShape12.waves > 0 && (kSgModeShape12.opt & 8) != 0);
static_assert(MoeSgModeShape(kMoeB128Stream).depth == kSgDepth && MoeSgModeShape(kMoeB128Stream).np == kSgNp &&
              MoeSgModeShape(kMoeB128Stream).waves == 0 && MoeSgModeShape(kMoeB128Stream).opt == 0);

// CRUCIBLE_MOE_B128 = 3 and 12 (Engine::MoeSgLaunch; bench/moe_sg_engine.hip variants 170 and 260): 12 runs the
// shape of MoeSgModeShape(12) with its grid cap at 2..8 rows, any other mode the default shape (3). MUT: the bench
// mutants. grid_cap > 0 (bench) replaces the mode's cap.
template <int R, int MUT = 0>
void MoeSgModeLaunch(int mode, const LinJob* gu, const LinJob* dn, unsigned* ctl, int pass_rows, int grid_cap = 0) {
  constexpr MoeSgShape s = kSgModeShape12;
  if (mode == kMoeB128Hold)
    MoeSgEngineLaunch<R, MUT, s.depth, s.waves, s.np, kSgDepthK6, s.opt>(
        gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : MoeSgGridCap(mode, pass_rows));
  else
    MoeSgEngineLaunch<R, MUT>(gu, dn, ctl, pass_rows, grid_cap);
}

// The grid of the instance that MoeSgModeLaunch<R>(mode, ...) launches (the same template arguments as above).
template <int R, int MUT = 0>
int MoeSgModeGrid(int mode) {
  constexpr MoeSgShape s = kSgModeShape12;
  return mode == kMoeB128Hold ? MoeSgGrid<R, MUT, s.depth, s.waves, s.np, kSgDepthK6, s.opt>() : MoeSgGrid<R, MUT>();
}

// Start-up check of the active mode (called where MoeSgCtlNeeded(mode) allocates the control words): the instance of
// every R the engine can launch, MoeSgFirstR to kMaxRows. Only the shape of the active mode is checked; the other
// shape is never launched.
template <int... I>
void MoeSgCheckGridsOf(int mode, std::integer_sequence<int, I...>) {
  (MoeSgModeGrid<MoeSgFirstR + I>(mode), ...);
}
inline void MoeSgCheckGrids(int mode) {
  MoeSgCheckGridsOf(mode, std::make_integer_sequence<int, kMaxRows - MoeSgFirstR + 1>());
}

}  // namespace crucible
