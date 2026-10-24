// Fused mul1 EXL3 prefill GEMM for gfx11 wave32 WMMA.
#pragma once
#include "exl3_gemv.h"

namespace crucible {
using GemmH16 = _Float16 __attribute__((ext_vector_type(16)));
using GemmF8 = float __attribute__((ext_vector_type(8)));

// A gathered job writes compact rows, so the engine can use partial[S][M][O]
// unchanged. row_ids indexes the already transformed x pool, not the output.
struct Exl3GemmJob {
  const uint32_t* tiles;
  const __half* x;
  float* partial;
  const int* row_ids;
  int I, O, M, S, K, Xrows;
  // > 0: tiles in the front order of engine/exl3/front_order.h with this split
  // (only PrefillWideBody<..., FRONT> reads it); 0: engine order.
  int front_s;
};
struct Exl3GemmItem { int job, s, m0, b0; };
constexpr int kGemmBStride = 12;
constexpr int kGemmAStride = 8;
constexpr int kGemmWaveWords = 16 * kGemmBStride;
constexpr int kGemmWeightWords = 4 * kGemmWaveWords;

// These exchanges publish LDS data only. Group fences must not invalidate
// read-only global inputs while they publish a new shared slice.
__device__ __forceinline__ void GemmWaveSync() {
#ifdef EXL3_GEMM_GLOBAL_FENCES
  __syncwarp();
#else
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
#endif
}
__device__ __forceinline__ void GemmGroupSync() {
#ifdef EXL3_GEMM_GLOBAL_FENCES
  __syncthreads();
#else
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#endif
}

template <bool LDS>
__device__ __forceinline__ GemmH16 GemmBFragment(const __half2 p[4], uint32_t* w, int lane) {
  uint32_t v[4];
#pragma unroll
  for (int z = 0; z < 4; ++z) v[z] = *reinterpret_cast<const uint32_t*>(&p[z]);
  GemmH16 bf;
  uint32_t* bp = reinterpret_cast<uint32_t*>(&bf);
  if constexpr (LDS) {
    // Padding rotates column bank groups without losing b128 alignment.
    // A stride of 8 conflicts; a stride of 9 forces scalarized fragment reads.
    const int q = lane & 3, c = lane / 4;
    w[c * kGemmBStride + q] = v[0];
    w[c * kGemmBStride + q + 4] = v[1];
    w[(c + 8) * kGemmBStride + q] = v[2];
    w[(c + 8) * kGemmBStride + q + 4] = v[3];
    GemmWaveSync();
#pragma unroll
    for (int z = 0; z < 8; ++z) bp[z] = w[(lane & 15) * kGemmBStride + z];
    GemmWaveSync();
  } else {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int src = 4 * (lane & 7) + q;
      const uint32_t lo0 = __shfl(v[0], src), hi0 = __shfl(v[2], src);
      const uint32_t lo1 = __shfl(v[1], src), hi1 = __shfl(v[3], src);
      bp[q] = (lane & 8) ? hi0 : lo0;
      bp[q + 4] = (lane & 8) ? hi1 : lo1;
    }
  }
  return bf;
}

template <bool STAGE_X>
__device__ __forceinline__ GemmH16 GemmAFragment(__amdgpu_buffer_rsrc_t rs,
    uint32_t offset, int a, const uint32_t* xs, int row, int astep = 32) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  GemmH16 af;
  auto* ap = reinterpret_cast<U4*>(&af);
  if constexpr (STAGE_X) {
    const auto* sp = reinterpret_cast<const U4*>(xs + row * kGemmAStride);
    ap[0] = sp[0]; ap[1] = sp[1];
  } else {
    ap[0] = __builtin_amdgcn_raw_buffer_load_b128(rs, offset, a * astep, 0);
    ap[1] = __builtin_amdgcn_raw_buffer_load_b128(rs, offset, a * astep + 16, 0);
  }
  return af;
}

// Each wave owns an output column tile and reuses its decoded B over RM row
// tiles. Default LDS windows are disjoint, so only wave barriers are needed.
// LDS=false is the direct lane-permute alternative; staged X uses group barriers.
// KEEP returns the accumulators in keep[] instead of storing partial sums.
template <int K, int RM, bool LDS, bool STAGE_X, int PF, bool REGISTER_A, bool PACKED_X, bool SKIP_PAD_RM, bool STRICT_A_ORDER,
          bool KEEP = false>
__device__ __forceinline__ void Exl3GemmBody(const Exl3GemmJob& j,
                                           const Exl3GemmItem& it, uint32_t* ws, GemmF8* keep = nullptr) {
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int b = it.b0 + wave;
  if (b * 16 >= j.O) return;
  const int A = j.I / 16;
  const int begin = A * it.s / j.S, end = A * (it.s + 1) / j.S;
  const LaneWindow<K> win(lane);
  constexpr int T = TileDwords(K);
  const auto rs = StreamRsrc(j.tiles + ((size_t)b * A + begin) * T,
                             (end - begin) * T * 4);
  GemmF8 acc[RM] = {};
  // Raw-buffer OOB zeros remove divergent tail-row branches from the hot loop.
  // Gather offsets are fixed for the entire K range, so route loads occur once.
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = PACKED_X ? j.Xrows * 32 : 32;
  uint32_t xo[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row = it.m0 + r * 16 + (lane & 15);
    const int xr = row < j.M ? (j.row_ids ? j.row_ids[row] : row) : j.Xrows;
    xo[r] = PACKED_X ? (row < j.M ? (uint32_t)xr * 32 : xbytes) : (uint32_t)xr * j.I * 2;
  }
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  uint32_t txo[(RM + 3) / 4];
  uint32_t* xs = STAGE_X ? ws + kGemmWeightWords : ws;
  if constexpr (STAGE_X) {
#pragma unroll
    for (int z = 0; z < (RM + 3) / 4; ++z) {
      const int row = it.m0 + (threadIdx.x >> 1) + z * 64;
      const int xr = row < j.M ? (j.row_ids ? j.row_ids[row] : row) : j.Xrows;
      txo[z] = (PACKED_X ? (row < j.M ? (uint32_t)xr * 32 : xbytes) : (uint32_t)xr * j.I * 2)
               + (threadIdx.x & 1) * 16;
    }
  }
  uint32_t* w = LDS ? ws + wave * kGemmWaveWords : ws;
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  // K3 windows start at offsets {3,11,19,27} and span 37 bits; K4
  // starts at 20 and spans 44. Both fit two words, including ring wrap.
  // The intrinsic documents bounds checks for voffset, not soffset. Keep
  // complete prefetch offsets in voffset. A hardware probe also confirmed
  // scalar-offset checks for our gfx1151 descriptor word3=0x31004000.
  uint32_t raw[PF][3];
#pragma unroll
  for (int f = 0; f < PF; ++f) {
    raw[f][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0 + f * T * 4, 0, 0);
    raw[f][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1 + f * T * 4, 0, 0);
    raw[f][2] = 0;
    if constexpr (K >= 5) raw[f][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2 + f * T * 4, 0, 0);
  }
  for (int base = begin; base < end; base += PF) {
#pragma unroll
    for (int f = 0; f < PF; ++f) {
    const int a = base + f;
    if (a >= end) break;
    if constexpr (STAGE_X) {
#pragma unroll
      for (int z = 0; z < (RM + 3) / 4; ++z) {
        const int idx = threadIdx.x + z * 128;
        if (idx < RM * 32)
          reinterpret_cast<U4*>(xs + (idx >> 1) * kGemmAStride + (idx & 1) * 4)[0] =
              __builtin_amdgcn_raw_buffer_load_b128(xrs, txo[z], a * astep, 0);
      }
      GemmGroupSync();
    }
    const uint32_t w0 = raw[f][0], w1 = raw[f][1], w2 = raw[f][2];
    const int next = (a - begin + PF) * T * 4;
    // REGISTER_A issues next weights after all current activation loads, so
    // operand waits can leave younger packed prefetches outstanding.
    auto refill = [&]() {
      raw[f][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0 + next, 0, 0);
      raw[f][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1 + next, 0, 0);
      if constexpr (K >= 5) raw[f][2] = __builtin_amdgcn_raw_buffer_load_b32(rs, o2 + next, 0, 0);
    };
    if constexpr (!REGISTER_A) refill();
    __half2 p[4];
    DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
    const GemmH16 bf = GemmBFragment<LDS>(p, w, lane);
    if constexpr (!REGISTER_A) {
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if constexpr (SKIP_PAD_RM && RM > 1) { if (it.m0 + r * 16 >= j.M) break; }
        const GemmH16 af = GemmAFragment<STAGE_X>(xrs, xo[r], a, xs, r * 16 + (lane & 15), astep);
        acc[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[r]);
      }
    } else {
      GemmH16 afr[RM];
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if constexpr (SKIP_PAD_RM && RM > 1) { if (it.m0 + r * 16 >= j.M) break; }
        afr[r] = GemmAFragment<STAGE_X>(xrs, xo[r], a, xs, r * 16 + (lane & 15), astep);
      }
      // The compiler may otherwise move the younger packed loads ahead of A.
      if constexpr (STRICT_A_ORDER) asm volatile("" ::: "memory");
      refill();
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if constexpr (SKIP_PAD_RM && RM > 1) { if (it.m0 + r * 16 >= j.M) break; }
        acc[r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bf, acc[r]);
      }
    }
    if constexpr (STAGE_X) GemmGroupSync();
    }
  }
  if constexpr (KEEP) {
#pragma unroll
    for (int r = 0; r < RM; ++r) keep[r] = acc[r];
    return;
  }
  // Raw-buffer stores drop rows past M by the range check, so the epilogue has
  // no per-element branch or 64-bit address math. The whole offset stays in
  // voffset, which the range check covers.
  const auto ps = StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial + (size_t)it.s * j.M * j.O),
                             (uint32_t)j.M * j.O * 4);
  const uint32_t po = ((uint32_t)(it.m0 + lane / 16) * j.O + b * 16 + (lane & 15)) * 4;
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    if constexpr (SKIP_PAD_RM && RM > 1) { if (it.m0 + r * 16 >= j.M) break; }
#pragma unroll
    for (int z = 0; z < 8; ++z)
      __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[r][z]), ps,
                                            po + (uint32_t)(r * 16 + 2 * z) * j.O * 4, 0, 0);
  }
}

// PACKED_X reads [I/16][Xrows][16], with the same fp16 values. Its input
// buffer must be at most 2 GiB so invalid-row offsets cannot wrap to valid data.
template <int RM = 2, bool LDS = true, bool STAGE_X = false, int PF = 1,
          bool REGISTER_A = false, bool PACKED_X = false, bool SKIP_PAD_RM = false, bool STRICT_A_ORDER = false>
__global__ void __launch_bounds__(128) Exl3GemmKernel(
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n) {
  static_assert(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16);
  static_assert(PF == 1 || PF == 4 || PF == 8);
  __shared__ uint32_t ws[STAGE_X ? kGemmWeightWords + RM * 16 * kGemmAStride : LDS ? kGemmWeightWords : 1];
  const int idx = blockIdx.x;
  if (idx >= n) return;
  const Exl3GemmItem it = items[idx];
  const Exl3GemmJob j = jobs[it.job];
  if (j.K == 3) Exl3GemmBody<3, RM, LDS, STAGE_X, PF, REGISTER_A, PACKED_X, SKIP_PAD_RM, STRICT_A_ORDER>(j, it, ws);
  else if (j.K == 4) Exl3GemmBody<4, RM, LDS, STAGE_X, PF, REGISTER_A, PACKED_X, SKIP_PAD_RM, STRICT_A_ORDER>(j, it, ws);
  else if (j.K == 5) Exl3GemmBody<5, RM, LDS, STAGE_X, PF, REGISTER_A, PACKED_X, SKIP_PAD_RM, STRICT_A_ORDER>(j, it, ws);
  else if (j.K == 6) Exl3GemmBody<6, RM, LDS, STAGE_X, PF, REGISTER_A, PACKED_X, SKIP_PAD_RM, STRICT_A_ORDER>(j, it, ws);
}
}  // namespace crucible
