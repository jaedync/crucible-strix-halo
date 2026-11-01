// moe-b128: the decode expert stage with b128 (and b64) weight loads and no LDS transpose (bench/verify_moe.hip
// variants 110 and up).
//
// moe-d1 found that the real kernel streams at the rate of the 3 x b32 window stream (72 to 83% of the floor),
// while a pure b128 stream reaches 88 to 95%. Its b128 attempt (t128) moved the bytes through an LDS transpose.
// Here the lanes are remapped instead, so each lane decodes the dwords it loaded:
//
// - Q4 (K4, 32 dwords per tile): lane 8 g + m loads dwords 4 m .. 4 m + 3 of the current tile of column g (one b128
//   per lane: 4 columns x 128 B per wave instruction). For K4 the engine lane t decodes (dword t - 1, dword t) with
//   shift 20 (LaneWindow<4>; w2 is unused). So lane (g, m) is the engine's lanes t = 4 m .. 4 m + 3 of column g:
//   the only foreign dword, 4 m - 1 (mod 32), comes from lane (g, m - 1 mod 8) by one DPP8 move. Those 4 engine
//   lanes are one column pair (t / 4 = m) with q = 0..3, so the lane keeps the 4 per-q Dot2 chains of the engine
//   and sums them as the engine's shuffles do: (a0 + a1) + (a2 + a3). Bitwise equal by construction.
//   A wave covers 4 columns, a block of W waves 4 W columns (item_blocks 4 W).
// - B64 (K4): lane 16 g + m loads dwords 2 m, 2 m + 1 of column g (columns b and b + 8, as the engine's C2): the
//   engine lanes t = 2 m, 2 m + 1. The foreign dword 2 m - 1 comes from lane m - 1 (mod 16) by v_permlane16; the
//   q pairs (0, 1) and (2, 3) are in lanes 2 k and 2 k + 1, summed in the engine order by one shuffle.
//   Drop-in for the engine item shape (8 waves, item_blocks 16).
// - The K6 shared expert keeps the engine body (MoeGemvCols, any CC gives the same per-column arithmetic).
//
// Timing knockouts (MODE, real grid): 1 = x from registers (no LDS reads), 2 = no decode (raw words as halves),
// 3 = decode only (no Dot2, no LDS). The stream kernel MoeStreamQ4Kernel adds the decode and Dot2 work to the
// b128 stream of moe-d1 (MoeStreamKernel SHAPE 0) step by step.
#pragma once
#include "moe_flow.h"
#include "../../engine/layers/moe_b128.h"
#include "../../engine/layers/moe_sg.h"

namespace crucible {

// The b128 stream of moe-d1 (MoeStreamKernel SHAPE 0: per-wave claims of one column block, D b128 loads in
// flight) plus the Q4 work per load: WORK 0 none, 1 decode (4 DecodeLane per lane), 2 decode + Dot2 with x in
// registers (R rows), 3 decode + Dot2 with x from LDS (2 ds_load_b128 per row, wave-uniform address),
// 4 Dot2 with x from LDS and no decode. Timing only.
template <int D, int WORK, int R>
__global__ void __launch_bounds__(256) MoeStreamQ4Kernel(const LinJob* __restrict__ gu, const LinJob* __restrict__ dn,
                                                         int jobs, int shared_job, unsigned* __restrict__ next,
                                                         uint32_t* __restrict__ sink) {
  __shared__ uint32_t xl[R > 0 ? R * 160 * 8 : 8];
  const int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < (R > 0 ? R * 160 * 8 : 8); i += 256) xl[i] = 0x3c003c00u ^ (i * 2654435761u & 0x03ff03ffu);
  __syncthreads();
  const int n = jobs * kFlowUnitsPerJob;
  uint32_t accx = 0;
  float acc[4][R > 0 ? R : 1][2] = {};
  B128u4 xreg[2];
  xreg[0] = *reinterpret_cast<const B128u4*>(xl);
  xreg[1] = *reinterpret_cast<const B128u4*>(xl + 4);
  for (;;) {
    int c = 0;
    if (lane == 0) c = (int)atomicAdd(next, 1u);
    c = __builtin_amdgcn_readfirstlane(__shfl(c, 0));
    if (c >= n) {
      if (lane == 0 && atomicAdd(next + 1, 1u) == gridDim.x * 8 - 1) next[0] = 0, next[1] = 0;
      break;
    }
    int u = c / kFlowUnitsPerJob;
    const int r0 = c - u * kFlowUnitsPerJob;
    u = u == 0 ? shared_job : u - 1;
    const LinJob& j = r0 < 80 ? gu[2 * u + (r0 >= 40)] : dn[u];
    const int K = __builtin_amdgcn_readfirstlane(j.K);
    if (K == 0) continue;
    const int cb = r0 < 80 ? r0 % 40 : r0 - 80;
    const int A = __builtin_amdgcn_readfirstlane(j.I >> 4), T = 8 * K;
    const uint32_t* tiles = UniformPtr(j.tiles);
    const uint32_t bytes = (uint32_t)A * T * 4;
    const auto rs = StreamRsrc(tiles + (size_t)cb * A * T, bytes);
    const int steps = (int)(bytes / 512);
    B128u4 ring[D];
#pragma unroll
    for (int k = 0; k < D; ++k) {
      ring[k] = __builtin_amdgcn_raw_buffer_load_b128(rs, lane * 16, k * 512, 0);
      __builtin_amdgcn_sched_barrier(0);
    }
    for (int s0 = 0; s0 < steps; s0 += D) {
#pragma unroll
      for (int k = 0; k < D; ++k) {
        const B128u4 w = ring[k];
        __half2 P[4][4];
        if constexpr (WORK == 0) {
          accx ^= w[0] ^ w[1] ^ w[2] ^ w[3];
        } else if constexpr (WORK == 4) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) P[q][kk] = B128AsH2(w[(q + kk) & 3]);
        } else {
          const uint32_t prev = (uint32_t)__builtin_amdgcn_mov_dpp8((int)w[3], kDpp8Prev);
          DecodeLane<4, kMulMode>(prev, w[0], 0u, 20, P[0]);
          DecodeLane<4, kMulMode>(w[0], w[1], 0u, 20, P[1]);
          DecodeLane<4, kMulMode>(w[1], w[2], 0u, 20, P[2]);
          DecodeLane<4, kMulMode>(w[2], w[3], 0u, 20, P[3]);
        }
        ring[k] = __builtin_amdgcn_raw_buffer_load_b128(rs, lane * 16, (s0 + k + D) * 512, 0);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (WORK == 1) {
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) accx ^= *reinterpret_cast<uint32_t*>(&P[q][kk]);
        } else if constexpr (WORK >= 2) {
#pragma unroll
          for (int r = 0; r < R; ++r) {
            B128u4 xa, xb;
            if constexpr (WORK == 2) {
              xa = xreg[0], xb = xreg[1];
            } else {
              const B128u4* xr = reinterpret_cast<const B128u4*>(xl + (r * 160 + ((s0 + k) & 127)) * 8);
              xa = xr[0], xb = xr[1];
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const __half2 ha = B128AsH2(xa[q]), hb = B128AsH2(xb[q]);
              acc[q][r][0] = Dot2(P[q][0], ha, acc[q][r][0]);
              acc[q][r][0] = Dot2(P[q][1], hb, acc[q][r][0]);
              acc[q][r][1] = Dot2(P[q][2], ha, acc[q][r][1]);
              acc[q][r][1] = Dot2(P[q][3], hb, acc[q][r][1]);
            }
          }
        }
      }
    }
  }
  float fs = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int r = 0; r < (R > 0 ? R : 1); ++r) fs += acc[q][r][0] + acc[q][r][1];
  sink[blockIdx.x * 256 + threadIdx.x] = accx ^ __float_as_uint(fs);
}

unsigned* g_sg_ctl = nullptr;

// Variants 110 to 139. Returns false for a variant this file does not own. R = 1, 2, 4, 8 only (compile time).
template <int R>
bool LaunchB128R(int v, const Group& gu, const Group& dn, unsigned* done) {
  if (!g_flow.next) {
    HIP_OK(hipMalloc(&g_flow.next, 64));
    HIP_OK(hipMalloc(&g_flow.sink, 4 << 20));
    HIP_OK(hipMemset(g_flow.next, 0, 64));
  }
  const size_t lds = MoeFusedLds(R, 0, gu.lds, dn.lds);
  [[maybe_unused]] auto real = [&](auto kern, int W) {
    hipLaunchKernelGGL(kern, dim3(gu.nitems + dn.nitems), dim3(32 * W), lds, 0, gu.d_jobs, gu.d_items, gu.nitems,
                       dn.d_jobs, dn.d_items, dn.nitems, done);
  };
  const int jobs = MoeExpertJobs(R), shared = R * kTopK;
  auto stream = [&](auto kern, int grid) {
    hipLaunchKernelGGL(kern, dim3(grid), dim3(256), 0, 0, gu.d_jobs, dn.d_jobs, jobs, shared, g_flow.next,
                       g_flow.sink);
  };
  // Stream-grid real kernels (moe_sg.h): 130 to 139. G is capped by the resident block count.
  if (!g_sg_ctl) {
    HIP_OK(hipMalloc(&g_sg_ctl, kSgCtlWords * 4));
    HIP_OK(hipMemset(g_sg_ctl, 0, kSgCtlWords * 4));
  }
  auto sg = [&](auto kern, int G) {
    const int g = std::min(G, FlowResident(kern, 0));
    hipLaunchKernelGGL(kern, dim3(g), dim3(256), 0, 0, gu.d_jobs, dn.d_jobs, g_sg_ctl, R);
  };
  switch (v) {
    // W: the VGPR cap of MoeSgKernel (12 waves per SIMD at R <= 4; 8 at R = 8, where 12 spills).
    case 130: sg(MoeSgKernel<R, 5, 1, 4, 0, (R <= 4 ? 12 : 8)>, 120); return true;
    case 131: sg(MoeSgKernel<R, 10, 1, 4, 0, 8>, 120); return true;
    case 132: sg(MoeSgKernel<R, 5, 1, 0, 0, (R <= 4 ? 12 : 8)>, 120); return true;
    case 133: sg(MoeSgKernel<R, 2, 1, 4, 0, 12>, 120); return true;
    case 134: sg(MoeSgKernel<R, 5, 1, 4, 0, 1>, 120); return true;
    case 135: sg(MoeSgKernel<R, 5, 1, 4, 1, (R <= 4 ? 12 : 8)>, 120); return true;
    case 136: sg(MoeSgKernel<R, 5, 1, 4, 2, (R <= 4 ? 12 : 8)>, 120); return true;
    case 137: sg(MoeSgKernel<R, 5, 1, 4, 3, (R <= 4 ? 12 : 8)>, 120); return true;
    case 138: sg(MoeSgKernel<R, 5, 2, 4, 0, (R <= 4 ? 12 : 8)>, 120); return true;
    // Long units: CBG / CBD column blocks per gate/up / down unit in one continuous stream.
    case 140: sg(MoeSgKernel<R, 5, 1, 4, 0, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
    case 141: sg(MoeSgKernel<R, 5, 1, 4, 0, 1, 1, 4>, 120); return true;
    case 142: sg(MoeSgKernel<R, 2, 1, 4, 0, 12, 1, 4>, 120); return true;
    case 143: sg(MoeSgKernel<R, 5, 1, 4, 0, (R <= 4 ? 12 : 8), 2, 4>, 120); return true;
    case 144: sg(MoeSgKernel<R, 5, 1, 4, 0, (R <= 4 ? 12 : 8), 1, 2>, 120); return true;
    case 145: sg(MoeSgKernel<R, 2, 1, 4, 0, 12, 1, 8>, 120); return true;
    case 146: sg(MoeSgKernel<R, 5, 1, 4, 0, 1, 1, 8>, 120); return true;
    case 147: sg(MoeSgKernel<R, 5, 1, 4, 1, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
    // Fewer K4 bodies (rows 1, 2, 4, R): instruction cache test.
    case 160: sg(MoeSgKernel<R, 5, 1, 4, 0, (R <= 4 ? 12 : 8), 1, 1, 1>, 120); return true;
    case 161: sg(MoeSgKernel<R, 2, 1, 4, 0, 12, 1, 1, 1>, 120); return true;
    case 162: sg(MoeSgKernel<R, 5, 1, 4, 0, 1, 1, 1, 1>, 120); return true;
    case 163: sg(MoeSgKernel<R, 5, 1, 4, 3, (R <= 4 ? 12 : 8), 1, 1, 1>, 120); return true;
    // Knockouts of 140 (timing only).
    case 150: sg(MoeSgKernel<R, 5, 1, 4, 10, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
    case 151: sg(MoeSgKernel<R, 5, 1, 4, 11, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
    case 152: sg(MoeSgKernel<R, 5, 1, 4, 12, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
    case 153: sg(MoeSgKernel<R, 5, 1, 4, 13, (R <= 4 ? 12 : 8), 1, 4>, 120); return true;
#ifndef VM_SG_LEAN
    case 110: real(MoeQ4Kernel<R, 4, 5, 0>, 5); return true;
    case 111: real(MoeQ4Kernel<R, 8, 5, 0>, 5); return true;
    case 112: real(MoeQ4Kernel<R, 2, 5, 0>, 5); return true;
    case 113: real(MoeQ4Kernel<R, 4, 8, 0>, 8); return true;
    case 114: real(MoeQ4Kernel<R, 4, 10, 0>, 10); return true;
    case 115: real(MoeQ4Kernel<R, 4, 5, 0, 1>, 5); return true;
    case 116: real(MoeQ4Kernel<R, 4, 5, 0, 2>, 5); return true;
    case 117: real(MoeQ4Kernel<R, 4, 5, 0, 3>, 5); return true;
    case 118: real(MoeQ4Kernel<R, 4, 8, 1>, 8); return true;
    case 119: real(MoeQ4Kernel<R, 8, 8, 1>, 8); return true;
    case 121: stream(MoeStreamQ4Kernel<8, 1, 0>, 120); return true;
    case 122: stream(MoeStreamQ4Kernel<8, 2, R>, 120); return true;
#endif
    case 120: stream(MoeStreamQ4Kernel<8, 0, 0>, 120); return true;
    case 123: stream(MoeStreamQ4Kernel<8, 3, R>, 120); return true;
#ifndef VM_SG_LEAN
    case 124: stream(MoeStreamQ4Kernel<8, 4, R>, 120); return true;
    case 125: stream(MoeStreamQ4Kernel<4, 3, R>, 120); return true;
    case 126: stream(MoeStreamQ4Kernel<8, 3, R>, 40); return true;
#endif
    default: return false;
  }
}

inline bool LaunchB128(int v, int R, const Group& gu, const Group& dn, unsigned* done) {
  if (v < 110 || v >= 170) return false;
  switch (R) {
    case 1: return LaunchB128R<1>(v, gu, dn, done);
    case 2: return LaunchB128R<2>(v, gu, dn, done);
    case 4: return LaunchB128R<4>(v, gu, dn, done);
    case 8: return LaunchB128R<8>(v, gu, dn, done);
    default:
      std::fprintf(stderr, "variants 110+ are built for R = 1, 2, 4, 8 only\n");
      std::exit(2);
  }
}
inline bool B128TimingOnly(int v) { return (v >= 115 && v <= 117) || (v >= 120 && v < 130) || (v >= 150 && v < 160); }  // 130-149, 160+: exact (or mutants)

}  // namespace crucible
