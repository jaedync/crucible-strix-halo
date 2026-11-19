// Shared device helpers for the Crucible engine (gfx1151, wave32).
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../model/flashnext/geometry.h"

#define HIP_OK(x)                                                                        \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, \
                   __LINE__, #x);                                                        \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace crucible {

__device__ __forceinline__ float Bf16(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ float RoundH(float v) { return __half2float(__float2half(v)); }
__device__ __forceinline__ float Sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }
__device__ __forceinline__ float Silu(float v) { return v / (1.f + __expf(-v)); }

typedef unsigned int U32x4 __attribute__((ext_vector_type(4)));

// 16-byte streaming load that bypasses cache retention (weights are read once).
__device__ __forceinline__ uint4 LoadNT(const void* p) {
  const U32x4 v = __builtin_nontemporal_load(reinterpret_cast<const U32x4*>(p));
  return make_uint4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) v += __shfl_xor(v, m);
  return v;
}

// Sum over a whole block of NT threads; `red` holds NT/32 floats. Every thread
// gets the same fixed-order result.
template <int NT>
__device__ __forceinline__ float BlockSum(float v, float* red) {
  v = WaveSum(v);
  const int w = threadIdx.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) t += red[i];
  return t;
}

// In-wave natural-order Sylvester Hadamard of 128 values, lane l holding
// elements 4l..4l+3. Strides ascend: 1, 2 in-lane, then 4..64 across lanes.
// Unnormalized; callers scale by kHadNorm.
__device__ __forceinline__ void Wht128(float v[4]) {
  const int lane = threadIdx.x & 31;
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float o = __shfl_xor(v[i], m);
      v[i] = (lane & m) ? (o - v[i]) : (v[i] + o);
    }
  }
}

// Value of lane ^ M. DPP row_xmask moves data within 16-lane rows and
// permlanex16 swaps the rows, so no ds_bpermute goes through the LDS
// crossbar. All 32 lanes must be active.
template <int M>
__device__ __forceinline__ float XorLane(float v) {
  const int x = __float_as_int(v);
  if constexpr (M == 16)
    return __int_as_float(__builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false));
  else
    return __int_as_float(__builtin_amdgcn_update_dpp(0, x, 0x160 + M, 0xf, 0xf, false));
}
// (lane & M) ? o - v : v + o as one fma with s = -1 or 1: s * v is exact, so
// the single rounding gives the same bits (including the sign of zero).
template <int M>
__device__ __forceinline__ void WhtStage(float v[4], int lane) {
  const float s = (lane & M) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __builtin_fmaf(s, v[i], XorLane<M>(v[i]));
}
// Wht128 with the same arithmetic in the same order (bitwise equal results),
// cross-lane moves by DPP. For full waves only. lane: threadIdx.x & 31 (Wht128DppLane takes it
// from the caller, so that a caller can make the sign terms opaque).
__device__ __forceinline__ void Wht128DppLane(float v[4], int lane) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
  WhtStage<1>(v, lane);
  WhtStage<2>(v, lane);
  WhtStage<4>(v, lane);
  WhtStage<8>(v, lane);
  WhtStage<16>(v, lane);
}
__device__ __forceinline__ void Wht128Dpp(float v[4]) { Wht128DppLane(v, threadIdx.x & 31); }
// Wht128Dpp with the lane as an argument and the row moves as mov_dpp with no
// old value (all 32 lanes are active and every row_xmask source lane exists):
// the same values, 2 VALU per value and stage instead of 3 (the X producer
// TileWht128, bytes checked). A caller that passes a lane from an opaque
// threadIdx (PcTid) keeps the lane terms out of its long-lived VGPRs.
template <int M>
__device__ __forceinline__ float XorLaneMov(float v) {
  const int x = __float_as_int(v);
  if constexpr (M == 16)
    return __int_as_float(__builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false));
  else
    return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 + M, 0xf, 0xf, true));
}
template <int M>
__device__ __forceinline__ void WhtStageMov(float v[4], int lane) {
  const float s = (lane & M) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __builtin_fmaf(s, v[i], XorLaneMov<M>(v[i]));
}
__device__ __forceinline__ void Wht128Lane(float v[4], int lane) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
  WhtStageMov<1>(v, lane);
  WhtStageMov<2>(v, lane);
  WhtStageMov<4>(v, lane);
  WhtStageMov<8>(v, lane);
  WhtStageMov<16>(v, lane);
}

}  // namespace crucible
