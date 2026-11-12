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
// cross-lane moves by DPP. For full waves only.
__device__ __forceinline__ void Wht128Dpp(float v[4]) {
  const int lane = threadIdx.x & 31;
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
// Wht128Lane of two vectors a and b up to a sign per lane, with the DPP move
// inside the fma. A stage keeps v + s o (s = -1 where lane & M) as one
// v_fmac_f32_dpp with o as the DPP operand (src0): one VALU per value and DPP
// stage (Wht128Lane: a mov_dpp and an fma; ROCm 7.2.4 does not fold an fma
// into the DPP move, only an add or a sub). Where lane & M that is -(o - v). A
// lane and its stage partner share the bits below M, so the same s is right in
// every stage, and lane l ends with Wht128Lane's values times
// (-1)^popcount(l). Each sum is the same exact value up to that sign with one
// rounding, and RNE is symmetric: a nonzero value is Wht128Lane's bits times
// the sign, a zero can have either sign. Returns the sign; the caller
// multiplies it into its next scale (exact where no zero's sign reaches the
// output, see the callers).
// One asm statement holds a stage of all 8 values, so a value's next stage is
// 8 VALU after it: the compiler has no latency for the asm and puts no
// s_delay_alu around it. gfx11 has no VALU write to DPP read hazard.
#define GFX_WHT_FOLD_STAGE(XM)                                                        \
  asm("v_fmac_f32_dpp %0, %0, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %4, %4, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %1, %1, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %5, %5, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %2, %2, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %6, %6, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %3, %3, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t" \
      "v_fmac_f32_dpp %7, %7, %8 row_xmask:" #XM " row_mask:0xf bank_mask:0xf bound_ctrl:1"   \
      : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3]), "+v"(b[0]), "+v"(b[1]), "+v"(b[2]), "+v"(b[3]) \
      : "v"((lane & XM) ? -1.f : 1.f))
__device__ __forceinline__ void WhtInLane(float v[4]) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
}
__device__ __forceinline__ float Wht128Fold2(float a[4], float b[4], int lane) {
  WhtInLane(a);
  WhtInLane(b);
  GFX_WHT_FOLD_STAGE(1);
  GFX_WHT_FOLD_STAGE(2);
  GFX_WHT_FOLD_STAGE(4);
  GFX_WHT_FOLD_STAGE(8);
  const float s = (lane & 16) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[i] = __builtin_fmaf(XorLaneMov<16>(a[i]), s, a[i]);
    b[i] = __builtin_fmaf(XorLaneMov<16>(b[i]), s, b[i]);
  }
  return (__builtin_popcount(lane & 31) & 1) ? -1.f : 1.f;
}
#undef GFX_WHT_FOLD_STAGE

}  // namespace crucible
