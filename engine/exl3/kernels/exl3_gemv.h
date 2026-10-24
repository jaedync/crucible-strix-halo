// EXL3 (mul1 codebook) matrix-vector device code for gfx1151, 1-8 activation rows.
// The kernels that only benches launch are in bench/kernels/exl3_gemv.h.
//
// Decode path, per 16x16 tile and lane t (EXL3's native fragment order):
// lane t owns positions 8t..8t+7, i.e. rows {r0, r0+1, r0+8, r0+9} of columns
// {c0, c0+8} with r0 = 2(t%4), c0 = t/4. Its eight 16-bit trellis states are
// one contiguous 7K+16 bit window, so three dword loads and two funnel shifts
// cover every state (K <= 8, lane_window.h). Products use v_dot2_f32_f16 into fp32, one accumulator
// chain per (row, column) in a fixed tile order: a row's result never depends
// on how many rows share the pass (exact batching / drafting).
//
// Weights are repacked column-block-major ("crucible order"): tile (a, b) of a
// [I/16, O/16] matrix lives at (b * A + a) * 8K dwords, so one wave streaming
// column block b over a K range reads contiguous memory.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "lane_window.h"

namespace crucible {

constexpr uint32_t kMul1 = 0x83DCD12Du;

enum MulMode { kMulLo = 0, kMulTrue16 = 1, kMulU24 = 2 };

// p = state * 0x83DCD12D mod 2^32 for a 16-bit state held in the low half.
template <int MODE>
__device__ __forceinline__ uint32_t Mul1Product(uint32_t x) {
  if constexpr (MODE == kMulLo) {
    return (x & 0xffffu) * kMul1;  // v_mul_lo_u32: quarter rate
  } else if constexpr (MODE == kMulU24) {
    uint32_t lo = __umul24(x & 0xffffu, 0xD12Du);
    uint32_t hi = __umul24(x & 0xffffu, 0x83DCu);
    return lo + (hi << 16);
  } else {
    // Packed 16-bit multiply with op_sel_hi picking x for the high lane:
    // hi = ((x * 0x83DC) mod 2^16) << 16 and the low lane x * 0 = 0. One 16x16
    // mad then adds x * 0xD12D. Two full-rate ops instead of a quarter-rate one.
    uint32_t hi, p;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(hi) : "v"(x), "s"(0x83DC0000u));
    asm("v_mad_u32_u16 %0, %1.l, 0xd12d, %2" : "=v"(p) : "v"(x), "v"(hi));
    return p;
  }
}

// fp16 code value: fma(1024 + bytesum(p), 0x1eee, 0xc931), one rounding.
__device__ __forceinline__ __half2 Mul1Pair(uint32_t p0, uint32_t p1) {
#ifndef EXL3_MUL1_SAD_HI
  uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0u, 0x6400u);
  uint32_t s1 = __builtin_amdgcn_sad_u8(p1, 0u, 0x6400u);
  uint32_t packed = __builtin_amdgcn_perm(s1, s0, 0x05040100u);
#else
  // Each byte sum is at most 1020, so neither half overflows or carries.
  // SAD_HI builds both half bit patterns without a separate permutation.
  uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
  uint32_t packed = __builtin_amdgcn_sad_hi_u8(p1, 0u, s0);
#endif
  __half2 h = *reinterpret_cast<__half2*>(&packed);
  const __half2 inv = __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee));
  const __half2 bias = __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931));
  return __hfma2(h, inv, bias);
}

// Decodes the lane's 8 weights of one tile into 4 half2:
// P0 = rows (r0, r0+1) col c0, P1 = rows (r0+8, r0+9) col c0,
// P2 = rows (r0, r0+1) col c0+8, P3 = rows (r0+8, r0+9) col c0+8.
template <int K, int MODE>
__device__ __forceinline__ void DecodeLane(uint32_t w0, uint32_t w1, uint32_t w2,
                                           int shift, __half2 P[4]) {
  uint32_t st[8];
  if constexpr (7 * K + 16 <= 64) {
    // K <= 6: the train 23 code, kept verbatim so that its kernels stay ISA-equal (the LaneStates
    // form computes the same states, but the compiler schedules it differently).
    // Aligned big-endian stream: bit 0 of the lane's range is a0's MSB.
    uint32_t a0 = __builtin_amdgcn_alignbit(w0, w1, 32 - shift);
    uint32_t a1 = __builtin_amdgcn_alignbit(w1, w2, 32 - shift);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      constexpr int kW = 32;
      int end = j * K + 16;  // state j = bits [jK, jK + 16)
      if (end <= kW) {
        st[j] = a0 >> (kW - end);
      } else {
        st[j] = __builtin_amdgcn_alignbit(a0, a1, 64 - end);
      }
    }
  } else {
    LaneStates<K>(w0, w1, w2, shift, st);  // K7, K8 and half widths (code 35 = K3.5): lane_window.h
  }
  uint32_t p[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) p[j] = Mul1Product<MODE>(st[j]);
  P[0] = Mul1Pair(p[0], p[1]);
  P[1] = Mul1Pair(p[2], p[3]);
  P[2] = Mul1Pair(p[4], p[5]);
  P[3] = Mul1Pair(p[6], p[7]);
}

__device__ __forceinline__ float Dot2(__half2 a, __half2 b, float c) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  return __builtin_amdgcn_fdot2(*reinterpret_cast<h2*>(&a), *reinterpret_cast<h2*>(&b), c,
                                false);
}

// gfx11 raw buffer descriptor word 3 (32-bit data, raw OOB checking).
constexpr int kBufferWord3 = 0x31004000;

// A wave's tile stream as a buffer resource: loads are
// buffer_load_b32 v, v_lane_off, s[rsrc], s_tile_off, i.e. no per-lane address
// math, and reads past the stream return 0 without touching memory, so the
// prefetch ring needs no clamping.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t StreamRsrc(const uint32_t* base,
                                                             uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<uint32_t*>(base), (short)0, (int)bytes,
                                           kBufferWord3);
}

}  // namespace crucible
