// EXL3 codebook tags (compile time) and the .mul1 marker that selects one. Host and device; no device intrinsics, so
// the portable kernels (kernels/hy4_kvb_exl3.h, HIP-CPU with g++) include it too.
//
//   kCbMul1       standard EXL3 mul1: state * 0x83DCD12D, tensor-core tile order (docs/EXL3_FORMAT.md 2 and 3). Every
//                 kernel instance that existed before the tags is this one.
//   kCbMul24      EXL3-INT8 mul24 (github.com/jyohukuchan/rocm_exl3_forme doc/exl3_int8.md, "mul24 形式"):
//                 state * 0x00DCD12D, the fp16 value fma(1024 + B, 0x1eee, 0xc931) of
//                 the byte sum B (the mul1 formula, the quantizer's reconstruction), WMMA B fragment tile order:
//                 stream position t = 8 L + j sits at row (input) 8 (L / 16) + j, column (output) L % 16.
//   kCbMul24Int8  the same states and order; the value is the INT8 grid q = (B - 509) >> 2 (-128 .. 127) times
//                 kMul24Step = 4 * 887 * 2^-17. The decode gives q as an exact fp16 integer; the matvec multiplies its
//                 fp32 sum by kMul24Step (the spec's fp32 epilogue scale).
// The marker cannot tell kCbMul24 from kCbMul24Int8 (one pack, two value forms): the caller picks the form.
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace crucible {

enum Exl3Codebook : int { kCbMul1 = 0, kCbMul24 = 1, kCbMul24Int8 = 2 };

constexpr uint32_t kMul1 = 0x83DCD12Du;
constexpr uint32_t kMul24 = 0x00DCD12Du;  // the low 24 bits of the mul1 constant: one v_mul_u32_u24
constexpr float kMul24Step = 0.027069091796875f;  // S_FIX = 4 * 887 * 2^-17, exact in fp32

constexpr bool IsMul24(int cb) { return cb == kCbMul24 || cb == kCbMul24Int8; }
// The tag of a tensor that no loader set: the default of the Hy4 EXL3 structs (Exl3Lin, Exl3Kvb, Hy4Exl3Mat). Every
// check and launch refuses it, so a loader that forgets the tag fails closed and never decodes a mul24 tensor as mul1.
constexpr int kCbUnset = -1;
constexpr bool IsKnownCodebook(int cb) { return cb == kCbMul1 || IsMul24(cb); }

// Row (input) and column (output) in the 16x16 tile of stream position t = 8 L + e (lane L, e = 0 .. 7; DecodeLane of
// kernels/exl3_gemv.h puts it in P[e / 2], the low half for even e). kCbMul1 (tensor-core order): row 2 (L % 4) +
// (e & 1) + 8 ((e >> 1) & 1), col L / 4 + 8 (e >> 2). mul24 (WMMA B fragment order): row 8 (L / 16) + e, col L % 16.
// constexpr: host and device code (clang HIP), HIP-CPU and plain C++.
constexpr void TilePos(int cb, int t, int* row, int* col) {
  const int L = t / 8, e = t % 8;
  if (IsMul24(cb)) {
    *row = 8 * (L / 16) + e;
    *col = L % 16;
  } else {
    *row = 2 * (L % 4) + (e & 1) + 8 * ((e >> 1) & 1);
    *col = L / 4 + 8 * (e >> 2);
  }
}

// The codebook of a tensor from its stored .mul1 value (the unsigned bits of the I32). 0x83DCD12D: kCbMul1;
// 0x00DCD12D: kCbMul24, or kCbMul24Int8 when int8_grid. Any other value throws: no fallback to another codebook.
inline int Exl3CodebookOfMarker(uint32_t marker, bool int8_grid = false, const std::string& name = "") {
  if (marker == kMul1) return kCbMul1;
  if (marker == kMul24) return int8_grid ? kCbMul24Int8 : kCbMul24;
  char hex[16];
  std::snprintf(hex, sizeof hex, "0x%08X", marker);
  throw std::runtime_error((name.empty() ? std::string("EXL3 tensor") : name) + ": .mul1 value " + hex +
                           " is no supported EXL3 codebook (mul1 0x83DCD12D, mul24 0x00DCD12D)");
}

}  // namespace crucible
