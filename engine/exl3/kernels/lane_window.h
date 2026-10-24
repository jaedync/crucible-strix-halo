// The lane geometry and trellis state extraction of DecodeLane (exl3_gemv.h), host-visible so the
// CPU tests check them against the tile codec (tests/lane_window_test.cpp).
#pragma once
#include <cstdint>

#ifdef __HIPCC__
#define EXL3_LANE_HD __host__ __device__ __attribute__((always_inline))
#else
#define EXL3_LANE_HD __attribute__((always_inline))
#endif

namespace crucible {

// v_alignbit_b32: the low 32 bits of (hi:lo) >> (s mod 32).
EXL3_LANE_HD inline uint32_t AlignBit(uint32_t hi, uint32_t lo, uint32_t s) {
#ifdef __HIP_DEVICE_COMPILE__
  return __builtin_amdgcn_alignbit(hi, lo, s);
#else
  return uint32_t(((uint64_t(hi) << 32) | lo) >> (s & 31));
#endif
}

// EXL3 width codes: 1..8 are integer K; 10 KA + 5 (15, 25, ..., 65) is K = KA + 0.5, the
// half-integer tiles of exllamav3 1.6 (pack_trellis_frac, MASK 0xAAAA): ring position i takes
// KA + (i odd) new bits, so a 256-weight tile has 16 (2 KA + 1) uint16 = 4 (2 KA + 1) dwords.
constexpr bool IsHalfK(int K) { return K > 8 && K % 10 == 5 && K / 10 >= 1 && K / 10 <= 6; }
constexpr int TileDwords(int K) { return IsHalfK(K) ? 4 * (2 * (K / 10) + 1) : 8 * K; }

// Lane-invariant geometry of the lane's 96-bit window inside a tile ring.
template <int K>
struct LaneWindow {
  // Three aligned dwords hold the state span (7K + 16 bits after `shift`): K=6 needs 58 bits,
  // K=7 65 (shift is at most 31) and K=8 72 (shift is always 24).
  static_assert((K > 0 && K <= 8) || IsHalfK(K));
  int d0;     // first dword (ring index) of the 3-dword window
  int shift;  // 1..32: the lane's first state bit is bit `shift` of w0:w1
  EXL3_LANE_HD explicit LaneWindow(int t) {
    if constexpr (IsHalfK(K)) {
      // Half K: positions 0..8t-1 take 4t (2 KA + 1) bits, position 8t (even) KA more.
      constexpr int KA = K / 10, T = TileDwords(K);
      int s = 4 * t * (2 * KA + 1) + KA - 16;
      int dw = (s - 1 >= 0) ? (s - 1) / 32 : -((32 - s) / 32);
      d0 = ((dw % T) + T) % T;
      shift = s - dw * 32;
    } else {
      int s = (8 * t + 1) * K - 16;  // first bit of position 8t's state
      // floor((s - 1) / 32) keeps shift in 1..32, so alignbit's 5-bit shift
      // amount 32 - shift never wraps and no lane needs a select.
      int dw = (s - 1 >= 0) ? (s - 1) / 32 : -((32 - s) / 32);
      d0 = ((dw % (8 * K)) + 8 * K) % (8 * K);
      shift = s - dw * 32;
    }
  }
};

// The lane's 8 trellis states (low 16 bits of st[j]; higher bits are not cleared).
template <int K>
EXL3_LANE_HD inline void LaneStates(uint32_t w0, uint32_t w1, uint32_t w2, int shift, uint32_t st[8]) {
  // Aligned big-endian stream: bit 0 of the lane's range is a0's MSB.
  uint32_t a0 = AlignBit(w0, w1, 32 - shift);
  uint32_t a1 = AlignBit(w1, w2, 32 - shift);
  if constexpr (IsHalfK(K)) {
    // State j = bits [j KA + (j + 1) / 2, + 16): the lane's 8 states span 7 KA + 20 <= 64 bits (KA <= 6).
    constexpr int KA = K / 10;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int end = j * KA + (j + 1) / 2 + 16;
      st[j] = end <= 32 ? a0 >> (32 - end) : AlignBit(a0, a1, 64 - end);
    }
    return;
  }
  // K7 and K8 only: the last state ends past bit 64 of the lane's range. For K <= 6 this word and
  // its branch do not exist, so their code is unchanged.
  uint32_t a2 = 0;
  if constexpr (7 * K + 16 > 64) a2 = AlignBit(w2, 0u, 32 - shift);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    constexpr int kW = 32;
    int end = j * K + 16;  // state j = bits [jK, jK + 16)
    if (end <= kW) {
      st[j] = a0 >> (kW - end);
    } else if (7 * K + 16 <= 64 || end <= 2 * kW) {
      st[j] = AlignBit(a0, a1, 64 - end);
    } else {
      st[j] = AlignBit(a1, a2, 96 - end);
    }
  }
}

}  // namespace crucible
