// The n-gram row codec of PleGatherKernel (ops.h), host-visible so the CPU tests check it against
// the exl3_ngram_trellis codec (tests/ple_row_codec_test.cpp).
// The row codec packs symbols LSB-first, unlike the linear tile codec.
// A state contains the current symbol, then older symbols, truncated to 16 bits.
#pragma once
#include <cstdint>

#ifdef __HIPCC__
#define PLE_ROW_HD __host__ __device__ __attribute__((always_inline))
#else
#define PLE_ROW_HD __attribute__((always_inline))
#endif

namespace crucible {

// row: word 0 is the fp16 scale, then 10K words of 160 K-bit symbols.
template <int K>
PLE_ROW_HD inline uint32_t PleRowSymbol(const uint16_t* row, int jj) {
  constexpr int Words = 10 * K;
  const int j = (jj + 160) % 160;
  const int bit = j * K, w = bit >> 4, sh = bit & 15;
  uint32_t v = row[1 + w];
  if (w + 1 < Words) v |= (uint32_t)row[2 + w] << 16;
  return (v >> sh) & ((1u << K) - 1);
}

template <int K>
PLE_ROW_HD inline uint32_t PleRowState(const uint16_t* row, int i) {
  static_assert(K == 5 || K == 6 || K == 8);
  auto sym = [&](int jj) -> uint32_t { return PleRowSymbol<K>(row, jj); };
  uint32_t state;
  if constexpr (K == 5)
    state = sym(i) | (sym(i - 1) << 5) | (sym(i - 2) << 10) | ((sym(i - 3) & 1u) << 15);
  else if constexpr (K == 6)
    state = sym(i) | (sym(i - 1) << 6) | ((sym(i - 2) & 15u) << 12);
  else  // K8: the 8-bit table of the h6_ng8 packs
    state = sym(i) | (sym(i - 1) << 8);
  return state;
}

}  // namespace crucible
