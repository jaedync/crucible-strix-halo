// PleRowState (the state of PleGatherKernel) against the exl3_ngram_trellis row codec
// (tools/ngram16/codec.py _state_sources): state bit m of position i is stream bit
// ((i - m / K) mod 160) * K + m % K, little-endian words, little bit order. K5, K6 and K8 rows.
#include "../engine/layers/ple_row_codec.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

template <int K>
static uint32_t RefState(const std::vector<uint16_t>& row, int i) {
  uint32_t state = 0;
  for (int m = 0; m < 16; ++m) {
    const int b = ((i - m / K) % 160 + 160) % 160 * K + m % K;
    state |= uint32_t((row[1 + b / 16] >> (b % 16)) & 1u) << m;
  }
  return state;
}

template <int K>
static long Check(std::mt19937& rng, int rows) {
  long checked = 0;
  for (int n = 0; n < rows; ++n) {
    std::vector<uint16_t> row(1 + 10 * K);
    for (uint16_t& w : row) w = uint16_t(rng());
    if (n == 0) for (uint16_t& w : row) w = 0xffff;
    for (int i = 0; i < 160; ++i, ++checked) {
      const uint32_t got = crucible::PleRowState<K>(row.data(), i), want = RefState<K>(row, i);
      if (got != want) {
        std::fprintf(stderr, "K%d row %d position %d: got %04x want %04x\n", K, n, i, got, want);
        return -1;
      }
    }
  }
  return checked;
}

int main() {
  std::mt19937 rng(461);
  long total = 0;
  for (const long n : {Check<5>(rng, 200), Check<6>(rng, 200), Check<8>(rng, 200)}) {
    if (n < 0) return 1;
    total += n;
  }
  std::printf("ple_row_codec_test: %ld states of K5, K6 and K8 rows match the codec\n", total);
}
