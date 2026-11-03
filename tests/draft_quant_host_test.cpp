// CPU checks of the low-bit draft head host code (engine/runtime/draft_quant_host.h): the CRUCIBLE_DRAFT_HEAD values,
// the bit layout, the quantizer error, the reference dot product and the rescoring block order.
#include "../engine/runtime/draft_quant_host.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace crucible;

static void Switch() {
  assert(ResolveDraftHeadBits(nullptr) == 0);
  assert(ResolveDraftHeadBits("") == 0);
  assert(ResolveDraftHeadBits("exl3") == 0);
  assert(ResolveDraftHeadBits("q4") == 4);
  assert(ResolveDraftHeadBits("q2") == 2);
  for (const char* bad : {"q3", "q8", "4", "Q4", "q4 ", " q4", "exl", "exl3x", "on", "1", "0", "q"})
    assert(ResolveDraftHeadBits(bad) == -1);
}

// Strided layout: every element has its own `bits` bits, every word is full, the pair (e, e + 1) shares a word at
// shifts s and s + 16, and the pairs of load c (words 4c .. 4c + 3) are pairs q * L + c, so that the lanes of a wave
// (consecutive c) read consecutive half2 words of x. One scale per load.
static void Layout(int bits) {
  constexpr int I = 2560;
  const int L = DraftQLoads(I, bits), words = DraftQWords(I, bits);
  assert(L * 4 == words && DraftQScales(I, bits) == L && L * (128 / bits) == I);
  std::vector<uint32_t> used(words, 0);
  for (int e = 0; e < I; ++e) {
    const int w = DraftQWord(e, I, bits), sh = DraftQShift(e, I, bits);
    assert(w >= 0 && w < words && sh >= 0 && sh + bits <= 32);
    const uint32_t mask = ((1u << bits) - 1) << sh;
    assert((used[w] & mask) == 0);
    used[w] |= mask;
    const int c = (e / 2) % L;
    assert(w / 4 == c && DraftQGroup(e, I, bits) == c);
    if (e % 2 == 0) assert(DraftQWord(e + 1, I, bits) == w && DraftQShift(e + 1, I, bits) == sh + 16);
    // Pair q of the load: word 4c + q / (16 / bits), pair slot q % (16 / bits).
    const int q = (e / 2) / L;
    assert(w == 4 * c + q / (16 / bits) && sh % 16 == (q % (16 / bits)) * bits);
  }
  for (uint32_t u : used) assert(u == 0xffffffffu);
}

// Gaussian rows (the EXL3 input transform makes the head rows close to Gaussian): relative RMS error near the
// uniform-quantizer bound (4 bits: 0.098, 2 bits: 0.34), and the reference dot equals the dequantized dot.
static void Quantizer(int bits, float bound) {
  constexpr int I = 2560, rows = 64;
  std::mt19937 rng(7 + bits);
  std::normal_distribution<float> normal(0.f, 0.02f);
  std::vector<float> w(I);
  std::vector<uint32_t> words(DraftQWords(I, bits));
  std::vector<_Float16> scales(DraftQScales(I, bits)), x(I);
  double err = 0, norm = 0, dot_err = 0, dot_norm = 0;
  for (int r = 0; r < rows; ++r) {
    for (auto& v : w) v = normal(rng);
    // One zero group and one outlier: the zero group stores scale 0, the outlier clips nothing else away.
    if (r == 0)
      for (int e = 0; e < I; ++e)
        if (DraftQGroup(e, I, bits) == 3) w[e] = 0.f;
    if (r == 1) w[100] = 0.5f;
    for (auto& v : x) v = _Float16(normal(rng) * 50.f);
    DraftQuantRow(w.data(), I, bits, words.data(), scales.data());
    if (r == 0) {
      assert(float(scales[3]) == 0.f);
      for (int e = 0; e < I; ++e)
        if (DraftQGroup(e, I, bits) == 3) assert(DraftQValue(words.data(), scales.data(), e, I, bits) == 0.f);
    }
    double exact = 0, deq = 0;
    for (int e = 0; e < I; ++e) {
      const float v = DraftQValue(words.data(), scales.data(), e, I, bits);
      if (r != 1) err += double(v - w[e]) * (v - w[e]), norm += double(w[e]) * w[e];
      exact += double(w[e]) * float(x[e]);
      deq += double(v) * float(x[e]);
    }
    const float ref = DraftQDotRef(words.data(), scales.data(), x.data(), I, bits);
    assert(std::fabs(ref - deq) <= 1e-3 * (std::fabs(deq) + 1.0));
    dot_err += (deq - exact) * (deq - exact);
    dot_norm += exact * exact;
  }
  const double rel = std::sqrt(err / norm);
  std::printf("draft q%d: weight relative RMS error %.4f (bound %.2f), dot relative RMS error %.4f\n", bits, rel,
              bound, std::sqrt(dot_err / dot_norm));
  assert(rel < bound);
}

// Ranks follow a stable sort by maximum (NaN last, ties by index); every rank below the cap is unique.
static void Order() {
  std::mt19937 rng(3);
  for (int trial = 0; trial < 200; ++trial) {
    const int n = 1 + int(rng() % 600);
    std::vector<float> m(n);
    for (auto& v : m) v = float(int(rng() % 40)) / 4.f;  // many ties
    if (trial % 5 == 0) m[rng() % n] = NAN;
    if (trial % 7 == 0) m[rng() % n] = -INFINITY;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return DraftQBefore(m[a], a, m[b], b); });
    for (int pos = 0; pos < n; ++pos) {
      assert(DraftQRank(m.data(), n, order[pos], n) == pos);
      assert(DraftQRank(m.data(), n, order[pos], kDraftQBlocks) == std::min(pos, kDraftQBlocks));
    }
  }
  assert(DraftQBefore(1.f, 5, NAN, 0) && !DraftQBefore(NAN, 0, 1.f, 5));
  assert(DraftQBefore(NAN, 0, NAN, 1) && DraftQBefore(-INFINITY, 0, NAN, 1));
}

int main() {
  Switch();
  Layout(4);
  Layout(2);
  Quantizer(4, 0.11f);
  Quantizer(2, 0.37f);
  Order();
  std::puts("draft_quant_host_test: ok");
  return 0;
}
