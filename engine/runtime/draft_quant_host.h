// Host side of the low-bit draft head (CRUCIBLE_DRAFT_HEAD): the switch, the row
// format and its quantizer, the reference dot product and the block order of the rescoring. No HIP code, so the
// CPU test (tests/draft_quant_host_test.cpp) checks the same functions that the loader and the kernels use.
//
// The draft head is one third of each MTP draft call (0.55 ms of 1.36 ms on 4.05): a GEMV over the 64K retained
// tokens, bound by its 126 MB of K6 EXL3 tiles. The low-bit head reads a copy of the same rows at 4 or 2 bits for
// coarse logits, then recomputes the 8 best 128-token blocks from the EXL3 tiles (the rescoring). The choice comes
// from the recomputed logits only. Drafts only: the verify pass checks every proposed token with the full head.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef __HIPCC__
#define DRAFTQ_HD __host__ __device__
#else
#define DRAFTQ_HD
#endif

namespace crucible {

// Blocks of 128 tokens that the rescoring recomputes from the EXL3 tiles, per scored row.
inline constexpr int kDraftQBlocks = 8;
// K splits of the rescoring GEMV: 160 input tiles of the 2560-wide head, 16 per split.
inline constexpr int kDraftQSplits = 10;

// CRUCIBLE_DRAFT_HEAD: unset, "" or "exl3" keep the EXL3 block head (0); "q4" and "q2" give the coarse width in
// bits. Any other text gives -1 (the caller stops).
inline int ResolveDraftHeadBits(const char* v) {
  if (!v || !*v || std::strcmp(v, "exl3") == 0) return 0;
  if (std::strcmp(v, "q4") == 0) return 4;
  if (std::strcmp(v, "q2") == 0) return 2;
  return -1;
}

// Row format. A row of I inputs is L = I * bits / 128 uint4 loads (4 words each); load c holds the input pairs
// q * L + c, q = 0 .. 64 / bits - 1 (pair h = inputs 2h, 2h + 1). So the 32 lanes of a wave, which read loads
// c .. c + 31, read 32 consecutive half2 words of x from LDS (no bank conflict). One fp16 scale per load: the
// elements of a load spread over the whole row, which the EXL3 input transform makes close to Gaussian.
constexpr int DraftQLoads(int I, int bits) { return I * bits / 128; }
constexpr int DraftQWords(int I, int bits) { return 4 * DraftQLoads(I, bits); }
constexpr int DraftQScales(int I, int bits) { return DraftQLoads(I, bits); }
// The level offset: a code q in [0, 2^bits) stands for (q - c) * scale, c = (2^bits - 1) / 2.
constexpr float DraftQCenter(int bits) { return ((1 << bits) - 1) * 0.5f; }

// Element e: pair h = e / 2 = q * L + c. Word 4c + q / (16 / bits); in the word, pair slot q % (16 / bits) holds
// the even element at bits [s, s + bits) and the odd one at [s + 16, s + 16 + bits), s = slot * bits: one shift and
// one mask give the half2 pair, and 0x6400 | q is the fp16 value 1024 + q (the device unpack in draft_quant.h).
inline int DraftQGroup(int e, int I, int bits) { return (e / 2) % DraftQLoads(I, bits); }
inline int DraftQWord(int e, int I, int bits) {
  const int L = DraftQLoads(I, bits), h = e / 2;
  return 4 * (h % L) + (h / L) / (16 / bits);
}
inline int DraftQShift(int e, int I, int bits) { return (e & 1) * 16 + ((e / 2) / DraftQLoads(I, bits)) % (16 / bits) * bits; }

// Quantizes one row: per load, the scale with the least squared error among 16 clip ratios of the absolute maximum.
// words: DraftQWords(I, bits) (overwritten); scales: DraftQScales(I, bits) fp16 values.
inline void DraftQuantRow(const float* w, int I, int bits, uint32_t* words, _Float16* scales) {
  const int levels = 1 << bits, L = DraftQLoads(I, bits), n = 128 / bits;
  const float c = DraftQCenter(bits);
  std::fill(words, words + DraftQWords(I, bits), 0u);
  float x[64];
  int at[64];
  for (int g = 0; g < L; ++g) {
    for (int i = 0; i < n; ++i) at[i] = 2 * ((i / 2) * L + g) + (i & 1), x[i] = w[at[i]];
    float amax = 0.f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    float best_s = 0.f, best_err = std::numeric_limits<float>::infinity();
    if (amax > 0.f && std::isfinite(amax)) {
      for (int k = 0; k < 16; ++k) {
        // The scale is rounded to fp16 first, so the error is the one of the stored value.
        const float s = float(_Float16(amax / c * (1.f - 0.03f * k)));
        if (!(s > 0.f)) continue;
        float err = 0.f;
        for (int i = 0; i < n; ++i) {
          const float q = std::clamp(std::nearbyint(x[i] / s + c), 0.f, float(levels - 1));
          const float d = x[i] - (q - c) * s;
          err += d * d;
        }
        if (err < best_err) best_err = err, best_s = s;
      }
    }
    scales[g] = _Float16(best_s);
    for (int i = 0; i < n; ++i) {
      const uint32_t q = best_s > 0.f ? uint32_t(std::clamp(std::nearbyint(x[i] / best_s + c), 0.f, float(levels - 1)))
                                      : uint32_t(c + 0.5f);
      words[DraftQWord(at[i], I, bits)] |= q << DraftQShift(at[i], I, bits);
    }
  }
}

// The stored code and value of element e.
inline uint32_t DraftQCode(const uint32_t* words, int e, int I, int bits) {
  return (words[DraftQWord(e, I, bits)] >> DraftQShift(e, I, bits)) & ((1u << bits) - 1);
}
inline float DraftQValue(const uint32_t* words, const _Float16* scales, int e, int I, int bits) {
  return (float(DraftQCode(words, e, I, bits)) - DraftQCenter(bits)) * float(scales[DraftQGroup(e, I, bits)]);
}

// The arithmetic of one row (draft_quant.h DraftQGemvKernel, up to the fp32 order inside a load): per load, fp16
// (q - c) times fp16 x summed in fp32, times the load's scale; the loads summed in order.
inline float DraftQDotRef(const uint32_t* words, const _Float16* scales, const _Float16* x, int I, int bits) {
  const int L = DraftQLoads(I, bits);
  float total = 0.f;
  for (int g = 0; g < L; ++g) {
    float part = 0.f;
    for (int q = 0; q < 64 / bits; ++q)
      for (int half = 0; half < 2; ++half) {
        const int e = 2 * (q * L + g) + half;
        part += float(_Float16(float(DraftQCode(words, e, I, bits)) - DraftQCenter(bits))) * float(x[e]);
      }
    total += part * float(scales[g]);
  }
  return total;
}

// The order of the rescoring: block a comes before block b when its coarse maximum is larger, or equal with the
// lower index. NaN counts as -inf. The rescoring takes the blocks of rank 0 .. kDraftQBlocks - 1.
DRAFTQ_HD inline bool DraftQBefore(float a, int ia, float b, int ib) {
  const float x = a == a ? a : -INFINITY, y = b == b ? b : -INFINITY;
  return x > y || (x == y && ia < ib);
}

// Rank of block j among n block maxima (the number of blocks before it); stops counting at `cap`.
DRAFTQ_HD inline int DraftQRank(const float* maxima, int n, int j, int cap) {
  int rank = 0;
  for (int k = 0; k < n && rank < cap; ++k) rank += k != j && DraftQBefore(maxima[k], k, maxima[j], j);
  return rank;
}

}  // namespace crucible
