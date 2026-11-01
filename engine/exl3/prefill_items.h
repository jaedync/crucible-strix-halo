// Input-transform item arithmetic shared by the prefill kernels and the CPU tests.
#pragma once

#ifdef __HIPCC__
#define PREFILL_ITEMS_HD __host__ __device__
#else
#define PREFILL_ITEMS_HD
#endif
#include <algorithm>
#include <vector>

namespace crucible {
// Item w of the list that PrefillLinear::Build wrote for jobs of R rows and I
// columns each: jobs in order, rows in order, then 128-column blocks.
struct PrefillInIndex { int job, r, blk; };
PREFILL_ITEMS_HD inline PrefillInIndex PrefillInItemAt(int R, int I, int w) {
  const int blocks = I / 128, per = R * blocks;
  const int j = w / per, rem = w - j * per;
  return {j, rem / blocks, rem % blocks};
}

// Fused output-kernel classes of a packed prefill group of n jobs with EXL3
// widths K. The fused kernels take K3..5 or K6 jobs, not both (a K6 branch
// next to K3..5 makes that code spill), so a mixed group (the 4.05 QSA input:
// q, k, v K6, the indexer K4) runs as two fused launches (split).
// fusable: packed and every O % 128 == 0.
struct PrefillFusedClasses { bool fused, k6, split; };
PREFILL_ITEMS_HD inline PrefillFusedClasses PrefillFusedClassesOf(const int* K, int n, bool fusable) {
  int six = 0;
  for (int i = 0; i < n; ++i) six += K[i] == 6;
  if (!fusable) return {false, false, false};
  return {true, six == n, six > 0 && six < n};
}

// The width of a fused group with K7 or K8 jobs (the h6_ng8 packs). Those groups run
// PrefillGemmOutWideKernel, one width per launch: 7 or 8 when every job has that width, 0 when no
// job is wider than K6, -1 when a K7 or K8 job shares the group with another width.
PREFILL_ITEMS_HD inline int PrefillWideKOf(const int* K, int n) {
  int wide = 0, other = 0;
  for (int i = 0; i < n; ++i) {
    if (K[i] == 7 || K[i] == 8) {
      if (wide && wide != K[i]) return -1;
      wide = K[i];
    } else {
      ++other;
    }
  }
  return wide && other ? -1 : wide;
}

// Items of the half prefill GEMM (prefill_half.h), 128 rows x 128 columns each: groups of G row
// blocks; inside a group every column block of every job, and for each the group's row blocks.
// Blocks in flight then share B in L2 and the group's A rows stay in the MALL. The order does not
// change any output. M[j], O[j]: rows and columns of job j (O % 128 == 0).
struct PrefillHalfItem { int job, m0, b0; };  // b0 in 16-column tiles
inline std::vector<PrefillHalfItem> PrefillHalfItemsOf(const int* M, const int* O, int n, int G) {
  std::vector<PrefillHalfItem> v;
  int rows = 0;
  for (int j = 0; j < n; ++j) rows = std::max(rows, M[j]);
  for (int g0 = 0; g0 < rows; g0 += 128 * G)
    for (int j = 0; j < n; ++j)
      for (int b = 0; b < O[j] / 16; b += 8)
        for (int m = g0; m < std::min(g0 + 128 * G, M[j]); m += 128) v.push_back({j, m, b});
  return v;
}
}  // namespace crucible
