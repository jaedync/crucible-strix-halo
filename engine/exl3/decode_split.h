// Decode split policy, shared by plan building and the front tile order.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "decode_support.h"

namespace crucible {

// A fixed span and divisor keep reductions independent of activation row count.
inline int SplitFor(int A, int target) {
  for (int d = 1; d <= A; ++d)
    if (A % d == 0 && A / d <= target) return d;
  return A;
}

// One span target per dense decode group: the longest span that still gives
// enough waves to cover the GPU. io holds (I, O) per matrix. The loader calls
// this too, because the front tile order depends on each matrix's split.
inline int DecodeTarget(const std::vector<std::pair<int, int>>& io) {
  int target = 20;
  for (int t : {160, 80, 40, 20}) {
    long waves = 0;
    for (const auto& [I, O] : io) waves += (long)(O / 16) * SplitFor(I / 16, t);
    target = t;
    if (waves >= 900) break;
  }
  return target;
}

// Front tile order geometry (front_order.h): 8 column blocks by 4 row tiles per chunk.
constexpr int kFrontCols = 8, kFrontTiles = 4;

// A front order applies when every split span and the column count divide.
inline bool FrontFitsShape(int A, int B, int S) {
  return S > 0 && A % S == 0 && (A / S) % kFrontTiles == 0 && B % kFrontCols == 0;
}

// The split S of each matrix of one dense decode group. The plan builder
// (Engine::Build) and the front tile order (Loader::FrontSplits) both call
// it, so a loaded front order matches the decode split by construction. S
// sets the order of the partial sums; it does not depend on K or tile order.
inline std::vector<int> DecodeSplits(const std::vector<std::pair<int, int>>& io) {
  const int target = DecodeTarget(io);
  std::vector<int> splits;
  for (const auto& [I, O] : io) splits.push_back(SplitFor(I / 16, target));
  return splits;
}

struct DenseMatrix {
  int I, O, K;
};

// Front-order splits for one dense decode group, or all 0 (engine order).
// A group that mixes K6 with K3..5 (4.05 QSA input: q, k, v K6, indexer K4)
// also loads in front order: the ring decoders take K per job, and the
// prefill GEMM runs such a group as one fused launch per width class, each
// with the front walk (PrefillLinear::Build). Until an earlier change the prefill
// GEMM could not read a mixed group in front order (78932a91).
inline std::vector<int> FrontGroupSplits(const std::vector<DenseMatrix>& group) {
  std::vector<std::pair<int, int>> io;
  for (const DenseMatrix& m : group) {
    if (!SupportedDecodeK(m.K)) return std::vector<int>(group.size(), 0);
    io.push_back({m.I, m.O});
  }
  return DecodeSplits(io);
}

// Loader::FrontSplits without the checkpoint: the trellis shapes (row tiles,
// column tiles, 16 K) of one group in group order. splits: the front splits,
// or all 0 (engine order); bad: the first matrix whose split does not fit the
// front order (the loader stops on it), or -1.
struct FrontLoad {
  std::vector<int> splits;
  int bad = -1;
};
inline FrontLoad FrontLoadSplits(const std::vector<std::vector<int64_t>>& trellis) {
  std::vector<DenseMatrix> group;
  for (const auto& shape : trellis) group.push_back({(int)shape[0] * 16, (int)shape[1] * 16, (int)shape[2] / 16});
  FrontLoad out{FrontGroupSplits(group)};
  if (out.splits[0] == 0) return out;
  for (size_t i = 0; i < group.size() && out.bad < 0; ++i)
    if (!FrontFitsShape(group[i].I / 16, group[i].O / 16, out.splits[i])) out.bad = (int)i;
  return out;
}

}  // namespace crucible
