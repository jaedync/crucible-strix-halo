// Pure load decisions of the Loader (model.h): a matrix's width, a dense group's front order and
// the n-gram row width. Host-only, so tests/load_plan_test.cpp checks them against the e80d21cc
// decisions on the real pack headers.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

#include "../../exl3/decode_split.h"
#include "../../exl3/decode_support.h"

namespace crucible {

// Width of a trellis with `words` int16 words per 16x16 tile; 0: not a width of this engine. A half
// width KA + 0.5 has 8 (2 KA + 1) words and returns its code 10 KA + 5 (routed experts only:
// ValidateLinear rejects it elsewhere).
inline int TrellisK(int64_t words) {
  if (words % 16 == 8 && words > 0 && words <= 8 * 13) {
    const int code = int(words / 8 - 1) / 2 * 10 + 5;
    return HalfDecodeK(code) ? code : 0;
  }
  if (words % 16) return 0;
  const int k = int(words / 16);
  return LoadableDecodeK(k) ? k : 0;
}

// Width of an n-gram table row with `words` int16 words (PleGatherKernel); 0: not supported.
inline int PleRowK(int64_t words) {
  if (words == 51 || words == 61 || words == 81) return int((words - 1) / 10);
  return 0;
}

struct GroupShape {
  int A, B, K;  // row tiles (I / 16), column tiles (O / 16), width
};

// Decode split of each matrix of a dense group in front order (front_order.h), or all 0 for engine
// order: the engine's rule (decode_split.h FrontGroupSplits, which Loader::FrontSplits calls through
// FrontLoadSplits). K7 and K8 groups stay in engine order: the ring kernels, the main front-order
// readers of decode, hold K <= 6 only (kSupportedDecodeK).
inline std::vector<int> FrontSplitsFor(const std::vector<GroupShape>& group) {
  std::vector<DenseMatrix> dense;
  for (const GroupShape& g : group) dense.push_back({g.A * 16, g.B * 16, g.K});
  return FrontGroupSplits(dense);
}

}  // namespace crucible
