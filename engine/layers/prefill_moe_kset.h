#pragma once
// M1: the K sets of the pair path. Host code only, no HIP, so the CPU
// test (tests/prefill_moe_kset_test.cpp) checks the engine's own rule.
#include "../exl3/decode_support.h"
namespace crucible {
// Only the K sets that the bench envelope byte-tested take the pair path:
// routed K3 with shared K5 (3.05) and routed K4 with shared K6 (4.05). k6 is
// the engine's flag (any K6 projection); it picks the set. Other K sets keep
// FX + Mid.
template<class Experts> inline bool PrefillMoePairKSet(bool k6, int sg, int su, int sd, const Experts& experts) {
  const int kr=k6 ? 4 : 3, ks=k6 ? 6 : 5;
  bool tested=sg==ks && su==ks && sd==ks;
  for (const auto& e : experts) tested=tested && e.K[0]==kr && e.K[1]==kr && e.K[2]==kr;
  return tested;
}
// prefill-moe: the KMAX codes (PrefillMoeKmax) that the 192-row tile kernels hold (prefill_moe_tile.h): every
// code of the RM6 MoE GEMM kernels, so every K set of kSupportedMoeWidths. 0 (no kernel set) keeps trunk's path.
constexpr bool PrefillMoeTileKmax(int kmax) {
  return kmax==5 || kmax==6 || kmax==7 || kmax==8 || kmax==kPrefillMoeK7Half || kmax==kPrefillMoeK6Low;
}
// prefill-rows: the fewest launch rows of the tile path. At 4,096 rows (about 80 rows per expert) the 192-row items
// have no decode advantage and the bench is slower (prefill-moe.md 14). Not a switch: SWITCH_CEILING allows no new
// name (tests/test_switch_ratchet.py).
constexpr int kPrefillMoeTileRows = 8192;
constexpr bool PrefillMoeTileLaunch(int rows) { return rows >= kPrefillMoeTileRows; }
}  // namespace crucible
