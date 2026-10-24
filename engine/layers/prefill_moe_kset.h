#pragma once
// M1: the K sets of the pair path. Host code only, no HIP, so the CPU
// test (tests/prefill_moe_kset_test.cpp) checks the engine's own rule.
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
}  // namespace crucible
