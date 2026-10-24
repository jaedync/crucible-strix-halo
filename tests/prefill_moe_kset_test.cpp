// The M1 pair path K-set allowlist (findings section 11.5 M4): only the
// K sets that the bench envelope byte-tested take the pair path.
#include "../engine/layers/prefill_moe_kset.h"
#include <cassert>
#include <vector>

namespace {
struct Expert { int K[3]; };
struct Proj { int K; };
struct Moe { Proj sg, su, sd; std::vector<Expert> experts; };

Moe Make(int routed, int shared) {
  Moe m{{shared},{shared},{shared},std::vector<Expert>(512,Expert{{routed,routed,routed}})};
  return m;
}
// The engine's k6 rule (prefill_engine.inc): any K6 projection.
bool K6(const Moe& m) {
  bool k6=m.sg.K==6 || m.su.K==6 || m.sd.K==6;
  for (const auto& e : m.experts) k6=k6 || e.K[0]==6 || e.K[1]==6 || e.K[2]==6;
  return k6;
}
bool Pair(const Moe& m) {
  return crucible::PrefillMoePairKSet(K6(m),m.sg.K,m.su.K,m.sd.K,m.experts);
}
}  // namespace

int main() {
  // Allowed: 3.05 (routed K3, shared K5) and 4.05 (routed K4, shared K6).
  assert(Pair(Make(3,5)));
  assert(Pair(Make(4,6)));
  // Every other uniform K set keeps FX + Mid.
  for (int r=2; r<=6; ++r)
    for (int s=2; s<=6; ++s)
      if (!((r==3 && s==5) || (r==4 && s==6))) assert(!Pair(Make(r,s)));
  // One routed projection or one shared projection off the set is enough.
  for (int p=0; p<3; ++p) {
    for (int k : {2,4,5,6}) {
      auto m=Make(3,5); m.experts[511].K[p]=k; assert(!Pair(m));
    }
    for (int k : {2,3,5}) {
      auto m=Make(4,6); m.experts[0].K[p]=k; assert(!Pair(m));
    }
  }
  for (int k : {3,4,6}) {
    auto a=Make(3,5); a.sg.K=k; assert(!Pair(a));
    auto b=Make(3,5); b.su.K=k; assert(!Pair(b));
    auto c=Make(3,5); c.sd.K=k; assert(!Pair(c));
  }
  for (int k : {3,4,5}) {
    auto a=Make(4,6); a.sg.K=k; assert(!Pair(a));
    auto b=Make(4,6); b.su.K=k; assert(!Pair(b));
    auto c=Make(4,6); c.sd.K=k; assert(!Pair(c));
  }
  // The k6 flag picks the set: a K4/K6 layer without k6 is not 3.05.
  assert(!crucible::PrefillMoePairKSet(false,6,6,6,Make(4,6).experts));
  assert(!crucible::PrefillMoePairKSet(true,5,5,5,Make(3,5).experts));
}
