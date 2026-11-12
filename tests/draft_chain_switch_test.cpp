// CPU checks of the CRUCIBLE_DRAFT_CHAIN switch resolution (draft_chain_switch.h): unset or "" keeps the default
// MTP draft calls; a comma list of the words hc8, lead and stop=NN turns each part on; any other text is an error.
// The words k1=N and sync were tried and removed: no gain.
#include "../engine/model/flashnext/draft_chain_switch.h"
#include <cassert>
#include <iostream>

using namespace crucible;
int main() {
  DraftChain d;
  assert(ResolveDraftChain(nullptr, d) && !d.hc8 && !d.lead && d.stop == 0 && d.Default());
  assert(ResolveDraftChain("", d) && d.Default());
  assert(ResolveDraftChain("hc8", d) && d.hc8 && !d.lead && !d.Default());
  assert(ResolveDraftChain("hc8,lead", d) && d.hc8 && d.lead && DraftChainLabel(d) == "hc8,lead");
  assert(ResolveDraftChain("lead,hc8", d) && d.hc8 && d.lead && DraftChainLabel(d) == "hc8,lead");
  // stop=NN: the prior of a draft not yet computed (the check before a chain call) is NN / 100.
  assert(ResolveDraftChain("stop=60", d) && d.stop == 60 && !d.Default() && d.Prior(0.75f) == 0.60f);
  assert(ResolveDraftChain("hc8,stop=05", d) && d.stop == 5 && d.hc8);
  assert(ResolveDraftChain("", d) && d.stop == 0 && d.Prior(0.75f) == 0.75f);
  for (const char* bad : {"stop=", "stop=0", "stop=00", "stop=100", "stop=6", "stop=6x", "stop=60,stop=50"}) {
    DraftChain e;
    assert(!ResolveDraftChain(bad, e));
  }
  assert(ResolveDraftChain("lead", d) && d.lead && !d.Default() && DraftChainLabel(d) == "lead");
  assert(ResolveDraftChain("stop=50,lead,hc8", d) && DraftChainLabel(d) == "hc8,lead,stop=50");
  for (const char* bad : {"lead,lead", "Lead", "lead=1", "hc4", "hc8,", ",hc8", "hc8,,lead", "hc8 lead", " hc8", "hc8,hc8", "on", "1"}) {
    DraftChain e;
    assert(!ResolveDraftChain(bad, e));
  }
  // The removed words stay errors, alone or in a list (a stale env line must stop the engine, not run the default).
  for (const char* bad : {"sync", "k1=6", "k1=10", "k1=0", "k1=", "k1", "hc8,sync", "k1=8,hc8", "sync,k1=1", "Sync", "hc8,lead,k1=6"}) {
    DraftChain e;
    assert(!ResolveDraftChain(bad, e));
  }
  assert(ResolveDraftChain(nullptr, d) && DraftChainLabel(d) == "default");
  std::cout << "draft_chain_switch_test: ok\n";
  return 0;
}
