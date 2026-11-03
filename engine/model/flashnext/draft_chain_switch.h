// Resolution of CRUCIBLE_DRAFT_CHAIN: proposer-only changes to the MTP draft calls.
// Every draft is verified by the full model, so these change the speed and the acceptance, never the output.
// Unset or "" keeps the default calls. Else a comma list of distinct words, in any order. The words k1=N (MTP layer
// MoE on the top N experts) and sync (one host wait per chain call) were tried and removed: no gain.
//   hc8: the MTP layer's HC sites and its mixer use int8 HC weights (the CRUCIBLE_HC_WMMA=2 read, hc_int8.h).
//   lead: a solo greedy step launches the next round's first MTP pass after its accept, so the pass runs while the
//          server handles the step's tokens (the GPU is idle then); the next step takes its choice if its state matches.
//   stop=NN (NN = 01..99): the greedy chain's check before a call uses the prior NN / 100 for the draft not yet
//          computed (default kDraftNextPrior, 0.75). A lower prior stops before more calls whose draft the check
//          after the call would drop (draft-cost.md lever 3); it changes which drafts are proposed.
// A pure function so a CPU test can check it (tests/draft_chain_switch_test.cpp).
#pragma once
#include <cctype>
#include <string>

namespace crucible {
struct DraftChain {
  bool hc8 = false;
  bool lead = false;
  int stop = 0;  // 0: the caller's default (kDraftNextPrior)
  bool Default() const { return !hc8 && !lead && stop == 0; }
  float Prior(float fallback) const { return stop ? float(stop) / 100.f : fallback; }
};

// Returns false (and leaves `out` unchanged) for a text that is not a valid list.
inline bool ResolveDraftChain(const char* v, DraftChain& out) {
  DraftChain d;
  if (!v || !*v) {
    out = d;
    return true;
  }
  const std::string text(v);
  size_t begin = 0;
  while (true) {
    const size_t end = text.find(',', begin);
    const std::string word = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (word == "hc8" && !d.hc8) d.hc8 = true;
    else if (word == "lead" && !d.lead) d.lead = true;
    else if (!d.stop && word.size() == 7 && word.compare(0, 5, "stop=") == 0 && std::isdigit((unsigned char)word[5]) &&
             std::isdigit((unsigned char)word[6]) && word != "stop=00")
      d.stop = (word[5] - '0') * 10 + (word[6] - '0');
    else return false;
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  out = d;
  return true;
}

inline std::string DraftChainLabel(const DraftChain& d) {
  if (d.Default()) return "default";
  std::string s;
  auto add = [&](const std::string& w) { s += (s.empty() ? "" : ",") + w; };
  if (d.hc8) add("hc8");
  if (d.lead) add("lead");
  if (d.stop) add("stop=" + std::string(d.stop < 10 ? "0" : "") + std::to_string(d.stop));
  return s;
}
}  // namespace crucible
