// The MoE kernel choice of every layer of the seven EXL3 packs (tests/data/packs/*.lin "moe" lines:
// each layer's routed width; the shared expert from its "lin" lines): each layer's widths must be a
// supported pair (decode_support.h) with exactly one decode kernel (MoeFusedKernel for the narrow
// pairs, one MoeFusedKernelWide launch per wide pair; engine MoeFusedR) and one prefill MoE kernel
// set (PrefillMoeKmax; the engine's prefill MoE launch). 3.05 and 4.05 keep their old choices.
#include "../engine/exl3/decode_support.h"
#include "../engine/model/flashnext/geometry.h"
#include "../engine/model/flashnext/load_plan.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace crucible;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
  }
}

std::string Pair(MoeWidths w) { return KName(w.kr) + "+" + KName(w.ks); }

bool Wide(MoeWidths w) {
  return std::find(kWideMoeWidths.begin(), kWideMoeWidths.end(), w) != kWideMoeWidths.end();
}

// Routed width of each layer prefix and the words of every other trellis, from a fixture.
struct Fixture {
  std::map<std::string, std::vector<int64_t>> routed;
  std::map<std::string, int64_t> lin;
};

Fixture Read(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "moe_widths_test: cannot read %s (run from the repository root)\n", path.c_str());
    std::exit(1);
  }
  Fixture x;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::string kind, name;
    s >> kind;
    int64_t a, b, words;
    if (kind == "moe" && s >> name >> words) x.routed[name].push_back(words);
    if (kind == "lin" && s >> name >> a >> b >> words) x.lin[name] = words;
  }
  return x;
}

// Every supported pair has exactly one decode kernel; no other pair has one.
void CheckKernelLists() {
  for (const MoeWidths& w : kSupportedMoeWidths)
    Expect(NarrowMoeWidths(w) != Wide(w), Pair(w) + " has exactly one decode MoE kernel");
  for (const MoeWidths& w : kWideMoeWidths) Expect(SupportedMoeWidths(w), Pair(w) + " wide launch is a supported pair");
  for (int kr : {2, 3, 4, 5, 6, 7, 8, 25, 35})
    for (int ks : {2, 3, 4, 5, 6, 7, 8, 25, 35}) {
      const MoeWidths w{kr, ks};
      if (!SupportedMoeWidths(w)) Expect(!NarrowMoeWidths(w) && !Wide(w), Pair(w) + " has no decode MoE kernel");
    }
  // Prefill: K3.5 only next to a K7 shared expert (the kPrefillMoeK7Half kernels).
  Expect(PrefillMoeKmax(std::vector<int>{7, 7, 7, 35, 35, 35}) == kPrefillMoeK7Half, "K3.5 + K7 prefill kernels");
  Expect(PrefillMoeKmax(std::vector<int>{8, 8, 8, 35, 35, 35}) == 0, "K3.5 + K8 has no prefill kernel");
  Expect(PrefillMoeKmax(std::vector<int>{6, 6, 6, 35, 35, 35}) == 0, "K3.5 + K6 has no prefill kernel");
  Expect(PrefillMoeKmax(std::vector<int>{5, 5, 5, 2, 2, 2}) == 0, "K2 + K5 has no prefill kernel");
  // K2 and K2.5 only next to a K6 shared expert (the kPrefillMoeK6Low kernels).
  Expect(PrefillMoeKmax(std::vector<int>{6, 6, 6, 2, 2, 2}) == kPrefillMoeK6Low, "K2 + K6 prefill kernels");
  Expect(PrefillMoeKmax(std::vector<int>{6, 6, 6, 25, 25, 25}) == kPrefillMoeK6Low, "K2.5 + K6 prefill kernels");
  Expect(PrefillMoeKmax(std::vector<int>{7, 7, 7, 25, 25, 25}) == 0, "K2.5 + K7 has no prefill kernel");
  Expect(PrefillMoeKmax(std::vector<int>{6, 6, 6, 35, 35, 25}) == 0, "K3.5 with K2.5 has no prefill kernel");
  Expect(PrefillMoeKmax(std::vector<int>{6, 6, 6, 3, 3, 3}) == 6, "K3 + K6 keeps the K6 kernels");
  Expect(WidthHalves(3) < WidthHalves(35) && WidthHalves(35) < WidthHalves(4), "K3 < K3.5 < K4");
  Expect(WidthHalves(2) < WidthHalves(25) && WidthHalves(25) < WidthHalves(3), "K2 < K2.5 < K3");
  Expect(KName(35) == "3.5" && KName(25) == "2.5" && KName(7) == "7", "width names");
}

struct Want {
  std::map<std::string, int> pairs;  // "kr+ks" -> layers (MTP included)
  std::map<std::string, int> kmax;   // "kr+ks" -> PrefillMoeKmax
};

void CheckPack(const std::string& name, const Want& want) {
  const Fixture x = Read("tests/data/packs/" + name + ".lin");
  std::vector<std::string> layers;
  for (int l = 0; l < kLayers; ++l) layers.push_back("layers." + std::to_string(l) + ".");
  layers.push_back("mtp.layers.0.");
  std::map<std::string, int> pairs;
  for (const std::string& L : layers) {
    const auto it = x.routed.find(L);
    if (it == x.routed.end() || it->second.size() != 1) {
      Expect(false, name + " " + L + " has one routed width");
      continue;
    }
    const int kr = TrellisK(it->second[0]);
    std::vector<int> ks;
    for (const char* p : {"gate_proj", "up_proj", "down_proj"}) ks.push_back(TrellisK(x.lin.at(L + "mlp.shared_expert." + p)));
    Expect(ks[0] == ks[1] && ks[1] == ks[2], name + " " + L + " shared expert has one width");
    const MoeWidths w{kr, ks[0]};
    const std::string pair = Pair(w);
    ++pairs[pair];
    Expect(SupportedMoeWidths(w), name + " " + L + " " + pair + " is supported");
    Expect(NarrowMoeWidths(w) || Wide(w), name + " " + L + " " + pair + " has a decode MoE kernel");
    std::vector<int> widths = ks;
    widths.insert(widths.end(), 3, kr);
    const int kmax = PrefillMoeKmax(widths);
    const auto k = want.kmax.find(pair);
    Expect(k != want.kmax.end() && kmax == k->second,
           name + " " + L + " " + pair + " prefill KMAX " + std::to_string(kmax));
  }
  Expect(pairs == want.pairs, name + " layer widths");
  std::string text;
  for (const auto& [pair, n] : pairs) text += " " + pair + " x" + std::to_string(n);
  std::printf("%s:%s\n", name.c_str(), text.c_str());
}

}  // namespace

int main() {
  CheckKernelLists();
  CheckPack("3.05bpw_h5_ng5", {{{"3+5", 49}}, {{"3+5", 5}}});
  CheckPack("4.05bpw_h6_ng6", {{{"4+6", 49}}, {{"4+6", 6}}});
  CheckPack("4.18bpw_h6_ng8", {{{"4+8", 46}, {"5+8", 3}}, {{"4+8", 8}, {"5+8", 8}}});
  CheckPack("3.18bpw_h6_ng8", {{{"3+7", 41}, {"3.5+7", 8}}, {{"3+7", 7}, {"3.5+7", kPrefillMoeK7Half}}});
  CheckPack("3.68bpw_h6_ng8", {{{"3.5+7", 41}, {"4+7", 8}}, {{"3.5+7", kPrefillMoeK7Half}, {"4+7", 7}}});
  CheckPack("2.18bpw_h6_ng8", {{{"2+6", 41}, {"2.5+6", 8}}, {{"2+6", kPrefillMoeK6Low}, {"2.5+6", kPrefillMoeK6Low}}});
  CheckPack("2.68bpw_h6_ng8", {{{"2.5+6", 41}, {"3+6", 8}}, {{"2.5+6", kPrefillMoeK6Low}, {"3+6", 6}}});
  if (failures) {
    std::printf("moe_widths_test: %d failures\n", failures);
    return 1;
  }
  std::printf("moe_widths_test: OK\n");
}
