// The Loader's load decisions (load_plan.h: matrix width, dense-group front order, n-gram row width)
// on the real tensor shapes of the five EXL3 packs (tests/data/packs/*.lin, from
// tools/packs/plan_fixture.py; its per-layer "moe" lines are for moe_widths_test). For 3.05 and
// 4.05 the plan must equal the trunk decisions (Legacy* below: front order from d36be2a6, widths from
// e80d21cc model.h) line for line, with pinned digests. 4.18, 3.18 and
// 3.68 must load; 3.18 and 3.68 hold K3.5 routed experts.
#include "../engine/model/flashnext/load_plan.h"
#include "../engine/model/flashnext/geometry.h"
#include "../engine/exl3/kernels/lane_window.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace crucible;

// d36be2a6 decode_split.h FrontGroupSplits (Loader::FrontSplits through FrontLoadSplits), without the
// shape fit check: a mixed K6/K4 group loads in front order since 5cb41d21.
std::vector<int> LegacyFrontSplits(const std::vector<GroupShape>& group) {
  std::vector<std::pair<int, int>> io;
  for (const GroupShape& g : group) {
    if (g.K < 2 || g.K > 6) return std::vector<int>(group.size(), 0);
    io.push_back({g.A * 16, g.B * 16});
  }
  const int target = DecodeTarget(io);
  std::vector<int> splits;
  for (const auto& [I, O] : io) splits.push_back(SplitFor(I / 16, target));
  return splits;
}

// e80d21cc model.h Loader::Trellis: "if (ti.shape[2] % 16 || K < 2 || K > 6) Die".
int LegacyTrellisK(int64_t words) {
  const int K = (int)words / 16;
  return words % 16 || K < 2 || K > 6 ? 0 : K;
}

// e80d21cc model.h Loader::PleW: rows of 51 or 61 words.
int LegacyPleRowK(int64_t words) { return words != 51 && words != 61 ? 0 : int((words - 1) / 10); }

struct Lin {
  int A, B;
  int64_t words;
};

struct Pack {
  std::map<std::string, Lin> lin;
  std::vector<std::pair<int64_t, long>> experts, ngram;
};

Pack Read(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "load_plan_test: cannot read %s (run from the repository root)\n", path.c_str());
    std::exit(1);
  }
  Pack p;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::string kind;
    s >> kind;
    if (kind == "lin") {
      std::string name;
      Lin l{};
      s >> name >> l.A >> l.B >> l.words;
      p.lin[name] = l;
    } else if (kind == "experts" || kind == "ngram") {
      int64_t words;
      long count;
      s >> words >> count;
      (kind == "experts" ? p.experts : p.ngram).push_back({words, count});
    }
  }
  return p;
}

struct Plan {
  std::vector<std::string> lines;  // one per matrix and table: name K front_s bytes
  std::string reject;              // the first decision that fails to load, or empty
};

// The Loader's order: each layer's front groups (LayerW), the MTP layer, then lm_head (Load).
std::vector<std::vector<std::string>> FrontGroups() {
  std::vector<std::vector<std::string>> groups;
  auto layer = [&](const std::string& L, bool qsa) {
    if (qsa) {
      const std::string A = L + "self_attn.";
      groups.push_back({A + "q_proj", A + "k_proj", A + "v_proj", A + "indexer.index_qk_proj"});
      groups.push_back({A + "o_proj"});
    } else {
      const std::string G = L + "linear_attn.";
      groups.push_back({G + "in_proj_qkv", G + "in_proj_z"});
      groups.push_back({G + "out_proj"});
    }
  };
  for (int l = 0; l < kLayers; ++l) layer("layers." + std::to_string(l) + ".", IsQsa(l));
  layer("mtp.layers.0.", IsQsa(kMtpLayer));
  groups.push_back({"lm_head"});
  return groups;
}

template <typename Split, typename KOf, typename PleK>
Plan Decide(const Pack& p, Split split, KOf k_of, PleK ple_k) {
  Plan plan;
  std::map<std::string, int> front;
  for (const auto& group : FrontGroups()) {
    std::vector<GroupShape> shapes;
    for (const auto& name : group) {
      const Lin& l = p.lin.at(name);
      shapes.push_back({l.A, l.B, k_of(l.words)});
    }
    const std::vector<int> s = split(shapes);
    for (size_t i = 0; i < group.size(); ++i) front[group[i]] = s[i];
  }
  auto reject = [&](const std::string& what) {
    if (plan.reject.empty()) plan.reject = what;
  };
  for (const auto& [name, l] : p.lin) {
    const int K = k_of(l.words);
    if (K == 0) reject(name + " words " + std::to_string(l.words));
    else if (2 * TileDwords(K) != l.words) reject(name + " K" + std::to_string(K) + " tile size");
    const auto it = front.find(name);
    plan.lines.push_back(name + " K" + std::to_string(K) + " front " + std::to_string(it == front.end() ? 0 : it->second) +
                         " bytes " + std::to_string((int64_t)l.A * l.B * l.words * 2));
  }
  for (const auto& [words, count] : p.experts) {
    const int K = k_of(words);
    if (K == 0) reject("routed experts words " + std::to_string(words));
    else if (2 * TileDwords(K) != words) reject("routed experts K" + std::to_string(K) + " tile size");
    plan.lines.push_back("experts K" + std::to_string(K) + " count " + std::to_string(count));
  }
  for (const auto& [words, count] : p.ngram) {
    const int K = ple_k(words);
    if (K == 0) reject("ngram words " + std::to_string(words));
    plan.lines.push_back("ngram K" + std::to_string(K) + " count " + std::to_string(count));
  }
  return plan;
}

uint64_t Digest(const Plan& plan) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a 64
  for (const auto& line : plan.lines)
    for (const char c : line + "\n") h = (h ^ (unsigned char)c) * 1099511628211ull;
  return h;
}

int failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  // Digests of the d36be2a6 plans of 3.05 and 4.05 (the Legacy* decisions on the fixtures). 4.05 changed
  // from 877fc0f3ad5d9126 at e80d21cc: 5cb41d21 loads its mixed K6/K4 QSA input group in front order.
  const std::map<std::string, uint64_t> pinned = {
      {"3.05bpw_h5_ng5", 0xd96c0ac32fc15987ull}, {"4.05bpw_h6_ng6", 0x2072807c5d48a836ull}};
  for (const std::string name : {"3.05bpw_h5_ng5", "4.05bpw_h6_ng6", "3.18bpw_h6_ng8", "3.68bpw_h6_ng8", "4.18bpw_h6_ng8"}) {
    const Pack p = Read("tests/data/packs/" + name + ".lin");
    const Plan legacy = Decide(p, LegacyFrontSplits, LegacyTrellisK, LegacyPleRowK);
    const Plan now = Decide(p, FrontSplitsFor, TrellisK, PleRowK);
    int fronts = 0;
    for (const auto& line : now.lines)
      fronts += line.find(" front ") != std::string::npos && line.find(" front 0 ") == std::string::npos;
    std::printf("%s: %zu lines, %d in front order, digest %016llx (e80d21cc %016llx)%s%s\n", name.c_str(),
                now.lines.size(), fronts, (unsigned long long)Digest(now), (unsigned long long)Digest(legacy),
                now.reject.empty() ? "" : ", rejects ", now.reject.c_str());
    if (pinned.count(name)) {
      Expect(legacy.reject.empty() && now.reject.empty(), name + " loads");
      Expect(now.lines == legacy.lines, name + " plan equals the e80d21cc plan");
      Expect(Digest(legacy) == pinned.at(name), name + " e80d21cc digest is the pinned one");
    } else if (name == "4.18bpw_h6_ng8") {
      Expect(!legacy.reject.empty(), name + " did not load at e80d21cc");
      Expect(now.reject.empty(), name + " loads");
      Expect(fronts == 1, name + ": only lm_head (K6) in front order");
    } else {
      Expect(!legacy.reject.empty(), name + " did not load at e80d21cc");
      Expect(now.reject.empty(), name + " loads");
      Expect(fronts == 1, name + ": only lm_head (K6) in front order");
      bool half = false;
      for (const auto& line : now.lines) half = half || line.rfind("experts K35 count ", 0) == 0;
      Expect(half, name + " has K3.5 (code 35) routed experts");
    }
  }
  if (failures) return 1;
  std::printf("load_plan_test: OK\n");
}
