// CPU test of the per-request draft counters.
#include <cassert>
#include <string>
#include <vector>
#include "../engine/serve/draft_tally.h"

using crucible::DraftTally;

static std::string Run(DraftTally& t, std::vector<int> drafts, int accepted, int target = -1, bool lookup = false) {
  t.Round(drafts, accepted, target, lookup);
  return {};
}

int main() {
  {  // all accepted, first rejected, middle rejected, depth 0
    DraftTally t;
    Run(t, {10, 11, 12}, 3);
    Run(t, {20, 21, 22}, 0, 99);
    Run(t, {30, 31, 32, 33}, 2, 77);
    Run(t, {}, 0);
    assert((t.proposed == std::array<long, 7>{3, 3, 3, 1, 0, 0, 0}));
    assert((t.tested == std::array<long, 7>{3, 2, 2, 0, 0, 0, 0}));
    assert((t.accepted == std::array<long, 7>{2, 2, 1, 0, 0, 0, 0}));
    assert((t.length == std::array<long, 8>{1, 0, 0, 2, 1, 0, 0, 0}));
    long p = 0, a = 0;
    for (int i = 0; i < 7; ++i) { p += t.proposed[size_t(i)]; a += t.accepted[size_t(i)]; assert(t.accepted[size_t(i)] <= t.tested[size_t(i)]); assert(t.tested[size_t(i)] <= t.proposed[size_t(i)]); }
    assert(p == 10 && a == 5 && t.tested[0] == t.proposed[0]);  // position 1 is always tested
    assert(t.Fields().find("draft_tally") == std::string::npos);    // detail off: no per-token members
  }
  {  // per-token maps: only head rounds, only tested positions, one miss per rejected round
    DraftTally t;
    t.detail = true;
    Run(t, {10, 11, 12}, 3);
    Run(t, {10, 21, 22}, 1, 99);          // 10 accepted, 21 rejected for 99, 22 never tested
    Run(t, {10, 11}, 0, 5, true);         // lookup round: positions count, tokens do not
    assert(t.lookup_rounds == 1 && t.proposed[0] == 3);
    assert((t.tokens.at({10, 1}) == std::array<long, 2>{2, 2}));
    assert((t.tokens.at({11, 2}) == std::array<long, 2>{1, 1}));
    assert((t.tokens.at({21, 2}) == std::array<long, 2>{1, 0}));
    assert(t.tokens.count({22, 3}) == 0 && t.tokens.count({12, 3}) == 1);
    assert(t.misses.size() == 1 && (t.misses.at({21, 2, 99}) == 1));
    const std::string f = t.Fields();
    assert(f.find("\"draft_miss\":[[21,2,99,1]]") != std::string::npos);
    assert(f.find("\"draft_lookup_rounds\":1") != std::string::npos);
  }
  {  // caps: a full map drops new keys and counts them; known keys still add
    DraftTally t;
    t.detail = true;
    for (int d = 0; d < int(crucible::kDraftDetailMax) + 5; ++d) Run(t, {d}, 0, 1000000 + d);
    assert(t.tokens.size() == crucible::kDraftDetailMax && t.misses.size() == crucible::kDraftDetailMax);
    assert(t.detail_cut == 10);
    Run(t, {0}, 1);
    assert((t.tokens.at({0, 1}) == std::array<long, 2>{2, 1}));
  }
  {  // head passes and the JSON members
    DraftTally t;
    t.Pass(1.25); t.Pass(0.5);
    Run(t, {1, 2}, 1, 3);
    assert(t.passes == 2);
    assert(t.Fields() == ",\"draft_pos_proposed\":[1,1,0,0,0,0,0],\"draft_pos_tested\":[1,1,0,0,0,0,0],"
                         "\"draft_pos_accepted\":[1,0,0,0,0,0,0],\"draft_len_hist\":[0,0,1,0,0,0,0,0],"
                         "\"draft_passes\":2,\"draft_ms\":1.750");
  }
  {  // a bad call is ignored, never thrown
    DraftTally t;
    t.Round(std::vector<int>{1}, 5, 0, false);
    t.Round(std::vector<int>{1}, -1, 0, false);
    assert((t.length == std::array<long, 8>{}));
  }
  return 0;
}
