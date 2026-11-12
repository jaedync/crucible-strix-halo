// Draft acceptance counters of one served request. Host only: the accept step
// feeds them values it already holds, so they never touch device state or the verified tokens. Counts only: no order
// of tokens and no time is kept, and the per-token maps are merged by (token, position) before they leave the engine.
#pragma once
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <span>
#include <string>

namespace crucible {

inline constexpr int kDraftPositions = 7;            // K: the deepest draft position (kDefaultSpecDepth)
inline constexpr size_t kDraftDetailMax = 2048;      // entries of each per-token map of one request
inline constexpr int kDraftRunBins = 9;              // runs of full-accept rounds of 1..8 rounds, last bin 9 or more

struct DraftTally {
  std::array<long, kDraftPositions> proposed{}, tested{}, accepted{};
  std::array<long, kDraftPositions + 1> length{};    // rounds by draft length 0..K
  std::array<long, kDraftPositions> lookup_proposed{}, lookup_accepted{};  // the lookup rounds alone, by position
  std::array<long, kDraftRunBins> run{};             // closed runs of full-accept rounds by length (plan wide C)
  long open_run = 0;                                 // full-accept rounds since the last miss
  long passes = 0, lookup_rounds = 0, detail_cut = 0;
  double head_ms = 0;
  bool detail = false;                               // per-token maps (serve --draft-stats)
  std::map<std::array<int, 2>, std::array<long, 2>> tokens;  // (draft, position 1..K) -> tested, accepted
  std::map<std::array<int, 3>, long> misses;                 // (draft, position, target) -> rounds

  // One finished round. `drafts` are the proposed tokens, `accepted` the length of the accepted prefix, and `target`
  // the token the target emitted at the first rejected position (ignored when every draft was accepted).
  void Round(std::span<const int> drafts, int accepted_count, int target, bool lookup) {
    const int depth = int(drafts.size());
    if (accepted_count < 0 || accepted_count > depth) return;  // a counter never fails a step
    ++length[size_t(std::min(depth, kDraftPositions))];
    const int tested_count = std::min(depth, accepted_count + 1);
    for (int i = 0; i < std::min(depth, kDraftPositions); ++i) {
      ++proposed[size_t(i)];
      if (i < tested_count) ++tested[size_t(i)];
      if (i < accepted_count) ++accepted[size_t(i)];
    }
    // A round with drafts either extends the run (all accepted) or ends it. A round without drafts does neither.
    if (depth > 0 && accepted_count == depth) ++open_run;
    else if (depth > 0) CloseRun();
    if (lookup) {
      ++lookup_rounds;
      for (int i = 0; i < std::min(depth, kDraftPositions); ++i) {
        ++lookup_proposed[size_t(i)];
        if (i < accepted_count) ++lookup_accepted[size_t(i)];
      }
      return;  // copied history tokens say nothing about the draft head
    }  // copied history tokens say nothing about the draft head
    if (!detail) return;
    for (int i = 0; i < std::min(depth, kDraftPositions); ++i) {
      if (i >= tested_count) break;  // positions after a miss were never checked
      Add(tokens, {drafts[size_t(i)], i + 1}, {1L, i < accepted_count ? 1L : 0L});
    }
    if (accepted_count < depth && accepted_count < kDraftPositions) {
      const std::array<int, 3> key{drafts[size_t(accepted_count)], accepted_count + 1, target};
      auto found = misses.find(key);
      if (found != misses.end()) ++found->second;
      else if (misses.size() < kDraftDetailMax) misses.emplace(key, 1);
      else ++detail_cut;
    }
  }
  // A head pass that served this request: its wall time share in ms.
  void Pass(double ms) { ++passes; head_ms += ms; }

  // The JSON members of the done event, with a leading comma. The per-token lists only when detail is on.
  std::string Fields() const {
    std::string out;
    auto list = [&](const char* name, const auto& values) {
      out += std::string(",\"") + name + "\":[";
      for (size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + std::to_string(values[i]);
      out += "]";
    };
    list("draft_pos_proposed", proposed);
    list("draft_pos_tested", tested);
    list("draft_pos_accepted", accepted);
    list("draft_len_hist", length);
    char ms[40];
    std::snprintf(ms, sizeof ms, "%.3f", head_ms);
    out += ",\"draft_passes\":" + std::to_string(passes) + ",\"draft_ms\":" + ms;
    auto runs = run;  // the open run counts here; the tally keeps it open
    if (open_run > 0) ++runs[size_t(std::min<long>(open_run, kDraftRunBins) - 1)];
    list("draft_run_hist", runs);
    out += ",\"draft_lookup_rounds\":" + std::to_string(lookup_rounds);
    list("draft_lookup_pos_proposed", lookup_proposed);
    list("draft_lookup_pos_accepted", lookup_accepted);
    if (!detail) return out;
    out += ",\"draft_tally\":[";
    bool first = true;
    for (const auto& [key, n] : tokens) {
      out += (first ? "[" : ",[") + std::to_string(key[0]) + "," + std::to_string(key[1]) + "," +
             std::to_string(n[0]) + "," + std::to_string(n[1]) + "]";
      first = false;
    }
    out += "],\"draft_miss\":[";
    first = true;
    for (const auto& [key, n] : misses) {
      out += (first ? "[" : ",[") + std::to_string(key[0]) + "," + std::to_string(key[1]) + "," +
             std::to_string(key[2]) + "," + std::to_string(n) + "]";
      first = false;
    }
    return out + "],\"draft_detail_cut\":" + std::to_string(detail_cut);
  }

 private:
  void CloseRun() {
    if (open_run > 0) ++run[size_t(std::min<long>(open_run, kDraftRunBins) - 1)];
    open_run = 0;
  }
  void Add(std::map<std::array<int, 2>, std::array<long, 2>>& map, std::array<int, 2> key, std::array<long, 2> n) {
    auto found = map.find(key);
    if (found != map.end()) { found->second[0] += n[0]; found->second[1] += n[1]; }
    else if (map.size() < kDraftDetailMax) map.emplace(key, n);
    else ++detail_cut;
  }
};

}  // namespace crucible
