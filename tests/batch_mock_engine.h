// The fake engine of the batch session tests (batch_session_test.cpp, end_resume_test.cpp): each row's greedy ID is
// (token + 1) % 16, and a logits row scores every ID by its distance from that row's greedy ID.
#pragma once
#include "../engine/serve/batch_session.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

struct MockEngine {
  int selected = 0, calls = 0, prefill_rows = 0, launch_rows = 0;
  std::map<int, std::vector<int>> histories;
  std::vector<int> last;
  int nan_row = -1;
  bool short_output = false;
  std::optional<int> forced_greedy;
  int context = 64;  // split2048_schedule_test.cpp runs prompts of 34,825 to 40,960 rows
  int Context() const { return context; }
  std::vector<int> capacities;
  int PrefillRows() const { return prefill_rows; }
  int PrefillLaunchRows() const { return launch_rows ? launch_rows : prefill_rows; }
  // x4-tail: join rows and the specs of the joined launches (Engine::SetPrefillJoin).
  int join_rows = 0;
  std::vector<crucible::PrefillJoinSpec> joins;
  int PrefillJoinRows() const { return join_rows; }
  void SetPrefillJoin(crucible::PrefillJoinSpec spec) { assert(join_rows > 0); joins.push_back(std::move(spec)); }
  void ConfigureServeStreams(int count, uint64_t = 0, int short_count = 0, int short_ctx = 0) {
    capacities.assign(size_t(count), Context());
    for (int slot = count - short_count; slot < count; ++slot) capacities[size_t(slot)] = short_ctx;
  }
  int StreamCapacity(int slot) const { return capacities.at(size_t(slot)); }
  void SelectStream(int slot) { selected = slot; }
  void Reset() { histories[selected].clear(); }
  std::vector<int> replays;
  std::vector<int> chunks;  // rows per Forward or Prefill call; negative for Prefill (the bulk path)
  bool in_prefill = false;
  std::vector<int> Forward(const int* ids, int rows, int replay, int) {
    ++calls; last.clear(); replays.push_back(replay); chunks.push_back(in_prefill ? -rows : rows);
    for (int row = 0; row < rows; ++row) {
      histories[selected].push_back(ids[row]); last.push_back(forced_greedy.value_or((ids[row] + 1) % 16));
    }
    return last;
  }
  int between_runs = 0;
  // As Engine::Prefill: decode passes between the 48 layers, then the chunk continues on its own stream.
  std::vector<int> keeps;  // per Prefill call: the first row whose final trunk stacks a caller reads
  std::vector<int> Prefill(const int* ids, int rows, const std::string&, bool predict,
                           const crucible::PrefillBetween* between = nullptr, int keep = 0) {
    keeps.push_back(keep);
    const int prefilling = selected;
    if (between && crucible::InterleaveAllowed(rows, crucible::kMaxRows, true))
      for (int layer = 0; layer < 48; ++layer)
        if (crucible::InterleaveAfterLayer(layer, between->every, 48)) {
          between->run(); ++between_runs; selected = prefilling;
        }
    in_prefill = true;
    const auto output = Forward(ids, rows, 0, rows);
    in_prefill = false;
    last = {output.back()};
    return predict ? last : std::vector<int>{};
  }
  std::vector<int> ForwardBatch(const std::vector<int>& ids, const std::vector<int>& slots) {
    ++calls; last.clear();
    for (size_t row = 0; row < ids.size(); ++row) {
      histories[slots[row]].push_back(ids[row]); last.push_back(forced_greedy.value_or((ids[row] + 1) % 16));
    }
    return short_output ? std::vector<int>{} : last;
  }
  int max_pass_rows = crucible::kPackedRows;  // as Engine::MaxPassRows
  mutable int max_pass_rows_calls = 0;          // the real call may build the 16-row plans
  int MaxPassRows() const { ++max_pass_rows_calls; return max_pass_rows; }
  std::vector<int> pass_widths;  // rows of each ForwardVerifySpans call
  std::vector<int> ForwardVerifySpans(const std::vector<crucible::VerifySpan>& spans) {
    ++calls; last.clear();
    int width = 0;
    for (const auto& span : spans) width += int(span.tokens.size());
    pass_widths.push_back(width);
    for (const auto& span : spans)
      for (const int token : span.tokens) { histories[span.stream].push_back(token); last.push_back((token + 1) % 16); }
    return last;
  }
  mutable int copied_logits = 0;  // CopyLogits calls (fix A copies the final row's logits for its End point)
  std::vector<float> CopyLogits(int, int row = 0) const {
    ++copied_logits;
    std::vector<float> scores(16);
    for (int id = 0; id < 16; ++id) scores[id] = -float(std::abs(last[row] - id));
    if (row == nan_row) scores[0] = std::numeric_limits<float>::quiet_NaN();
    return scores;
  }
};
inline crucible::ServeRequest Request(int token, int n = 5, double temperature = 0, uint64_t seed = 0) {
  crucible::ServeRequest r; r.ids = {token}; r.max_tokens = n; r.spec = 0; r.stop_ids.clear();
  r.sampling.temperature = temperature; r.sampling.seed = seed; return r;
}
template <class Engine = MockEngine>
std::vector<int> Complete(crucible::BatchSession<Engine>& session, uint64_t id,
                                 const crucible::ServeRequest& request) {
  std::vector<int> ids;
  auto events = session.Open(id, request);
  while (true) {
    bool done = false;
    for (const auto& event : events) {
      if (event.token) ids.push_back(*event.token);
      if (event.done) { done = true; assert(event.completion_tokens == ids.size()); }
    }
    if (done) return ids;
    events = session.Step({id});
  }
}
