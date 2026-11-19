// Penalized requests in the batch session: the tokens follow the penalty formula on the
// host's logits row, for greedy and sampled requests, and they do not depend on the batch or on the step order.
// The mock's logits row scores ID i by -|g - i|, where g is the row's greedy ID (batch_mock_engine.h).
#include "batch_mock_engine.h"
#include <cassert>
#include <cstdio>
#include <set>

namespace {
using crucible::ServeRequest;

ServeRequest Penalized(std::vector<int> prompt, int n, double temperature, double repetition, double presence,
                       double frequency) {
  ServeRequest r = Request(prompt.front(), n, temperature, 99);
  r.ids = std::move(prompt);
  r.sampling.repetition_penalty = repetition;
  r.sampling.presence_penalty = presence;
  r.sampling.frequency_penalty = frequency;
  return r;
}

// The formula written out once more, apart from sampling_penalty.h: every row of the mock has its greedy ID `center`.
std::vector<int> Reference(const ServeRequest& r, int center) {
  std::vector<int> output;
  std::vector<int> count(16, 0);
  const std::set<int> prompt(r.ids.begin(), r.ids.end());
  const auto& o = r.sampling;
  for (int step = 0; step < r.max_tokens; ++step) {
    std::vector<float> row(16);
    for (int id = 0; id < 16; ++id) {
      float v = -float(std::abs(center - id));
      if (prompt.count(id) || count[size_t(id)]) {
        const float rp = float(o.repetition_penalty);
        v = v > 0 ? v / rp : v * rp;
      }
      if (count[size_t(id)]) {
        v -= float(o.frequency_penalty) * float(count[size_t(id)]);
        v -= float(o.presence_penalty);
      }
      row[size_t(id)] = v;
    }
    crucible::sampling::Options plain;
    plain.temperature = o.temperature; plain.seed = o.seed;
    const int token = crucible::sampling::FromLogits(row, plain).KeyedSample(plain.seed, r.ids.size() + output.size());
    output.push_back(token);
    ++count[size_t(token)];
  }
  return output;
}

// Opens every request, then steps all active requests together; `order` reverses the step order.
std::map<uint64_t, std::vector<int>> RunBatch(const std::vector<ServeRequest>& requests, bool reverse) {
  MockEngine engine;
  engine.forced_greedy = 5;  // every row's greedy ID is 5, so an unpenalized request repeats 5
  crucible::BatchSession<MockEngine> session(engine, int(requests.size()));
  std::map<uint64_t, std::vector<int>> out;
  std::set<uint64_t> active;
  const auto take = [&](const std::vector<crucible::BatchEvent>& events) {
    for (const auto& event : events) {
      if (event.token) out[event.request_id].push_back(*event.token);
      if (event.done) active.erase(event.request_id);
    }
  };
  for (size_t i = 0; i < requests.size(); ++i) {
    active.insert(i + 1);
    take(session.Open(i + 1, requests[i]));
  }
  while (!active.empty()) {
    std::vector<uint64_t> ids(active.begin(), active.end());
    if (reverse) std::reverse(ids.begin(), ids.end());
    take(session.Step(ids));
  }
  return out;
}

void Tests() {
  const std::vector<ServeRequest> requests{
      Penalized({5, 4}, 12, 0, 1, 0.4, 0.3),     // greedy, presence and frequency
      Penalized({5, 6, 4}, 12, 0, 1.5, 0, 0),    // greedy, repetition only: the prompt holds 4, 5 and 6
      Penalized({5}, 12, 0.8, 1.2, 0.5, 0.25),   // sampled, all three
      Penalized({3, 5}, 12, 0, 1, -0.5, 0),      // greedy, a negative presence penalty
  };
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto alone = RunBatch({requests[i]}, false);
    assert(alone.at(1) == Reference(requests[i], 5));
  }
  // The penalties take effect: the greedy run without them repeats 5.
  assert(Reference(Penalized({5, 4}, 12, 0, 1, 0, 0), 5) == std::vector<int>(12, 5));
  assert(Reference(requests[0], 5) != std::vector<int>(12, 5));
  // Batch of four, both step orders: the same tokens as alone.
  for (const bool reverse : {false, true}) {
    const auto together = RunBatch(requests, reverse);
    for (size_t i = 0; i < requests.size(); ++i) assert(together.at(i + 1) == Reference(requests[i], 5));
  }
  // A neutral request keeps the device's greedy ID and copies no logits row.
  MockEngine engine;
  engine.forced_greedy = 5;
  crucible::BatchSession<MockEngine> session(engine, 1);
  assert(Complete(session, 1, Request(5, 6)) == std::vector<int>(6, 5));
  assert(engine.copied_logits == 0);
  // The prompt boundary: IDs after it count as output (a server continuation keeps the earlier job's counts).
  ServeRequest continued = Penalized({5, 4, 5, 5}, 6, 0, 1, 0.4, 0.3);
  continued.sampling.penalty_prompt = 2;
  ServeRequest whole = Penalized({5, 4}, 8, 0, 1, 0.4, 0.3);
  const auto first = RunBatch({whole}, false).at(1);
  assert(first[0] == 5 && first[1] == 5);
  const auto rest = RunBatch({continued}, false).at(1);
  assert(std::equal(rest.begin(), rest.end(), first.begin() + 2));
}
}  // namespace

int main() {
  Tests();
  std::printf("batch_penalty_test: ok\n");
  return 0;
}
