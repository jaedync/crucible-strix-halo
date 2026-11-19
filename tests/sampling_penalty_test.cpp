// CPU reference tests of the sampling penalties (engine/runtime/sampling_penalty.h).
// The semantics: repetition_penalty over the prompt and the output tokens (divide a positive
// logit, multiply the others), then frequency_penalty * count and presence_penalty over the output tokens only.
#include "../engine/runtime/sampling_penalty.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace crucible::sampling;

static Options Penalties(double repetition, double presence, double frequency) {
  Options o;
  o.repetition_penalty = repetition; o.presence_penalty = presence; o.frequency_penalty = frequency;
  return o;
}

static std::string Message(auto&& call) {
  try { call(); } catch (const std::invalid_argument& error) { return error.what(); }
  return "";
}

// Hand-computed rows on a vocabulary of six tokens.
static void HandComputed() {
  // Prompt {0, 1, 1}; output {3, 3, 2}.
  PenaltyHistory history(std::vector<int>{0, 1, 1}, 3, 6);
  for (const int token : {3, 3, 2}) history.Add(token);
  assert(history.Count(3) == 2 && history.Count(2) == 1 && history.Count(0) == 0);
  assert(history.InPrompt(0) && history.InPrompt(1) && !history.InPrompt(3));
  std::vector<float> logits{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  history.Apply(logits, Penalties(2, 0.5, 0.25));
  // 0: prompt, 2 / 2. 1: prompt, -1 * 2. 2: output once, 0.5 / 2 - 0.25 - 0.5. 3: output twice, 3 / 2 - 0.5 - 0.5.
  // 4 and 5: not seen, unchanged.
  const std::vector<float> expected{1.0f, -2.0f, -0.5f, 0.5f, -2.0f, 0.0f};
  assert(logits == expected);
  // Presence and frequency count the output only: a prompt token gets the repetition penalty only.
  std::vector<float> only_counts{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  history.Apply(only_counts, Penalties(1, 0.5, 0.25));
  assert((only_counts == std::vector<float>{2.0f, -1.0f, -0.25f, 2.0f, -2.0f, 0.0f}));
  // A negative presence or frequency penalty makes a seen token more probable (OpenAI range [-2, 2]).
  std::vector<float> negative{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  history.Apply(negative, Penalties(1, -1, -0.5));
  assert((negative == std::vector<float>{2.0f, -1.0f, 2.0f, 5.0f, -2.0f, 0.0f}));
  // A repetition penalty below 1 also makes a seen token more probable; zero stays zero.
  std::vector<float> below{2.0f, -1.0f, 0.0f, 3.0f, -2.0f, 0.0f};
  history.Apply(below, Penalties(0.5, 0, 0));
  assert((below == std::vector<float>{4.0f, -0.5f, 0.0f, 6.0f, -2.0f, 0.0f}));
  // Greedy: the penalized argmax moves from token 3 to token 0.
  const std::vector<float> raw{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  assert(PenalizedArgmax(raw, Penalties(2, 0.5, 0.25), history) == 0);
  assert(Draw(raw, Penalties(2, 0.5, 0.25), history, 9) == 0);
  // The prompt boundary: the IDs after it are output tokens of an earlier job of the same request.
  PenaltyHistory continued(std::vector<int>{0, 1, 1, 3, 3}, 3, 6);
  continued.Add(2);
  std::vector<float> same{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  continued.Apply(same, Penalties(2, 0.5, 0.25));
  assert(same == expected);
  assert(Message([] { PenaltyHistory(std::vector<int>{0, 1}, 3, 6); }) == "penalty prompt boundary is outside the IDs");
  assert(Message([] { PenaltyHistory(std::vector<int>{0, 6}, 2, 6); }) == "penalty token is outside the vocabulary");
  // Remove undoes Add (a draft that the verify step did not keep).
  history.Add(4); history.Add(3);
  history.Remove(3); history.Remove(4);
  std::vector<float> undone{2.0f, -1.0f, 0.5f, 3.0f, -2.0f, 0.0f};
  history.Apply(undone, Penalties(2, 0.5, 0.25));
  assert(undone == expected);
  assert(Message([&] { history.Remove(4); }) == "penalty history has no such output token");
}

static void Ranges() {
  Options o;
  assert(!o.Penalized() && !o.HostDraw() && o.Error(4).empty());
  o.temperature = 0.5;
  assert(!o.Penalized() && o.HostDraw());
  o.temperature = 0;
  o.presence_penalty = -0.0;
  assert(!o.Penalized());
  for (const double good : {-2.0, -0.5, 0.25, 2.0}) {
    o = Options{}; o.presence_penalty = good; assert(o.Error(4).empty() && o.Penalized() && o.HostDraw());
    o = Options{}; o.frequency_penalty = good; assert(o.Error(4).empty() && o.Penalized());
  }
  const double nan = std::nan(""), inf = std::numeric_limits<double>::infinity();
  for (const double bad : {-2.01, 2.01, nan, inf, -inf}) {
    o = Options{}; o.presence_penalty = bad; assert(o.Error(4) == "presence_penalty must be in [-2, 2]");
    o = Options{}; o.frequency_penalty = bad; assert(o.Error(4) == "frequency_penalty must be in [-2, 2]");
  }
  for (const double good : {0.01, 0.9, 1.1, 2.0}) {
    o = Options{}; o.repetition_penalty = good; assert(o.Error(4).empty() && o.Penalized());
  }
  // A tiny value (1e-50 is 0 as a float) took a positive logit to +inf. The floor is 0.01.
  for (const double bad : {0.0, -0.0, -1.0, 2.01, nan, inf, 1e-50, 1e-38, 0.0099}) {
    o = Options{}; o.repetition_penalty = bad; assert(o.Error(4) == "repetition_penalty must be in [0.01, 2]");
  }
}

// PenalizedArgmax and Draw equal FromLogits on a penalized copy: the same token, the same ties, the same errors.
static void MatchesFilteredCopy() {
  std::mt19937 random(7);
  std::uniform_real_distribution<float> value(-4, 4);
  std::uniform_int_distribution<int> pick(0, 31);
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<int> prompt;
    for (int i = 0; i < 6; ++i) prompt.push_back(pick(random));
    PenaltyHistory history(prompt, prompt.size(), 32);
    for (int i = 0; i < trial % 9; ++i) history.Add(pick(random));
    Options o = Penalties(trial % 3 ? 1.3 : 1, trial % 4 ? 0.4 : 0, trial % 5 ? -0.3 : 0.7);
    std::vector<float> logits(32);
    for (float& x : logits) x = trial % 7 ? value(random) : std::round(value(random));  // rounding makes ties
    if (trial % 11 == 0) logits[size_t(pick(random))] = -std::numeric_limits<float>::infinity();
    std::vector<float> copy = logits;
    history.Apply(copy, o);
    assert(PenalizedArgmax(logits, o, history) == FromLogits(copy, o).Ids().front());
    o.temperature = 0.8; o.top_k = trial % 3 ? 0 : 5; o.top_p = trial % 2 ? 1 : 0.9; o.seed = uint64_t(trial);
    assert(Draw(logits, o, history, 100 + uint64_t(trial)) ==
           FromLogits(copy, o).KeyedSample(o.seed, 100 + uint64_t(trial)));
    // A mapped row (the draft head's subset): the row's index i holds token mapping[i].
    const std::vector<int> mapping{1, 4, 9, 16, 25, 31};
    std::vector<float> sub{logits[1], logits[4], logits[9], logits[16], logits[25], logits[31]};
    std::vector<float> sub_copy = sub;
    history.Apply(sub_copy, o, mapping);
    for (size_t i = 0; i < mapping.size(); ++i) assert(sub_copy[i] == copy[size_t(mapping[i])]);
    o.temperature = 0;
    assert(PenalizedArgmax(sub, o, history, mapping) == FromLogits(sub_copy, o, mapping).Ids().front());
  }
  PenaltyHistory history(std::vector<int>{0}, 1, 4);
  const Options o = Penalties(1.5, 0, 0);
  const float nan_row[] = {1, std::nanf(""), 0, 0};
  assert(Message([&] { PenalizedArgmax(nan_row, o, history); }) == "nonfinite model logits");
  const float inf_row[] = {1, 0, std::numeric_limits<float>::infinity(), 0};
  assert(Message([&] { PenalizedArgmax(inf_row, o, history); }) == "nonfinite model logits");
  const float inf = std::numeric_limits<float>::infinity();
  const float masked[] = {-inf, -inf, -inf, -inf};
  assert(Message([&] { PenalizedArgmax(masked, o, history); }) == "model logits have no finite value");
  // A finite logit stays finite: the penalty clamps it to the float range, so no request's options can
  // make a +inf (an error that stops the batch session) or a -inf (a masked token).
  const float huge[] = {3e38f, 0, 0, 0};
  assert(PenalizedArgmax(huge, Penalties(0.5, 0, 0), history) == 0);
  const int bad_mapping[] = {0, -1};
  const float two[] = {0, 1};
  assert(Message([&] { PenalizedArgmax(two, o, history, bad_mapping); }) == "negative token mapping");
}

// A model whose next logits depend on the whole history: speculation with match acceptance gives the tokens of
// serial decoding, for greedy and sampled penalized requests, whatever the drafts are.
static std::vector<float> ModelRow(const std::vector<int>& sequence) {
  uint64_t hash = 1469598103934665603ull;
  for (const int token : sequence) hash = (hash ^ uint64_t(token)) * 1099511628211ull;
  std::vector<float> row(16);
  for (int t = 0; t < 16; ++t) row[size_t(t)] = float((Mix64(hash + uint64_t(t)) >> 40) % 1000) / 250.0f - 2.0f;
  row[size_t(sequence.back() % 16)] += 3;  // a strong repeat tendency that the penalties must fight
  return row;
}

static void SpeculationMatchesSerial() {
  const std::vector<int> prompt{3, 5, 3, 7};
  for (const double temperature : {0.0, 0.9}) {
    Options o = Penalties(1.25, 0.3, 0.2);
    o.temperature = temperature; o.seed = 41;
    std::vector<int> serial = prompt;
    PenaltyHistory serial_history(prompt, prompt.size(), 16);
    for (int i = 0; i < 40; ++i) {
      const int token = Draw(ModelRow(serial), o, serial_history, serial.size());
      serial.push_back(token);
      serial_history.Add(token);
    }
    // Neutral options pick other tokens: the penalties take effect on this model.
    std::vector<int> plain = prompt;
    Options neutral; neutral.temperature = temperature; neutral.seed = 41;
    PenaltyHistory none;
    for (int i = 0; i < 40; ++i) plain.push_back(Draw(ModelRow(plain), neutral, none, plain.size()));
    assert(plain != serial);
    for (const int depth : {1, 3, 7}) {
      std::mt19937 random{uint32_t(depth)};
      std::vector<int> sequence = prompt;
      PenaltyHistory history(prompt, prompt.size(), 16);
      while (sequence.size() < serial.size()) {
        // Drafts: some right (copied from the serial run), some wrong.
        std::vector<int> drafts;
        for (int k = 0; k < depth; ++k) {
          const size_t at = sequence.size() + size_t(k);
          drafts.push_back(at < serial.size() && random() % 4 ? serial[at] : int(random() % 16));
        }
        const auto verified = MatchAccept(drafts, &history, [&](size_t row) {
          std::vector<int> rows(sequence.begin(), sequence.end());
          rows.insert(rows.end(), drafts.begin(), drafts.begin() + long(row));
          return Draw(ModelRow(rows), o, history, rows.size());
        });
        assert(!verified.empty() && verified.size() <= drafts.size() + 1);
        for (const int token : verified) {
          if (sequence.size() == serial.size()) break;
          sequence.push_back(token);
          history.Add(token);
        }
      }
      assert(sequence == serial);
    }
  }
}

// Penalize maps every finite logit to a finite logit, also for options that Error refuses (defense in depth: one
// request must never poison the batch session). A nonfinite model logit stays as it is (the model's own error).
static void FiniteStaysFinite() {
  constexpr float kMax = std::numeric_limits<float>::max(), kInf = std::numeric_limits<float>::infinity();
  PenaltyHistory history(std::vector<int>{0, 1, 2, 3, 4, 5}, 6, 8);
  for (const int token : {0, 1, 2}) for (int i = 0; i < 1000; ++i) history.Add(token);
  for (const double r : {1e-50, 1e-38, 0.01, 2.0, 1e30}) {
    for (const double counts : {-2.0, 0.0, 2.0}) {
      const Options o = Penalties(r, counts, counts);
      std::vector<float> row{kMax, -kMax, 3e38f, -3e38f, 1.0f, -1.0f, 0.0f, 5.0f};
      history.Apply(row, o);
      for (const float x : row) assert(std::isfinite(x));
    }
  }
  std::vector<float> model{kInf, -kInf, std::nanf(""), 1.0f, 0, 0, 0, 0};
  history.Apply(model, Penalties(1e-50, 2, 2));
  assert(model[0] == kInf && model[1] == -kInf && std::isnan(model[2]) && std::isfinite(model[3]));
  // The order of the clamped values stays the order of the penalized values: 3e38 / 0.5 and FLT_MAX / 0.5 tie at
  // FLT_MAX, and the lower ID wins.
  PenaltyHistory prompt(std::vector<int>{0, 1}, 2, 4);
  const float tied[] = {3e38f, kMax, 1, 0};
  assert(PenalizedArgmax(tied, Penalties(0.5, 0, 0), prompt) == 0);
}

int main() {
  HandComputed();
  Ranges();
  MatchesFilteredCopy();
  SpeculationMatchesSerial();
  FiniteStaysFinite();
  std::printf("sampling_penalty_test: ok\n");
  return 0;
}
