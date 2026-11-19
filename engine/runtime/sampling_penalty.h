// Sampling penalties per request. The semantics are those of the Hugging Face
// repetition penalty and the OpenAI penalties, in this order, on float logits:
//   1. repetition_penalty r, for each token in the prompt or in the output: a logit > 0 becomes logit / r, any other
//      logit becomes logit * r (Hugging Face RepetitionPenaltyLogitsProcessor, Keskar et al. 2019, with the sign fix).
//   2. frequency_penalty f and presence_penalty p (the OpenAI definition), for each token with c > 0 output copies:
//      logit - f * c, then - p.
// Prompt and output have no window: every ID counts, special tokens too. The output is the request's own tokens,
// with the earlier jobs of a server continuation (Options::penalty_prompt). A neutral request never reaches this file.
#pragma once
#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace crucible::sampling {

// The prompt tokens and the output counts of one request. A default-constructed history is inactive (no memory).
class PenaltyHistory {
 public:
  PenaltyHistory() = default;
  // ids: the request's IDs; the first `prompt` of them are the prompt, the rest are output of an earlier job.
  PenaltyHistory(std::span<const int> ids, size_t prompt, int vocabulary)
      : count_(size_t(vocabulary), 0), flags_(size_t(vocabulary), 0) {
    if (prompt > ids.size()) throw std::invalid_argument("penalty prompt boundary is outside the IDs");
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i < prompt) Mark(Checked(ids[i]), kPrompt);
      else Add(ids[i]);
    }
  }
  bool Active() const { return !count_.empty(); }
  void Add(int token) {
    Mark(Checked(token), 0);
    ++count_[size_t(token)];
  }
  // Undoes one Add: a draft row that the verify step did not keep.
  void Remove(int token) {
    if (count_[size_t(Checked(token))] == 0) throw std::invalid_argument("penalty history has no such output token");
    --count_[size_t(token)];
  }
  int Count(int token) const { return count_[size_t(Checked(token))]; }
  bool InPrompt(int token) const { return flags_[size_t(Checked(token))] & kPrompt; }
  // True for a token that the prompt or the output ever held; any other token keeps its logit.
  bool Listed(int token) const {
    return token >= 0 && size_t(token) < flags_.size() && (flags_[size_t(token)] & kListed);
  }
  // The penalized logit of a listed token, in float. A finite logit stays finite: the
  // result is clamped to the float range, so no request's options can make a +inf (an error that stops the batch
  // session for every user) or a -inf (a masked token). A nonfinite model logit is returned as it is.
  float Penalize(float logit, int token, const Options& o) const {
    if (!std::isfinite(logit)) return logit;
    const float raw = logit;
    const int count = count_[size_t(token)];
    if ((flags_[size_t(token)] & kPrompt) || count > 0) {
      const float r = float(o.repetition_penalty);
      if (r != 1) logit = logit > 0 ? logit / r : logit * r;
    }
    if (count > 0) {
      logit -= float(o.frequency_penalty) * float(count);
      logit -= float(o.presence_penalty);
    }
    constexpr float kMax = std::numeric_limits<float>::max();
    return std::isnan(logit) ? raw : std::clamp(logit, -kMax, kMax);  // NaN only from NaN options, which Error refuses
  }
  // Penalizes a logits row in place. mapping: the row's token for each index (a draft head subset); empty is the
  // identity.
  void Apply(std::span<float> logits, const Options& o, std::span<const int> mapping = {}) const {
    if (mapping.empty()) {
      for (const int token : listed_) {
        if (size_t(token) >= logits.size()) throw std::invalid_argument("penalty token is outside the logits row");
        logits[size_t(token)] = Penalize(logits[size_t(token)], token, o);
      }
      return;
    }
    if (mapping.size() != logits.size()) throw std::invalid_argument("empty logits or mismatched token mapping");
    for (size_t i = 0; i < logits.size(); ++i)
      if (Listed(mapping[i])) logits[i] = Penalize(logits[i], mapping[i], o);
  }

 private:
  static constexpr uint8_t kPrompt = 1, kListed = 2;
  int Checked(int token) const {
    if (token < 0 || size_t(token) >= count_.size()) throw std::invalid_argument("penalty token is outside the vocabulary");
    return token;
  }
  void Mark(int token, uint8_t flag) {
    if (!(flags_[size_t(token)] & kListed)) {
      flags_[size_t(token)] |= kListed;
      listed_.push_back(token);
    }
    flags_[size_t(token)] |= flag;
  }
  std::vector<int32_t> count_;
  std::vector<uint8_t> flags_;
  std::vector<int> listed_;  // the listed tokens, in first-seen order
};

// The history of a request: inactive (no memory) for neutral options.
inline PenaltyHistory RequestPenalties(std::span<const int> ids, const Options& o, int vocabulary) {
  if (!o.Penalized()) return {};
  const size_t prompt = o.penalty_prompt < 0 ? ids.size() : size_t(o.penalty_prompt);
  return PenaltyHistory(ids, prompt, vocabulary);
}

// Equals FromLogits(penalized copy, o with temperature 0, mapping): the same token, ties to the lower ID, the same
// errors in the same row order. One pass, with no copy and no candidate list.
inline int PenalizedArgmax(std::span<const float> logits, const Options& o, const PenaltyHistory& history,
                           std::span<const int> mapping = {}) {
  if (logits.empty() || (!mapping.empty() && mapping.size() != logits.size()))
    throw std::invalid_argument("empty logits or mismatched token mapping");
  if (const std::string error = o.Error(std::numeric_limits<int>::max()); !error.empty())
    throw std::invalid_argument(error);
  bool found = false;
  int best_token = 0;
  float best = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const int token = mapping.empty() ? int(i) : mapping[i];
    const float value = history.Listed(token) ? history.Penalize(logits[i], token, o) : logits[i];
    if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
      throw std::invalid_argument("nonfinite model logits");
    if (!std::isfinite(value)) continue;
    if (token < 0) throw std::invalid_argument("negative token mapping");
    if (!found || value > best || (value == best && token < best_token)) {
      found = true; best = value; best_token = token;
    }
  }
  if (!found) throw std::invalid_argument("model logits have no finite value");
  return best_token;
}

// The target's own draw at `position` (the keyed law, sampling.h). Neutral options take the old call unchanged.
inline int Draw(std::span<const float> logits, const Options& o, const PenaltyHistory& history, uint64_t position,
                std::span<const int> mapping = {}) {
  if (!o.Penalized()) return FromLogits(logits, o, mapping).KeyedSample(o.seed, position);
  if (!history.Active()) throw std::logic_error("penalized request without a penalty history");
  if (o.temperature == 0) return PenalizedArgmax(logits, o, history, mapping);
  std::vector<float> copy(logits.begin(), logits.end());
  history.Apply(copy, o, mapping);
  return FromLogits(copy, o, mapping).KeyedSample(o.seed, position);
}

// The filtered distribution of a penalized row (FromLogits on a penalized copy); neutral options take FromLogits as is.
inline Distribution PenalizedDistribution(std::span<const float> logits, const Options& o, const PenaltyHistory& history,
                                          std::span<const int> mapping = {}) {
  if (!o.Penalized()) return FromLogits(logits, o, mapping);
  if (!history.Active()) throw std::logic_error("penalized request without a penalty history");
  std::vector<float> copy(logits.begin(), logits.end());
  history.Apply(copy, o, mapping);
  return FromLogits(copy, o, mapping);
}

// Draft tokens that a draft chain counts while it proposes the next ones (a draft at depth k sees the drafts before
// it, as the verify row of depth k does). The destructor takes them out again. Does nothing for an inactive history.
class ProvisionalTokens {
 public:
  explicit ProvisionalTokens(PenaltyHistory& history) : history_(history) {}
  ProvisionalTokens(const ProvisionalTokens&) = delete;
  ProvisionalTokens& operator=(const ProvisionalTokens&) = delete;
  ~ProvisionalTokens() { for (const int token : added_) history_.Remove(token); }
  void Add(int token) {
    if (!history_.Active()) return;
    history_.Add(token);
    added_.push_back(token);
  }

 private:
  PenaltyHistory& history_;
  std::vector<int> added_;
};

// Match acceptance over the verify rows of one speculative step. target(row) gives the target's own draw at row
// `row`; while it runs, the history holds the drafts before that row, as a serial run would hold them (a row runs
// only when every draft before it equals the target's token). The history is the same again at the return; the
// caller adds the verified tokens it emits.
template <class Target>
std::vector<int> MatchAccept(std::span<const int> drafts, PenaltyHistory* history, Target&& target) {
  const bool counted = history && history->Active();
  std::vector<int> verified;
  for (size_t row = 0;; ++row) {
    const int token = target(row);
    verified.push_back(token);
    if (row == drafts.size() || token != drafts[row]) break;
    if (counted) history->Add(token);
  }
  if (counted)
    for (size_t i = 0; i + 1 < verified.size(); ++i) history->Remove(verified[i]);
  return verified;
}

}  // namespace crucible::sampling
