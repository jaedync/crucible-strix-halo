#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible::sampling {

struct Options {
  double temperature = 0;
  int top_k = 0;
  double top_p = 1, min_p = 0;
  uint64_t seed = 0;
  // Penalties (sampling_penalty.h). The neutral values (0, 0, 1) leave every path as it
  // was: no logits copy, the device's greedy IDs.
  double presence_penalty = 0, frequency_penalty = 0, repetition_penalty = 1;
  // The IDs of the request before this boundary are its prompt; the IDs after it are output of an earlier job of the
  // same request (a server continuation). -1: every ID is prompt.
  int penalty_prompt = -1;

  bool Penalized() const { return presence_penalty != 0 || frequency_penalty != 0 || repetition_penalty != 1; }
  // The token comes from the host's copy of the logits row, not from the device's greedy ID.
  bool HostDraw() const { return temperature > 0 || Penalized(); }

  std::string Error(int vocabulary) const {
    if (!std::isfinite(temperature) || temperature < 0) return "temperature must be finite and nonnegative";
    if (top_k < 0 || top_k > vocabulary) return "top_k must be in [0, vocabulary]";
    if (!std::isfinite(top_p) || top_p < 0 || top_p > 1) return "top_p must be in [0, 1]";
    if (!std::isfinite(min_p) || min_p < 0 || min_p > 1) return "min_p must be in [0, 1]";
    if (!(presence_penalty >= -2 && presence_penalty <= 2)) return "presence_penalty must be in [-2, 2]";
    if (!(frequency_penalty >= -2 && frequency_penalty <= 2)) return "frequency_penalty must be in [-2, 2]";
    // The floor 0.01 (review H1): a smaller value divides a logit past the float range (1e-50 is 0 as a float).
    if (!(repetition_penalty >= 0.01 && repetition_penalty <= 2)) return "repetition_penalty must be in [0.01, 2]";
    if (penalty_prompt < -1) return "penalty_prompt must be -1 or a prompt length";
    return {};
  }
};

class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed) {}
  // Avoid standard-library distribution differences between host platforms.
  double Unit() { return double(state_() >> 11) * 0x1p-53; }
 private:
  std::mt19937_64 state_;
};

// Keyed sampling law (G6b). A draw depends only on (seed, absolute position, token),
// so a sampled request's output does not depend on speculation depth, spec=0, or batch packing.
inline uint64_t Mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
inline double KeyedUniform(uint64_t seed, uint64_t position, int token) {
  const uint64_t hash = Mix64(Mix64(Mix64(seed) ^ position) ^ uint64_t(uint32_t(token)));
  // 52 bits keep k + 0.5 exact, so U lies in [2^-53, 1 - 2^-53] and both logarithms stay finite.
  return (double(hash >> 12) + 0.5) * 0x1p-52;
}

class Distribution {
 public:
  Distribution(std::vector<int> ids, std::vector<double> weights) {
    if (ids.size() != weights.size() || ids.empty()) throw std::invalid_argument("empty or mismatched distribution");
    double sum = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 0 || (i && ids[i] <= ids[i - 1]) || !std::isfinite(weights[i]) || weights[i] < 0)
        throw std::invalid_argument("invalid distribution IDs or weights");
      if (weights[i] > 0) { ids_.push_back(ids[i]); probabilities_.push_back(weights[i]); sum += weights[i]; }
    }
    if (!(sum > 0) || !std::isfinite(sum)) throw std::invalid_argument("distribution has no finite positive mass");
    for (double& p : probabilities_) p /= sum;
  }

  static Distribution Point(int token) { return Distribution({token}, {1}); }
  double Probability(int token) const {
    const auto found = std::lower_bound(ids_.begin(), ids_.end(), token);
    return found == ids_.end() || *found != token ? 0 : probabilities_[found - ids_.begin()];
  }
  int Sample(Rng& rng) const {
    const double uniform = rng.Unit();
    double prefix = 0;
    for (size_t i = 0; i < ids_.size(); ++i) {
      prefix += probabilities_[i];
      if (uniform < prefix) return ids_[i];
    }
    return ids_.back();  // A normalized sum can round just below one.
  }
  // Exponential race over the support with keyed noise: the token with the largest p_i / E_i wins, where
  // E_i = -log U_i. This is Gumbel-max (log p_i - log E_i) without two logarithms per token, and it is an
  // exact draw from these probabilities. A draft with the same key matches the target's own draw whenever
  // both pick the same token.
  int KeyedSample(uint64_t seed, uint64_t position) const {
    const auto rate = [&](size_t i) { return probabilities_[i] / -std::log(KeyedUniform(seed, position, ids_[i])); };
    // Start from the most probable token. E_i is never below kMinExponential, so a token with
    // p_i < best * kMinExponential cannot win and needs no hash. The factor 0.5 keeps the cut safe
    // against rounding; the result equals a full scan with ties to the lower ID.
    const size_t top = size_t(std::max_element(probabilities_.begin(), probabilities_.end()) - probabilities_.begin());
    double best_rate = rate(top);
    int best = ids_[top];
    const double floor = best_rate * kMinExponential * 0.5;
    for (size_t i = 0; i < ids_.size(); ++i) {
      if (i == top || probabilities_[i] < floor) continue;
      const double value = rate(i);
      if (value > best_rate || (value == best_rate && ids_[i] < best)) { best_rate = value; best = ids_[i]; }
    }
    return best;
  }
  // KeyedSample over the tokens that `legal` takes (structured output, engine/runtime/constraint.h), with the same
  // rates and tie rule, so the result equals a full scan of the legal support: when the unrestricted winner is legal,
  // it is this result too. -1 when no token of the support is legal. KeyedSample itself is unchanged.
  template <class Legal>
  int KeyedSampleWhere(uint64_t seed, uint64_t position, Legal&& legal) const {
    const auto rate = [&](size_t i) { return probabilities_[i] / -std::log(KeyedUniform(seed, position, ids_[i])); };
    size_t top = ids_.size();
    for (size_t i = 0; i < ids_.size(); ++i)
      if (legal(ids_[i]) && (top == ids_.size() || probabilities_[i] > probabilities_[top])) top = i;
    if (top == ids_.size()) return -1;
    double best_rate = rate(top);
    int best = ids_[top];
    const double floor = best_rate * kMinExponential * 0.5;
    for (size_t i = 0; i < ids_.size(); ++i) {
      if (i == top || probabilities_[i] < floor || !legal(ids_[i])) continue;
      const double value = rate(i);
      if (value > best_rate || (value == best_rate && ids_[i] < best)) { best_rate = value; best = ids_[i]; }
    }
    return best;
  }
  // The smallest exponential noise: -log U at the largest KeyedUniform output.
  static inline const double kMinExponential = -std::log((double((uint64_t(1) << 52) - 1) + 0.5) * 0x1p-52);
  const std::vector<int>& Ids() const { return ids_; }
  const std::vector<double>& Probabilities() const { return probabilities_; }

 private:
  std::vector<int> ids_;
  std::vector<double> probabilities_;
};

struct Candidate { int token; double value; };

inline bool Higher(const Candidate& a, const Candidate& b) {
  return a.value > b.value || (a.value == b.value && a.token < b.token);
}

inline void Nucleus(std::vector<Candidate>& candidates, double total, double top_p) {
  constexpr size_t kInitialNucleus = 64;
  const double target = top_p * total;
  size_t window = std::min(kInitialNucleus, candidates.size());
  for (;;) {
    if (window < candidates.size())
      std::nth_element(candidates.begin(), candidates.begin() + window, candidates.end(), Higher);
    std::sort(candidates.begin(), candidates.begin() + window, Higher);
    double prefix = 0;
    size_t keep = 0;
    do { prefix += candidates[keep++].value; }
    while (keep < window && prefix < target);
    if (prefix >= target || window == candidates.size()) { candidates.resize(keep); return; }
    // A broad distribution should fall back to one full sort, not many scans.
    const double estimated = std::ceil(double(window) * target / prefix);
    window = size_t(std::min(double(candidates.size()), std::max(double(window * 4), estimated)));
  }
}

// The greedy head (ArgmaxKernel, layers/ops.h) writes a negative ID for a row with a NaN or +Inf logit.
// A target row must end the turn with the error of the sampled path, never with a normal token.
inline void RequireFiniteHeadIds(std::span<const int> ids) {
  for (const int id : ids)
    if (id < 0) throw std::invalid_argument("nonfinite model logits");
}

// A draft only proposes: its row with a NaN or +Inf logit proposes token 0, so a request's output
// stays the same with speculation on and off.
inline void ReplaceNonfiniteDraftIds(std::span<int> ids) {
  for (int& id : ids)
    if (id < 0) id = 0;
}

inline std::vector<Candidate> AllCandidates(std::span<const float> logits, std::span<const int> mapping) {
  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity())
      throw std::invalid_argument("nonfinite model logits");
    if (!std::isfinite(logits[i])) continue;
    const Candidate item{mapping.empty() ? int(i) : mapping[i], logits[i]};
    if (item.token < 0) throw std::invalid_argument("negative token mapping");
    candidates.push_back(item);
  }
  return candidates;
}

// The exact top-k under Higher in one pass, sorted by Higher. Most logits fall below the current k-th
// candidate and cost one compare; the full candidate list and its selection cost about 2.4 ms per row.
// NaN and +inf never compare below the floor, so they still reach the nonfinite check.
inline std::vector<Candidate> TopCandidates(std::span<const float> logits, std::span<const int> mapping, size_t k) {
  for (const int token : mapping)
    if (token < 0) throw std::invalid_argument("negative token mapping");
  std::vector<Candidate> heap;
  heap.reserve(k);
  float floor = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < logits.size(); ++i) {
    const float value = logits[i];
    if (value < floor) continue;
    if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
      throw std::invalid_argument("nonfinite model logits");
    if (!std::isfinite(value)) continue;
    const Candidate item{mapping.empty() ? int(i) : mapping[i], value};
    // A heap ordered by Higher keeps its lowest candidate at the front.
    if (heap.size() < k) {
      heap.push_back(item);
      std::push_heap(heap.begin(), heap.end(), Higher);
    } else if (Higher(item, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), Higher);
      heap.back() = item;
      std::push_heap(heap.begin(), heap.end(), Higher);
    }
    if (heap.size() == k) floor = float(heap.front().value);
  }
  std::sort(heap.begin(), heap.end(), Higher);
  return heap;
}

// Filter order: temperature, top-k, top-p, min-p, then final normalization.
// top-p keeps the crossing token. Ties use the lower global token ID.
inline Distribution FromLogits(std::span<const float> logits, const Options& options,
                               std::span<const int> mapping = {}) {
  if (logits.empty() || (!mapping.empty() && mapping.size() != logits.size()))
    throw std::invalid_argument("empty logits or mismatched token mapping");
  if (const std::string error = options.Error(std::numeric_limits<int>::max()); !error.empty())
    throw std::invalid_argument(error);
  const bool fast_top_k = options.temperature > 0 && options.top_k > 0 && size_t(options.top_k) < logits.size();
  std::vector<Candidate> candidates = fast_top_k ? TopCandidates(logits, mapping, size_t(options.top_k))
                                                 : AllCandidates(logits, mapping);
  if (candidates.empty()) throw std::invalid_argument("model logits have no finite value");
  const Candidate best = fast_top_k ? candidates.front()
                                    : *std::min_element(candidates.begin(), candidates.end(), Higher);
  if (options.temperature == 0) return Distribution::Point(best.token);
  if (options.top_k > 0 && size_t(options.top_k) < candidates.size()) {
    std::nth_element(candidates.begin(), candidates.begin() + options.top_k, candidates.end(), Higher);
    candidates.resize(options.top_k);
  }
  double sum = 0;
  for (Candidate& item : candidates) {
    item.value = std::exp((item.value - best.value) / options.temperature);
    sum += item.value;
  }
  if (options.top_p < 1) Nucleus(candidates, sum, options.top_p);
  std::vector<Candidate> kept;
  kept.reserve(candidates.size());
  for (const Candidate& item : candidates)
    if (item.value > 0 && item.value >= options.min_p) kept.push_back(item);
  const auto by_token = [](const Candidate& a, const Candidate& b) { return a.token < b.token; };
  if (!std::is_sorted(kept.begin(), kept.end(), by_token)) std::sort(kept.begin(), kept.end(), by_token);
  std::vector<int> ids;
  std::vector<double> weights;
  ids.reserve(kept.size()); weights.reserve(kept.size());
  for (const Candidate& item : kept) { ids.push_back(item.token); weights.push_back(item.value); }
  return Distribution(std::move(ids), std::move(weights));
}

inline Distribution Residual(const Distribution& p, const Distribution& q) {
  std::vector<int> ids;
  std::vector<double> weights;
  const auto& pi = p.Ids();
  const auto& qi = q.Ids();
  const auto& pp = p.Probabilities();
  const auto& qp = q.Probabilities();
  size_t j = 0;
  for (size_t i = 0; i < pi.size(); ++i) {
    while (j < qi.size() && qi[j] < pi[i]) ++j;
    const double other = j < qi.size() && qi[j] == pi[i] ? qp[j] : 0;
    const double mass = std::max(0., pp[i] - other);
    if (mass > 0) { ids.push_back(pi[i]); weights.push_back(mass); }
  }
  return Distribution(std::move(ids), std::move(weights));
}

struct Decision { int token; bool accepted; };

inline Decision Verify(const Distribution& p, const Distribution& q, int proposed, Rng& rng) {
  const double proposal_probability = q.Probability(proposed);
  if (!(proposal_probability > 0)) throw std::invalid_argument("proposal has no draft probability");
  const double acceptance = std::min(1., p.Probability(proposed) / proposal_probability);
  if (rng.Unit() < acceptance) return {proposed, true};
  return {Residual(p, q).Sample(rng), false};
}

}  // namespace crucible::sampling
