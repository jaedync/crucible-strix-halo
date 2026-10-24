// The opt-in scoring position: the top-N log-probabilities of one logits row, computed on the host.
// It reads the row that the sampler already copies, so no device code changes and the sampled token is untouched.
#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace crucible {
struct TokenLogprob {
  int id = 0;
  float logprob = 0;
};
inline constexpr int kMaxTopLogprobs = 20;

// The `n` largest entries of log_softmax(logits), best first. A tie keeps the lower ID first.
inline std::vector<TokenLogprob> TopLogprobs(std::span<const float> logits, int n) {
  if (n < 1 || n > kMaxTopLogprobs) throw std::invalid_argument("top_logprobs must be 1 to 20");
  float peak = -INFINITY;
  for (const float x : logits) {
    if (!std::isfinite(x)) throw std::runtime_error("logits are not finite");
    peak = std::max(peak, x);
  }
  double sum = 0;
  for (const float x : logits) sum += std::exp(double(x) - double(peak));
  const double lse = double(peak) + std::log(sum);
  std::vector<int> order(logits.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  const size_t count = std::min(size_t(n), order.size());
  std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
  });
  std::vector<TokenLogprob> top;
  for (size_t i = 0; i < count; ++i) top.push_back({order[i], float(double(logits[order[i]]) - lse)});
  return top;
}
}  // namespace crucible
