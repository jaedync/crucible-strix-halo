// Host top-N log-softmax for the opt-in scoring position. CPU only.
#include "../engine/serve/logprobs.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

int main() {
  const std::vector<float> logits{1.f, 3.f, 3.f, -2.f, 0.5f, 10.f};
  double sum = 0;
  for (float x : logits) sum += std::exp(double(x) - 10.0);
  const double lse = 10.0 + std::log(sum);
  const auto top = crucible::TopLogprobs(logits, 4);
  assert(top.size() == 4);
  const int want[4] = {5, 1, 2, 0};  // a tie keeps the lower ID first
  for (int i = 0; i < 4; ++i) {
    assert(top[i].id == want[i]);
    assert(std::fabs(double(top[i].logprob) - (double(logits[want[i]]) - lse)) < 1e-6);
  }
  assert(crucible::TopLogprobs(logits, 20).size() == logits.size());  // fewer logits than N: all of them
  // A huge shift must not overflow: the result is shift invariant.
  std::vector<float> shifted(logits); for (auto& x : shifted) x += 80.f;
  assert(std::fabs(crucible::TopLogprobs(shifted, 1)[0].logprob - top[0].logprob) < 1e-4);
  auto bad = logits; bad[2] = std::numeric_limits<float>::quiet_NaN();
  try { crucible::TopLogprobs(bad, 2); assert(false); } catch (const std::runtime_error&) {}
  try { crucible::TopLogprobs(logits, 0); assert(false); } catch (const std::invalid_argument&) {}
  return 0;
}
