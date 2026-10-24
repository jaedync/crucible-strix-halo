#include "../engine/runtime/generation.h"
#include "../engine/runtime/retained_prefix.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
using namespace crucible;
struct FakeEngine {
  int ctx = 64, resets = 0, mtp_calls = 0;
  bool reject_draft = false;
  std::vector<int> hist;
  std::vector<float> hidden = std::vector<float>(kMaxRows * kWide);
  RetainedPrefix retained;
  void Reset() { ++resets; hist.clear(); retained.Reset(); }
  void EnableProfile() {}
  int Context() { return ctx; }
  float* H() { return hidden.data(); }
  size_t Length() { return hist.size(); }
  void Truncate(size_t n) { assert(retained.Truncate(n).empty()); hist.resize(n); }
  std::vector<int> Forward(const int* t, int R, int replay, int commit, const std::string& = "") {
    assert(retained.Forward(R, replay, commit).empty());
    hist.insert(hist.end(), t, t + R);
    std::vector<int> ids;
    for (int r = 0; r < R; ++r) ids.push_back(t[r] + 1);
    return ids;
  }
  std::vector<int> Mtp(const float*, const int* t, int R, int) {
    ++mtp_calls;
    std::vector<int> ids;
    for (int r = 0; r < R; ++r) ids.push_back(t[r] + (reject_draft ? 2 : 1));
    return ids;
  }
};
// A GEMM-prefill engine: Prefill commits R rows and predicts only the last.
struct FakePrefillEngine : FakeEngine {
  int prefill_rows = 5, prefill_calls = 0, mtp_rows = 0;
  std::vector<float> stacks = std::vector<float>(5 * kWide);
  int PrefillRows() const { return prefill_rows; }
  float* PromptStacks() { return stacks.data(); }
  void CopyPrefillStacks(float* dst, int first, int rows) {
    std::memcpy(dst, hidden.data() + (size_t)first * kWide, (size_t)rows * kWide * sizeof(float));
  }
  std::vector<int> Prefill(const int* t, int R, const std::string&, bool predict, const void* = nullptr, int = 0) {
    ++prefill_calls;
    assert(retained.Forward(R, 0, R).empty());
    hist.insert(hist.end(), t, t + R);
    return predict ? std::vector<int>{t[R - 1] + 1} : std::vector<int>{};
  }
  std::vector<int> Mtp(const float* s, const int* t, int R, int pos) {
    assert(R >= 1 && R <= kMaxRows);
    mtp_rows += R;
    return FakeEngine::Mtp(s, t, R, pos);
  }
};
static_assert(HasGemmPrefill<FakePrefillEngine> && !HasGemmPrefill<FakeEngine>);

// Records the rows of each Forward call (the verify rows) for the draft-stop checks.
struct RecordingEngine : FakeEngine {
  std::vector<std::vector<int>> calls;
  std::vector<int> Forward(const int* t, int R, int replay, int commit, const std::string& dump = "") {
    calls.emplace_back(t, t + R);
    return FakeEngine::Forward(t, R, replay, commit, dump);
  }
};

int main() {
  std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
  auto copy = [](float* d, const float* s, size_t n) { std::memcpy(d, s, n * sizeof(float)); };
  FakeEngine e;
  const std::vector<int> prompt{1,2,3};
  std::vector<int> emitted;
  auto emit = [&](int id) { emitted.push_back(id); };
  const auto serial = GenerateExact(e, prompt, 20, 8, 0, {}, "", stk.data(), carry.data(), copy, emit);
  assert(serial.ids == emitted && serial.ids.size() == 20);
  for (const int spec : {1,2,7}) {
    for (const bool rejection : {false,true}) {
      e.reject_draft = rejection;
      emitted.clear();
      const auto result = GenerateExact(e, prompt, 20, 2, spec, {}, "", stk.data(), carry.data(), copy, emit);
      assert(result.ids == serial.ids && emitted == serial.ids);
    }
  }
  emitted.clear();
  const std::vector<int> stop{8};
  e.reject_draft = false;
  const auto stopped = GenerateExact(e, prompt, 20, 8, 2, stop, "", stk.data(), carry.data(), copy, emit);
  assert(stopped.stopped && stopped.ids.size() == 5 && stopped.ids.back() == 8);
  assert(emitted == stopped.ids);
  e.ctx = 7;
  const auto near_end = GenerateExact(e, prompt, 5, 8, 2, {}, "", stk.data(), carry.data(), copy, [](int){});
  assert(near_end.ids == stopped.ids);
  const std::vector<int> one{5};
  const auto first_stop = GenerateExact(e, prompt, 1, 8, 0, one, "", stk.data(), carry.data(), copy, [](int){});
  assert(first_stop.ids == std::vector<int>{4});
  assert(e.resets == 10);
  {
    // 12 prompt rows in Prefill chunks of 5, 5, 2: same IDs as Forward, and
    // MTP sees the 11 (stack, next token) pairs in pieces of at most kMaxRows.
    const std::vector<int> longer{1,2,3,4,5,6,7,8,9,10,11,12};
    FakeEngine plain;
    const auto want = GenerateExact(plain, longer, 20, 8, 0, {}, "", stk.data(), carry.data(), copy, [](int){});
    for (const int spec : {0, 2}) {
      FakePrefillEngine p;
      const auto got = GenerateExact(p, longer, 20, 8, spec, {}, "", stk.data(), carry.data(), copy, [](int){});
      assert(got.ids == want.ids && p.prefill_calls == 3);
      if (spec) assert(p.mtp_rows >= 11);
      else assert(p.mtp_rows == 0);
    }
  }
  {
    // Draft stop (draft_stop.h): the fake engine drafts the true continuation 5, 6, 7, 8, 9, ... after the first token 4.
    // A verify row never follows a stop ID, and no row passes the token limit; the committed IDs equal serial IDs.
    e.ctx = 64; e.reject_draft = false;
    RecordingEngine at_stop;
    const auto cut = GenerateExact(at_stop, prompt, 20, 8, 7, stop, "", stk.data(), carry.data(), copy, [](int){});
    assert(cut.ids == stopped.ids);
    assert((at_stop.calls.back() == std::vector<int>{4, 5, 6, 7, 8}));
    for (const auto& call : at_stop.calls)
      for (size_t i = 0; i + 1 < call.size(); ++i) assert(call[i] != 8);
    for (const int limit : {2, 3, 4, 9}) {
      RecordingEngine budget;
      const auto got = GenerateExact(budget, prompt, limit, 8, 7, {}, "", stk.data(), carry.data(), copy, [](int){});
      assert(got.ids == std::vector<int>(serial.ids.begin(), serial.ids.begin() + limit));
      assert(int(budget.calls.back().size()) == std::min(limit - 1, 8));
    }
  }
  try {
    GenerateExact(e, {}, 1, 8, 0, {}, "", stk.data(), carry.data(), copy, [](int){});
    assert(false);
  } catch (const std::runtime_error&) {
    assert(e.resets == 10);
  }
}
