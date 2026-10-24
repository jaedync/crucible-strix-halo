#include "../engine/runtime/sampling_generation.h"
#include "../engine/runtime/retained_prefix.h"
#include <cassert>
#include <cstring>
#include <array>
#include <cstdio>

using namespace crucible;

struct Metrics {
  enum Kind { kVerify, kMtpFirst, kMtpChain };
  void Begin() {} void End(Kind, int) {} void Margin(int, float) {}
  void Accepted(int, int) {} void StepDone() {}
};

struct Engine {
  int context = 64, resets = 0, mtp_base = 0, mtp_resets = 0;
  void ResetMtp() { ++mtp_resets; }
  bool sparse = false, identical = false;
  float stale_margin = 2;
  std::vector<int> history;
  std::vector<float> hidden = std::vector<float>(kMaxRows * kWide);
  std::vector<float> target, draft;
  std::vector<std::pair<size_t, std::vector<int>>> verifies;  // (history length before, rows) of each Forward
  RetainedPrefix retained;
  int Context() const { return context; }
  int Vocabulary() const { return 3; }
  size_t Length() const { return history.size(); }
  float* H() { return hidden.data(); }
  float DraftMargin() const { return stale_margin; }
  void ClearExpertCounts() {} void EnableProfile() {} void ObserveDraft(const int*, int) {}
  void Synchronize() {}
  void Reset() { ++resets; history.clear(); retained.Reset(); }
  void Truncate(size_t n) { assert(retained.Truncate(n).empty()); history.resize(n); }
  static std::array<float, 3> Logits(int token, bool proposal) {
    const std::array<float, 3> p = token == 0 ? std::array<float, 3>{1.1, 0.2, -0.8} :
        token == 1 ? std::array<float, 3>{-0.3, 0.2, 1.4} : std::array<float, 3>{0.5, 0.8, 0.5};
    return proposal ? std::array<float, 3>{p[2], p[0], p[1]} : p;
  }
  std::vector<int> Forward(const int* tokens, int rows, int replay, int commit, const std::string& = "") {
    assert(retained.Forward(rows, replay, commit).empty());
    verifies.emplace_back(history.size(), std::vector<int>(tokens, tokens + rows));
    history.insert(history.end(), tokens, tokens + rows);
    target.clear();
    for (int r = 0; r < rows; ++r) {
      const auto row = Logits(tokens[r], false);
      target.insert(target.end(), row.begin(), row.end());
      hidden[r * kWide] = tokens[r];
    }
    return std::vector<int>(rows, 0);
  }
  std::vector<int> Mtp(const float* stacks, const int* tokens, int rows, int mpos, bool score = true) {
    for (int r = 0; r < rows && size_t(mtp_base + mpos + r) < history.size(); ++r)
      assert(stacks[r * kWide] == history[mtp_base + mpos + r]);
    for (int r = 0; r < rows; ++r) hidden[r * kWide] = tokens[r];
    if (score) {
      const auto row = Logits(tokens[rows - 1], !identical);
      draft = sparse ? std::vector<float>{row[0], row[2]} : std::vector<float>(row.begin(), row.end());
    }
    return std::vector<int>(rows, 0);
  }
  std::vector<float> CopyLogits(int rows, int first = 0) const {
    assert(target.size() >= size_t(rows + first) * 3);
    return {target.begin() + first * 3, target.begin() + (first + rows) * 3};
  }
  std::pair<std::vector<float>, std::vector<int>> DraftLogits(int) const {
    return {draft, sparse ? std::vector<int>{0, 2} : std::vector<int>{}};
  }
};

int main() {
  assert(SampledRawMargin(std::array<float, 3>{2, 2, -1}) == 0);
  assert(SampledRawMargin(std::array<float, 3>{-INFINITY, 2, -1}) == 3);
  assert(std::isinf(SampledRawMargin(std::array<float, 1>{2})));
  Engine engine;
  Metrics metrics;
  std::vector<float> stacks((kMaxRows + 1) * kWide), carry(kWide);
  const auto copy = [](float* dst, const float* src, size_t n) { std::memcpy(dst, src, n * sizeof(float)); };
  SampledGenerationOptions options;
  options.sampling.temperature = 0.7;
  options.sampling.seed = 87326;
  options.n = 24; options.chunk = 2;
  const std::vector<int> prompt{0, 1, 2, 0, 1, 2};
  std::vector<int> emitted;
  auto emit = [&](int token) { emitted.push_back(token); };
  // Keyed law: the depth, spec=0, the draft head, and draft_q change only speed, never the sampled IDs.
  std::vector<int> keyed_reference;
  for (int spec : {0, 1, 2, 7}) {
    for (bool sparse : {false, true}) {
      engine.sparse = sparse; options.spec = spec;
      emitted.clear();
      const auto first = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, emit, metrics);
      assert(first.ids == emitted && first.ids.size() == size_t(options.n));
      if (keyed_reference.empty()) keyed_reference = first.ids;
      assert(first.ids == keyed_reference);
      const auto second = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
      assert(first.ids == second.ids);
    }
  }
  options.spec = 7; options.draft_q = 0.3; engine.sparse = false;
  engine.stale_margin = -100;
  const auto low_stale = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  engine.stale_margin = 100;
  const auto high_stale = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(low_stale.ids == high_stale.ids && low_stale.attempted == high_stale.attempted);
  assert(low_stale.ids == keyed_reference);
  for (const float q : {0.f, 0.1f, 0.9f}) {
    options.draft_q = q; options.draft_margin = q ? 0 : 0.5f;
    assert(GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics).ids == keyed_reference);
  }
  options.draft_q = 0.3; options.draft_margin = 0;
  options.draft_q = 0;
  options.lookup = 7; options.lookup_min = 1; options.spec = 7; options.n = 50;
  const auto lookup = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(lookup.lookup_steps > 0);
  options.stop_ids = {lookup.ids[0]}; emitted.clear();
  const auto stop = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, emit, metrics);
  assert(stop.stopped && stop.ids.size() == 1 && emitted == stop.ids);
  options.stop_ids.clear();
  {
    // Draft stop (draft_stop.h): no verify row follows a stop ID or needs a token past the limit, and the IDs equal
    // the run without the stop for as long as it lasts. Copies (lookup) and MTP chains both pass here.
    for (const int stop_id : {0, 1, 2}) {
      for (const int n : {2, 3, 5, 9, 50}) {
        options.n = n; options.lookup = 7; options.spec = 7; options.stop_ids = {};
        const auto plain = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
        options.stop_ids = {stop_id};
        engine.verifies.clear();
        const auto cut = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
        size_t keep = plain.ids.size();
        for (size_t i = 0; i < plain.ids.size(); ++i) if (plain.ids[i] == stop_id) { keep = i + 1; break; }
        assert(cut.ids == std::vector<int>(plain.ids.begin(), plain.ids.begin() + keep));
        for (const auto& [before, rows] : engine.verifies) {
          if (before < prompt.size()) continue;  // prompt prefill
          const size_t committed = before - prompt.size() + 1;
          assert(rows.size() <= size_t(n) - committed);
          for (size_t i = 0; i + 1 < rows.size(); ++i) assert(rows[i] != stop_id);
        }
      }
    }
    options.stop_ids.clear();
  }
  std::vector<std::vector<float>> trace;
  options.n = 24;
  options.target_trace = [&](std::span<const float> logits) { trace.emplace_back(logits.begin(), logits.end()); };
  const auto audited = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(trace.size() == audited.ids.size());
  engine.Reset();
  int last_rows = 0;
  for (size_t i = 0; i < prompt.size(); i += options.chunk) {
    last_rows = int(std::min<size_t>(options.chunk, prompt.size() - i));
    engine.Forward(prompt.data() + i, last_rows, 0, last_rows);
  }
  assert(engine.CopyLogits(1, last_rows - 1) == trace[0]);
  for (size_t i = 1; i < audited.ids.size(); ++i) {
    engine.Forward(&audited.ids[i - 1], 1, 0, 1);
    assert(engine.CopyLogits(1) == trace[i]);
  }
  options.target_trace = {};
  engine.context = 9; options.n = 4;
  const auto boundary = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(boundary.ids.size() == 4);
  const int resets = engine.resets;
  try { GenerateSampled(engine, {}, options, stacks.data(), carry.data(), copy, [](int) {}, metrics); assert(false); }
  catch (const std::runtime_error&) { assert(engine.resets == resets); }
  engine.context = 64;
  options.lookup = 0; options.spec = 2; options.n = 9; options.stop_ids.clear();
  struct Snapshot { Engine engine; ConversationResume resume; std::vector<float> stacks, carry; };
  std::optional<Snapshot> saved;
  ConversationHooks hooks;
  hooks.boundary = 4;
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    if (state.length == 4)
      saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  options.conversation = &hooks;
  GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(saved && saved->resume.length == 4 && saved->resume.mpos == 3);
  const std::vector<int> appended{0,1,2,0,1,2,1,0};
  hooks.save = {};
  const auto reference = GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine;
    copy(s, saved->stacks.data(), saved->stacks.size());
    copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  const auto restored = GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(reference.ids == restored.ids);
  // Exact prompt reuse must sample saved target logits with a fresh request RNG.
  hooks.restore = {};
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    if (state.length == prompt.size())
      saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  const auto same_reference = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  hooks.save = {};
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine;
    copy(s, saved->stacks.data(), saved->stacks.size());
    copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  const auto same_restored = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(same_reference.ids == same_restored.ids);
  hooks.restore = {};
  engine.identical = true; engine.sparse = false; options.n = 3;
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    if (state.length > prompt.size())
      saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(saved->resume.length > prompt.size());
  std::vector<int> next = saved->engine.history;
  next.insert(next.end(), {1,2,0,1});
  hooks.save = {};
  hooks.restore = [&](float*, float*) { return saved->resume; };
  const auto continued_tail = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine;
    copy(s, saved->stacks.data(), saved->stacks.size());
    copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  const auto restored_tail = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(continued_tail.ids == restored_tail.ids);
  // Session replay (G15): when the warm-up window reaches into the restored tail, a generated-tail restore is refused unless the hooks mark it
  // as a continued tail. Then the MTP state continues, and the IDs equal the unbounded continuation (drafts change
  // only speed).
  const int saved_prefix = options.mtp_prefix;
  options.mtp_prefix = 6;  // longer than the 4-token suffix, so the tail overlaps the warm-up window
  try { GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics); assert(false); }
  catch (const std::runtime_error&) {}
  hooks.continue_tail = true;
  const auto replayed_tail = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(replayed_tail.ids == restored_tail.ids);
  // A tail without MTP state (saved by a spec 0 turn, or by batch serve) restarts the MTP warm-up at the restored
  // length: a reset, no overlap error, the same IDs.
  {
    const auto plain_restore = hooks.restore;
    hooks.restore = [&](float* s, float* c) {
      const auto state = ConversationTargetResume(plain_restore(s, c));
      engine.mtp_base = int(state.length);  // MTP position 0 pairs the first restored-suffix row
      return state;
    };
    const int resets = engine.mtp_resets;
    const auto plain_tail = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
    assert(plain_tail.ids == restored_tail.ids && engine.mtp_resets == resets + 1);
    hooks.restore = plain_restore;
    engine.mtp_base = 0;
  }
  // A suffix longer than the window takes the usual fresh MTP warm-up, with the same IDs.
  options.mtp_prefix = 2;
  assert(GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics).ids ==
         restored_tail.ids);
  hooks.continue_tail = false;
  options.mtp_prefix = saved_prefix;
  hooks.restore = {}; hooks.save = {};
  options.n = 7;
  std::vector<int> stop_reference;
  for (uint64_t seed = 1; seed < 64; ++seed) {
    options.sampling.seed = seed;
    stop_reference = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics).ids;
    if (stop_reference[1] != stop_reference[0]) break;
  }
  assert(stop_reference[1] != stop_reference[0]);
  options.stop_ids = {stop_reference[1]};
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    if (state.length > prompt.size())
      saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  const auto middle_stop = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(middle_stop.stopped && middle_stop.ids.size() == 2);
  assert(saved->resume.length == prompt.size() + 1 && saved->resume.replay == 0);
  assert(saved->resume.pending_tokens.size() == 1 && saved->resume.next_id == middle_stop.ids.back());
  next = saved->engine.history;
  next.insert(next.end(), {middle_stop.ids.back(), 2, 0});
  options.stop_ids.clear(); hooks.save = {};
  hooks.restore = [&](float*, float*) { return saved->resume; };
  const auto stop_continued = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine;
    copy(s, saved->stacks.data(), saved->stacks.size());
    copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  const auto stop_restored = GenerateSampled(engine, next, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(stop_continued.ids == stop_restored.ids);
  hooks.restore = {};
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  engine.identical = false; options.n = 9;
  for (uint64_t seed = 0; seed < 8; ++seed) {
    options.sampling.seed = seed;
    GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
    assert(saved->resume.length == engine.Length());
    assert(saved->resume.length == prompt.size() + options.n - 1);
    assert(saved->resume.mpos + saved->resume.pending_tokens.size() == saved->resume.length);
  }
  hooks.restore = {}; hooks.boundary = 4;
  options.mtp_prefix = 2; options.n = 9;
  engine.mtp_base = int(prompt.size()) - 2;
  hooks.save = [&](const ConversationResume& state, float* s, float* c) {
    if (state.length == 4)
      saved = Snapshot{engine, state, {s, s + stacks.size()}, {c, c + carry.size()}};
  };
  GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(saved->resume.length == 4 && saved->resume.mpos == 0);
  hooks.save = {};
  engine.mtp_base = int(appended.size()) - 2;
  const auto capped_ref = GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  // Old proposal metadata must not survive a new relative window.
  saved->resume.mpos = 3; saved->resume.have_carry = true; saved->resume.pending_tokens = {1, 2};
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine; engine.mtp_base = int(appended.size()) - 2;
    copy(s, saved->stacks.data(), saved->stacks.size()); copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  for (uint64_t seed = 0; seed < 32; ++seed) {
    options.sampling.seed = seed;
    hooks.restore = {};
    engine.mtp_base = int(appended.size()) - 2;
    const auto fresh = GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
    hooks.restore = [&](float* s, float* c) {
      engine = saved->engine; engine.mtp_base = int(appended.size()) - 2;
      copy(s, saved->stacks.data(), saved->stacks.size()); copy(c, saved->carry.data(), saved->carry.size());
      return saved->resume;
    };
    const auto hit = GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
    assert(fresh.ids == hit.ids && engine.mtp_resets == saved->engine.mtp_resets + 1);
  }
  saved->resume.length = appended.size() - 1;
  try { GenerateSampled(engine, appended, options, stacks.data(), carry.data(), copy, [](int) {}, metrics); assert(false); }
  catch (const std::runtime_error&) {}
  hooks.restore = {}; hooks.boundary = 4;
  hooks.defer_prompt_end_save = true; hooks.skip_prompt_interior_save = true;
  options.n = 9; options.spec = 2; options.mtp_prefix = 0;
  const auto streamed_fresh = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  engine.mtp_base = 0;
  int writes = 0;
  hooks.save_generated_tail = false;
  bool token_sent = false;
  hooks.save = [&](const ConversationResume& state, float*, float*) {
    assert(state.prompt_only);
    assert(token_sent && state.length == prompt.size());
    assert(engine.Length() == prompt.size());
    assert(state.mpos == int(prompt.size()) - 1 && state.replay == 0 && state.have_carry && state.pending_tokens.empty());
    const auto expected = Engine::Logits(prompt.back(), false);
    assert(state.prompt_boundary == 4 && state.logits == std::vector<float>(expected.begin(), expected.end()));
    ++writes;
    saved = Snapshot{engine, state, {stacks.begin(), stacks.end()}, {carry.begin(), carry.end()}};
  };
  const auto delayed = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy,
      [&](int) { if (!token_sent) assert(writes == 0); else assert(writes == 1); token_sent = true; }, metrics);
  assert(writes == 1 && delayed.ids.size() == 9 && delayed.ids == streamed_fresh.ids);
  hooks.save = {};
  hooks.restore = [&](float* s, float* c) {
    engine = saved->engine;
    copy(s, saved->stacks.data(), saved->stacks.size());
    copy(c, saved->carry.data(), saved->carry.size());
    return saved->resume;
  };
  const auto delayed_hit = GenerateSampled(engine, prompt, options, stacks.data(), carry.data(), copy, [](int) {}, metrics);
  assert(delayed.ids == delayed_hit.ids);
  assert(!hooks.deferred_prompt);
  std::puts("PASS: sampled restore, pending replay, streamed saves, stops, and 32 seeded bounded-MTP comparisons");
}
