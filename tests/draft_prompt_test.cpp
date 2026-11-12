#include "../engine/runtime/generation.h"
#include "../engine/runtime/draft_warmup.h"
#include "../engine/runtime/conversation_admission.h"
#include "../engine/runtime/prefill_interleave.h"
#include <cassert>
#include <numeric>
using namespace crucible;

struct PromptEngine {
  int prefill = 0, length = 0, launch = 0;
  mutable int logit_copies = 0;
  std::vector<float> h, snapshots;
  std::vector<int> chunks;  // rows of every Forward or Prefill call, in order
  std::vector<bool> bulk;   // per chunk: true for Prefill (the bulk path), false for Forward (the decode path)
  bool in_prefill = false;
  int pruned = 0;  // final trunk stack rows that Prefill was allowed to skip
  explicit PromptEngine(int p, int l = 0) : prefill(p), launch(l),
      h((size_t)std::max({p, l, kMaxRows}) * kWide), snapshots(h.size()) {}
  int PrefillRows() const { return prefill; }
  int PrefillLaunchRows() const { return launch ? launch : prefill; }
  size_t Length() const { return size_t(length); }
  float* PromptStacks() { return snapshots.data(); }
  float* H() { return h.data(); }
  std::vector<float> CopyLogits(int, int) const { ++logit_copies; return {float(length)}; }
  std::vector<int> Forward(const int* tokens, int rows, int, int, const std::string&) {
    chunks.push_back(rows);
    bulk.push_back(in_prefill);
    for (int r = 0; r < rows; ++r) {
      assert(tokens[r] == length + r + 1);
      h[(size_t)r * kWide] = tokens[r] * 3 + 1;
      h[(size_t)(r + 1) * kWide - 1] = tokens[r] * 5 + 2;
    }
    length += rows;
    return {tokens[rows - 1] + 1};
  }
  // Rows below keep have no final trunk stacks (last-layer pruning); the head row stays when predict is set.
  // As in the engine, a bulk chunk's stacks stay in a padded buffer; H() does not hold them.
  std::vector<int> Prefill(const int* tokens, int rows, const std::string& dump, bool predict,
                           const PrefillBetween* = nullptr, int keep = 0) {
    in_prefill = true;
    const auto result = Forward(tokens, rows, 0, rows, dump);
    in_prefill = false;
    const int skip = std::min(keep, predict ? rows - 1 : rows);
    for (int r = 0; r < skip; ++r) h[(size_t)r * kWide] = h[(size_t)(r + 1) * kWide - 1] = -7;
    pruned += std::max(skip, 0);
    padded.assign((size_t)rows * kPadded, -3);
    for (int r = 0; r < rows; ++r) {
      padded[(size_t)r * kPadded] = h[(size_t)r * kWide];
      padded[(size_t)r * kPadded + kWide - 1] = h[(size_t)(r + 1) * kWide - 1];
      h[(size_t)r * kWide] = h[(size_t)(r + 1) * kWide - 1] = -9;
    }
    return predict ? result : std::vector<int>{};
  }
  void CopyPrefillStacks(float* dst, int first, int rows) {
    assert(first >= 0 && rows >= 1 && (size_t)(first + rows) * kPadded <= padded.size());
    for (int r = 0; r < rows; ++r) {
      dst[(size_t)r * kWide] = padded[(size_t)(first + r) * kPadded];
      dst[(size_t)(r + 1) * kWide - 1] = padded[(size_t)(first + r) * kPadded + kWide - 1];
    }
  }
  static constexpr size_t kPadded = kWide + 64;
  std::vector<float> padded;
};

void CheckPrompt(PromptEngine& eng, int length, int chunk, int limit, bool mtp_on) {
  eng.length = 0;
  eng.pruned = 0;
  std::vector<int> ids(length);
  std::iota(ids.begin(), ids.end(), 1);
  std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
  int mpos = 0, paired = 0, finals = 0;
  const size_t start = DraftWarmupStart(ids.size(), limit);
  const auto copy = [](float* dst, const float* src, size_t count) {
    assert(count % kWide == 0);
    for (size_t r = 0; r < count / kWide; ++r) {
      dst[r * kWide] = src[r * kWide];
      dst[(r + 1) * kWide - 1] = src[(r + 1) * kWide - 1];
    }
  };
  const auto mtp = [&](float* stacks, const int* tokens, int count, int pos) {
    assert(count > 0 && count <= kMaxRows && pos == paired);
    for (int r = 0; r < count; ++r) {
      const int source = (int)start + pos + r + 1;
      assert(tokens[r] == source + 1);
      assert(stacks[(size_t)r * kWide] == source * 3 + 1);
      assert(stacks[(size_t)(r + 1) * kWide - 1] == source * 5 + 2);
    }
    paired += count;
    for (size_t r = 0; r < eng.h.size() / kWide; ++r)
      eng.h[r * kWide] = eng.h[(r + 1) * kWide - 1] = -1;
  };
  const auto final = [&](int row) {
    ++finals;
    assert(row == (eng.prefill ? 0 : (length - 1) % chunk));
  };
  const auto last = RunPrompt(eng, ids, chunk, mtp_on, "", stk.data(), carry.data(), mpos,
                              copy, mtp, final, limit);
  assert(last == std::vector<int>{length + 1} && eng.length == length && finals == 1);
  assert(mpos == paired && paired == (mtp_on ? length - (int)start - 1 : 0));
  if (mtp_on) assert(carry[0] == length * 3 + 1 && carry[kWide - 1] == length * 5 + 2);
  // Every bulk row that no MTP pair and no head reads is offered for pruning.
  if (eng.prefill) assert(eng.pruned == (mtp_on ? std::min((int)start, length - 1) : length - 1));
}

void CheckCachedBulk() {
  PromptEngine eng(16);
  std::vector<int> ids(64); std::iota(ids.begin(), ids.end(), 1);
  std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
  const auto copy = [](float* d, const float* s, size_t count) { std::copy_n(s, count, d); };
  // The opt-in split: an unaligned boundary snapshot exists and is incompatible with another boundary.
  ConversationHooks hooks; hooks.boundary = 21; hooks.split_at_boundary = true;
  std::vector<ConversationResume> saved;
  hooks.save = [&](const ConversationResume& state, float*, float*) { saved.push_back(state); };
  int mpos = 0;
  RunPrompt(eng, ids, 8, true, "", stk.data(), carry.data(), mpos, copy,
            [](float*, const int*, int, int) {}, [](int) {}, 8, {}, &hooks);
  const auto found = std::find_if(saved.begin(), saved.end(), [](const auto& r) { return r.length == 16; });
  assert(found != saved.end() && found->prompt_only && found->mpos == 0);
  assert(ConversationPromptCompatible(*found, 96, 70, 16, true));
  const auto unstable = std::find_if(saved.begin(), saved.end(), [](const auto& r) { return r.length == 21; });
  assert(unstable != saved.end() && !ConversationPromptCompatible(*unstable, 96, 70, 16, true));
  ids.resize(96); std::iota(ids.begin(), ids.end(), 1);
  eng.length = 16; hooks.save = {}; hooks.boundary = 70;
  auto resume = ConversationTargetResume(*found);
  mpos = 0;
  const auto got = RunPrompt(eng, ids, 8, true, "", stk.data(), carry.data(), mpos, copy,
      [&](float* stacks, const int* tokens, int count, int pos) {
        for (int r = 0; r < count; ++r) {
          const int source = 89 + pos + r;
          assert(tokens[r] == source + 1 && stacks[size_t(r) * kWide] == source * 3 + 1);
        }
      }, [](int) {}, 8, resume, &hooks);
  assert(got == std::vector<int>{97} && eng.length == 96 && mpos == 7);
}

// One canonical schedule (design decision, Oct 4): the chat boundary picks snapshot positions only. The prompt chunks stay
// on the zero-origin grid unless a caller opts in to the schedule-changing boundary split.
void CheckCanonicalSchedule() {
  for (const bool split : {false, true}) for (const int prefill : {0, 16}) {
    PromptEngine eng(prefill);
    std::vector<int> ids(64); std::iota(ids.begin(), ids.end(), 1);
    std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
    ConversationHooks hooks; hooks.boundary = 21; hooks.split_at_boundary = split;
    const int rows = prefill ? prefill : 8;
    const auto admission = ConversationAdmission::ForPrompt(ids.size(), hooks.boundary, rows, 0);
    hooks.admit_prompt = [admission](size_t n) { return admission.Allows(n); };
    std::vector<size_t> writes;
    hooks.save = [&](const ConversationResume& r, float*, float*) { writes.push_back(r.length); };
    int mpos = 0;
    RunPrompt(eng, ids, 8, false, "", stk.data(), carry.data(), mpos, [](float*, const float*, size_t) {},
              [](float*, const int*, int, int) {}, [](int) {}, 0, {}, &hooks);
    const std::vector<int> grid = prefill ? std::vector<int>{16, 16, 16, 16} : std::vector<int>(8, 8);
    const std::vector<int> cut = prefill ? std::vector<int>{16, 5, 16, 16, 11}
                                         : std::vector<int>{8, 8, 5, 8, 8, 8, 8, 8, 3};
    assert(eng.chunks == (split ? cut : grid));
    // The admitted frontier is a grid position, so both schedules save it.
    assert(writes == std::vector<size_t>{admission.frontier});
  }
}

// Session replay (G15): after a tail restore the first chunk is always one small-row decode-path Forward, with or
// without pending GDN rows. That count depends on the spec and packing schedule (load), not on the history, so it
// must not pick the path. A prompt-state restore keeps the bulk grid.
void CheckTailSchedule() {
  for (const bool tail : {true, false}) for (const int replay : {0, 2}) {
    if (!tail && replay) continue;
    PromptEngine eng(16); eng.length = 21;
    std::vector<int> ids(64); std::iota(ids.begin(), ids.end(), 1);
    std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
    ConversationHooks hooks; hooks.continue_tail = tail;
    ConversationResume resume; resume.length = 21; resume.replay = replay; resume.prompt_only = !tail;
    int mpos = 0;
    const auto got = RunPrompt(eng, ids, 8, false, "", stk.data(), carry.data(), mpos,
                               [](float*, const float*, size_t) {}, [](float*, const int*, int, int) {}, [](int) {},
                               0, resume, &hooks);
    assert(got == std::vector<int>{65} && eng.length == 64);
    if (tail) assert(eng.chunks == (std::vector<int>{8, 16, 16, 3}) &&
                     eng.bulk == (std::vector<bool>{false, true, true, true}));
    else assert(eng.chunks == (std::vector<int>{16, 16, 11}) && eng.bulk == (std::vector<bool>{true, true, true}));
  }
}

void CheckDeferredPrompt() {
  PromptEngine eng(16);
  std::vector<int> ids(12); std::iota(ids.begin(), ids.end(), 1);
  std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
  ConversationHooks hooks; hooks.boundary = 5;
  hooks.defer_prompt_end_save = true; hooks.skip_prompt_interior_save = true;
  int writes = 0, mpos = 0;
  bool published = false;
  hooks.save = [&](const ConversationResume& r, float* s, float* c) {
    assert(published && r.prompt_only && r.length == 12 && r.next_id == 13);
    assert(r.prompt_boundary == 5 && r.logits == std::vector<float>{12});
    assert(r.mpos == 0 && !r.have_carry && r.pending_tokens.empty());
    assert(s == stk.data() && c == carry.data());
    ++writes;
  };
  const auto last = RunPrompt(eng, ids, 8, false, "", stk.data(), carry.data(), mpos,
      [](float*, const float*, size_t) {}, [](float*, const int*, int, int) {}, [](int) {}, 0, {}, &hooks);
  assert(writes == 0 && last == std::vector<int>{13});
  published = true;
  FlushDeferredPrompt(&hooks, stk.data(), carry.data());
  assert(writes == 1 && !hooks.deferred_prompt);
  FlushDeferredPrompt(&hooks, stk.data(), carry.data());
  assert(writes == 1);
}

void CheckAdmission() {
  for (int prefill : {0, 16}) for (int length : {12, 48, 64}) {
    PromptEngine eng(prefill);
    std::vector<int> ids(length); std::iota(ids.begin(), ids.end(), 1);
    std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
    ConversationHooks hooks; hooks.boundary = size_t(length - 1);
    const auto admission = ConversationAdmission::ForPrompt(ids.size(), hooks.boundary, prefill ? prefill : 8, 0);
    hooks.admit_prompt = [admission](size_t n) { return admission.Allows(n); };
    std::vector<size_t> writes;
    hooks.save = [&](const ConversationResume& r, float*, float*) { writes.push_back(r.length); };
    int mpos = 0;
    const auto ids_out = RunPrompt(eng, ids, 8, false, "", stk.data(), carry.data(), mpos,
        [](float*, const float*, size_t) {}, [](float*, const int*, int, int) {}, [](int) {}, 8, {}, &hooks);
    assert(ids_out == std::vector<int>{length + 1});
    assert(writes == (admission.frontier ? std::vector<size_t>{admission.frontier} : std::vector<size_t>{}));
    assert(eng.logit_copies == int(writes.size()));
  }
}

// Merged bulk launches (prefill-moe): grid chunks merge up to the launch rows, but no launch crosses the MTP
// warm-up start, so the warm-up pairs (checked by CheckPrompt) are those of the grid schedule. A caller that saves
// snapshots keeps the grid.
void CheckMergedLaunch() {
  PromptEngine eng(16, 64);
  CheckPrompt(eng, 150, 8, 20, true);  // warm-up from 130
  assert((eng.chunks == std::vector<int>{64, 64, 16, 6}));
  eng.chunks.clear();
  CheckPrompt(eng, 150, 8, 0, false);
  assert((eng.chunks == std::vector<int>{64, 64, 16, 6}));  // F3: the 6-row tail is its own (small-row) launch
  eng.chunks.clear();
  CheckPrompt(eng, 150, 8, 0, true);  // warm-up from 0: the grid
  assert((eng.chunks == std::vector<int>{16, 16, 16, 16, 16, 16, 16, 16, 16, 6}));
  for (int length : {1, 15, 16, 17, 63, 64, 65, 129, 200, 1000})
    for (int limit : {0, 1, 8, 16, 64, 128, 256}) CheckPrompt(eng, length, 8, limit, true);
  eng.chunks.clear(); eng.length = 0;
  std::vector<int> ids(150); std::iota(ids.begin(), ids.end(), 1);
  std::vector<float> stk((kMaxRows + 1) * kWide), carry(kWide);
  ConversationHooks hooks;
  hooks.save = [](const ConversationResume&, float*, float*) {};
  int mpos = 0;
  RunPrompt(eng, ids, 8, false, "", stk.data(), carry.data(), mpos, [](float*, const float*, size_t) {},
            [](float*, const int*, int, int) {}, [](int) {}, 0, {}, &hooks);
  assert((eng.chunks == std::vector<int>{16, 16, 16, 16, 16, 16, 16, 16, 16, 6}));
}

int main() {
  CheckMergedLaunch();
  CheckCanonicalSchedule();
  CheckTailSchedule();
  CheckCachedBulk();
  CheckAdmission();
  CheckDeferredPrompt();
  for (int prefill : {0, 16, 64, 4096}) {
    PromptEngine eng(prefill);
    for (int length : {1,2,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256,4096,16384})
      for (int chunk : {1, 8}) {
        CheckPrompt(eng, length, chunk, 0, false);
        for (int limit : {0,1,2,7,8,16,64,128,256}) CheckPrompt(eng, length, chunk, limit, true);
      }
  }
}
