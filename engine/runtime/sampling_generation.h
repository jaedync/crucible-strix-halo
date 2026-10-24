// Sampling owns probabilities and RNG state. The engine owns recurrent rollback.
#pragma once
#include "generation.h"
#include "sampling.h"
#include "draft_policy.h"
#include <optional>
#include "conversation_resume.h"
#include <functional>

namespace crucible {

struct SampledGenerationOptions {
  sampling::Options sampling;
  int n = 32, chunk = 8, spec = 0, lookup = 0, lookup_min = 8, mtp_prefix = 0;
  float draft_margin = 0, draft_q = 0;
  bool profile = false, score_prefill = false;
  std::vector<int> stop_ids;
  std::string dump;
  ConversationHooks* conversation = nullptr;
  std::function<void(std::span<const float>)> target_trace;
};

struct SampledGenerationResult : GenerationResult {
  long lookup_steps = 0, lookup_accepted = 0;
};

inline float SampledRawMargin(std::span<const float> logits) {
  float best = -INFINITY, second = -INFINITY;
  for (float value : logits) {
    if (!std::isfinite(value)) continue;
    if (value > best) { second = best; best = value; }
    else if (value > second) second = value;
  }
  return best - second;
}

// This estimate controls only work depth. It never replaces the p/q rule.
inline float SampledDepthConfidence(float margin) {
  static constexpr float edges[] = {1, 2, 3, 4, 6, 8, 12};
  static constexpr float values[] = {0.35f, 0.43f, 0.51f, 0.75f, 0.87f, 0.94f, 0.99f, 0.99f};
  int bucket = 0;
  while (bucket < 7 && margin >= edges[bucket]) ++bucket;
  return values[bucket];
}

template <class Engine, class Copy, class Emit, class Metrics>
class SampledGeneration {
 public:
  SampledGeneration(Engine& engine, std::span<const int> ids, const SampledGenerationOptions& options,
                    float* stacks, float* carry, Copy copy, Emit emit, Metrics& metrics)
      : eng_(engine), ids_(ids), o_(options), stk_(stacks), carry_(carry), copy_(copy),
        emit_(emit), metrics_(metrics), history_(std::max(1, options.lookup_min)) {
    // Validate() checks lookup_min later; the index is fed only when lookup is on.
    if (options.lookup > 0) history_.Append(ids);
  }

  SampledGenerationResult Run() {
    Validate();
    const double prefill_start = GenerationNow();
    if (o_.conversation && o_.conversation->restore) resume_ = o_.conversation->restore(stk_, carry_);
    else eng_.Reset();
    // A replayed tail inside the warm-up window keeps its gap-free MTP state. A tail without MTP state (a spec 0
    // turn) restarts the warm-up at the restored length, as batch serve does. Otherwise a fresh window as usual.
    const size_t mtp_start = DraftWarmupStart(ids_.size(), o_.mtp_prefix);
    const bool tail = o_.conversation && o_.conversation->continue_tail;
    const bool continue_mtp = tail && resume_.length > mtp_start && resume_.mpos > 0;
    if (o_.spec && o_.mtp_prefix && resume_.length && !continue_mtp) {
      if (resume_.length > mtp_start && !tail)
        throw std::runtime_error("snapshot overlaps the new MTP warmup window");
      eng_.ResetMtp();
      resume_ = ConversationTargetResume(resume_);
    }
    mpos_ = resume_.mpos; replay_ = resume_.replay;
    if (resume_.length == ids_.size()) {
      tail_logits_ = resume_.logits;
      first_ = sampling::FromLogits(tail_logits_, o_.sampling);
    }
    Prefill();
    eng_.Synchronize();
    result_.prefill_seconds = GenerationNow() - prefill_start;
    eng_.ClearExpertCounts();
    if (o_.profile) eng_.EnableProfile();
    const double decode_start = GenerationNow();
    Push(Draw(*first_, Next()));
    FlushDeferredPrompt(o_.conversation, stk_, carry_);
    Trace(tail_logits_);
    if (o_.spec > 0) {
      if (resume_.length == ids_.size() && !resume_.pending_tokens.empty()) {
        ctoks_ = resume_.pending_tokens;
        ctoks_.back() = result_.ids.back();
      } else { copy_(stk_, carry_, kWide); ctoks_ = {result_.ids.back()}; }
    }
    while (!Done()) {
      if (o_.spec == 0) { Serial(); continue; }
      const int bound = BoundedDraftDepth(std::max(o_.spec, o_.lookup), eng_.Context(),
                                          int(eng_.Length()), mpos_, int(ctoks_.size()));
      const int depth = std::min(bound, o_.n - int(result_.ids.size()) - 1);
      if (depth == 0) { Serial(); continue; }
      Step(depth);
    }
    result_.decode_seconds = GenerationNow() - decode_start;
    if (o_.conversation && o_.conversation->save && o_.conversation->save_generated_tail &&
        result_.forwards && valid_mtp_tail_) {
      const size_t logical = ids_.size() + result_.ids.size() - 1;
      if (eng_.Length() > logical) {
        const int keep = ConversationRetainedRows(eng_.Length(), logical, ctoks_.size());
        if (!keep) valid_mtp_tail_ = false;
        else {
          eng_.Truncate(logical);
          replay_ = keep - 1;
          ctoks_.resize(keep);
          tail_logits_ = eng_.CopyLogits(1, keep - 1);
        }
      }
      if (valid_mtp_tail_) {
        ConversationResume state{eng_.Length(), mpos_, replay_, result_.ids.back(), false, ctoks_, tail_logits_};
        o_.conversation->save(state, stk_, carry_);
      }
    }
    return result_;
  }

 private:
  void Validate() const {
    if (!ValidTokenIds(ids_, kVocab) || !ValidTokenIds(o_.stop_ids, kVocab) ||
        !RequestFitsContext(ids_.size(), o_.n, eng_.Context()) || o_.chunk < 1 || o_.chunk > kMaxRows ||
        o_.spec < 0 || o_.spec >= kMaxRows || o_.lookup < 0 || o_.lookup >= kMaxRows ||
        o_.lookup_min < 1 || o_.lookup_min > 32 || o_.mtp_prefix < 0 || o_.mtp_prefix > 262144 ||
        o_.sampling.temperature <= 0 ||
        !o_.sampling.Error(kVocab).empty() || !std::isfinite(o_.draft_margin) || o_.draft_margin < 0 ||
        !std::isfinite(o_.draft_q) || o_.draft_q < 0 || o_.draft_q > 1)
      throw std::runtime_error("invalid sampled generation request; check sampling, IDs, lengths, and context");
  }

  void Rows(float* destination, int destination_row, int first, int count) {
    copy_(destination + size_t(destination_row) * kWide, eng_.H() + size_t(first) * kWide, size_t(count) * kWide);
  }

  // The shared prompt path keeps GEMM prefill, MTP pairing, and the warmup cap equal to greedy RunGen.
  void Prefill() {
    RunPrompt(eng_, ids_, o_.chunk, o_.spec > 0, o_.dump, stk_, carry_, mpos_, copy_,
        [&](float* stacks, const int* tokens, int count, int position) {
          eng_.Mtp(stacks, tokens, count, position, o_.score_prefill);
        },
        [&](int row) {
          tail_logits_ = eng_.CopyLogits(1, row);
          first_ = sampling::FromLogits(tail_logits_, o_.sampling);
        }, o_.mtp_prefix, resume_, o_.conversation);
    if (resume_.length < ids_.size()) replay_ = 0;
  }

  void Trace(std::span<const float> logits) const { if (o_.target_trace) o_.target_trace(logits); }
  bool Done() const { return result_.stopped || int(result_.ids.size()) >= o_.n; }
  // Absolute sequence position of the next generated token.
  uint64_t Next() const { return ids_.size() + result_.ids.size(); }
  int Draw(const sampling::Distribution& distribution, uint64_t position) const {
    return distribution.KeyedSample(o_.sampling.seed, position);
  }
  void Push(int token) {
    if (Done()) return;
    result_.ids.push_back(token);
    if (o_.lookup > 0) history_.Append(std::span<const int>(&token, 1));
    eng_.ObserveDraft(&token, 1);
    result_.stopped = std::find(o_.stop_ids.begin(), o_.stop_ids.end(), token) != o_.stop_ids.end();
    emit_(token);
  }

  void Serial() {
    metrics_.Begin();
    eng_.Forward(&result_.ids.back(), 1, replay_, 1, o_.dump);
    const auto logits = eng_.CopyLogits(1);
    tail_logits_ = logits;
    const bool save = o_.conversation && o_.conversation->save;
    const bool catchup = o_.spec && save;
    // Keep the new trunk stack before predictor catch-up overwrites H().
    if (save) Rows(carry_, 0, 0, 1);
    if (catchup) {
      eng_.Mtp(stk_, ctoks_.data(), int(ctoks_.size()), mpos_, false);
      mpos_ += int(ctoks_.size());
      copy_(stk_, carry_, kWide);
    }
    if (o_.spec) valid_mtp_tail_ = catchup;
    metrics_.End(Metrics::kVerify, 1);
    Push(Draw(sampling::FromLogits(logits, o_.sampling), Next()));
    if (catchup) ctoks_ = {result_.ids.back()};
    Trace(logits);
    replay_ = 0;
    ++result_.forwards;
    metrics_.StepDone();
  }

  std::pair<sampling::Distribution, float> DraftDistribution(int rows) {
    const auto [logits, mapping] = eng_.DraftLogits(rows);
    auto distribution = sampling::FromLogits(logits, o_.sampling, mapping);
    // Greedy head scoring can leave DraftMargin stale in a sampled request.
    return {std::move(distribution), SampledRawMargin(logits)};
  }

  struct Proposals {
    std::vector<int> ids;
    std::vector<sampling::Distribution> probabilities;
    bool lookup = false;
  };

  Proposals Drafts(int limit) {
    Proposals out;
    // Selecting lookup after comparing with a random MTP draw would change q.
    auto copied = history_.Draft(std::min(o_.lookup, limit));
    out.lookup = copied.size() > 1;
    // A copy that crosses a stop ID: the rows after it cannot commit a token (draft_stop.h).
    TrimDrafts(copied, o_.stop_ids, limit);
    const int rows = int(ctoks_.size());
    metrics_.Begin();
    eng_.Mtp(stk_, ctoks_.data(), rows, mpos_, !out.lookup);
    metrics_.End(Metrics::kMtpFirst, rows);
    mpos_ += rows;
    if (out.lookup) {
      out.ids = copied;
      for (int token : copied) out.probabilities.push_back(sampling::Distribution::Point(token));
      metrics_.Margin(0, -1);
      ++result_.lookup_steps;
      return out;
    }
    auto [first_distribution, margin] = DraftDistribution(rows);
    metrics_.Margin(0, margin);
    out.probabilities.push_back(std::move(first_distribution));
    out.ids.push_back(Draw(out.probabilities.back(), Next()));
    float confidence = SampledDepthConfidence(margin);
    for (int k = 1; k < std::min(o_.spec, limit) && ChainMayGrow(out.ids, o_.stop_ids, limit); ++k) {
      if (o_.draft_q > 0 ? confidence * 0.75f < o_.draft_q : margin < o_.draft_margin) break;
      metrics_.Begin();
      Rows(stk_, kMaxRows, k == 1 ? rows - 1 : 0, 1);
      eng_.Mtp(stk_ + size_t(kMaxRows) * kWide, &out.ids.back(), 1, mpos_ + k - 1);
      metrics_.End(Metrics::kMtpChain, 1);
      auto [next_distribution, next_margin] = DraftDistribution(1);
      margin = next_margin;
      confidence *= SampledDepthConfidence(margin);
      if (o_.draft_q > 0 && confidence < o_.draft_q) break;
      metrics_.Margin(k, margin);
      out.probabilities.push_back(std::move(next_distribution));
      out.ids.push_back(Draw(out.probabilities.back(), Next() + out.ids.size()));
    }
    return out;
  }

  void Step(int limit) {
    const size_t base = eng_.Length();
    const uint64_t next = Next();
    const Proposals proposals = Drafts(limit);
    const int depth = int(proposals.ids.size());
    std::vector<int> input{result_.ids.back()};
    input.insert(input.end(), proposals.ids.begin(), proposals.ids.end());
    metrics_.Begin();
    eng_.Forward(input.data(), depth + 1, replay_, 1);
    const auto logits = eng_.CopyLogits(depth + 1);
    metrics_.End(Metrics::kVerify, depth + 1);
    std::vector<int> verified;
    int accepted = 0;
    for (int row = 0; row <= depth; ++row) {
      const auto p = sampling::FromLogits(std::span(logits).subspan(size_t(row) * eng_.Vocabulary(), eng_.Vocabulary()), o_.sampling);
      // Match acceptance: every emitted token is the target's own keyed draw at its position.
      const int target = Draw(p, next + row);
      verified.push_back(target);
      if (row == depth || target != proposals.ids[row]) break;
      ++accepted;
    }
    tail_logits_.assign(logits.begin() + size_t(accepted) * eng_.Vocabulary(),
                        logits.begin() + size_t(accepted + 1) * eng_.Vocabulary());
    eng_.Truncate(base + accepted + 1);
    Rows(stk_, 0, 0, accepted + 1);
    ctoks_ = verified;
    for (size_t i = 0; i < verified.size() && !Done(); ++i) {
      Push(verified[i]);
      Trace(std::span(logits).subspan(i * eng_.Vocabulary(), eng_.Vocabulary()));
    }
    replay_ = accepted;
    result_.accepted += accepted;
    result_.attempted += depth;
    if (proposals.lookup) result_.lookup_accepted += accepted;
    ++result_.forwards;
    metrics_.Accepted(depth, accepted);
    metrics_.StepDone();
  }

  Engine& eng_;
  std::span<const int> ids_;
  const SampledGenerationOptions& o_;
  float* stk_;
  float* carry_;
  Copy copy_;
  Emit emit_;
  Metrics& metrics_;
  std::optional<sampling::Distribution> first_;
  LookupIndex history_;
  std::vector<int> ctoks_;
  std::vector<float> tail_logits_;
  ConversationResume resume_;
  bool valid_mtp_tail_ = true;
  int mpos_ = 0, replay_ = 0;
  SampledGenerationResult result_;
};

template <class Engine, class Copy, class Emit, class Metrics>
SampledGenerationResult GenerateSampled(Engine& engine, std::span<const int> ids,
    const SampledGenerationOptions& options, float* stacks, float* carry, Copy copy, Emit emit, Metrics& metrics) {
  return SampledGeneration<Engine, Copy, Emit, Metrics>(engine, ids, options, stacks, carry, copy, emit, metrics).Run();
}

}  // namespace crucible
