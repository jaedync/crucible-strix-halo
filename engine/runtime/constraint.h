// Structured output: the request's grammar state and the masked choice of a target.
// No library header here: constraint_vocab.h builds the grammar state from xgrammar (constraint_xgr.cc), and the
// tests build it from small vocabularies. A request with no constraint holds a null pointer and runs no code of
// this file, so its tokens stay those of an engine without it.
#pragma once
#include "sampling.h"
#include "sampling_penalty.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crucible {
// The error event of a request whose grammar has no legal token (an empty grammar node): only that request ends.
inline constexpr const char* kConstraintDeadEndMessage = "constraint_dead_end: the grammar has no legal token";
// The checkpoint's `</think>` token (one added token): a request with reasoning arms its grammar after it.
inline constexpr int kConstraintThinkEnd = 248069;
// The rollback depth a grammar state keeps: a verify span has at most 8 rows, and the draft walk rolls back 7.
inline constexpr int kConstraintRollback = 32;
// The longest run of JSON whitespace a json_schema or json_object grammar allows (a model cannot loop on spaces).
inline constexpr int kConstraintMaxWhitespace = 16;

// The request line's constraint: kind json_schema, json_object or structural_tag; spec is the JSON text of the
// schema or of the structural tag (empty for json_object). reasoning: the output starts in the reasoning phase.
struct ConstraintSpec {
  std::string kind;
  std::string spec;
  bool reasoning = false;
};

// A fixed-message error: the text never holds schema or output text.
struct ConstraintError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// One grammar state machine with rollback (the library side).
class GrammarState {
 public:
  virtual ~GrammarState() = default;
  // false: the grammar refuses the token; the state does not change.
  virtual bool Accept(int token) = 0;
  virtual void Rollback(int tokens) = 0;
  // Bit (t % 32) of words[t / 32] is set when token t is legal now.
  virtual void FillMask(std::span<uint32_t> words) = 0;
  // The stop token was accepted: no token is legal.
  virtual bool Terminated() const = 0;
};

// Counters of one constrained request (numbers only).
struct ConstraintCounts {
  long tokens = 0;      // tokens committed while armed
  long refusals = 0;    // targets whose unmasked choice the grammar refused
  long cut_drafts = 0;  // drafts cut before a verify pass
  double fallback_us = 0;  // host time of the masked choices after a refusal
  double compile_us = 0;
  int cache_hit = 0;
};

// The phases of a request: reasoning (no mask) until the think-end token, then the grammar. Rollback undoes both.
class Constraint {
 public:
  Constraint(std::unique_ptr<GrammarState> grammar, int vocab, int stop_token, bool reasoning,
             int think_end = kConstraintThinkEnd)
      : grammar_(std::move(grammar)), vocab_(vocab), stop_(stop_token), think_end_(think_end), armed_(!reasoning) {
    if (!grammar_ || vocab_ <= 0) throw std::invalid_argument("constraint needs a grammar and a vocabulary");
  }
  bool Armed() const { return armed_; }
  int Vocab() const { return vocab_; }
  int StopToken() const { return stop_; }
  bool Terminated() const { return armed_ && grammar_->Terminated(); }
  // No token was legal at a choice (DeadEndToken): the session ends the request with kConstraintDeadEndMessage.
  bool DeadEnd() const { return dead_end_; }
  void MarkDeadEnd() { dead_end_ = true; }
  // Commits one token. false: refused, nothing changes.
  bool Accept(int token) {
    if (token < 0 || token >= vocab_) return false;
    if (!armed_) {
      if (token == think_end_) { armed_ = true; history_.push_back(kArm); }
      else history_.push_back(kFree);
      return true;
    }
    if (grammar_->Terminated() || !grammar_->Accept(token)) return false;
    history_.push_back(kGrammar);
    ++counts.tokens;
    return true;
  }
  bool Allows(int token) {
    if (!armed_) return token >= 0 && token < vocab_;
    if (!Accept(token)) return false;
    Rollback(1);
    return true;
  }
  void Rollback(int tokens) {
    if (tokens < 0 || size_t(tokens) > history_.size()) throw std::logic_error("constraint rollback exceeds its history");
    int grammar = 0;
    for (int i = 0; i < tokens; ++i) {
      const uint8_t kind = history_.back();
      history_.pop_back();
      if (kind == kGrammar) { ++grammar; --counts.tokens; }
      else if (kind == kArm) {
        if (grammar) grammar_->Rollback(std::exchange(grammar, 0));
        armed_ = false;
      }
    }
    if (grammar) grammar_->Rollback(grammar);
    // Keep only the rollback window of grammar tokens; older history never rolls back.
    if (history_.size() > size_t(4 * kConstraintRollback)) history_.erase(history_.begin(), history_.end() - kConstraintRollback);
  }
  // The legal prefix of a draft chain; the state is the same after the call.
  int WalkDrafts(std::span<const int> drafts) {
    int legal = 0;
    for (const int token : drafts) {
      if (!Accept(token)) break;
      ++legal;
    }
    Rollback(legal);
    return legal;
  }
  // The legal tokens now, as 32-bit words (all legal in the reasoning phase).
  void FillMask(std::vector<uint32_t>& words) {
    words.assign(size_t(vocab_ + 31) / 32, 0);
    if (!armed_) {
      for (int t = 0; t < vocab_; ++t) words[size_t(t) / 32] |= 1u << (t % 32);
      return;
    }
    if (!grammar_->Terminated()) grammar_->FillMask(words);
    // Rows past the vocabulary (pad bits of the last word) are never legal.
    if (vocab_ % 32) words.back() &= (1u << (vocab_ % 32)) - 1;
  }
  ConstraintCounts counts;

 private:
  static constexpr uint8_t kFree = 0, kArm = 1, kGrammar = 2;
  std::unique_ptr<GrammarState> grammar_;
  int vocab_, stop_, think_end_;
  bool armed_;
  bool dead_end_ = false;
  std::vector<uint8_t> history_;
};

// Builds the request's constraint (compile, cache). Throws ConstraintError with a fixed message.
using ConstraintFactory = std::function<std::shared_ptr<Constraint>(const ConstraintSpec&)>;

inline bool MaskAllows(std::span<const uint32_t> mask, int token) {
  return token >= 0 && size_t(token) / 32 < mask.size() && (mask[size_t(token) / 32] >> (token % 32) & 1u);
}
// The masked greedy target: the largest logit among legal IDs, ties to the lower ID (as ArgmaxKernel). -1 if no ID
// is legal. A -inf logit can win only when every legal logit is -inf.
inline int MaskedArgmax(std::span<const float> row, std::span<const uint32_t> mask) {
  int best = -1;
  float value = -std::numeric_limits<float>::infinity();
  for (size_t w = 0; w < mask.size(); ++w) {
    uint32_t bits = mask[w];
    while (bits) {
      const int bit = __builtin_ctz(bits);
      bits &= bits - 1;
      const size_t token = w * 32 + size_t(bit);
      if (token >= row.size()) return best;
      if (best < 0 || row[token] > value) { best = int(token); value = row[token]; }
    }
  }
  return best;
}

namespace constraint_detail {
inline double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}
// No token is legal (an empty grammar node): the request is marked and the choice returns the stop token, which
// ends the span (AcceptRows) and which BatchSession::Publish turns into the request's error event. Nothing else
// of the batch changes (owner rule, Nov 16: a dead end ends only its own request).
inline int DeadEndToken(Constraint& c) {
  c.MarkDeadEnd();
  return c.StopToken();
}
}  // namespace constraint_detail

// Greedy: the device ID is the masked argmax when the grammar takes it, because
// the device argmax is the maximum of a superset with ties to the lower ID. Else `row()` copies the row's logits.
// Commits the returned token. A negative device ID (a nonfinite row) passes through to the caller's error.
template <class Row>
int ChooseGreedy(Constraint& c, int greedy, Row&& row) {
  if (greedy < 0 || c.Accept(greedy)) return greedy;
  const auto start = std::chrono::steady_clock::now();
  ++c.counts.refusals;
  std::vector<uint32_t> mask;
  c.FillMask(mask);
  const auto logits = row();
  const int target = MaskedArgmax(std::span<const float>(logits.data(), logits.size()), mask);
  if (target < 0 || !c.Accept(target)) return constraint_detail::DeadEndToken(c);
  c.counts.fallback_us += constraint_detail::Since(start);
  return target;
}

// Sampled (3.b): filters, then the mask, then the keyed race. If the unmasked draw is legal it wins the masked race
// too (the race takes the largest p/E, and the legal set is a subset), so only a refused draw scans again. When no
// kept token is legal, the mask comes first and the filters after it. Commits the returned token.
inline int MaskedKeyedDraw(std::span<const float> logits, const sampling::Options& options,
                           std::span<const uint32_t> mask, uint64_t seed, uint64_t position) {
  const auto kept = sampling::FromLogits(logits, options);
  const int token = kept.KeyedSampleWhere(seed, position, [&](int id) { return MaskAllows(mask, id); });
  if (token >= 0) return token;
  std::vector<float> masked(logits.begin(), logits.end());
  bool any = false;
  for (size_t id = 0; id < masked.size(); ++id) {
    if (MaskAllows(mask, int(id)) && !std::isinf(masked[id])) any = true;
    else masked[id] = -std::numeric_limits<float>::infinity();
  }
  if (!any) return -1;
  return sampling::FromLogits(masked, options).KeyedSample(seed, position);
}
inline int ChooseKeyed(Constraint& c, std::span<const float> logits, const sampling::Options& options,
                       uint64_t seed, uint64_t position) {
  const int drawn = sampling::FromLogits(logits, options).KeyedSample(seed, position);
  if (c.Accept(drawn)) return drawn;
  const auto start = std::chrono::steady_clock::now();
  ++c.counts.refusals;
  std::vector<uint32_t> mask;
  c.FillMask(mask);
  const int target = MaskedKeyedDraw(logits, options, mask, seed, position);
  if (target < 0 || !c.Accept(target)) return constraint_detail::DeadEndToken(c);
  c.counts.fallback_us += constraint_detail::Since(start);
  return target;
}

// Penalties with structured output: the penalties change a copy of the row first, then
// the grammar's rule above chooses on the penalized row (filters, mask, keyed race; or the masked argmax). The choice
// reads legal IDs only, so a penalty can change which legal ID wins but never makes a refused ID possible. A neutral
// request never calls this: it keeps ChooseGreedy (the device ID) or ChooseKeyed on the raw row. Commits the token.
inline int ChoosePenalized(Constraint& c, std::span<const float> logits, const sampling::Options& options,
                           const sampling::PenaltyHistory& history, uint64_t position) {
  if (!options.Penalized() || !history.Active()) throw std::logic_error("ChoosePenalized needs an active penalty");
  if (options.temperature == 0) {
    // The penalized argmax of the full row stands in for the device ID: legal, it is the masked argmax too.
    const int greedy = sampling::PenalizedArgmax(logits, options, history);
    return ChooseGreedy(c, greedy, [&] {
      std::vector<float> row(logits.begin(), logits.end());
      history.Apply(row, options);
      return row;
    });
  }
  std::vector<float> row(logits.begin(), logits.end());
  history.Apply(row, options);
  return ChooseKeyed(c, row, options, options.seed, position);
}

// Cuts a draft chain at its first draft that the grammar refuses: that draft never equals a masked target, so the
// cut changes the rows of the pass only. Returns the drafts cut.
inline int CutDrafts(Constraint& c, std::vector<int>& drafts) {
  const int legal = c.WalkDrafts(drafts);
  const int cut = int(drafts.size()) - legal;
  drafts.resize(size_t(legal));
  c.counts.cut_drafts += cut;
  return cut;
}

// The verified tokens of one proposal of a constrained request (input: the last token, then the drafts): row r's
// masked target, up to the first target that differs from draft r + 1. Each target is committed to the grammar, so
// after the call the grammar holds the emitted tokens. greedy: the device IDs of the rows; logits: all rows when
// sampled (vocab floats each); copy_row(r): row r's logits for a greedy refusal. BatchSpecDecoder::Accept and the
// session tests call this, so both apply one rule. The span ends at an accepted stop token or a terminated grammar
// (review H1): the request ends there, and a row after it has no legal token, so it is never masked or sampled.
// penalties: a penalized request's history (its prompt and emitted tokens); row r then takes ChoosePenalized on
// logits row r, with the drafts before r counted (as a run without drafts counts them), and the history is the
// same again at the return. Null or inactive: the neutral rule above, unchanged.
template <class CopyRow>
std::vector<int> AcceptRows(Constraint* c, std::span<const int> input, std::span<const int> greedy,
                            const sampling::Options& options, uint64_t next, std::span<const float> logits, size_t vocab,
                            CopyRow&& copy_row, sampling::PenaltyHistory* penalties = nullptr) {
  if (!c) throw std::logic_error("AcceptRows needs a constraint");
  const int depth = int(input.size()) - 1;
  const bool sampled = options.temperature > 0;
  const bool penalized = penalties && penalties->Active();
  if (penalized && logits.size() < size_t(depth + 1) * vocab)
    throw std::logic_error("a penalized verify span needs the logits of every row");
  std::optional<sampling::ProvisionalTokens> drafts_seen;
  if (penalized) drafts_seen.emplace(*penalties);
  std::vector<int> verified;
  for (int row = 0; row <= depth; ++row) {
    const auto row_logits = [&] { return logits.subspan(size_t(row) * vocab, vocab); };
    const int target = penalized ? ChoosePenalized(*c, row_logits(), options, *penalties, next + uint64_t(row))
        : sampled ? ChooseKeyed(*c, row_logits(), options, options.seed, next + uint64_t(row))
                  : ChooseGreedy(*c, greedy[size_t(row)], [&] { return copy_row(row); });
    verified.push_back(target);
    if (row == depth || target != input[size_t(row) + 1] || target == c->StopToken() || c->Terminated() ||
        c->DeadEnd())
      break;
    if (drafts_seen) drafts_seen->Add(target);
  }
  return verified;
}

// The done event's counters: numbers only.
inline std::string ConstraintCountsJson(const ConstraintCounts& k) {
  return "{\"tokens\":" + std::to_string(k.tokens) + ",\"refusals\":" + std::to_string(k.refusals) +
         ",\"cut_drafts\":" + std::to_string(k.cut_drafts) + ",\"fallback_us\":" + std::to_string(long(k.fallback_us)) +
         ",\"compile_us\":" + std::to_string(long(k.compile_us)) + ",\"cache_hit\":" + std::to_string(k.cache_hit) + "}";
}
}  // namespace crucible
