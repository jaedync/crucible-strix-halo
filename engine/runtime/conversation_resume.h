// A request owns its sampler. A prefix owns only model and proposal continuation state.
#pragma once
#include <algorithm>
#include <functional>
#include <optional>
#include <cstdint>
#include <vector>
#include "../model/flashnext/geometry.h"
#include "tenant.h"

namespace crucible {
struct ConversationPolicy {
  int spec = 0, chunk = 8, prefill_rows = 0, lookup = 0, lookup_min = 8;
  bool sampled = false, dense = false, adaptive = false, last_head = true, overlap = false;
  uint32_t margin_bits = 0, confidence_bits = 0;
  std::vector<int> draft_selection;
  bool head_auto = false;
  int mtp_prefix = 0;
  // W1: who the state belongs to. Every cache, slot and disk match compares the whole policy, so a state
  // matches only inside its tenant. Last, so that the positional initializations keep their meaning.
  StateTenant tenant = kLocalTenant;
  bool operator==(const ConversationPolicy&) const = default;
};
// The bytes that the cache counts for an entry's policy: train 23's count, without the tenant key. With the local
// tenant every budget decision then stays the same as before W1 (the key is 32 bytes of an entry of 100 MB or more).
inline constexpr size_t kConversationPolicyBytes = sizeof(ConversationPolicy) - sizeof(StateTenant);
struct ConversationResume {
  size_t length = 0;
  int mpos = 0, replay = 0, next_id = -1;
  bool have_carry = false;
  std::vector<int> pending_tokens;
  std::vector<float> logits;
  bool prompt_only = false;
  size_t prompt_boundary = 0;
};
// Bulk recurrence can change with chunk boundaries. Reuse only the same prefix schedule.
inline bool ConversationPromptCompatible(const ConversationResume& saved, size_t prompt_length,
                                           size_t boundary, int prefill_rows, bool split_at_boundary = false) {
  if (prefill_rows <= 0) return true;
  if (!saved.prompt_only) return false;
  const size_t length = saved.length;
  const size_t step = size_t(prefill_rows);
  // Canonical schedule: chunks end only on the zero-origin grid, so only a grid position or the whole prompt (an
  // exact retry) resumes. The boundary rules below belong to the opt-in split schedule.
  if (!split_at_boundary) return length == prompt_length || length % step == 0;
  const auto cut = [&](size_t b) { return b && b < length && b % step ? b : size_t(0); };
  if (cut(saved.prompt_boundary) != cut(boundary)) return false;
  const size_t origin = boundary && boundary < length ? boundary : 0;
  return length == prompt_length || length == boundary || (length - origin) % step == 0;
}
// A new bounded window cannot inherit auxiliary positions from an older request.
inline ConversationResume ConversationTargetResume(const ConversationResume& saved) {
  auto target = saved;
  target.mpos = 0;
  target.have_carry = false;
  target.pending_tokens.clear();
  return target;
}
struct ConversationHooks {
  size_t boundary = 0;
  // The boundary picks snapshot positions only. Opt-in and schedule-changing: also end a prompt chunk there, which
  // changes the arithmetic of every later row (outputs then differ from the same request on another server mode).
  bool split_at_boundary = false;
  bool defer_prompt_end_save = false;
  bool skip_prompt_interior_save = false;
  bool save_generated_tail = true;
  // Session replay (G15): the restored state is a previous turn's generated tail. When the new suffix is shorter
  // than the MTP warm-up window, the tail's MTP state continues instead of the overlap error.
  bool continue_tail = false;
  std::function<bool(size_t)> admit_prompt;
  std::optional<ConversationResume> deferred_prompt;
  std::function<ConversationResume(float*, float*)> restore;
  std::function<void(const ConversationResume&, float*, float*)> save;
};
inline int ConversationRetainedRows(size_t length, size_t logical, size_t pending_rows) {
  if (!pending_rows || pending_rows > length || logical > length || logical <= length - pending_rows) return 0;
  return int(logical - (length - pending_rows));
}
// Session replay (G15): the generated tail a finished request keeps. One rule for serial and batch serve, so a
// later turn finds the same tail in both. It needs at least one forwarded output row (serial: forwards > 0). A stop
// or limit inside an accepted window drops the rows after the last published token, as serial RunGen does.
struct ConversationTail { size_t length; int replay; };
inline std::optional<ConversationTail> ConversationTailState(size_t length, size_t prompt, size_t generated,
                                                             int replay) {
  if (generated < 2 || replay < 0) return std::nullopt;
  const size_t logical = prompt + generated - 1;
  const int keep = ConversationRetainedRows(length, logical, size_t(replay) + 1);
  if (!keep) return std::nullopt;
  return ConversationTail{logical, keep - 1};
}
// Session replay (G15): tails match on tokens only. Drafts change only speed, so the request's draft and sampling
// policy must not decide whether a later turn replays (batch serve keys tails the same way).
inline ConversationPolicy ConversationTailPolicy(int prefill_rows, int chunk) {
  ConversationPolicy policy;
  policy.prefill_rows = prefill_rows;
  policy.chunk = chunk;
  return policy;
}
inline int ConversationRows(size_t at, size_t end, int chunk, size_t boundary) {
  const size_t limit = boundary > at && boundary < end ? boundary : end;
  return int(std::min<size_t>(chunk, limit - at));
}
}  // namespace crucible
