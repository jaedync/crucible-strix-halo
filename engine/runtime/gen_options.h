// Generation options and the draft-selection rules around them. Serial and batched serve and the CLI read
// them; only the argument parser (cli/gen_cli.h) is CLI-specific. Global scope, like gen_cli.h.
// The crucible:: prefixes below name the same entities that gen_cli.h reached through
// engine.hip's `using namespace crucible;` (this header also compiles on its own).
#pragma once
#include <functional>
#include <span>
#include <string>
#include <vector>
#include "conversation_resume.h"
#include "decode_policy.h"
#include "draft_vocab.h"
#include "prefill_rows.h"
#include "sampling.h"

struct GenOptions {
  std::string dump, draft_blocks, state_dump, logits_dump, draft_head = "full";
#if CRUCIBLE_DEBUG_NONFINITE & 4
  std::string inject;  // --inject KIND,LAYER,POS[,STREAM] (debug build only)
#endif
  int prefill_rows = kPrefillAuto;
  bool chunk_set = false;
  bool draft_dense = false, adapt_set = false;
  int adapt = 0, head_auto = 0, mtp_prefix = 0;
  float draft_margin = 0.f, draft_q = 0.f;
  int lookup = 0, lookup_min = 8, last_head = 1, overlap = 0, selection_options = 0;
  std::vector<int> ids;
  int n = 32, chunk = 8, ctx = 32768, spec = 0;
  bool steps = false;  // per-call GPU and wall timing of the decode loop
  crucible::sampling::Options sampling;
  std::vector<int> stop_ids;
  std::function<void(int)> emit;
  std::function<void(std::span<const float>)> sampling_trace;
  crucible::ConversationHooks* conversation = nullptr;
  // gen --debug-schedule diagnostics (cli/gen_debug_schedule.h); empty or 0 is off.
  std::string debug_schedule, step_logits;
  size_t debug_restore_at = 0, debug_state_at = 0;
};

inline GenOptions CliGenOptions() {
  GenOptions o;
  o.spec = crucible::kDefaultSpecDepth;
  o.draft_q = crucible::kDefaultDraftQ;
  o.draft_head = crucible::kDefaultDraftHead;
  o.mtp_prefix = crucible::kDefaultMtpPrefix;
  return o;
}

bool BuiltinDraft(const GenOptions& o) {
  return o.draft_blocks.empty() && o.draft_head == "64k";
}

bool AdaptiveDraft(const GenOptions& o) {
  return o.adapt != 0 || (!o.adapt_set && BuiltinDraft(o));
}

// Serving can call RunGen directly. Resolve builtins before any sampler dispatch.
std::vector<int> ResolveDraftBlocks(const GenOptions& o, const std::vector<int>& supplied) {
  if (o.spec == 0 || o.n == 1) return {};
  if (supplied.empty() && BuiltinDraft(o))
    return {crucible::kDefaultDraftBlocks.begin(), crucible::kDefaultDraftBlocks.end()};
  return supplied;
}
