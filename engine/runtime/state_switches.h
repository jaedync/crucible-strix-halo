// Which CRUCIBLE_* switches are part of a state's identity.
//
// A switch of kind kNumerics, kLayout or kProfile changes state bytes or the state layout on purpose: it stays in the
// identity. A switch outside the registry stays in too. A kBuild name is a compile-time macro that the environment does
// not set; the machine-code fingerprint of the state epoch covers it. The kExactSpeed and kDiagnostic switches are
// listed here, each with a reason. `in_identity = false`: the switch cannot change the bytes of a stored state, so a
// state written with it restores under any value of it. `in_identity = true`: it can, or nothing shows that it
// cannot, so it stays (the safe side).
//
// A new kExactSpeed or kDiagnostic switch needs a row here: tests/state_epoch_test.cpp fails without one. Moving a
// switch to `false` needs the state-equality gate (tools/state_equality_gate.py, arm `neutral`) to show equal bytes.
// Sorted by name.
#pragma once
#include "../switches.h"
#include <string_view>

namespace crucible {
struct StateSwitchRule { const char* name; bool in_identity; const char* reason; };
inline constexpr StateSwitchRule kStateSwitchRules[] = {
  {"CRUCIBLE_ABLATE_ANY_CHECKPOINT", true, "lets an ablation run on a checkpoint that fails the geometry check; the state then comes from an unchecked ablation (measurement tool, never in a daily profile)"},
  {"CRUCIBLE_DECODE_RING", false, "load order of the decode ring; a bitwise speed A/B (registry kind kExactSpeed); the gate arm neutral sets 0"},
  {"CRUCIBLE_DRAFT_CHAIN", true, "hc8 runs the MTP sites with int8 HC weights, and the MTP layer rows are part of the stored state; lead and stop move work in time only, but one word holds all three"},
  {"CRUCIBLE_DRAFT_HEAD", true, "a coarse head chooses the draft tokens; the MTP rows of drafted positions can stay in a saved tail; no gate shows otherwise"},
  {"CRUCIBLE_EXACT_RESUME", true, "chooses the rows where a prefill resumes from a checkpoint; a resumed prefill is not always byte equal to a cold one, so it decides which bytes a state holds"},
  {"CRUCIBLE_FUSE", false, "row policy fuse level, a bitwise speed A/B (registry kind kExactSpeed); the gate arm neutral sets 0"},
  {"CRUCIBLE_GDN_VERIFY", false, "GDN verify core in one launch, bitwise equal to GdnKernel; the gate arm neutral sets 4"},
  {"CRUCIBLE_INTERLEAVE_CHECK", false, "appends token IDs and prompt hashes to a file; reads only"},
  {"CRUCIBLE_LAZY_SHADOW", false, "compares the lazy spill pieces with an eager copy and refuses a write on a difference; it never changes what a written state holds"},
  {"CRUCIBLE_LOOKAHEAD", false, "prefetch depth of the row policy; schedule only"},
  {"CRUCIBLE_LOOKAHEAD_MIB", false, "prefetch size of the row policy; schedule only"},
  {"CRUCIBLE_MOE_B128", false, "decode expert body variants, bitwise to the v0 body (registry kind kExactSpeed); the gate arm neutral sets 0"},
  {"CRUCIBLE_MTP_SIDECAR", true, "other head file; the MTP rows are in the state, and StateDiskSettings hashes the file content"},
  {"CRUCIBLE_PACKED_ROWS", false, "rows of a packed verify pass; per-row results do not depend on the row grouping (batch invariance); the gate arm neutral sets 8"},
  {"CRUCIBLE_PFBENCH_PASSES", false, "timing passes of pfbench; the serve path never reads it"},
  {"CRUCIBLE_PF_FX_LEAD", false, "queue lead of the prefill MoE; schedule only"},
  {"CRUCIBLE_PF_LAUNCH_ROWS", true, "sets the rows of a prefill launch and the cold-tail join, so the chunk boundaries of the prefill; no gate shows equal state bytes over its values"},
  {"CRUCIBLE_PF_MOE_RM", false, "prefill MoE tile rows; a row result does not depend on the tile (registry kind kExactSpeed); the gate arm neutral sets 4"},
  {"CRUCIBLE_PF_OUT", false, "output path of pfbench; the serve path never reads it"},
  {"CRUCIBLE_PF_PRUNE", false, "drops the last layer's FFN rows that no logit needs; the state rows of attention and GDN are written before it; the gate arm neutral sets 0"},
  {"CRUCIBLE_PF_TILES", true, "prefill GEMM tile shapes can regroup the K loop of a GEMM; the registry calls it bitwise, but the gate arm neutral cannot set it (the valid sets are tuned per pack), so nothing shows equal bytes"},
  {"CRUCIBLE_PLE_FETCH", false, "how the host gather of the PLE table reads pages (hints and parts); same bytes"},
  {"CRUCIBLE_PLE_TRACE", false, "gather timings on stderr"},
  {"CRUCIBLE_PROFILE", false, "stage timings on stderr"},
  {"CRUCIBLE_QSA_CAPTURE_AT", true, "capture of attention tensors; it can select another QSA path while it runs, so a state written under capture stays apart"},
  {"CRUCIBLE_QSA_CAPTURE_DIR", true, "capture of attention tensors; it can select another QSA path while it runs, so a state written under capture stays apart"},
  {"CRUCIBLE_QSA_CAPTURE_INDEX_ONLY", true, "capture of attention tensors; it can select another QSA path while it runs, so a state written under capture stays apart"},
  {"CRUCIBLE_ROUTE_DUMP", false, "appends the MoE routing records to a file; reads only"},
  {"CRUCIBLE_SERVE_PROFILE", false, "packed-pass timings on stderr"},
  {"CRUCIBLE_SPEC_SPAN", true, "a span above 8 rows gives every sequence 16 pending GDN rows a head (the stored fixed state grows) and a session tail up to 13 replay rows, so a state of one value does not restore under another"},
  {"CRUCIBLE_SPEC_STATS", false, "draft counts on stderr"},
  {"CRUCIBLE_STATE_CACHE", false, "row policy state cache level; schedule only"},
  {"CRUCIBLE_STATIC_CACHE", false, "row policy static cache level; schedule only"},
  {"CRUCIBLE_TUNE_ROWS", false, "switches the tuned row policy; schedule only"},
  {"CRUCIBLE_VERIFY_QSA_CHECK", false, "kill check counts; it reads and synchronizes"},
  {"CRUCIBLE_WARMUP", false, "startup buffer warm-up; no state is written by it"},
};

namespace state_switch_detail {
constexpr bool Equal(std::string_view a, const char* b) { return a == std::string_view(b); }
}  // namespace state_switch_detail

// True when a state written with this switch set must not restore under another value of it.
inline bool StateSwitchInIdentity(std::string_view name) {
  for (const StateSwitchRule& rule : kStateSwitchRules)
    if (state_switch_detail::Equal(name, rule.name)) return rule.in_identity;
  for (const SwitchInfo& info : kSwitches)
    if (state_switch_detail::Equal(name, info.name)) return info.kind != SwitchKind::kBuild;
  return true;
}
}  // namespace crucible
