// The CRUCIBLE_* switch registry: the only list of engine switches and
// the only reader of the environment. A read goes through SwitchText (or a
// parser that takes a SwitchName): a name that is not in kSwitches or
// kExternalVariables stops the build. tests/test_switch_registry.py checks
// this table against the engine's reads and against the hand-kept lists in
// tools/gates.py, tools/batch_gate_report.py and server/backend.py.
// This header includes no engine code, so kernels/ may include it.
//
// Columns: name; kind; default (the behavior when unset); module; issue (0:
// the switch is older than the issue backlog, Oct 9); decide_by (the train
// that makes an exact-speed or layout switch the code default or deletes it;
// 0: no deadline); use (kServe: serve passes it to the engine; kRecord: gate
// records publish its value); note.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <unistd.h>

namespace crucible {
enum class SwitchKind {
  kNumerics,    // selects a non-bitwise path; docs/ARCHITECTURE.md (Exactness rules), off when CRUCIBLE_EXACT=1
  kExactSpeed,  // a bitwise speed A/B or rollback
  kLayout,      // a bitwise memory or storage layout that a profile selects
  kDiagnostic,  // prints, counts, captures or dumps; outputs do not change
  kBuild,       // a compile-time macro; the environment does not set it
  kProfile,     // changes the model on purpose (a user profile input)
};
enum SwitchUse : unsigned { kNone = 0, kServe = 1, kRecord = 2 };

struct SwitchInfo {
  const char* name;
  SwitchKind kind;
  const char* default_value;
  const char* module;
  int issue;
  int decide_by;
  unsigned use;
  const char* note;
};

namespace switch_table {
using K = SwitchKind;
constexpr unsigned N = kNone, S = kServe, R = kRecord, SR = kServe | kRecord;
// Sorted by name (a static_assert below checks it).
inline constexpr SwitchInfo kSwitches[] = {
  {"CRUCIBLE_ABLATE", K::kProfile, "unset", "model", 0, 0, SR, "abliterated profile: direction file (read only)"},
  {"CRUCIBLE_ABLATE_ANY_CHECKPOINT", K::kDiagnostic, "off", "model", 0, 0, R, "measurement-only bypass of the ablation geometry check"},
  {"CRUCIBLE_COMPACT_STATE", K::kLayout, "0", "runtime", 0, 20, SR, "1: compact sequence storage"},
  {"CRUCIBLE_DEBUG_NONFINITE", K::kBuild, "unset", "runtime", 116, 0, N, "compile-time mask (debug build): 1 non-finite probes, 2 NaN poison of scratch, 4 inject probe, 8 FLIP control, 16 NaN poison of the KV tail"},
  {"CRUCIBLE_DECODE_RING", K::kExactSpeed, "front", "exl3", 0, 20, SR, "decode ring order: 0 off, 1 on"},
  {"CRUCIBLE_EXACT", K::kNumerics, "0", "runtime", 0, 0, SR, "master switch: 1 is option A, bitwise to the baseline"},
  {"CRUCIBLE_FUSE", K::kExactSpeed, "tuned", "runtime", 0, 20, SR, "row policy fuse level 0..3"},
  {"CRUCIBLE_GDN_VSCAN_MUTANT", K::kBuild, "unset", "layers", 69, 0, N, "compile-time macro (bench/gdn_vscan.hip mutant: other conv tap order)"},
  {"CRUCIBLE_HC_WMMA", K::kNumerics, "1", "layers", 0, 0, SR, "0: VALU HC kernels"},
  {"CRUCIBLE_INTERLEAVE_CHECK", K::kDiagnostic, "unset", "runtime", 0, 0, N, "appends verified rows' token IDs and prompt hashes to a file"},
  {"CRUCIBLE_LOOKAHEAD", K::kExactSpeed, "tuned", "runtime", 0, 20, SR, "row policy prefetch blocks 0..8"},
  {"CRUCIBLE_LOOKAHEAD_MIB", K::kExactSpeed, "tuned", "runtime", 0, 20, SR, "row policy prefetch MiB 1..32"},
  {"CRUCIBLE_MOE_WINDOW_PROBE", K::kBuild, "1", "exl3", 0, 0, N, "compile-time macro (moe_window_policy.h)"},
  {"CRUCIBLE_MTP_SIDECAR", K::kExactSpeed, "unset", "model", 0, 20, SR, "tuned MTP draft head file (read only; drafts only)"},
  {"CRUCIBLE_PACKED_ROWS", K::kExactSpeed, "auto", "serve", 0, 20, SR, "packed verify rows: unset 16 from 4 decoders, else 8; 8..16 fixed"},
  {"CRUCIBLE_PERSISTENT_GDN", K::kNumerics, "off", "layers", 0, 0, SR, "1: persistent GDN kernel"},
  {"CRUCIBLE_PERSISTENT_GDN_DEPENDENT_LDS", K::kBuild, "unset", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PERSISTENT_GDN_LIMIT_LOADS", K::kBuild, "unset", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PERSISTENT_GDN_PREFETCH", K::kBuild, "1", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PERSISTENT_GDN_SERIAL", K::kBuild, "1", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PERSISTENT_GDN_WAVES", K::kBuild, "9", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PERSISTENT_HC_PIPE", K::kBuild, "2", "layers", 0, 0, N, "compile-time macro"},
  {"CRUCIBLE_PFBENCH_PASSES", K::kDiagnostic, "2", "cli", 0, 0, N, "pfbench timing passes"},
  {"CRUCIBLE_PF_EXACT_SPLIT", K::kNumerics, "off", "layers", 0, 0, SR, "1: prefill split numerics A/B"},
  {"CRUCIBLE_PF_FX_LEAD", K::kExactSpeed, "kFxLead", "exl3", 0, 20, SR, "prefill MoE queue lead 0..1048576"},
  {"CRUCIBLE_PF_GDN_EAGER", K::kNumerics, "off", "layers", 0, 0, SR, "1: prefill GDN eager scan"},
  {"CRUCIBLE_PF_GDN_MICRO", K::kNumerics, "off", "layers", 0, 0, SR, "1: prefill GDN micro kernel"},
  {"CRUCIBLE_PF_HC_FUSED", K::kNumerics, "off", "layers", 0, 0, SR, "1: prefill HC dots kernel"},
  {"CRUCIBLE_PF_LAUNCH_ROWS", K::kExactSpeed, "8192+128", "runtime", 24, 20, SR, "prefill rows per bulk launch, \"L+T\" joins a cold tail of up to T rows (8192: no join; 4096: one grid chunk, the off value; costs 1,677 MiB of arena at 8192, and the join a cut buffer from the cache budget)"},
  {"CRUCIBLE_PF_MOE_RM", K::kExactSpeed, "6", "exl3", 0, 20, SR, "prefill MoE tile rows"},
  {"CRUCIBLE_PF_OUT", K::kDiagnostic, "unset", "cli", 0, 0, R, "pfbench output path (writes a file)"},
  {"CRUCIBLE_PF_PAD_H", K::kLayout, "1", "runtime", 0, 20, SR, "0: tight prefill H storage"},
  {"CRUCIBLE_PF_PRUNE", K::kExactSpeed, "on", "runtime", 0, 20, SR, "0: no last-layer row pruning"},
  {"CRUCIBLE_PF_QSA_ROWS", K::kNumerics, "off", "layers", 0, 0, SR, "1: prefill QSA rows per part"},
  {"CRUCIBLE_PF_QSA_SCORE_MODE", K::kBuild, "1", "layers", 0, 0, N, "compile-time macro (prefill_qsa_select.h)"},
  {"CRUCIBLE_PF_STAGE_SHARE", K::kLayout, "on", "runtime", 0, 20, SR, "0: separate prefill stage arenas"},
  {"CRUCIBLE_PF_TILES", K::kExactSpeed, "auto", "exl3", 0, 20, SR, "prefill GEMM tiles: six integers"},
  {"CRUCIBLE_PLE_BF16", K::kProfile, "unset", "model", 129, 24, SR, "BF16 n-gram table of the original weights (file, read only; needs CRUCIBLE_PLE_HOST=1)"},
  {"CRUCIBLE_PLE_HOST", K::kLayout, "0", "model", 0, 20, SR, "1: PLE table on the host (the long profiles set it)"},
  {"CRUCIBLE_PLE_TRACE", K::kDiagnostic, "off", "model", 0, 0, SR, "PLE host gather timings on stderr"},
  {"CRUCIBLE_PROFILE", K::kDiagnostic, "off", "cli", 0, 0, SR, "stage timings on stderr"},
  {"CRUCIBLE_QSA_CAPTURE_AT", K::kDiagnostic, "unset", "layers", 0, 0, R, "QSA capture: writes attention tensors to files"},
  {"CRUCIBLE_QSA_CAPTURE_DIR", K::kDiagnostic, "unset", "layers", 0, 0, R, "QSA capture: writes attention tensors to files"},
  {"CRUCIBLE_QSA_CAPTURE_INDEX_ONLY", K::kDiagnostic, "unset", "layers", 0, 0, R, "QSA capture: writes attention tensors to files"},
  {"CRUCIBLE_QSA_POOL_TILED", K::kBuild, "1", "layers", 0, 0, N, "compile-time macro (ops.h)"},
  {"CRUCIBLE_QSA_PREFETCH", K::kBuild, "64", "layers", 0, 0, N, "compile-time macro (qsa_long.h)"},
  {"CRUCIBLE_QSA_SPLITS", K::kBuild, "32", "layers", 0, 0, N, "compile-time macro (qsa_long.h)"},
  {"CRUCIBLE_QSA_TRACE", K::kBuild, "unset", "layers", 0, 0, N, "compile-time macro (trace build)"},
  {"CRUCIBLE_ROUTE_DUMP", K::kDiagnostic, "unset", "runtime", 0, 0, N, "appends every MoE routing record to a file"},
  {"CRUCIBLE_SERVE_PROFILE", K::kDiagnostic, "0", "serve", 0, 0, S, "packed-pass timings on stderr every N passes"},
  {"CRUCIBLE_SPEC_STATS", K::kDiagnostic, "off", "serve", 0, 0, S, "draft counts on stderr"},
  {"CRUCIBLE_STATE_CACHE", K::kExactSpeed, "tuned", "runtime", 0, 20, SR, "row policy state cache 0..6"},
  {"CRUCIBLE_STATIC_CACHE", K::kExactSpeed, "tuned", "runtime", 0, 20, SR, "row policy static cache 0..4"},
  {"CRUCIBLE_TUNE_ROWS", K::kExactSpeed, "1", "runtime", 0, 20, SR, "0: untuned row policy"},
  {"CRUCIBLE_VERIFY_QSA_CHECK", K::kDiagnostic, "off", "serve", 0, 0, S, "kill check counts; synchronizes"},
  {"CRUCIBLE_WARMUP", K::kExactSpeed, "on", "runtime", 0, 20, SR, "0: no startup buffer warm-up"},
};
}  // namespace switch_table
using switch_table::kSwitches;

// Variables that the engine reads but does not own (the ROCm runtime and the
// state disk root). Sorted by name.
inline constexpr const char* kExternalVariables[] = {
    "GPU_PINNED_MIN_XFER_SIZE", "HOME", "HSA_USERPTR_FOR_PAGED_MEM", "HSA_USE_SVM",
};

namespace switch_detail {
constexpr int Compare(const char* a, const char* b) {
  while (*a && *a == *b) ++a, ++b;
  return (unsigned char)*a - (unsigned char)*b;
}
constexpr bool Sorted() {
  for (std::size_t i = 1; i < sizeof(kSwitches) / sizeof(kSwitches[0]); ++i)
    if (Compare(kSwitches[i - 1].name, kSwitches[i].name) >= 0) return false;
  for (std::size_t i = 1; i < sizeof(kExternalVariables) / sizeof(kExternalVariables[0]); ++i)
    if (Compare(kExternalVariables[i - 1], kExternalVariables[i]) >= 0) return false;
  return true;
}
// A build macro is not an environment variable, so it is not readable.
constexpr bool Readable(const char* name) {
  for (const SwitchInfo& s : kSwitches)
    if (Compare(s.name, name) == 0) return s.kind != SwitchKind::kBuild;
  for (const char* v : kExternalVariables)
    if (Compare(v, name) == 0) return true;
  return false;
}
// Not constexpr: a call during constant evaluation stops the build.
inline void UnregisteredSwitch() {}
}  // namespace switch_detail
static_assert(switch_detail::Sorted(), "keep kSwitches and kExternalVariables sorted and unique");

// A registered switch name. The constructor runs at compile time, so a name
// outside the registry does not build.
class SwitchName {
 public:
  consteval SwitchName(const char* name) : text_(name) {
    if (!switch_detail::Readable(name)) switch_detail::UnregisteredSwitch();
  }
  // C++ tests only: a fixture name outside the registry. engine/ and kernels/
  // never use it (tests/test_switch_registry.py).
  static constexpr SwitchName Unchecked(const char* name) { return SwitchName(name, 0); }
  constexpr const char* text() const { return text_; }

 private:
  constexpr SwitchName(const char* name, int) : text_(name) {}
  const char* text_;
};

// The environment text of a switch, or null when it is unset. Every parser
// starts here; this is the engine's only getenv.
inline const char* SwitchText(SwitchName name) { return std::getenv(name.text()); }

// The text of a registry row, for code that walks the table (ValidateNumericsSwitches).
inline const char* SwitchText(const SwitchInfo& info) { return std::getenv(info.name); }

// The whole environment, for the state disk settings record.
inline char** SwitchEnvironment() { return ::environ; }
}  // namespace crucible
