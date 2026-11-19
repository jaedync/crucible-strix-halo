// The grammar compile helper. The grammar compiler (xgrammar) runs on text from the
// network, so it never runs in the engine process: a helper process compiles each grammar in a child with CPU-time,
// address-space and stack limits, and passes the result back in a binary form. A crash, a timeout or an
// out-of-memory of the compiler is a ConstraintError for that one request. The engine process only loads the result
// and runs the per-token matcher.
//
// The helper is one program built into the engine (an ELF image in the constraint library, started from a sealed
// memfd), so it needs no install step and always matches the engine's build. Its first process loads the vocabulary
// once and forks one child for each compile; it never reads a grammar itself.
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace crucible {
// The real bounds of one compile (the server's count limits in server/grammar_limits.py are a first filter).
inline constexpr int kHelperDeadlineMs = 5000;                      // wall time of one compile, from the request
inline constexpr uint64_t kHelperMemoryBytes = uint64_t(2) << 30;   // address space a compile may add (RLIMIT_AS)
inline constexpr int kHelperCpuSeconds = 30;                        // RLIMIT_CPU of a compile child (all threads)
inline constexpr uint64_t kHelperStackBytes = uint64_t(8) << 20;    // RLIMIT_STACK of a compile child
inline constexpr size_t kMaxGrammarBytes = size_t(128) << 20;       // the compiled grammar in its binary form
inline constexpr size_t kMaxSpecBytes = size_t(1) << 20;            // the spec text of one request
inline constexpr int kHelperStartMs = 60000;                        // the helper builds its vocabulary
inline constexpr int kHelperWalkTokens = 16;                        // the child's matcher check (first legal tokens)
// The engine's grammar cache: compiled grammars by spec, bounded by bytes (MemorySizeBytes) and entries.
inline constexpr size_t kGrammarCacheBytes = size_t(512) << 20;
inline constexpr size_t kGrammarCacheEntries = 64;
// The engine's compile queue and its cache of refused specs (a refused spec is refused again at once).
inline constexpr size_t kCompileQueue = 8;
inline constexpr size_t kRefusalCacheEntries = 256;

// The fixed results of one compile and their texts (HelperMessage); no library text and
// no spec text crosses the process boundary.
enum class HelperStatus : uint32_t {
  kOk = 0,
  kSchema = 1,        // the schema does not compile to a grammar
  kTag = 2,           // the structural tag does not compile
  kKind = 3,          // an unknown kind
  kSpec = 4,          // json_object with a spec
  kCrash = 5,         // the compiler stopped (a signal)
  kTime = 6,          // over kHelperDeadlineMs or kHelperCpuSeconds
  kMemory = 7,        // over kHelperMemoryBytes
  kTooBig = 8,        // the compiled grammar is over kMaxGrammarBytes
  kMatcher = 9,       // the matcher check failed
  kHelper = 10,       // the helper does not run or answer
  kLoad = 11,         // the engine could not load the helper's result
};

inline const char* HelperMessage(HelperStatus status) {
  switch (status) {
    case HelperStatus::kOk: return "";
    case HelperStatus::kSchema: return "constraint_compile: the schema does not compile to a grammar";
    case HelperStatus::kTag: return "constraint_compile: the structural tag does not compile";
    case HelperStatus::kKind: return "constraint_kind: kind must be json_schema, json_object or structural_tag";
    case HelperStatus::kSpec: return "constraint_spec: json_object takes no spec";
    case HelperStatus::kCrash: return "constraint_compile: the grammar compiler stopped on this schema";
    case HelperStatus::kTime: return "constraint_limit: the grammar compile exceeds kHelperDeadlineMs = 5000";
    case HelperStatus::kMemory: return "constraint_limit: the grammar compile exceeds kHelperMemoryBytes = 2 GiB";
    case HelperStatus::kTooBig: return "constraint_limit: the compiled grammar exceeds kMaxGrammarBytes = 128 MiB";
    case HelperStatus::kMatcher: return "constraint_compile: the grammar has no matcher";
    case HelperStatus::kHelper: return "constraint_compile: the grammar helper does not run";
    case HelperStatus::kLoad: return "constraint_compile: the compiled grammar does not load";
  }
  return "constraint_compile: the grammar helper does not run";
}
// The serve loop's refusals (constraint_vocab.h ServeConstraintFactory): the server opens a pending spec again later.
inline constexpr const char* kPendingMessage = "constraint_pending: the grammar is not compiled yet";
inline constexpr const char* kBusyMessage = "constraint_busy: the grammar compile queue is full (kCompileQueue = 8)";

// The helper program's ELF image (constraint_helper_image.S in the engine library). Empty inside the helper itself.
std::span<const unsigned char> ConstraintHelperImage();
// The helper's main loop on the socket `fd` (constraint_helper_main.cc calls it). Returns the exit code.
int ConstraintHelperMain(int fd);
}  // namespace crucible
