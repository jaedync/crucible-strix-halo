// The grammar library side of structured output: a vocabulary and its compiled grammars (xgrammar 0.2.9,
// third_party/xgrammar, built into build/third_party/libcrucible_constraint.a by constraint_xgr.cc).
// The library compiles grammars only in the helper process (constraint_helper.h); this process loads the results.
#pragma once
#include "constraint.h"
#include "constraint_helper.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace crucible {
// Opaque: the tokenizer view, the helper process, and the grammar cache.
struct ConstraintVocab;

// The model's vocabulary from the checkpoint's tokenizer.json: byte-level BPE tokens, `rows` IDs (pad rows are never
// legal). Of the added tokens only the call tags (<tool_call>, </tool_call>) are text; `stop` is the one stop token
// (<|im_end|>). Every other added token is never legal inside a grammar. Throws ConstraintError.
std::shared_ptr<ConstraintVocab> LoadConstraintVocab(const std::string& tokenizer_json_path, int rows, int stop);
// A raw vocabulary (tests): token i is the text tokens[i]; an empty text is never legal; `stop` ends the grammar.
std::shared_ptr<ConstraintVocab> MakeConstraintVocab(const std::vector<std::string>& tokens, int stop);
int ConstraintVocabRows(const ConstraintVocab& vocab);
// Compiles (in the helper; this call waits for it) or takes from the cache the grammar of `spec`, and returns the
// request's constraint in its first phase. Throws ConstraintError with a fixed message (no schema text). The serve
// loop never calls this: it uses ConstraintService.
std::shared_ptr<Constraint> MakeConstraint(ConstraintVocab& vocab, const ConstraintSpec& spec,
                                           int think_end = kConstraintThinkEnd);
// The vendored library version (a CPU test checks the pin).
std::string ConstraintLibraryVersion();

// The last compile of this vocabulary in the helper (tests and the fuzz runner): its result, wall time and the
// child's peak RSS. And the bytes and entries of the grammar cache.
struct HelperRun {
  HelperStatus status = HelperStatus::kOk;
  double wall_ms = 0;
  uint64_t max_rss_kb = 0;
};
HelperRun LastHelperRun(const ConstraintVocab& vocab);
size_t ConstraintCacheBytes(const ConstraintVocab& vocab);
size_t ConstraintCacheEntries(const ConstraintVocab& vocab);

// The serve path: the compile never runs on the serve loop. A worker thread loads the
// vocabulary (at the first compile) and compiles in the helper; the serve loop only asks for the state and opens a
// grammar that is ready. Both calls return at once.
enum class CompileState { kReady, kCompiling, kRefused };
struct CompileStatus {
  CompileState state = CompileState::kCompiling;
  std::string message;  // kRefused: the fixed ConstraintError text
};
class ConstraintService;
std::shared_ptr<ConstraintService> StartConstraintService(std::function<std::shared_ptr<ConstraintVocab>()> load);
// The compile state of `spec`; the first call queues its compile. A full queue refuses (constraint_busy).
CompileStatus CompileConstraint(ConstraintService& service, const ConstraintSpec& spec);
// The constraint of a compiled `spec`. Throws ConstraintError: constraint_pending when it is not compiled (the serve
// loop never waits for a compile), or the refusal of its compile.
std::shared_ptr<Constraint> OpenCompiledConstraint(ConstraintService& service, const ConstraintSpec& spec,
                                                   int think_end = kConstraintThinkEnd);
// The serve loop's factory (BatchSession::SetConstraintFactory): it never waits. A spec that is not compiled yet
// queues its compile and throws ConstraintError kPendingMessage (the server opens it again, server/batch_backend.py);
// a refused compile throws its refusal at once.
ConstraintFactory ServeConstraintFactory(std::shared_ptr<ConstraintService> service,
                                         int think_end = kConstraintThinkEnd);
}  // namespace crucible
