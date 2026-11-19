// Grammar compiler guard: a fault of the grammar compiler (crash, endless time,
// memory) is a ConstraintError for one request, never the end of the engine process. The compile runs in a helper
// process; this test process stands for the engine.
#include "../engine/runtime/constraint_vocab.h"
#include <xgrammar/xgrammar.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <picojson.h>
#include <sstream>
#include <random>
#include <signal.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace crucible;

namespace {
constexpr int kStop = 97;
std::vector<std::string> CharTokens() {
  std::vector<std::string> tokens;
  for (int c = 32; c < 127; ++c) tokens.push_back(std::string(1, char(c)));
  tokens.push_back("\t");
  tokens.push_back("\n");
  tokens.push_back("<|end|>");
  tokens.push_back("</think>");
  tokens.push_back("\":");
  return tokens;
}
int Id(char c) { return c - 32; }

// The code before the colon of a ConstraintError, or "" when the call did not throw one.
std::string Refusal(ConstraintVocab& vocab, const ConstraintSpec& spec) {
  try {
    MakeConstraint(vocab, spec);
  } catch (const ConstraintError& error) {
    const std::string text = error.what();
    return text.substr(0, text.find(':'));
  }
  return "";
}

void Works(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", R"({"type":"string","pattern":"a+"})", false});
  for (const char ch : std::string("\"aa\"")) assert(c->Accept(Id(ch)));
  assert(c->Accept(kStop) && c->Terminated());
}

// Test 1 (grammar probe): 40,000 nested groups crash RegexIR::visit(Bracket) (fsm_builder.cc:955) with SIGSEGV.
void Nested40000(ConstraintVocab& vocab) {
  const std::string pattern = std::string(40000, '(') + "a" + std::string(40000, ')');
  const std::string schema = R"({"type":"string","pattern":")" + pattern + R"("})";
  assert(Refusal(vocab, {"json_schema", schema, false}) == "constraint_compile");
  // The engine (this process) lives, and the next request compiles.
  Works(vocab);
  assert(Refusal(vocab, {"json_schema", schema, false}) == "constraint_compile");
  Works(vocab);
  std::puts("PASS: 40,000 nested groups are a constraint_compile refusal, and the next request compiles");
}
using Clock = std::chrono::steady_clock;
double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// The full refusal text of a ConstraintError, or "".
std::string RefusalText(ConstraintVocab& vocab, const ConstraintSpec& spec) {
  try {
    MakeConstraint(vocab, spec);
  } catch (const ConstraintError& error) {
    return error.what();
  }
  return "";
}

// A compile over kHelperDeadlineMs is a constraint_limit refusal that names the deadline, within the deadline and 1 s.
// The slow spec takes over 300 s (grammar probe). "(a|b|c){1,5000}" was host-bound: 2.7 s on an idle machine, a pass.
void Deadline(ConstraintVocab& vocab) {
  const auto start = Clock::now();
  const std::string text = RefusalText(vocab, {"json_schema", R"({"type":"string","pattern":"[ab]{1,50000}x"})", false});
  const double ms = MsSince(start);
  assert(text.rfind("constraint_limit:", 0) == 0 && text.find("kHelperDeadlineMs = 5000") != std::string::npos);
  assert(ms >= kHelperDeadlineMs && ms < kHelperDeadlineMs + 1000);
  assert(LastHelperRun(vocab).status == HelperStatus::kTime);
  Works(vocab);
  std::printf("PASS: a compile over the deadline is refused after %.0f ms; the next request compiles\n", ms);
}

// The helper's first process ends (here: SIGKILL from outside): the next compile starts a new helper.
std::vector<pid_t> Children() {
  std::vector<pid_t> children;
  const std::string path = "/proc/" + std::to_string(getpid()) + "/task/" + std::to_string(getpid()) + "/children";
  FILE* file = std::fopen(path.c_str(), "r");
  if (!file) return children;
  long pid;
  while (std::fscanf(file, "%ld", &pid) == 1) children.push_back(pid_t(pid));
  std::fclose(file);
  return children;
}
void Restart(ConstraintVocab& vocab) {
  Works(vocab);
  const auto children = Children();
  assert(!children.empty());
  for (const pid_t child : children) kill(child, SIGKILL);
  // The dead helper costs at most one refused compile (kHelper); then a new one runs.
  const std::string text = RefusalText(vocab, {"json_schema", R"({"type":"string","pattern":"b+"})", false});
  assert(text.empty() || text == "constraint_compile: the grammar helper does not run");
  Works(vocab);
  std::puts("PASS: a helper that ends is started again at the next compile");
}

// The masks of a grammar that the helper compiled and this process loaded equal the masks of the same grammar
// compiled in this process, along seeded random walks (the binary form keeps every mask).
std::vector<uint32_t> MatcherMask(xgrammar::GrammarMatcher& matcher, int rows) {
  std::vector<uint32_t> words(size_t(xgrammar::GetBitmaskSize(rows)));
  int64_t shape[1] = {int64_t(words.size())};
  DLTensor tensor{};
  tensor.data = words.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = DLDataType{kDLInt, 32, 1};
  tensor.shape = shape;
  matcher.FillNextTokenBitmask(&tensor);
  if (rows % 32) words.back() &= (1u << (rows % 32)) - 1;
  return words;
}
int SameWalks(ConstraintVocab& vocab, const xgrammar::TokenizerInfo& info, const std::vector<uint8_t>& never,
              const std::vector<ConstraintSpec>& specs, int rows, int steps) {
  xgrammar::GrammarCompiler compiler(info, 8, false);
  int compared = 0;
  for (size_t s = 0; s < specs.size(); ++s) {
    const auto& spec = specs[s];
    xgrammar::Grammar grammar = spec.kind == "structural_tag"
        ? std::get<xgrammar::Grammar>(xgrammar::Grammar::FromStructuralTag(spec.spec))
        : xgrammar::Grammar::FromJSONSchema(spec.spec, true, std::nullopt, std::nullopt, true, kConstraintMaxWhitespace);
    xgrammar::GrammarMatcher local(compiler.CompileGrammar(grammar), std::nullopt, false, kConstraintRollback);
    auto loaded = MakeConstraint(vocab, spec);
    std::mt19937 random(uint32_t(17 + s));
    for (int step = 0; step < steps && !local.IsTerminated(); ++step) {
      const auto mine = MatcherMask(local, rows);
      std::vector<uint32_t> theirs;
      loaded->FillMask(theirs);
      assert(mine == theirs);
      ++compared;
      std::vector<int> legal;
      for (int t = 0; t < rows; ++t)
        if ((theirs[size_t(t) / 32] >> (t % 32) & 1u) && !never[size_t(t)]) legal.push_back(t);
      if (legal.empty()) break;
      const int pick = legal[random() % legal.size()];
      assert(local.AcceptToken(pick) && loaded->Accept(pick));
    }
  }
  return compared;
}
const std::vector<ConstraintSpec> kWalkSpecs = {
    {"json_schema", R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"enum":["a","b","c"]}},"zip":{"type":"string","pattern":"[0-9]{5}"}},"required":["name","age"]})", false},
    {"json_schema", R"({"type":"string","pattern":"[a-z]{2,40}x(ab|cd)*"})", false},
    {"json_schema", R"({"type":"array","items":{"anyOf":[{"type":"number"},{"type":"boolean"},{"type":"null"}]},"maxItems":5})", false},
    {"json_schema", R"({"type":"string","minLength":3,"maxLength":300})", false},
    {"json_schema", R"({"type":"object","properties":{"u":{"type":"string","format":"email"},"d":{"type":"string","format":"date-time"}},"required":["u","d"]})", false},
    {"structural_tag", R"({"type":"structural_tag","format":{"type":"tag","begin":"<a>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"k":{"type":"integer"}},"required":["k"]}},"end":"</a>"}})", false},
};
void RoundTrip(ConstraintVocab& vocab) {
  const auto tokens = CharTokens();
  xgrammar::TokenizerInfo info(tokens, xgrammar::VocabType::RAW, int(tokens.size()), std::vector<int32_t>{kStop}, false);
  std::vector<uint8_t> never(tokens.size(), 0);
  const int compared = SameWalks(vocab, info, never, kWalkSpecs, int(tokens.size()), 200);
  std::printf("PASS: loaded grammars give the masks of an in-process compile (%d masks, character vocabulary)\n", compared);
}

std::string RealTokenizer() {
  if (const char* path = std::getenv("CRUCIBLE_TOKENIZER_JSON")) return path;
  const char* home = std::getenv("HOME");
  const std::string local = std::string(home ? home : "") + "/.cache/mtp-slot/ckpt305/tokenizer.json";
  return std::filesystem::exists(local) ? local : "";
}
// The real vocabulary uses the bitset masks (above 1000 accepted and rejected tokens), which the small one never
// makes. The in-process TokenizerInfo reads the file as LoadConstraintVocab does.
void RealRoundTrip() {
  const std::string path = RealTokenizer();
  if (path.empty()) { std::puts("SKIP: real tokenizer round trip (set CRUCIBLE_TOKENIZER_JSON)"); return; }
  constexpr int kRows = 248320, kRealStop = 248046;
  auto vocab = LoadConstraintVocab(path, kRows, kRealStop);
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  picojson::value root;
  assert(picojson::parse(root, text.str()).empty());
  std::vector<std::string> tokens(static_cast<size_t>(kRows));
  for (const auto& [token, id] : root.get("model").get("vocab").get<picojson::object>()) tokens[size_t(id.get<int64_t>())] = token;
  for (const auto& entry : root.get("added_tokens").get<picojson::array>()) {
    const int64_t id = entry.get("id").get<int64_t>();
    tokens[size_t(id)] = id == 248058 || id == 248059 || id == kRealStop ? entry.get("content").get<std::string>() : "";
  }
  xgrammar::TokenizerInfo info(tokens, xgrammar::VocabType::BYTE_LEVEL, kRows, std::vector<int32_t>{kRealStop}, false);
  std::vector<uint8_t> never(static_cast<size_t>(kRows), 1);
  for (int t = 0; t < kRows; ++t) never[size_t(t)] = tokens[size_t(t)].empty() && t != kRealStop;
  const int compared = SameWalks(*vocab, info, never, kWalkSpecs, kRows, 60);
  std::printf("PASS: loaded grammars give the masks of an in-process compile (%d masks, real tokenizer)\n", compared);
}

// The serve path: the compile runs on the service's worker, never on the caller (the serve loop). While a compile
// takes its full deadline, each state poll returns at once, a ready grammar opens at once, and the slow spec ends
// as a refusal for itself only.
void Service() {
  auto service = StartConstraintService([] { return MakeConstraintVocab(CharTokens(), kStop); });
  const ConstraintSpec fast{"json_schema", R"({"type":"string","pattern":"a+"})", false};
  const ConstraintSpec slow{"json_schema", R"({"type":"string","pattern":"[ab]{1,50000}x"})", false};
  bool pending = false;
  try {
    OpenCompiledConstraint(*service, fast);
  } catch (const ConstraintError& error) {
    pending = std::string(error.what()).rfind("constraint_pending:", 0) == 0;
  }
  assert(pending);
  auto start = Clock::now();
  while (CompileConstraint(*service, fast).state != CompileState::kReady) {
    assert(MsSince(start) < 30000);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  start = Clock::now();
  assert(CompileConstraint(*service, slow).state == CompileState::kCompiling);
  double worst_poll = 0, worst_step = 0;
  int passes = 0;
  CompileStatus status;
  for (;;) {
    auto t = Clock::now();
    status = CompileConstraint(*service, slow);
    worst_poll = std::max(worst_poll, MsSince(t));
    if (status.state != CompileState::kCompiling) break;
    // One "decode pass" of another session: open its ready grammar and take tokens.
    t = Clock::now();
    auto c = OpenCompiledConstraint(*service, fast);
    for (const char ch : std::string("\"aaa")) assert(c->Accept(Id(ch)));
    worst_step = std::max(worst_step, MsSince(t));
    ++passes;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const double ms = MsSince(start);
  assert(status.state == CompileState::kRefused && status.message.rfind("constraint_limit:", 0) == 0);
  assert(ms >= kHelperDeadlineMs - 50 && ms < kHelperDeadlineMs + 1500);
  assert(worst_poll < 5 && worst_step < 100 && passes > 100);
  // The refused spec stays refused at once, also at open; the other spec still opens.
  start = Clock::now();
  assert(CompileConstraint(*service, slow).state == CompileState::kRefused && MsSince(start) < 5);
  bool refused = false;
  try {
    OpenCompiledConstraint(*service, slow);
  } catch (const ConstraintError& error) {
    refused = std::string(error.what()).rfind("constraint_limit:", 0) == 0;
  }
  assert(refused);
  OpenCompiledConstraint(*service, fast);
  std::printf("PASS: a %.0f ms compile runs off the caller: %d passes, worst poll %.2f ms, worst pass %.2f ms\n", ms,
              passes, worst_poll, worst_step);
}

// A full queue refuses at once (constraint_busy) and never waits.
void Busy() {
  auto service = StartConstraintService([] { return MakeConstraintVocab(CharTokens(), kStop); });
  int busy = 0;
  for (int i = 0; i < int(kCompileQueue) + 4; ++i) {
    const ConstraintSpec spec{"json_schema", R"({"type":"string","pattern":"[ab]{1,50000}x)" + std::to_string(i) + R"("})", false};
    const auto start = Clock::now();
    const auto status = CompileConstraint(*service, spec);
    assert(MsSince(start) < 5);
    busy += status.state == CompileState::kRefused && status.message.rfind("constraint_busy:", 0) == 0;
  }
  assert(busy >= 3);
  std::printf("PASS: a full compile queue refuses at once (%d of %d)\n", busy, int(kCompileQueue) + 4);
}
}  // namespace

int main(int argc, char** argv) {
  auto vocab = MakeConstraintVocab(CharTokens(), kStop);
  Works(*vocab);
  Nested40000(*vocab);
  RoundTrip(*vocab);
  Restart(*vocab);
  // The slow cases take one deadline each; CONSTRAINT_GUARD_FAST=1 skips them in an edit loop.
  if (!std::getenv("CONSTRAINT_GUARD_FAST")) {
    Deadline(*vocab);
    Service();
    Busy();
  }
  RealRoundTrip();
  std::puts("PASS: constraint_guard");
}
