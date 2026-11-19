// Hostile structured output specs. The server takes JSON schemas from API clients, and
// the engine compiles them on the serve loop (BatchSession::OpenConstraint). The contract that this test checks: for
// every spec, the compile returns a grammar or a ConstraintError within kDeadlineSeconds, and the process lives.
// Each spec runs in a child process, so a crash or a hang fails this one case and the test reports every case.
// RED on xgrammar 0.2.9 as vendored: a nested pattern overflows the stack in RegexIR::visit (fsm_builder.cc:955),
// and a 45-byte bounded repeat takes about 13 s.
#include "../engine/runtime/constraint_vocab.h"
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace crucible;

namespace {
// The helper's compile deadline (constraint_helper.h) and 2 s for the helper start and this child's own work.
constexpr unsigned kDeadlineSeconds = unsigned(kHelperDeadlineMs / 1000) + 2;
constexpr rlim_t kMemoryBytes = rlim_t(2) << 30;

std::vector<std::string> CharTokens() {
  std::vector<std::string> tokens;
  for (int c = 32; c < 127; ++c) tokens.push_back(std::string(1, char(c)));
  tokens.push_back("\t");
  tokens.push_back("\n");
  tokens.push_back("<|end|>");
  return tokens;
}

std::string Repeat(const std::string& text, int count) {
  std::string out;
  out.reserve(text.size() * size_t(count));
  for (int i = 0; i < count; ++i) out += text;
  return out;
}

struct Case {
  const char* name;
  ConstraintSpec spec;
};

// True when the child compiled the spec or refused it with a ConstraintError in time.
bool Survives(const Case& c, std::string& how) {
  std::fflush(stdout);
  const pid_t pid = fork();
  if (pid < 0) {
    how = "fork failed";
    return false;
  }
  if (pid == 0) {
    rlimit memory{kMemoryBytes, kMemoryBytes};
    setrlimit(RLIMIT_AS, &memory);
    alarm(kDeadlineSeconds);
    try {
      auto vocab = MakeConstraintVocab(CharTokens(), 97);
      MakeConstraint(*vocab, c.spec);
    } catch (const ConstraintError&) {
    }
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
  if (WIFSIGNALED(status))
    how = WTERMSIG(status) == SIGALRM ? "over the deadline" : std::string("signal ") + strsignal(WTERMSIG(status));
  else
    how = "exit " + std::to_string(WEXITSTATUS(status));
  return false;
}
}  // namespace

int main() {
  const std::string nest = Repeat("(", 40000) + "a" + Repeat(")", 40000);
  const std::string star = Repeat("(", 20000) + "a" + Repeat(")*", 20000);
  const std::vector<Case> cases = {
      // The control: an ordinary pattern compiles.
      {"control pattern", {"json_schema", R"({"type":"string","pattern":"^[a-z]{1,16}$"})", false}},
      {"pattern nested 40000", {"json_schema", R"({"type":"string","pattern":")" + nest + R"("})", false}},
      {"pattern nested quantifiers 20000", {"json_schema", R"({"type":"string","pattern":")" + star + R"("})", false}},
      {"pattern in an object property",
       {"json_schema", R"({"type":"object","properties":{"q":{"type":"string","pattern":")" + nest + R"("}},"required":["q"]})",
        false}},
      {"pattern in a strict tool parameter (structural tag)",
       {"structural_tag",
        R"({"type":"structural_tag","format":{"type":"tag","begin":"<tool_call>\n<function=f>\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<parameter=q>\n"},{"type":"json_schema","json_schema":{"type":"object","properties":{"r":{"type":"string","pattern":")" +
            nest + R"("}}},"max_whitespace_cnt":16},{"type":"const_string","value":"\n</parameter>\n"}]}]},"end":"</function>\n</tool_call>"}})",
        false}},
      {"bounded repeat (a|b|c){1,5000}", {"json_schema", R"({"type":"string","pattern":"(a|b|c){1,5000}"})", false}},
  };
  int failed = 0;
  for (const auto& c : cases) {
    std::string how;
    if (Survives(c, how)) {
      std::printf("PASS: hostile spec: %s\n", c.name);
    } else {
      std::printf("FAIL: hostile spec: %s (%s)\n", c.name, how.c_str());
      ++failed;
    }
  }
  if (failed) {
    std::printf("FAIL: constraint_hostile: %d of %zu cases\n", failed, cases.size());
    return 1;
  }
  std::puts("PASS: constraint_hostile");
}
