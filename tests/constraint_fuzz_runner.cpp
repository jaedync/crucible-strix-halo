// The fuzz runner of the grammar guard (tools/constraint_fuzz.py drives it). It stands
// for the engine process: each spec compiles in the helper (MakeConstraint), and the per-token matcher runs here, as
// it does on the serve loop. A crash of this process is a finding (the engine would end).
// Usage: constraint_fuzz_runner raw|TOKENIZER_JSON [WALK_TOKENS] < lines "KIND<TAB>REASONING<TAB>SPEC"
// Writes "S <i>" before each spec and "R <i> <status> <make_ms> <child_rss_kb> <grammar_bytes> <cache_bytes>
// <open_ms> <tokens> <max_token_ms> <walk_ms>" after it. status: ok or the refusal code.
#include "../engine/runtime/constraint_vocab.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace crucible;
using Clock = std::chrono::steady_clock;

namespace {
double MsSince(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }

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
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "constraint_fuzz_runner raw|TOKENIZER_JSON [WALK_TOKENS]\n");
    return 2;
  }
  const bool raw = std::string(argv[1]) == "raw";
  const int walk = argc > 2 ? std::atoi(argv[2]) : 256;
  auto vocab = raw ? MakeConstraintVocab(CharTokens(), 97) : LoadConstraintVocab(argv[1], 248320, 248046);
  const int rows = ConstraintVocabRows(*vocab);
  const int stop = raw ? 97 : 248046;
  const int think = raw ? 98 : kConstraintThinkEnd;
  std::string line;
  for (long i = 0; std::getline(std::cin, line); ++i) {
    const size_t a = line.find('\t'), b = a == std::string::npos ? a : line.find('\t', a + 1);
    if (b == std::string::npos) continue;
    ConstraintSpec spec{line.substr(0, a), line.substr(b + 1), line.substr(a + 1, b - a - 1) == "1"};
    std::printf("S %ld\n", i);
    std::fflush(stdout);
    const size_t before = ConstraintCacheBytes(*vocab);
    std::shared_ptr<Constraint> c;
    std::string status = "ok";
    // The time to a compiled grammar as the caller sees it (helper, transfer and load); a cache hit is fast. The
    // helper's own run (LastHelperRun) is the last compile, which a cache hit does not change.
    const auto make_start = Clock::now();
    try {
      c = MakeConstraint(*vocab, spec, think);
    } catch (const ConstraintError& error) {
      const std::string text = error.what();
      status = text.substr(0, text.find(':'));
      if (text.find("kHelperDeadlineMs") != std::string::npos) status += "/time";
      else if (text.find("kHelperMemoryBytes") != std::string::npos) status += "/memory";
      else if (text.find("kMaxGrammarBytes") != std::string::npos) status += "/size";
      else if (text.find("stopped") != std::string::npos) status += "/crash";
      else if (text.find("helper does not run") != std::string::npos) status += "/helper";
      else if (text.find("does not load") != std::string::npos) status += "/load";
    }
    const double make_ms = MsSince(make_start);
    const HelperRun run = LastHelperRun(*vocab);
    const size_t after = ConstraintCacheBytes(*vocab);
    double open_ms = 0, max_token_ms = 0, walk_ms = 0;
    int tokens = 0;
    if (c) {
      // The serve loop's work: an open of a cached grammar, then masks and tokens.
      auto t = Clock::now();
      c = MakeConstraint(*vocab, spec, think);
      open_ms = MsSince(t);
      if (spec.reasoning) c->Accept(think);
      std::mt19937 random(uint32_t(i * 7919 + 1));
      std::vector<uint32_t> mask;
      const auto walk_start = Clock::now();
      for (int step = 0; step < walk && !c->Terminated(); ++step) {
        t = Clock::now();
        c->FillMask(mask);
        std::vector<int> legal;
        for (int id = 0; id < rows; ++id)
          if (MaskAllows(mask, id) && id != stop) legal.push_back(id);
        if (legal.empty()) {
          if (!c->Accept(stop)) break;
        } else if (!c->Accept(legal[random() % legal.size()])) {
          std::printf("E %ld a legal token was refused\n", i);
          break;
        }
        // Rollback as the verify span does (at most 8 tokens).
        if (step % 16 == 15) c->Rollback(std::min(int(random() % 8) + 1, step + 1));
        max_token_ms = std::max(max_token_ms, MsSince(t));
        ++tokens;
      }
      walk_ms = MsSince(walk_start);
    }
    std::printf("R %ld %s %.0f %llu %zu %zu %.3f %d %.3f %.1f\n", i, status.c_str(), make_ms,
                static_cast<unsigned long long>(run.max_rss_kb), after > before ? after - before : size_t(0), after,
                open_ms, tokens, max_token_ms, walk_ms);
    std::fflush(stdout);
  }
}
