// The Python runner includes the real driver, without GPU CLI entry points.
#include "gen_cli_cpu_fixture.h"
#include "gen_cli_cpu.h"

// The prefill 16 and 64 cases must run the bulk path of RunPrompt, not the Forward fallback.
static_assert(HasGemmPrefill<Engine>);

int main() {
  for (double temperature : {0., 0.7}) for (int prefill : {0, 16, 64})
    for (int prompt : {1, 24, 65, 257, 3000}) for (int cap : {0, 8, 256}) {
      GenOptions options = CliGenOptions();
      options.n = 1; options.ctx = 4096; options.ids.assign(prompt, 1);
      options.sampling.temperature = temperature; options.mtp_prefix = cap;
      std::vector<int> emitted;
      options.emit = [&](int token) { emitted.push_back(token); };
      Engine engine(4096, prefill);
      const auto first = RunGen(engine, options, {});
      assert(first.gen.size() == 1 && engine.mtp_calls == 0 && first.gen == emitted);
      emitted.clear();
      const auto repeated = RunGen(engine, options, {});
      assert(first.gen == repeated.gen && engine.mtp_calls == 0 && repeated.gen == emitted);
      if (prompt > 1) {
        options.n = 2;
        const auto longer = RunGen(engine, options, {});
        assert(longer.gen.size() == 2 && engine.mtp_calls > 0);
      }
    }
  // Draft stop (draft_stop.h): no verify row follows a stop ID or goes past the token limit. The target always
  // answers 2 and the draft head always answers 7.
  const auto run = [](std::vector<int> ids, std::vector<int> stops, int lookup, int n) {
    GenOptions options = CliGenOptions();
    options.n = n; options.ctx = 4096; options.ids = ids; options.stop_ids = stops;
    options.spec = 7; options.lookup = lookup; options.lookup_min = 1; options.draft_q = 0;
    Engine engine(4096, 0);
    engine.forward_token = 2; engine.mtp_token = 7;
    RunGen(engine, options, {});
    return engine.forwards;
  };
  const auto decode = [](const std::vector<std::vector<int>>& forwards, size_t prompt) {
    return std::vector<std::vector<int>>(forwards.begin() + prompt, forwards.end());  // prompt rows go through Forward when prefill is 0
  };
  const std::vector<int> plain_prompt{1, 3, 4};
  const size_t prompt_calls = run(plain_prompt, {}, 0, 1).size();
  // MTP chain: the first draft is a stop ID, so the chain stops and the step carries one draft.
  for (const auto& rows : decode(run(plain_prompt, {7}, 0, 20), prompt_calls)) {
    assert(rows.size() <= 2);
    for (size_t i = 0; i + 1 < rows.size(); ++i) assert(rows[i] != 7);
  }
  // Lookup copy: the prompt holds "1 2" followed by 7 3 3 3, and the copy crosses the stop ID 3.
  const std::vector<int> copy_prompt{1, 2, 7, 3, 3, 3, 1, 2};
  const size_t copy_calls = run(copy_prompt, {}, 0, 1).size();
  bool copied = false;
  for (const auto& rows : decode(run(copy_prompt, {3}, 7, 20), copy_calls)) {
    for (size_t i = 0; i + 1 < rows.size(); ++i) assert(rows[i] != 3);
    copied = copied || rows.size() == 3;
  }
  assert(copied);
  // Token limit: a step never carries more rows than the tokens still allowed.
  for (int n : {2, 3, 4, 5, 9}) {
    size_t committed = 1;
    for (const auto& rows : decode(run(plain_prompt, {}, 0, n), prompt_calls)) {
      assert(rows.size() <= size_t(n) - committed);
      ++committed;  // the draft 7 never matches the target 2, so each step commits one token
    }
  }
  std::puts("PASS: actual greedy and sampled RunGen skip all MTP work for one-token requests");
}
