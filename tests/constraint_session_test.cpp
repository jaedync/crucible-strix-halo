// Structured output in the batch session (plan tests 1 to 6) with the fake engine of batch_mock_engine.h: its greedy
// ID is (token + 1) % 16, so it prefers tokens that the grammar refuses. The 16 IDs are characters of a JSON
// object; 14 is the think end and 15 the stop token.
#include "batch_mock_engine.h"
#include "../engine/runtime/constraint_vocab.h"
#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <map>
#include <set>
#include <cmath>
#include <tuple>
#include <string>
#include <vector>

using namespace crucible;

namespace {
constexpr int kThink = 14, kStop = 15;
const std::vector<std::string> kTokens{"{", "\"", "k", "\":", "1", "2", ",", "}", " ", "a", "b", "[", "]", "x",
                                       "</think>", "<|end|>"};
const char* kSchema = R"({"type":"object","properties":{"k":{"type":"integer"}},"required":["k"],"additionalProperties":false})";
std::shared_ptr<ConstraintVocab>& Vocab() {
  static auto vocab = MakeConstraintVocab(kTokens, kStop);
  return vocab;
}
ConstraintFactory Factory() {
  return [](const ConstraintSpec& spec) { return MakeConstraint(*Vocab(), spec, kThink); };
}
ServeRequest Constrained(int token, int n, bool reasoning = false, double temperature = 0, uint64_t seed = 0) {
  auto r = Request(token, n, temperature, seed);
  r.stop_ids = {kStop};
  r.constraint = ConstraintSpec{"json_schema", kSchema, reasoning};
  return r;
}
// Every token is legal for a fresh state of the same grammar (a legal prefix; complete when it ends on the stop).
void CheckLegal(const std::vector<int>& tokens, bool reasoning) {
  auto c = MakeConstraint(*Vocab(), {"json_schema", kSchema, reasoning}, kThink);
  for (const int token : tokens) assert(c->Accept(token));
  if (!tokens.empty() && tokens.back() == kStop) assert(c->Terminated());
}
// A fake speculative decoder over the shared helpers: drafts are the mock's own chain (token + 1), cut by the grammar
// before the pass; acceptance takes the masked targets (AcceptRows), as BatchSpecDecoder does.
struct FakeSpec {
  std::map<uint64_t, std::shared_ptr<Constraint>> grammar;
  std::map<uint64_t, ServeRequest> requests;
  int cut = 0;
  bool cut_on = true;  // off: the illegal drafts reach the pass (the reference of the cut test)
  BatchSpecHooks Hooks(MockEngine& engine) {
    BatchSpecHooks hooks;
    hooks.check = [](const ServeRequest&) { return std::string(); };
    hooks.open = [this](uint64_t id, int, const ServeRequest& r, size_t) {
      grammar[id] = r.grammar;
      requests[id] = r;
      penalties[id] = sampling::RequestPenalties(r.ids, r.sampling, 16);  // as BatchSpecDecoder's State
    };
    hooks.chunk = [](uint64_t, size_t, int, bool) {};
    hooks.first = [this](uint64_t id, int token) { Emitted(penalties.at(id), {token}); };
    hooks.step = [](uint64_t, int last, size_t, uint64_t) { return std::vector<int>{last}; };
    hooks.close = [this](uint64_t id) { grammar.erase(id); requests.erase(id); penalties.erase(id); };
    hooks.propose = [this](uint64_t id, int last, size_t remaining, uint64_t position, int cap) {
      std::vector<int> drafts;
      for (int k = 0; k < std::min<int>(cap, int(remaining) - 1); ++k) drafts.push_back(((drafts.empty() ? last : drafts.back()) + 1) % 16);
      if (grammar[id] && cut_on) cut += CutDrafts(*grammar[id], drafts);
      std::vector<int> rows{last};
      rows.insert(rows.end(), drafts.begin(), drafts.end());
      pending[id] = {rows, position};
      return VerifySpan{-1, rows, 0, 1};
    };
    hooks.accept = [this, &engine](uint64_t id, int first, std::span<const int> greedy) {
      const auto& [rows, position] = pending.at(id);
      const ServeRequest& r = requests.at(id);
      const bool sampled = r.sampling.HostDraw();
      auto& history = penalties.at(id);
      // The mock copies one row per call; collect the rows of the span.
      std::vector<float> logits;
      if (sampled) {
        for (size_t row = 0; row < rows.size(); ++row) {
          const auto one = engine.CopyLogits(1, first + int(row));
          logits.insert(logits.end(), one.begin(), one.end());
        }
      }
      if (!grammar.at(id)) {
        // As BatchSpecDecoder::Accept for a request with no constraint.
        const auto verified = sampling::MatchAccept(std::span<const int>(rows).subspan(1), &history, [&](size_t row) {
          return sampled ? sampling::Draw(std::span<const float>(logits).subspan(row * 16, 16), r.sampling, history,
                                          position + row)
                         : greedy[row];
        });
        Emitted(history, verified);
        return verified;
      }
      const auto verified = AcceptRows(grammar.at(id).get(), rows, greedy, r.sampling, position, logits, 16,
                                       [&](int row) { return engine.CopyLogits(1, first + row); }, &history);
      Emitted(history, verified);
      return verified;
    };
    hooks.settle = [] {};
    return hooks;
  }
  std::map<uint64_t, std::pair<std::vector<int>, uint64_t>> pending;
  std::map<uint64_t, sampling::PenaltyHistory> penalties;
  // As BatchSpecDecoder::Emit: the emitted tokens join the history.
  static void Emitted(sampling::PenaltyHistory& history, const std::vector<int>& tokens) {
    if (history.Active())
      for (const int token : tokens) history.Add(token);
  }
};

std::vector<int> Tokens(const std::vector<BatchEvent>& events, uint64_t id) {
  std::vector<int> tokens;
  for (const auto& event : events) if (event.request_id == id && event.token) tokens.push_back(*event.token);
  return tokens;
}
// Runs the requests to their ends with packed verify passes (spec) or packed plain steps.
std::map<uint64_t, std::vector<int>> Run(const std::map<uint64_t, ServeRequest>& requests, bool verify,
                                         int* pass_rows = nullptr, int* cut = nullptr, bool cut_on = true) {
  MockEngine engine; engine.prefill_rows = 4;
  BatchSession<MockEngine> session(engine, 4);
  session.SetConstraintFactory(Factory());
  FakeSpec spec;
  spec.cut_on = cut_on;
  session.SetSpecHooks(spec.Hooks(engine));
  std::map<uint64_t, std::vector<int>> out;
  std::set<uint64_t> live;
  for (const auto& [id, r] : requests) {
    const auto events = session.Open(id, r);
    for (const int t : Tokens(events, id)) out[id].push_back(t);
    bool done = false;
    for (const auto& e : events) done |= e.done;
    if (!done) live.insert(id);
  }
  while (!live.empty()) {
    const std::vector<uint64_t> ids(live.begin(), live.end());
    const auto events = verify ? session.VerifyStep(ids) : session.Step(ids);
    for (const auto& e : events) {
      if (e.token) out[e.request_id].push_back(*e.token);
      if (e.done || e.error) live.erase(e.request_id);  // an error: a grammar dead end ends its own request
    }
  }
  if (pass_rows) { *pass_rows = 0; for (const int w : engine.pass_widths) *pass_rows += w; }
  if (cut) *cut = spec.cut;
  return out;
}

void Legal() {
  int ended = 0;
  for (const double temperature : {0.0, 1.0})
    for (const int prompt : {0, 5, 13}) {
      const auto tokens = Run({{1, Constrained(prompt, 24, false, temperature, 3)}}, false)[1];
      assert(tokens.size() == 24 || tokens.back() == kStop);
      // An output that ends is a complete object; one that the token limit cuts is a legal prefix. (The greedy fake
      // model loops on a digit: its favorite legal token.)
      CheckLegal(tokens, false);
      ended += tokens.back() == kStop;
    }
  assert(ended >= 3);
  std::puts("PASS: greedy and sampled constrained requests give only legal tokens (complete or a legal prefix)");
}

void Exact() {
  for (const double temperature : {0.0, 1.0}) {
    auto solo = Constrained(5, 24, false, temperature, 9);
    const auto alone = Run({{1, solo}}, false)[1];
    auto spec = solo; spec.spec = 7;
    int rows_spec = 0, cut = 0;
    const auto drafted = Run({{1, spec}}, true, &rows_spec, &cut)[1];
    assert(drafted == alone);
    assert(cut > 0);  // the grammar cut drafts before the pass
    // Packed beside three unconstrained requests (two speculative).
    std::map<uint64_t, ServeRequest> mixed{{1, spec}};
    for (uint64_t id = 2; id <= 4; ++id) {
      auto r = Request(int(id) * 3, 12, temperature, id);
      if (id != 4) r.spec = 7;
      mixed[id] = r;
    }
    const auto packed = Run(mixed, true);
    assert(packed.at(1) == alone);
    // The neighbors give the tokens they give with no constrained request beside them.
    std::map<uint64_t, ServeRequest> without(mixed);
    without.erase(1);
    const auto plain = Run(without, true);
    for (uint64_t id = 2; id <= 4; ++id) assert(packed.at(id) == plain.at(id));
    // Packed plain steps (spec 0 for all) give the same tokens again.
    std::map<uint64_t, ServeRequest> stepped{{1, solo}};
    for (uint64_t id = 2; id <= 4; ++id) { auto r = mixed[id]; r.spec = 0; stepped[id] = r; }
    const auto steps = Run(stepped, false);
    assert(steps.at(1) == alone);
    for (uint64_t id = 2; id <= 4; ++id) assert(steps.at(id) == plain.at(id));
  }
  std::puts("PASS: spec 0, spec 7 and packed passes give equal tokens; neighbors are unchanged; drafts are cut");
}

void DraftCut() {
  for (const double temperature : {0.0, 1.0}) {
    auto spec = Constrained(5, 24, false, temperature, 4); spec.spec = 7;
    int cut_rows = 0, full_rows = 0, cut = 0;
    const auto a = Run({{1, spec}}, true, &cut_rows, &cut);
    const auto b = Run({{1, spec}}, true, &full_rows, nullptr, false);
    // The cut changes no token, and the passes hold exactly the cut drafts fewer rows.
    assert(a == b && cut > 0 && full_rows - cut_rows == cut);
  }
  std::puts("PASS: an illegal draft is cut before the pass (same tokens, fewer rows)");
}

// Review H1 (session): a constrained request drafts the stop token when its document is complete. Its verify span
// ends at the accepted stop: the request ends there, the session is not poisoned, and a speculative peer in the
// same packed passes keeps the tokens it gives alone. Before the fix the row after the stop had no legal token, the
// step threw and the poisoned session failed every request of the batch.
std::map<uint64_t, std::vector<int>> RunStopDraft(const std::map<uint64_t, ServeRequest>& requests) {
  MockEngine engine; engine.prefill_rows = 4;
  BatchSession<MockEngine> session(engine, 2);
  session.SetConstraintFactory(Factory());
  FakeSpec spec;
  auto hooks = spec.Hooks(engine);
  const auto chain = hooks.propose;
  hooks.propose = [&spec, chain](uint64_t id, int last, size_t remaining, uint64_t position, int cap) {
    auto span = chain(id, last, remaining, position, cap);
    if (cap >= 1 && spec.grammar[id] && spec.grammar[id]->Allows(kStop)) {
      span.tokens = {last, kStop};
      spec.pending[id].first = span.tokens;
    }
    return span;
  };
  session.SetSpecHooks(hooks);
  std::map<uint64_t, std::vector<int>> out;
  std::set<uint64_t> live;
  for (const auto& [id, r] : requests) {
    bool done = false;
    for (const auto& e : session.Open(id, r)) {
      if (e.token) out[id].push_back(*e.token);
      done |= e.done;
    }
    if (!done) live.insert(id);
  }
  while (!live.empty()) {
    for (const auto& e : session.VerifyStep(std::vector<uint64_t>(live.begin(), live.end()))) {
      if (e.token) out[e.request_id].push_back(*e.token);
      if (e.done) live.erase(e.request_id);
    }
    assert(!session.Poisoned());
  }
  return out;
}
void StopDraft() {
  for (const double temperature : {0.0, 1.0}) {
    auto r = Constrained(5, 8, false, temperature, 5);
    r.spec = 7;
    r.constraint->spec = R"({"const":{}})";
    r.ids = {5, 0};  // the output so far: "{"
    r.constraint_from = 1;
    auto peer = Request(3, 12, temperature, 2);
    peer.spec = 7;
    const auto alone = RunStopDraft({{2, peer}});
    const auto both = RunStopDraft({{1, r}, {2, peer}});
    assert((both.at(1) == std::vector<int>{7, kStop}));  // "}", then the stop draft, accepted
    assert(both.at(2) == alone.at(2) && alone.at(2).size() == 12);
  }
  std::puts("PASS: an accepted stop draft ends its request only; the session stays healthy and a peer keeps its tokens");
}

// A grammar that takes any token but the stop for `left` tokens and then none (a dead end in the middle of an output,
// as an empty grammar node gives). The owner's rule (Nov 16): a dead end ends ONLY its own request, with an error
// event and its slot free; the session is not poisoned and a peer keeps its tokens. Before the fix the step threw
// "constraint has no legal token" and the poisoned session failed every request.
class DeadEndGrammar : public GrammarState {
 public:
  explicit DeadEndGrammar(int left) : left_(left) {}
  bool Accept(int token) override {
    if (taken_ >= left_ || token == kStop) return false;
    ++taken_;
    return true;
  }
  void Rollback(int tokens) override { taken_ -= tokens; }
  void FillMask(std::span<uint32_t> words) override {
    for (auto& word : words) word = 0;
    if (taken_ < left_) words[0] = 0xffffu & ~(1u << kStop);
  }
  bool Terminated() const override { return false; }

 private:
  int left_, taken_ = 0;
};
ConstraintFactory DeadEndFactory() {
  return [](const ConstraintSpec& spec) -> std::shared_ptr<Constraint> {
    if (spec.spec.rfind("dead:", 0) == 0)
      return std::make_shared<Constraint>(std::make_unique<DeadEndGrammar>(std::stoi(spec.spec.substr(5))), 16, kStop,
                                          spec.reasoning, kThink);
    return Factory()(spec);
  };
}
struct Outcome {
  std::vector<int> tokens;
  int errors = 0, dones = 0;
  std::string error;
};
std::map<uint64_t, Outcome> RunDeadEnd(const std::map<uint64_t, ServeRequest>& requests, bool verify) {
  MockEngine engine; engine.prefill_rows = 4;
  BatchSession<MockEngine> session(engine, 2);
  session.SetConstraintFactory(DeadEndFactory());
  FakeSpec spec;
  session.SetSpecHooks(spec.Hooks(engine));
  std::map<uint64_t, Outcome> out;
  std::set<uint64_t> live;
  const auto take = [&](const std::vector<BatchEvent>& events) {
    for (const auto& e : events) {
      auto& o = out[e.request_id];
      if (e.token) o.tokens.push_back(*e.token);
      if (e.done) { ++o.dones; live.erase(e.request_id); }
      if (e.error) { ++o.errors; o.error = *e.error; live.erase(e.request_id); }
    }
  };
  for (const auto& [id, r] : requests) {
    live.insert(id);
    take(session.Open(id, r));
  }
  while (!live.empty()) {
    take(verify ? session.VerifyStep(std::vector<uint64_t>(live.begin(), live.end()))
                : session.Step(std::vector<uint64_t>(live.begin(), live.end())));
    assert(!session.Poisoned());
  }
  // The dead request's slot is free: two more opens fit in the two slots.
  for (uint64_t id = 10; id < 12; ++id) take(session.Open(id, Request(3, 1)));
  return out;
}
void DeadEnd() {
  for (const bool verify : {false, true})
    for (const double temperature : {0.0, 1.0}) {
      auto r = Constrained(5, 20, false, temperature, 3);
      r.constraint->spec = "dead:4";
      auto peer = Request(3, 12, temperature, 2);
      if (verify) r.spec = peer.spec = 7;
      const auto alone = RunDeadEnd({{2, peer}}, verify);
      const auto both = RunDeadEnd({{1, r}, {2, peer}}, verify);
      const auto& dead = both.at(1);
      assert(dead.errors == 1 && dead.dones == 0 && dead.tokens.size() == 4);
      assert(dead.error == std::string(kConstraintDeadEndMessage));
      assert(both.at(2).tokens == alone.at(2).tokens && alone.at(2).tokens.size() == 12 && both.at(2).dones == 1);
    }
  std::puts("PASS: a grammar dead end ends its own request with an error event; the session and a peer go on");
}

// Penalties with structured output: the penalties change the row first, then the grammar
// chooses among its legal IDs only, so a penalty never makes a refused ID possible. The reference writes the formula
// out apart from sampling_penalty.h; the choice is the trunk's masked rule (masked argmax, or the keyed draw).
ServeRequest PenalizedConstrained(double temperature, double repetition, double presence, double frequency) {
  auto r = Constrained(5, 24, false, temperature, 7);
  r.ids = {5, 13, 5};  // the prompt holds 5 and 13 ("x", which the grammar never takes)
  r.sampling.repetition_penalty = repetition;
  r.sampling.presence_penalty = presence;
  r.sampling.frequency_penalty = frequency;
  return r;
}
// A grammar dead end (no legal ID, for example 16 spaces where only ":" may follow and the vocabulary has no ":")
// ends the reference as the session ends the request: with the tokens before it.
std::vector<int> PenalizedReference(const ServeRequest& r, bool* dead = nullptr) {
  auto c = MakeConstraint(*Vocab(), *r.constraint, kThink);
  const std::set<int> prompt(r.ids.begin(), r.ids.end());
  std::vector<int> output, count(16, 0);
  int last = r.ids.back();
  sampling::Options plain;
  plain.temperature = r.sampling.temperature;
  plain.seed = r.sampling.seed;
  while (int(output.size()) < r.max_tokens) {
    const int center = (last + 1) % 16;
    std::vector<float> row(16);
    for (int id = 0; id < 16; ++id) {
      float v = -float(std::abs(center - id));
      if (prompt.count(id) || count[size_t(id)]) {
        const float rp = float(r.sampling.repetition_penalty);
        v = v > 0 ? v / rp : v * rp;
      }
      if (count[size_t(id)]) v = v - float(r.sampling.frequency_penalty) * float(count[size_t(id)]) -
                                 float(r.sampling.presence_penalty);
      row[size_t(id)] = v;
    }
    std::vector<uint32_t> mask;
    c->FillMask(mask);
    if (std::all_of(mask.begin(), mask.end(), [](uint32_t word) { return word == 0; })) {
      if (dead) *dead = true;
      break;
    }
    const int token = plain.temperature == 0
        ? MaskedArgmax(row, mask)
        : MaskedKeyedDraw(row, plain, mask, plain.seed, r.ids.size() + output.size());
    assert(token >= 0 && MaskAllows(mask, token) && c->Accept(token));
    output.push_back(token);
    ++count[size_t(token)];
    last = token;
    if (token == kStop) break;
  }
  return output;
}
void Penalties() {
  int changed = 0, dead_ends = 0;
  for (const double temperature : {0.0, 1.0})
    for (const auto& [repetition, presence, frequency] :
         std::vector<std::tuple<double, double, double>>{{1.6, 0.8, 0.5}, {1, -2, -2}, {0.5, 0, 0}, {2, 2, 2}}) {
      const auto r = PenalizedConstrained(temperature, repetition, presence, frequency);
      const auto tokens = Run({{1, r}}, false)[1];
      CheckLegal(tokens, false);
      bool dead = false;
      assert(tokens == PenalizedReference(r, &dead));
      dead_ends += dead;
      auto neutral = r;
      neutral.sampling.repetition_penalty = 1;
      neutral.sampling.presence_penalty = neutral.sampling.frequency_penalty = 0;
      const auto plain = Run({{1, neutral}}, false)[1];
      changed += tokens != plain;
      // Packed beside a neutral constrained request and a plain one: both sides keep their tokens.
      const auto packed = Run({{1, r}, {2, neutral}, {3, Request(4, 12, temperature, 3)}}, false);
      assert(packed.at(1) == tokens && packed.at(2) == plain);
      assert(packed.at(3) == Run({{3, Request(4, 12, temperature, 3)}}, false).at(3));
      // Verify passes with drafts (the grammar cuts them) give the tokens of plain steps: row r counts the drafts
      // before it as output, as a run without drafts does.
      auto spec = r;
      spec.spec = 7;
      int cut = 0;
      assert(Run({{1, spec}}, true, nullptr, &cut)[1] == tokens);
      auto peer = Request(4, 12, temperature, 3);
      peer.spec = 7;
      const auto verified = Run({{1, spec}, {2, peer}}, true);
      assert(verified.at(1) == tokens && verified.at(2) == Run({{2, peer}}, true).at(2));
    }
  // 5 of the 8 runs differ from the neutral run; negative penalties repeat a space into a dead end (3 runs).
  assert(changed >= 4 && dead_ends >= 1);
  std::puts("PASS: penalized constrained requests follow penalty-then-mask, stay legal, and change no neighbor");
}

void Thinking() {
  auto r = Constrained(5, 30, true);
  const auto tokens = Run({{1, r}}, false)[1];
  auto free = Request(5, 30); free.stop_ids = {kStop};
  const auto unconstrained = Run({{1, free}}, false)[1];
  // Before the think end the tokens are the model's own: 6, 7, ..., 14.
  size_t arm = 0;
  while (arm < tokens.size() && tokens[arm] != kThink) ++arm;
  assert(arm < tokens.size() && arm > 0);
  for (size_t i = 0; i <= arm; ++i) assert(tokens[i] == unconstrained[i]);
  CheckLegal(tokens, true);
  std::puts("PASS: reasoning tokens are not masked; the grammar arms at the think end");
}

void Continuation() {
  const auto full = Run({{1, Constrained(5, 20, true)}}, false)[1];
  assert(full.size() > 12);
  auto first = Constrained(5, 12, true);
  const auto head = Run({{1, first}}, false)[1];
  assert(std::equal(head.begin(), head.end(), full.begin()));
  auto next = Constrained(5, 20 - int(head.size()), true);
  next.ids.insert(next.ids.end(), head.begin(), head.end());
  next.constraint_from = 1;
  const auto tail = Run({{1, next}}, false)[1];
  std::vector<int> joined = head;
  joined.insert(joined.end(), tail.begin(), tail.end());
  assert(joined == full);
  std::puts("PASS: a continuation with constraint_from gives the tokens of one long job");
}

void Refusals() {
  MockEngine engine; engine.prefill_rows = 4;
  BatchSession<MockEngine> session(engine, 2);
  const auto refused = [&](const ServeRequest& r) {
    try { session.Open(7, r); } catch (const RecoverableOpenError&) { return !session.Poisoned(); }
    return false;
  };
  assert(refused(Constrained(5, 4)));  // no factory: this engine has no structured output
  session.SetConstraintFactory(Factory());
  auto no_stop = Constrained(5, 4); no_stop.stop_ids = {};
  assert(refused(no_stop));
  auto bad = Constrained(5, 4); bad.constraint->spec = "{";
  assert(refused(bad));
  auto beyond = Constrained(5, 4); beyond.constraint_from = 2;
  assert(refused(beyond));
  auto illegal = Constrained(5, 4); illegal.ids = {5, 13}; illegal.constraint_from = 1;  // "x" is no JSON start
  assert(refused(illegal));
  auto scoring = Constrained(5, 1); scoring.top_logprobs = 2;
  assert(refused(scoring));
  // Review H1, plain path: an output so far that ends on the stop token leaves no legal token for the first sample.
  // The open refuses it; before the fix the first sample threw no legal token and poisoned the session.
  auto complete = Constrained(5, 4); complete.ids = {5, 0, 7, kStop}; complete.constraint_from = 1;
  complete.constraint->spec = R"({"const":{}})";
  assert(refused(complete));
  // The done event carries the counters (numbers only).
  const auto events = session.Open(8, Constrained(5, 24));
  std::vector<BatchEvent> all(events);
  for (;;) {
    bool done = false;
    for (const auto& e : all) done |= e.done;
    if (done) break;
    const auto more = session.Step({8});
    all.insert(all.end(), more.begin(), more.end());
  }
  for (const auto& e : all)
    if (e.done) {
      assert(e.constraint && e.constraint->tokens > 0 && e.constraint->refusals > 0);
      assert(BatchConstraintFields(e).find("\"constraint\":{\"tokens\":") == 1);
    }
  // A request with no constraint has no constraint fields: its done line stays byte-identical.
  BatchEvent plain; plain.done = true;
  assert(BatchConstraintFields(plain).empty());
  std::puts("PASS: bad constraints are refused at open; the done event carries the counters");
}

void Protocol() {
  const auto r = ParseServeRequest(
      R"({"ids":[1,2,3],"max_tokens":4,"constraint":{"kind":"json_schema","spec":"{\"type\":\"object\"}","reasoning":1},"constraint_from":2})",
      64);
  assert(r.constraint && r.constraint->kind == "json_schema" && r.constraint->spec == R"({"type":"object"})" &&
         r.constraint->reasoning && r.constraint_from == 2);
  const auto plain = ParseServeRequest(R"({"ids":[1,2,3],"max_tokens":4})", 64);
  assert(!plain.constraint && plain.constraint_from == -1 && !plain.grammar);
  for (const char* bad : {R"({"ids":[1],"constraint":{"kind":"regex","spec":"a"}})",
                          R"({"ids":[1],"constraint":{"kind":"json_object","spec":"","extra":1}})",
                          R"({"ids":[1],"constraint_from":0})",
                          R"({"ids":[1,2],"constraint":{"kind":"json_object","spec":""},"constraint_from":3})"}) {
    bool thrown = false;
    try { ParseServeRequest(bad, 64); } catch (const std::runtime_error&) { thrown = true; }
    assert(thrown);
  }
  assert(BatchReadyConstraintField() == ",\"constraint\":1");
  std::puts("PASS: protocol keys constraint and constraint_from");
}
// The serve loop never waits for a compile.
// The serve factory queues the compile and refuses the open as constraint_pending; the server opens it again. While
// one compile takes its full deadline, the other requests keep their pass rate, and the slow spec ends as a
// constraint_limit refusal for its own open only.
void ServeLoop() {
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
  MockEngine engine; engine.prefill_rows = 4; engine.context = 1 << 16;
  BatchSession<MockEngine> session(engine, 4);
  session.SetConstraintFactory(ServeConstraintFactory(StartConstraintService([] { return Vocab(); }), kThink));
  const auto try_open = [&](uint64_t id, const ServeRequest& r, std::string& refusal) {
    try {
      session.Open(id, r);
      return true;
    } catch (const RecoverableOpenError& error) {
      refusal = error.what();
      return false;
    }
  };
  // The first open of a spec is pending; it opens once the worker has compiled it.
  std::string refusal;
  auto t = Clock::now();
  assert(!try_open(1, Constrained(0, 40), refusal) && refusal.rfind("constraint_pending:", 0) == 0);
  assert(ms(t) < 100);
  while (!try_open(1, Constrained(0, 40), refusal)) {
    assert(refusal.rfind("constraint_pending:", 0) == 0 && ms(t) < 30000);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto plain = Request(3, 60000);
  session.Open(2, plain);
  // A slow spec: its compile takes the helper's full deadline (over 300 s without it).
  auto slow = Constrained(0, 8);
  slow.constraint->spec = R"({"type":"string","pattern":"[ab]{1,50000}x"})";
  const auto start = Clock::now();
  double worst_open = 0, worst_step = 0;
  int steps = 0;
  bool live1 = true;
  for (;;) {
    t = Clock::now();
    const bool opened = try_open(3, slow, refusal);
    worst_open = std::max(worst_open, ms(t));
    assert(!opened && !session.Poisoned());
    if (refusal.rfind("constraint_pending:", 0) != 0) break;
    t = Clock::now();
    const auto events = session.Step(live1 ? std::vector<uint64_t>{1, 2} : std::vector<uint64_t>{2});
    worst_step = std::max(worst_step, ms(t));
    for (const auto& e : events) if (e.done && e.request_id == 1) live1 = false;
    ++steps;
    assert(ms(start) < 30000);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  const double compile_ms = ms(start);
  assert(refusal.rfind("constraint_limit:", 0) == 0 && refusal.find("kHelperDeadlineMs") != std::string::npos);
  assert(compile_ms >= kHelperDeadlineMs - 100);
  assert(worst_open < 100 && worst_step < 100 && steps > 500);
  // The refusal is for that spec only: it stays refused at once; the plain request still steps.
  t = Clock::now();
  assert(!try_open(4, slow, refusal) && refusal.rfind("constraint_limit:", 0) == 0 && ms(t) < 100);
  assert(!session.Step({2}).empty());
  std::printf("PASS: a %.0f ms compile never blocks the serve loop: %d steps, worst open %.2f ms, worst step %.2f ms\n",
              compile_ms, steps, worst_open, worst_step);
}
}  // namespace

int main() {
  Protocol();
  Legal();
  Exact();
  DraftCut();
  StopDraft();
  DeadEnd();
  Thinking();
  Continuation();
  Penalties();
  Refusals();
  ServeLoop();
  std::puts("PASS: constraint session");
}
