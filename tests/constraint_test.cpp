// Structured output: the grammar state (engine/runtime/constraint.h) on xgrammar 0.2.9 (constraint_vocab.h).
// A synthetic character vocabulary always runs. The real tokenizer runs when CRUCIBLE_TOKENIZER_JSON names it, or
// from the test machine copy of the checkpoint folder; otherwise that part prints SKIP.
#include "../engine/runtime/constraint_vocab.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace crucible;

namespace {
// Token i < 95 is chr(32 + i); 95 TAB, 96 newline, 97 the stop token, 98 the think end, 99 a two-character token.
constexpr int kTab = 95, kNewline = 96, kStop = 97, kThink = 98, kQuoteColon = 99;
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
int Id(char c) {
  assert(c >= 32 && c < 127);
  return c - 32;
}
std::vector<int> Ids(const std::string& text) {
  std::vector<int> ids;
  for (const char c : text) ids.push_back(c == '\t' ? kTab : c == '\n' ? kNewline : Id(c));
  return ids;
}
std::vector<uint32_t> Mask(Constraint& c) {
  std::vector<uint32_t> words;
  c.FillMask(words);
  return words;
}
const char* kPerson =
    R"({"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer"}},"required":["name","n"]})";

void Stream(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, false});
  assert(c->Armed() && !c->Terminated());
  for (const int id : Ids(R"({"name":"ab","n":12})")) assert(c->Accept(id));
  assert(c->Allows(kStop));
  assert(c->Accept(kStop));
  assert(c->Terminated());
  assert(!c->Allows(Id(' ')));
  std::puts("PASS: a schema stream is legal and ends on the stop token");
}

void Refused(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, false});
  assert(c->Accept(Id('{')));
  const auto before = Mask(*c);
  assert(!c->Accept(Id('x')));
  assert(!c->Allows(Id('x')));
  assert(!c->Allows(kStop));  // the stop token is legal only where the grammar accepts
  assert(Mask(*c) == before);
  assert(c->Accept(Id('"')));
  std::puts("PASS: a refused token returns false and changes nothing");
}

void Walk(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, false});
  for (const int id : Ids(R"({"name":"ab","n":)")) assert(c->Accept(id));
  const auto before = Mask(*c);
  const std::vector<int> drafts{Id('1'), Id('2'), Id('3'), Id('x'), Id('4')};
  assert(c->WalkDrafts(drafts) == 3);
  assert(Mask(*c) == before);
  std::vector<int> cut = drafts;
  assert(CutDrafts(*c, cut) == 2 && cut.size() == 3 && c->counts.cut_drafts == 2);
  assert(Mask(*c) == before);
  // Rollback of 8 restores the state: the next mask equals the mask before.
  for (const int id : Ids("1234567")) assert(c->Accept(id));
  assert(c->Accept(Id('}')));
  c->Rollback(8);
  assert(Mask(*c) == before);
  std::puts("PASS: the draft walk stops at the first illegal draft; rollback of 8 restores the mask");
}

// The defect of xgrammar 0.2.9 that the property prototype found (findings 4.2): a raw control character inside a
// string with minLength or maxLength. The vendored copy is patched.
void ControlCharacters(ConstraintVocab& vocab) {
  for (const char* schema : {R"({"type":"string","maxLength":40})", R"({"type":"string","minLength":1})",
                             R"({"type":"string"})", R"({"type":"string","minLength":1,"maxLength":3})"}) {
    auto c = MakeConstraint(vocab, {"json_schema", schema, false});
    assert(c->Accept(Id('"')) && c->Accept(Id('a')));
    assert(!c->Allows(kTab));
    assert(!c->Allows(kNewline));
    assert(c->Accept(Id('b')) && c->Accept(Id('"')) && c->Accept(kStop));
  }
  std::puts("PASS: no raw control character inside a length-bounded string");
}

void Thinking(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, true}, kThink);
  assert(!c->Armed());
  assert(c->Allows(Id('x')) && c->Allows(kStop));
  const auto all = Mask(*c);
  for (int t = 0; t < ConstraintVocabRows(vocab); ++t) assert(MaskAllows(all, t));
  assert(c->Accept(Id('x')) && c->Accept(Id('y')));
  assert(c->Accept(kThink) && c->Armed());
  assert(!c->Allows(Id('x')));
  assert(c->Allows(kNewline));  // the template's newlines after </think> are leading whitespace
  assert(c->Accept(Id('{')));
  // A rollback across the think end leaves the grammar again.
  c->Rollback(2);
  assert(!c->Armed() && c->Allows(Id('x')));
  assert(c->Accept(kThink) && c->Accept(Id('{')) && c->Accept(kQuoteColon) == false);
  // The draft walk arms in the middle of a chain.
  auto d = MakeConstraint(vocab, {"json_schema", kPerson, true}, kThink);
  assert(d->WalkDrafts(std::vector<int>{Id('a'), kThink, Id('{'), Id('x')}) == 3);
  assert(!d->Armed());
  std::puts("PASS: reasoning tokens are not masked; the grammar arms at the think end");
}

void JsonObject(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_object", "", false});
  assert(!c->Allows(Id('[')) && !c->Allows(Id('1')));
  for (const int id : Ids(R"({"a":[1,2],"b":{"c":null}})")) assert(c->Accept(id));
  assert(c->Accept(kStop));
  std::puts("PASS: json_object takes an object only");
}

void StructuralTag(ConstraintVocab& vocab) {
  // A call tag with the template's exact newlines and one enum parameter.
  const std::string tag = R"({"type":"structural_tag","format":{"type":"tag","begin":"<f>\n","content":)"
                          R"({"type":"or","elements":[{"type":"const_string","value":"m1"},{"type":"const_string","value":"m2"}]},)"
                          R"("end":"\n</f>"}})";
  auto c = MakeConstraint(vocab, {"structural_tag", tag, false});
  for (const int id : Ids("<f>\nm")) assert(c->Accept(id));
  assert(!c->Allows(Id('3')) && c->Allows(Id('2')));
  for (const int id : Ids("2\n</f>")) assert(c->Accept(id));
  assert(c->Accept(kStop));
  auto d = MakeConstraint(vocab, {"structural_tag", tag, false});
  for (const int id : Ids("<f>\n")) assert(d->Accept(id));
  assert(!d->Allows(Id(' ')) && !d->Allows(kNewline));  // no whitespace that the call parser refuses
  std::puts("PASS: a structural tag keeps its exact newlines");
}

void Errors(ConstraintVocab& vocab) {
  for (const auto& spec : std::vector<ConstraintSpec>{{"json_schema", "{not json", false}, {"regex", "a", false},
                                                      {"structural_tag", R"({"type":"structural_tag"})", false},
                                                      {"json_schema", R"({"type":"string","minLength":5,"maxLength":2})", false}}) {
    bool thrown = false;
    try { MakeConstraint(vocab, spec); } catch (const ConstraintError& error) {
      thrown = true;
      // Fixed messages: no part of the spec text.
      assert(std::string(error.what()).find("not json") == std::string::npos);
    }
    assert(thrown);
  }
  std::puts("PASS: a bad constraint gives a fixed-message ConstraintError");
}

void Cache(ConstraintVocab& vocab) {
  auto a = MakeConstraint(vocab, {"json_schema", kPerson, false});
  auto b = MakeConstraint(vocab, {"json_schema", kPerson, false});
  assert(b->counts.cache_hit == 1);
  assert(a->Accept(Id('{')) && !b->Allows(Id('x')) && b->Allows(Id('{')));  // two states of one grammar
  std::puts("PASS: a repeated spec uses the compiled grammar cache; each request has its own state");
}

void Argmax() {
  const std::vector<float> row{1, 3, 3, 2, -std::numeric_limits<float>::infinity()};
  std::vector<uint32_t> all{0x1f}, no1{0x1d}, only03{0x09}, only4{0x10}, none{0};
  assert(MaskedArgmax(row, all) == 1);
  assert(MaskedArgmax(row, no1) == 2);
  assert(MaskedArgmax(row, only03) == 3);
  assert(MaskedArgmax(row, only4) == 4);
  assert(MaskedArgmax(row, none) == -1);
  std::puts("PASS: masked argmax keeps the device tie rule (lower ID)");
}

void Greedy(ConstraintVocab& vocab) {
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, false});
  int copies = 0;
  std::vector<float> logits(size_t(ConstraintVocabRows(vocab)), 0);
  logits[size_t(Id('x'))] = 9;
  logits[size_t(Id('{'))] = 1;
  const auto row = [&] { ++copies; return logits; };
  assert(ChooseGreedy(*c, Id('x'), row) == Id('{') && copies == 1 && c->counts.refusals == 1);
  assert(ChooseGreedy(*c, Id('"'), row) == Id('"') && copies == 1 && c->counts.refusals == 1);
  assert(ChooseGreedy(*c, -1, row) == -1);  // a nonfinite row keeps the device's error
  std::puts("PASS: a legal device ID is kept with no copy; a refused one takes the masked argmax");
}

// Penalties, then the mask (ChoosePenalized): after "{" the grammar takes '"' and whitespace, never 'x'. A negative
// frequency and presence penalty lifts the output token 'x' over every legal ID, and the choice still takes a legal
// ID; a repetition penalty on '"' (in the prompt) makes the space win. Greedy and 2,000 keyed draws.
void PenaltyMask(ConstraintVocab& vocab) {
  const int rows = ConstraintVocabRows(vocab);
  std::vector<float> logits(size_t(rows), -4);
  logits[size_t(Id('x'))] = -1;
  logits[size_t(Id('"'))] = 3;
  logits[size_t(Id(' '))] = 2;
  sampling::Options o;
  o.frequency_penalty = -2;
  o.presence_penalty = -2;
  o.repetition_penalty = 2;
  const std::vector<int> ids{Id('"'), Id('x'), Id('x'), Id('x')};
  const sampling::PenaltyHistory history(ids, 1, rows);  // '"' is prompt; 'x' is output 3 times
  std::vector<float> penalized = logits;
  history.Apply(penalized, o);
  assert(penalized[size_t(Id('x'))] == 6 && penalized[size_t(Id('"'))] == 1.5f);  // -1 * 2 + 6 + 2; 3 / 2
  auto c = MakeConstraint(vocab, {"json_schema", kPerson, false});
  assert(c->Accept(Id('{')) && !c->Allows(Id('x')) && c->Allows(Id(' ')) && c->Allows(Id('"')));
  std::vector<uint32_t> mask = Mask(*c);
  assert(sampling::PenalizedArgmax(logits, o, history) == Id('x'));  // the unmasked choice is refused
  assert(ChoosePenalized(*c, logits, o, history, 4) == Id(' '));       // the masked argmax of the penalized row
  int legal = 0;
  for (uint64_t seed = 0; seed < 2000; ++seed) {
    auto s = MakeConstraint(vocab, {"json_schema", kPerson, false});
    assert(s->Accept(Id('{')));
    sampling::Options t = o;
    t.temperature = 1;
    t.seed = seed;
    const int token = ChoosePenalized(*s, logits, t, history, 4);
    assert(MaskAllows(mask, token) && token == MaskedKeyedDraw(penalized, t, mask, seed, 4));
    legal += token == Id(' ') || token == Id('"');
  }
  assert(legal > 1000);
  // Neutral options never take this path.
  bool thrown = false;
  try { ChoosePenalized(*c, logits, sampling::Options{}, history, 4); } catch (const std::logic_error&) { thrown = true; }
  assert(thrown);
  std::puts("PASS: penalties change the row first; the mask then keeps every choice legal (greedy and sampled)");
}

// A penalized verify span (AcceptRows with a history): row r counts the drafts before r as output, as a run without
// drafts does, and the history is the same again at the return.
void PenaltySpan(ConstraintVocab& vocab) {
  const size_t rows = size_t(ConstraintVocabRows(vocab));
  sampling::Options o;
  o.presence_penalty = 1.5;
  const std::vector<int> prompt{Id('a')};
  // Every row prefers '1' (3) over '2' (2): after one '1' is counted, the presence penalty makes '2' win.
  std::vector<float> logits(rows * 3, -8);
  for (size_t r = 0; r < 3; ++r) { logits[r * rows + size_t(Id('1'))] = 3; logits[r * rows + size_t(Id('2'))] = 2; }
  const auto no_copy = [](int) -> std::vector<float> { throw std::logic_error("a penalized span has every row"); };
  auto c = MakeConstraint(vocab, {"json_schema", R"({"type":"integer"})", false});
  sampling::PenaltyHistory history(prompt, 1, int(rows));
  const std::vector<int> input{Id('a'), Id('1'), Id('2')}, greedy{Id('1'), Id('1'), Id('1')};
  const auto verified = AcceptRows(c.get(), input, greedy, o, 1, logits, rows, no_copy, &history);
  assert((verified == std::vector<int>{Id('1'), Id('2'), Id('1')}));  // row 2 counts '1' and '2'
  assert(history.Count(Id('1')) == 0 && history.Count(Id('2')) == 0);
  // Without the history the neutral rule takes the device IDs.
  auto n = MakeConstraint(vocab, {"json_schema", R"({"type":"integer"})", false});
  sampling::Options neutral;
  assert((AcceptRows(n.get(), input, greedy, neutral, 1, logits, rows, no_copy) == std::vector<int>{Id('1'), Id('1')}));
  std::puts("PASS: a penalized verify span counts the drafts before each row and leaves the history as it was");
}

// Review H1: a verify span whose draft is the stop token at a complete document. The rows after an accepted stop
// are never masked or sampled: before the fix the row after it had no legal token and threw.
void StopDraft(ConstraintVocab& vocab) {
  const size_t rows = size_t(ConstraintVocabRows(vocab));
  const auto no_copy = [](int) -> std::vector<float> { throw std::logic_error("no row copy past the stop"); };
  for (const double temperature : {0.0, 1.0}) {
    sampling::Options o;
    o.temperature = temperature;
    o.seed = 3;
    for (const auto& input : {std::vector<int>{Id('1'), kStop}, std::vector<int>{Id('1'), kStop, Id(' ')}}) {
      auto c = MakeConstraint(vocab, {"json_schema", R"({"const":1})", false});
      assert(c->Accept(Id('1')));
      std::vector<int> drafts(input.begin() + 1, input.end());
      CutDrafts(*c, drafts);
      assert(drafts.size() == 1 && drafts[0] == kStop);  // the stop draft is legal; nothing after it is
      std::vector<int> greedy{kStop, Id(' '), Id('{')};
      greedy.resize(input.size());
      const std::vector<float> logits(rows * input.size(), 0);
      const auto verified = AcceptRows(c.get(), input, greedy, o, 0, logits, rows, no_copy);
      assert(verified == std::vector<int>{kStop} && c->Terminated());
    }
  }
  // In the reasoning phase the stop token ends the span too (the request ends there).
  auto r = MakeConstraint(vocab, {"json_schema", kPerson, true});
  sampling::Options greedy_options;
  const std::vector<int> input{Id('a'), kStop, Id('b')}, greedy{kStop, Id('b'), Id('c')};
  assert((AcceptRows(r.get(), input, greedy, greedy_options, 0, {}, rows, no_copy) == std::vector<int>{kStop}));
  std::puts("PASS: a verify span ends at an accepted stop draft (greedy, sampled, reasoning); no row after it runs");
}

// The lazy rule (3.b): the masked draw equals the full masked race over the kept and legal tokens, on 10,000 random
// rows and masks; and when the unmasked draw is legal, it is that draw.
int FullMaskedRace(const sampling::Distribution& kept, std::span<const uint32_t> mask, uint64_t seed, uint64_t position) {
  int best = -1;
  double rate = 0;
  for (size_t i = 0; i < kept.Ids().size(); ++i) {
    const int id = kept.Ids()[i];
    if (!MaskAllows(mask, id)) continue;
    const double value = kept.Probabilities()[i] / -std::log(sampling::KeyedUniform(seed, position, id));
    if (best < 0 || value > rate || (value == rate && id < best)) { best = id; rate = value; }
  }
  return best;
}
void LazyDraw() {
  std::mt19937_64 rng(7);
  std::normal_distribution<float> normal(0, 2);
  int lazy = 0, fallback = 0;
  for (int trial = 0; trial < 10000; ++trial) {
    const int vocab = 64;
    std::vector<float> logits(vocab);
    for (float& v : logits) v = normal(rng);
    std::vector<uint32_t> mask(2, 0);
    const double density = std::uniform_real_distribution<double>(0.02, 1.0)(rng);
    for (int t = 0; t < vocab; ++t)
      if (std::uniform_real_distribution<double>(0, 1)(rng) < density) mask[size_t(t) / 32] |= 1u << (t % 32);
    if (mask[0] == 0 && mask[1] == 0) mask[0] = 1;
    sampling::Options o;
    o.temperature = std::uniform_real_distribution<double>(0.3, 1.5)(rng);
    o.top_k = trial % 3 == 0 ? 8 : 0;
    o.top_p = trial % 4 == 0 ? 0.9 : 1;
    o.min_p = trial % 5 == 0 ? 0.05 : 0;
    const uint64_t seed = rng(), position = rng() % 100000;
    const auto kept = sampling::FromLogits(logits, o);
    int expected = FullMaskedRace(kept, mask, seed, position);
    if (expected < 0) {
      std::vector<float> masked = logits;
      for (int t = 0; t < vocab; ++t) if (!MaskAllows(mask, t)) masked[size_t(t)] = -std::numeric_limits<float>::infinity();
      expected = FullMaskedRace(sampling::FromLogits(masked, o), mask, seed, position);
      ++fallback;
    }
    assert(MaskedKeyedDraw(logits, o, mask, seed, position) == expected);
    const int drawn = kept.KeyedSample(seed, position);
    if (MaskAllows(mask, drawn)) { assert(drawn == expected); ++lazy; }
  }
  assert(lazy > 1000 && fallback > 10);
  std::printf("PASS: masked keyed draw equals the full masked race on 10000 rows (%d lazy, %d mask-first)\n", lazy,
              fallback);
}

std::string RealTokenizer() {
  if (const char* path = std::getenv("CRUCIBLE_TOKENIZER_JSON")) return path;
  const char* home = std::getenv("HOME");
  const std::string local = std::string(home ? home : "") + "/.cache/mtp-slot/ckpt305/tokenizer.json";
  return std::filesystem::exists(local) ? local : "";
}
void Real() {
  const std::string path = RealTokenizer();
  if (path.empty()) { std::puts("SKIP: real tokenizer (set CRUCIBLE_TOKENIZER_JSON)"); return; }
  auto vocab = LoadConstraintVocab(path, 248320, 248046);
  assert(ConstraintVocabRows(*vocab) == 248320);
  const char* schema = R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},)"
                       R"("required":["name","age"],"additionalProperties":false})";
  auto c = MakeConstraint(*vocab, {"json_schema", schema, false});
  // {"name": "Ada", "age": 36} as the checkpoint's tokenizer encodes it.
  for (const int id : {4754, 591, 763, 328, 92055, 487, 328, 416, 763, 220, 18, 21, 92}) assert(c->Accept(id));
  assert(!c->Allows(248044) && !c->Allows(248319) && !c->Allows(248045));
  assert(c->Accept(248046) && c->Terminated());
  // The call tags are text; the other added tokens and the pad rows are never legal.
  auto any = MakeConstraint(*vocab, {"structural_tag",
      R"({"type":"structural_tag","format":{"type":"tag","begin":"<tool_call>","content":{"type":"any_text"},"end":"</tool_call>"}})", false});
  assert(any->Allows(248058) && !any->Allows(248069) && !any->Allows(248319));
  std::vector<uint32_t> mask;
  c = MakeConstraint(*vocab, {"json_schema", schema, false});
  c->FillMask(mask);
  assert(mask.size() == 248320 / 32 && MaskAllows(mask, 4754) && !MaskAllows(mask, 248046));
  std::puts("PASS: real tokenizer: recorded stream, stop token, added tokens and pad rows");
}
std::string Hy4Tokenizer() {
  if (const char* path = std::getenv("CRUCIBLE_HY4_TOKENIZER_JSON")) return path;
  const char* home = std::getenv("HOME");
  const std::string local = std::string(home ? home : "") + "/.cache/hy4-server/hf/tokenizer.json";
  return std::filesystem::exists(local) ? local : "";
}
// Hy4-flash (tencent/Hy4-flash tokenizer.json): the call tags of server/hy4_structured.py are text in a grammar, the
// reasoning phase ends at </think:opensource>, and <|hy_end:opensource|> is the stop token.
void Hy4Real() {
  const std::string path = Hy4Tokenizer();
  if (path.empty()) { std::puts("SKIP: Hy4-flash tokenizer (set CRUCIBLE_HY4_TOKENIZER_JSON)"); return; }
  const ConstraintFamily family = Hy4ConstraintFamily();
  assert(family.rows == 120832 && family.stop == 120025 && family.think_end == 120030);
  auto vocab = LoadConstraintVocab(path, family);
  assert(ConstraintVocabRows(*vocab) == 120832);
  const char* spec = R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                     R"({"type":"const_string","value":"<tool_calls:opensource><tool_call:opensource>read"},)"
                     R"({"type":"tag","begin":"<arg_key:opensource>x</arg_key:opensource><arg_value:opensource>",)"
                     R"("content":{"type":"any_text"},"end":"</arg_value:opensource>"},)"
                     R"({"type":"const_string","value":"</tool_call:opensource></tool_calls:opensource>"}]}})";
  auto c = MakeConstraint(*vocab, {"structural_tag", spec, true}, family.think_end);
  assert(c->Allows(120035) && c->Allows(87));  // the reasoning phase: no mask
  assert(c->Accept(87) && c->Accept(120030));  // </think:opensource> arms the grammar
  assert(!c->Allows(87) && !c->Allows(120029) && !c->Allows(120025));
  for (const int id : {120035, 120033, 1629, 120040, 87, 120041, 120042, 816, 120043, 120034, 120036})
    assert(c->Accept(id));
  assert(c->Accept(120025) && c->Terminated());
  // The Qwen form of the loader keeps only the Qwen call tags: the Hy4 block tag is never legal there.
  auto qwen = LoadConstraintVocab(path, 120832, 120025);
  auto old = MakeConstraint(*qwen, {"structural_tag", spec, false});
  assert(!old->Allows(120035));
  std::puts("PASS: Hy4-flash tokenizer: call tags are text, </think:opensource> arms, <|hy_end|> stops");
}
}  // namespace

int main() {
  assert(ConstraintLibraryVersion() == "xgrammar 0.2.9 crucible-1");
  auto vocab = MakeConstraintVocab(CharTokens(), kStop);
  assert(ConstraintVocabRows(*vocab) == 100);
  Stream(*vocab);
  Refused(*vocab);
  Walk(*vocab);
  ControlCharacters(*vocab);
  Thinking(*vocab);
  JsonObject(*vocab);
  StructuralTag(*vocab);
  Errors(*vocab);
  Cache(*vocab);
  Argmax();
  Greedy(*vocab);
  StopDraft(*vocab);
  PenaltyMask(*vocab);
  PenaltySpan(*vocab);
  LazyDraw();
  Real();
  Hy4Real();
  std::puts("PASS: constraint");
}
