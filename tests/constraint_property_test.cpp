// Structured output property driver (plan 4.2): masked greedy decode with fake logits through the real grammar.
// With no arguments (make test-fast) it runs a small self-check on a character vocabulary. With
// `constraint_property_test TOKENIZER_JSON` it reads JSON lines {"kind","spec","reasoning","seed","mode","max"} from
// stdin and writes {"ids":[...],"ended":0|1} for each: server/tests/test_structured_property.py checks the outputs
// with the server's own parsers. Modes (fake logits, a keyed hash per step and token):
//   0 random, the stop token and the template's end pieces get +3, so outputs end;
//   1 adversarial: the device argmax is a token the grammar refuses (+10 on a random refused token), so every
//     step takes the masked fallback (ChooseGreedy);
//   2 long strings: the same as 0 but no bonus to end pieces;
//   3 replay: no decode; the line's "ids" (a string "1,2,3") go through the grammar until the first refused token.
//     The output names the accepted ids and "ended" 1 when the grammar terminated (the stop token was accepted).
#include "../engine/runtime/constraint_vocab.h"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace crucible;

namespace {
double Noise(uint64_t seed, uint64_t step, int token) {
  return -std::log(sampling::KeyedUniform(seed, step, token));  // exponential: a heavy right tail like real logits
}
// A tiny JSON reader for the driver's own input lines (flat object of strings and integers).
struct Line { std::string kind, spec; int reasoning = 0, mode = 0, max = 150; uint64_t seed = 0; std::vector<int> replay; };
std::string ReadString(const std::string& s, size_t& i) {
  std::string out;
  for (++i; i < s.size() && s[i] != '"'; ++i) {
    if (s[i] != '\\') { out += s[i]; continue; }
    const char e = s[++i];
    if (e == 'n') out += '\n'; else if (e == 't') out += '\t'; else if (e == 'r') out += '\r';
    else if (e == 'b') out += '\b'; else if (e == 'f') out += '\f';
    else if (e == 'u') {
      unsigned code = std::stoul(s.substr(i + 1, 4), nullptr, 16);
      i += 4;
      if (code >= 0xD800 && code < 0xDC00 && s.compare(i + 1, 2, "\\u") == 0) {
        const unsigned low = std::stoul(s.substr(i + 3, 4), nullptr, 16);
        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        i += 6;
      }
      if (code < 0x80) out += char(code);
      else if (code < 0x800) { out += char(0xC0 | (code >> 6)); out += char(0x80 | (code & 63)); }
      else if (code < 0x10000) { out += char(0xE0 | (code >> 12)); out += char(0x80 | ((code >> 6) & 63)); out += char(0x80 | (code & 63)); }
      else { out += char(0xF0 | (code >> 18)); out += char(0x80 | ((code >> 12) & 63)); out += char(0x80 | ((code >> 6) & 63)); out += char(0x80 | (code & 63)); }
    } else out += e;
  }
  ++i;
  return out;
}
Line Parse(const std::string& s) {
  Line line;
  size_t i = s.find('{') + 1;
  while (i < s.size() && s[i] != '}') {
    while (s[i] == ' ' || s[i] == ',') ++i;
    if (s[i] == '}') break;
    const std::string key = ReadString(s, i);
    while (s[i] == ' ' || s[i] == ':') ++i;
    if (s[i] == '"') {
      const std::string value = ReadString(s, i);
      if (key == "kind") line.kind = value; else if (key == "spec") line.spec = value;
      else if (key == "ids") {
        for (size_t at = 0; at < value.size();) {
          const size_t comma = value.find(',', at);
          line.replay.push_back(std::stoi(value.substr(at, comma - at)));
          at = comma == std::string::npos ? value.size() : comma + 1;
        }
      }
    } else {
      size_t end = i;
      while (end < s.size() && (isdigit(s[end]) || s[end] == '-')) ++end;
      const long long value = std::stoll(s.substr(i, end - i));
      i = end;
      if (key == "reasoning") line.reasoning = int(value); else if (key == "mode") line.mode = int(value);
      else if (key == "max") line.max = int(value); else if (key == "seed") line.seed = uint64_t(value);
    }
  }
  return line;
}

// One decode: returns the ids and whether the stop token ended it.
std::pair<std::vector<int>, bool> Decode(ConstraintVocab& vocab, const Line& line, const std::vector<int>& end_ids,
                                         int think_end) {
  auto c = MakeConstraint(vocab, {line.kind, line.spec, line.reasoning != 0}, think_end);
  if (line.mode == 3) {
    std::vector<int> taken;
    for (const int id : line.replay) {
      if (!c->Accept(id)) break;
      taken.push_back(id);
    }
    return {taken, c->Terminated()};
  }
  const int rows = ConstraintVocabRows(vocab), stop = c->StopToken();
  std::vector<double> bonus(size_t(rows), 0);
  bonus[size_t(stop)] = 3;
  if (line.mode != 2) for (const int id : end_ids) bonus[size_t(id)] += 3;
  std::vector<int> ids;
  std::vector<uint32_t> mask;
  if (line.reasoning) {
    // A short reasoning phase, then the think end (not masked; the grammar arms after it).
    for (const int id : end_ids) { if (!c->Accept(id)) std::abort(); ids.push_back(id); }
    if (!c->Accept(think_end)) std::abort();
    ids.push_back(think_end);
  }
  for (int step = 0; step < line.max; ++step) {
    const auto score = [&](int t) { return Noise(line.seed, uint64_t(step), t) + bonus[size_t(t)]; };
    c->FillMask(mask);
    int token = -1;
    if (line.mode == 1) {
      // The device argmax: a refused token; the full row exists only for the fallback.
      int refused = -1;
      for (int probe = 0; probe < 64 && refused < 0; ++probe) {
        const int t = int(sampling::Mix64(line.seed ^ uint64_t(step * 64 + probe)) % uint64_t(rows));
        if (!MaskAllows(mask, t)) refused = t;
      }
      if (refused >= 0) {
        token = ChooseGreedy(*c, refused, [&] {
          std::vector<float> row(static_cast<size_t>(rows));
          for (int t = 0; t < rows; ++t) row[size_t(t)] = float(score(t) + (t == refused ? 10 : 0));
          return row;
        });
      }
    }
    if (token < 0) {
      // Masked argmax over the legal tokens only (the same choice as a full row of these scores).
      double best = -1;
      for (size_t w = 0; w < mask.size(); ++w)
        for (uint32_t bits = mask[w]; bits; bits &= bits - 1) {
          const int t = int(w * 32 + size_t(__builtin_ctz(bits)));
          const double value = score(t);
          if (token < 0 || value > best) { token = t; best = value; }
        }
      if (token < 0 || !c->Accept(token)) return {ids, false};
    }
    ids.push_back(token);
    if (token == stop) return {ids, true};
  }
  return {ids, false};
}

void SelfCheck() {
  std::vector<std::string> tokens;
  for (int ch = 32; ch < 127; ++ch) tokens.push_back(std::string(1, char(ch)));
  tokens.push_back("<|end|>");
  tokens.push_back("</think>");
  const int stop = 95, think = 96;
  auto vocab = MakeConstraintVocab(tokens, stop);
  int ended = 0;
  for (int seed = 0; seed < 40; ++seed) {
    Line line{"json_schema", R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string","maxLength":3}},"required":["a","b"]})",
              seed % 2, seed % 3, 120, uint64_t(seed)};
    const auto [ids, end] = Decode(*vocab, line, {'}' - 32, '"' - 32}, think);
    auto check = MakeConstraint(*vocab, {line.kind, line.spec, line.reasoning != 0}, think);
    for (const int id : ids) assert(check->Accept(id));
    if (end) { assert(check->Terminated()); ++ended; }
  }
  assert(ended >= 20);
  std::printf("PASS: constraint property self-check (%d of 40 ended; every output a legal prefix)\n", ended);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { SelfCheck(); return 0; }
  auto vocab = LoadConstraintVocab(argv[1], 248320, 248046);
  // The pieces of "\n</parameter>\n</function>\n</tool_call>", "}" and "\"" (checkpoint token IDs).
  std::vector<int> end_ids;
  for (int i = 2; i < argc; ++i) end_ids.push_back(std::atoi(argv[i]));
  std::string text;
  while (std::getline(std::cin, text)) {
    if (text.empty()) continue;
    const Line line = Parse(text);
    std::string out = "{\"ids\":[";
    bool ended = false;
    std::string error;
    try {
      auto [ids, end] = Decode(*vocab, line, end_ids, kConstraintThinkEnd);
      ended = end;
      for (size_t i = 0; i < ids.size(); ++i) out += (i ? "," : "") + std::to_string(ids[i]);
    } catch (const std::exception& e) {
      error = e.what();
    }
    out += "],\"ended\":" + std::to_string(int(ended));
    if (!error.empty()) out += ",\"error\":\"" + error.substr(0, error.find(':')) + "\"";
    std::cout << out << "}\n" << std::flush;
  }
}
