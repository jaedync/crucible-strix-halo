// The text checker of structured output (review H2): does the engine grammar of a spec
// take a text? server/tests/test_structured_strict.py builds the specs with server/structured.py and checks pairs of
// texts that the schema allows and forbids. A character vocabulary: token i < 95 is chr(32 + i), 95 TAB, 96 newline,
// 97 the stop token, 98 `</think>`, then the other control characters (U+0000 to U+001F without TAB and newline) and
// DEL, so a test can see a grammar that takes a raw control character inside a JSON string; then three characters
// outside ASCII of 2, 3 and 4 bytes (U+00E9, U+20AC, U+1F600).
// Usage: constraint_text_check < lines "KIND<TAB>SPEC<TAB>TEXT_HEX"
// Writes one line for each: "accept" (the text, then the stop token), "prefix" (the text, but no stop after it),
// "reject" (a refused character), "unsupported" (a character outside the vocabulary) or "refused <code>".
#include "../engine/runtime/constraint_vocab.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace crucible;

namespace {
constexpr int kStop = 97;
constexpr const char* kWide[] = {"\xC3\xA9", "\xE2\x82\xAC", "\xF0\x9F\x98\x80"};
constexpr int kFirstWide = 130;
std::vector<std::string> CharTokens() {
  std::vector<std::string> tokens;
  for (int c = 32; c < 127; ++c) tokens.push_back(std::string(1, char(c)));
  tokens.push_back("\t");
  tokens.push_back("\n");
  tokens.push_back("<|end|>");
  tokens.push_back("</think>");
  for (int c = 0; c < 32; ++c)
    if (c != '\t' && c != '\n') tokens.push_back(std::string(1, char(c)));
  tokens.push_back(std::string(1, char(127)));
  for (const char* wide : kWide) tokens.push_back(wide);
  return tokens;
}
std::string FromHex(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) out += char(std::stoi(hex.substr(i, 2), nullptr, 16));
  return out;
}
int Token(char c) {
  if (c == '\t') return 95;
  if (c == '\n') return 96;
  if (c >= 32 && c < 127) return c - 32;
  if (c == 127) return 129;
  if (c >= 0 && c < 32) return 99 + c - (c > '\n' ? 2 : c > '\t' ? 1 : 0);
  return -1;
}
std::string Check(ConstraintVocab& vocab, const ConstraintSpec& spec, const std::string& text) {
  std::shared_ptr<Constraint> c;
  try {
    c = MakeConstraint(vocab, spec, 98);
  } catch (const ConstraintError& error) {
    const std::string message = error.what();
    return "refused " + message.substr(0, message.find(':'));
  }
  for (size_t i = 0; i < text.size();) {
    int token = Token(text[i]);
    size_t width = 1;
    for (int w = 0; token < 0 && w < 3; ++w) {
      const std::string wide = kWide[w];
      if (text.compare(i, wide.size(), wide) == 0) token = kFirstWide + w, width = wide.size();
    }
    if (token < 0) return "unsupported";
    if (!c->Accept(token)) return "reject";
    i += width;
  }
  return c->Accept(kStop) ? "accept" : "prefix";
}
}  // namespace

int main() {
  auto vocab = MakeConstraintVocab(CharTokens(), kStop);
  std::string line;
  while (std::getline(std::cin, line)) {
    const size_t a = line.find('\t'), b = a == std::string::npos ? a : line.find('\t', a + 1);
    if (b == std::string::npos) { std::puts("bad line"); continue; }
    std::puts(Check(*vocab, {line.substr(0, a), line.substr(a + 1, b - a - 1), false}, FromHex(line.substr(b + 1))).c_str());
    std::fflush(stdout);
  }
}
