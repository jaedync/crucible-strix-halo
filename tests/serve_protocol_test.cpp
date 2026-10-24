#include "../engine/serve/serve_protocol.h"
#include <cassert>
#include <stdexcept>
#include <string>
using crucible::ParseServeRequest;
static void Rejected(const std::string& s, const char* reason) {
  try { ParseServeRequest(s, 32); assert(false); }
  catch (const std::runtime_error& e) { assert(std::string(e.what()).find(reason) != std::string::npos); }
}
int main() {
  // The ready event names the model, so the server takes its metadata from the engine.
  assert(crucible::ReadyModelJson() ==
         R"("model":{"arch":"qwen4_exp","vocab":248320,"eos":248044,"layers":48,"hidden":2560})");
  auto r = ParseServeRequest(R"({"ids":[1,2],"max_tokens":4,"spec":2,"stop_ids":[248046,248044]})", 32);
  assert(r.ids.size() == 2 && r.max_tokens == 4 && r.spec == 2);
  assert(r.stop_ids.size() == 2);
  const auto cached = ParseServeRequest(R"({"ids":[1,2],"max_tokens":1,"cache_control":"refresh","cache_prefix_tokens":1})", 32);
  assert(cached.cache_control == "refresh" && cached.cache_prefix_tokens == 1);
  assert(ParseServeRequest(R"({"ids":[1],"max_tokens":1,"top_logprobs":20})", 32).top_logprobs == 20);
  assert(ParseServeRequest(R"({"ids":[1],"max_tokens":1})", 32).top_logprobs == 0);
  Rejected(R"({"ids":[1],"max_tokens":1,"top_logprobs":21})", "top_logprobs");
  Rejected(R"({"ids":[1],"max_tokens":2,"top_logprobs":3})", "top_logprobs");
  Rejected(R"({"ids":[1],"max_tokens":1,"temperature":0.5,"top_logprobs":3})", "top_logprobs");
  Rejected(R"({"ids":[1],"cache_control":"purge"})", "cache_control");
  assert(ParseServeRequest(R"({"ids":[1,2],"max_tokens":1,"cache_tail":"guard"})", 32).cache_tail == "guard");
  assert(ParseServeRequest(R"({"ids":[1,2],"max_tokens":1})", 32).cache_tail == "auto");
  Rejected(R"({"ids":[1],"cache_tail":"keep"})", "cache_tail");
  assert(ParseServeRequest(R"({"ids":[1,2],"max_tokens":1,"draft_detail":1})", 32).draft_detail == 1);
  assert(ParseServeRequest(R"({"ids":[1,2],"max_tokens":1})", 32).draft_detail == 0);
  Rejected(R"({"ids":[1],"draft_detail":2})", "draft_detail");
  Rejected(R"({"ids":[1],"cache_prefix_tokens":2})", "cache_prefix_tokens");
  Rejected(R"({"ids":[1],"cache_prefix_tokens":-1})", "cache_prefix_tokens");
  assert(ParseServeRequest(R"({"ids":[1,2],"max_tokens":1,"shared_prefix_tokens":2})", 32).shared_prefix_tokens == 2);
  Rejected(R"({"ids":[1],"shared_prefix_tokens":2})", "shared_prefix_tokens");
  Rejected(R"({"ids":[1],"shared_prefix_tokens":-1})", "shared_prefix_tokens");
  const auto sampled = ParseServeRequest(R"({"ids":[1],"temperature":0.7,"top_k":17,"top_p":0.9,"min_p":0.05,"seed":18446744073709551615})", 32);
  assert(sampled.sampling.temperature == 0.7 && sampled.sampling.top_k == 17);
  assert(sampled.sampling.top_p == 0.9 && sampled.sampling.min_p == 0.05);
  assert(sampled.sampling.seed == UINT64_MAX);
  Rejected(R"({"ids":[1],"seed":18446744073709551616})", "integer");
  Rejected(R"({"ids":[1],"seed":-1})", "integer");
  Rejected(R"({"ids":[1],"temperature":1e309})", "number");
  Rejected(R"({"ids":[1],"temperature":-0.1})", "temperature");
  Rejected(R"({"ids":[1],"top_k":248321})", "top_k");
  Rejected(R"({"ids":[1],"top_p":1.01})", "top_p");
  Rejected(R"({"ids":[1],"min_p":-0.1})", "min_p");
  Rejected("{\"ids\":[1", "expected");
  Rejected(R"({"ids":[-1]})", "token IDs");
  Rejected(R"({"ids":[248320]})", "token IDs");
  Rejected(R"({"ids":[1],"spec":8})", "spec");
  Rejected(R"({"ids":[],"max_tokens":1})", "context");
  Rejected(R"({"ids":[1],"max_tokens":0})", "context");
  Rejected(R"({"ids":[1],"max_tokens":2147483648})", "integer");
  Rejected(R"({"ids":[1],"max_tokens":33})", "context");
  Rejected(R"({"ids":[1],"stop_ids":[-1]})", "stop IDs");
  Rejected(R"({"ids":[1],"ids":[2]})", "duplicate");
  Rejected(R"({"ids":[1],"top_a":1})", "unknown");
  Rejected(R"({"ids":[1]}) garbage)", "trailing");
  assert(crucible::JsonQuote("a\n\"b") == "\"a\\u000a\\\"b\"");
}
