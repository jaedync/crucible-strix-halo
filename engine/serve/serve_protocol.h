// Recoverable JSON-line input. All request checks precede state reset and GPU writes.
#pragma once
#include "../exl3/safetensors.h"
#include "../runtime/bounds.h"
#include "../model/flashnext/geometry.h"
#include "../runtime/decode_policy.h"
#include "../runtime/sampling.h"
#include "../runtime/tenant.h"
#include "logprobs.h"
#include "../model/flashnext/multimodal.h"
#include <set>
#include <stdexcept>

namespace crucible {
// The model fields of the ready event: the server takes its model metadata from them.
inline std::string ReadyModelJson() {
  return std::string("\"model\":{\"arch\":\"") + kModelArch + "\",\"vocab\":" + std::to_string(kVocab) +
         ",\"eos\":" + std::to_string(kEos) + ",\"layers\":" + std::to_string(kLayers) +
         ",\"hidden\":" + std::to_string(kHidden) + "}";
}
struct ServeRequest : DecodePolicy {
  std::vector<int> ids;
  int max_tokens = 32;
  int cache_prefix_tokens = 0;
  // End of the system-and-tools rendering that sessions share (G7).
  int shared_prefix_tokens = 0;
  std::string cache_control = "auto";
  // "guard": the client may not send this output back (reasoning, or the server's continuation job), so the
  // request's replay tail never replaces an older resume point.
  std::string cache_tail = "auto";
  std::vector<int> stop_ids{kEos, 248046};
  sampling::Options sampling;
  // The scoring position: 0 is off. N asks for the first token's top-N log-probabilities.
  int top_logprobs = 0;
  // Only `serve --vision 1` accepts images (docs: server/README.md).
  std::vector<ImageInput> images;
  // 1: the done event adds per-token draft counters (serve --draft-stats).
  int draft_detail = 0;
  // W1: the request's tenant key, 64 hex characters on the line; no field is the local tenant. Every cache,
  // slot and disk match stays inside it. Never logged.
  StateTenant tenant = kLocalTenant;
};
// A tenant field that is not 64 hex characters. Only that request is refused (the batch line: a recoverable open
// error); its message is fixed and holds no part of the value.
struct ServeTenantError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// The conversation-cache key of `tokens` (the request's prompt, a prefix of it, or its generated tail). Two images
// have the same pad token IDs, so image tokens key by image digest (CacheKey); text keys equal the token IDs.
inline std::vector<int> RequestCacheKey(const ServeRequest& request, std::span<const int> tokens) {
  return CacheKey(tokens, request.images);
}
inline int RequestInt(st::Scanner& scanner) {
  const int64_t value = scanner.Int();
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::runtime_error("request integer exceeds the supported range");
  return int(value);
}
inline std::vector<int> RequestIds(st::Scanner& scanner) {
  const auto wide = scanner.IntArray();
  std::vector<int> ids;
  for (const int64_t value : wide) {
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      throw std::runtime_error("token integer exceeds the supported range");
    ids.push_back(int(value));
  }
  return ids;
}
inline ServeRequest ParseServeRequest(const std::string& line, int ctx, const DecodePolicy& defaults = {}) {
  st::Scanner scanner(line, "request", true);
  ServeRequest request;
  static_cast<DecodePolicy&>(request) = defaults;
  std::set<std::string> fields;
  std::optional<std::string> tenant;
  scanner.Expect('{');
  while (!scanner.Peek('}')) {
    const std::string key = scanner.Str();
    if (!fields.insert(key).second && key != "tenant") throw std::runtime_error("duplicate request field: " + key);
    scanner.Expect(':');
    if (key == "ids") request.ids = RequestIds(scanner);
    else if (key == "max_tokens") request.max_tokens = RequestInt(scanner);
    else if (key == "spec") request.spec = RequestInt(scanner);
    else if (key == "draft_q") request.draft_q = scanner.Real();
    else if (key == "draft_head") request.draft_head = scanner.Str();
    else if (key == "mtp_prefix") request.mtp_prefix = RequestInt(scanner);
    else if (key == "draft_adapt") {
      request.draft_adapt = RequestInt(scanner);
      if (request.draft_adapt < 0) throw std::runtime_error("draft_adapt must be 0 or 1");
    }
    else if (key == "lookup") request.lookup = RequestInt(scanner);
    else if (key == "lookup_min") request.lookup_min = RequestInt(scanner);
    else if (key == "draft_value") request.draft_value = RequestInt(scanner);
    else if (key == "draft_detail") {
      request.draft_detail = RequestInt(scanner);
      if (request.draft_detail < 0 || request.draft_detail > 1) throw std::runtime_error("draft_detail must be 0 or 1");
    }
    else if (key == "stop_ids") request.stop_ids = RequestIds(scanner);
    else if (key == "cache_control") request.cache_control = scanner.Str();
    else if (key == "cache_tail") request.cache_tail = scanner.Str();
    else if (key == "cache_prefix_tokens") request.cache_prefix_tokens = RequestInt(scanner);
    else if (key == "shared_prefix_tokens") request.shared_prefix_tokens = RequestInt(scanner);
    else if (key == "temperature") request.sampling.temperature = scanner.Real();
    else if (key == "top_k") request.sampling.top_k = RequestInt(scanner);
    else if (key == "top_p") request.sampling.top_p = scanner.Real();
    else if (key == "min_p") request.sampling.min_p = scanner.Real();
    else if (key == "seed") request.sampling.seed = scanner.Uint();
    else if (key == "top_logprobs") request.top_logprobs = RequestInt(scanner);
    else if (key == "images") request.images = ParseImages(scanner);
    // A tenant that is not one JSON string (another type, or a second tenant field) is a bad tenant: the empty value
    // fails ParseTenantHex below, so only this request is refused, with the fixed message.
    else if (key == "tenant") tenant = !tenant && scanner.Peek('"') ? scanner.Str() : (scanner.Skip(), std::string());
    else throw std::runtime_error("unknown request field; check the serve protocol documentation");
    scanner.Separator('}');
  }
  scanner.Expect('}');
  scanner.Finish();
  if (request.top_logprobs < 0 || request.top_logprobs > kMaxTopLogprobs ||
      (request.top_logprobs && (request.max_tokens != 1 || request.sampling.temperature != 0)))
    throw std::runtime_error("top_logprobs must be 0 to 20 and needs max_tokens 1 and temperature 0");
  if (request.cache_control != "auto" && request.cache_control != "bypass" && request.cache_control != "refresh")
    throw std::runtime_error("cache_control must be auto, bypass, or refresh");
  if (request.cache_tail != "auto" && request.cache_tail != "guard")
    throw std::runtime_error("cache_tail must be auto or guard");
  if (request.cache_prefix_tokens < 0 || size_t(request.cache_prefix_tokens) > request.ids.size())
    throw std::runtime_error("cache_prefix_tokens must be inside the prompt");
  if (request.shared_prefix_tokens < 0 || size_t(request.shared_prefix_tokens) > request.ids.size())
    throw std::runtime_error("shared_prefix_tokens must be inside the prompt");
  if (!ValidTokenIds(request.ids, kVocab)) throw std::runtime_error("prompt token IDs must be in [0,248320)");
  if (!ValidTokenIds(request.stop_ids, kVocab)) throw std::runtime_error("stop IDs must be in [0,248320)");
  CheckImages(request.images, request.ids);
  CheckDecodePolicy(request);
  if (const std::string error = request.sampling.Error(kVocab); !error.empty()) throw std::runtime_error(error);
  if (!RequestFitsContext(request.ids.size(), request.max_tokens, ctx))
    throw std::runtime_error("nonempty prompt and positive max_tokens must fit the configured context");
  // Last, so that a refused tenant means the rest of the request is valid.
  if (tenant) {
    const auto parsed = ParseTenantHex(*tenant);
    if (!parsed) throw ServeTenantError("tenant must be 64 hex characters");
    request.tenant = *parsed;
  }
  return request;
}
inline std::string JsonQuote(const std::string& value) {
  std::string out = "\"";
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c < 32) {
      char escaped[7];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
      out += escaped;
    } else out += c;
  }
  return out + '"';
}
}  // namespace crucible
