// Y2 host policy for the RoPE mode of a request. CPU only.
//  - The length gate: a request's fed positions (prompt + max_tokens - 1, fixed at its open) pick its mode. Up to
//    262,144 it is plain, the same bits as an engine without YaRN; above that, the smallest YaRN factor that holds it
//    and that the process allows (CRUCIBLE_ROPE_MODE=auto:F). The forced test word puts every request in its mode.
//  - The state tenant: a YaRN request keeps its states under a tenant of its own, so no slot, checkpoint, frontier,
//    host-cache or disk match crosses the modes (as AblatedTenant does for the ablated mode). Plain keeps the tenant.
//    The tenant also carries a mark, so the disk tier can see the mode of a state from its policy alone.
#pragma once
#include "rope_mode.h"
#include "sha256.h"
#include "tenant.h"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace crucible {
// The positions that a request feeds: its prompt and every generated token but the last.
inline size_t RopeFedPositions(size_t prompt, int max_tokens) {
  return prompt + size_t(max_tokens > 0 ? max_tokens : 1) - 1;
}
// The mode of a request, or nullopt when no mode of this process holds it.
inline std::optional<RopeMode> RopeModeFor(size_t prompt, int max_tokens, const RopeSetting& setting) {
  const size_t fed = RopeFedPositions(prompt, max_tokens);
  if (setting.Forced()) return fed <= size_t(RopeMaxPositions(setting.start)) ? std::optional(setting.start) : std::nullopt;
  if (fed <= size_t(kRopeNativePositions)) return RopeMode::Plain;
  for (const RopeMode mode : {RopeMode::Yarn2, RopeMode::Yarn4})
    if (RopeFactor(mode) <= RopeFactor(setting.allow) && fed <= size_t(RopeMaxPositions(mode))) return mode;
  return std::nullopt;
}

// The mode word of the server ("rope" in the request: "plain", "yarn2", "yarn4"). The server decides the mode of a
// request once, and every job of it (a first job bounded at the think cap, a think-cap or end-of-turn continuation)
// sends that word, so a job that feeds fewer positions than its request keeps the request's mode. Only an auto
// process takes it: the mode must be allowed and must hold the job. nullopt: refused.
inline std::optional<RopeMode> RopeModeForWord(std::string_view word, size_t prompt, int max_tokens,
                                               const RopeSetting& setting) {
  const std::optional<RopeMode> mode = word == "plain" ? std::optional(RopeMode::Plain)
                                       : word == "yarn2" ? std::optional(RopeMode::Yarn2)
                                       : word == "yarn4" ? std::optional(RopeMode::Yarn4)
                                                         : std::nullopt;
  if (!mode || setting.Forced() || setting.allow == RopeMode::Plain) return std::nullopt;
  if (RopeFactor(*mode) > RopeFactor(setting.allow)) return std::nullopt;
  if (RopeFedPositions(prompt, max_tokens) > size_t(RopeMaxPositions(*mode))) return std::nullopt;
  return mode;
}

// The mark of a YaRN state tenant: bytes [0, 8) of the tenant; byte 8 is the mode. The other 23 bytes are the
// digest of the tag, the mode and the request's tenant. A plain tenant (the local tenant, a gateway tenant, an
// ablated one) has this mark with a chance of 2^-64; such a tenant would only lose the disk tier.
inline constexpr uint8_t kRopeTenantMark[8] = {'c', 'r', 'o', 'p', 'e', 'y', 'r', 'n'};
inline StateTenant RopeTenant(const StateTenant& tenant, RopeMode mode) {
  if (mode == RopeMode::Plain) return tenant;
  static constexpr char kTag[] = "crucible-rope-request-v1";
  const uint8_t word = uint8_t(mode);
  Sha256 sha;
  sha.Update(reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
  sha.Update(&word, 1);
  sha.Update(tenant.data(), tenant.size());
  StateTenant out = sha.Digest();
  for (size_t i = 0; i < sizeof(kRopeTenantMark); ++i) out[i] = kRopeTenantMark[i];
  out[sizeof(kRopeTenantMark)] = word;
  return out;
}
// The mode of a state from its tenant: Plain unless the tenant has the YaRN mark.
inline RopeMode RopeTenantMode(const StateTenant& tenant) {
  for (size_t i = 0; i < sizeof(kRopeTenantMark); ++i)
    if (tenant[i] != kRopeTenantMark[i]) return RopeMode::Plain;
  const uint8_t word = tenant[sizeof(kRopeTenantMark)];
  return word == uint8_t(RopeMode::Yarn2) ? RopeMode::Yarn2 : word == uint8_t(RopeMode::Yarn4) ? RopeMode::Yarn4
                                                                                                : RopeMode::Plain;
}
// A closed-set word for log lines and events: "plain", "yarn2", "yarn4".
inline const char* RopeWord(RopeMode mode) {
  return mode == RopeMode::Yarn4 ? "yarn4" : mode == RopeMode::Yarn2 ? "yarn2" : "plain";
}
// Serial serve: put the selected stream in the request's mode. A change resets the stream first (the engine refuses a
// change on a sequence that holds a token); the same mode keeps the state. An engine without the setter is plain only.
template <class E>
void ApplyStreamRope(E& eng, int stream, RopeMode mode) {
  if constexpr (requires { eng.SetStreamRope(stream, mode); eng.StreamRope(stream); }) {
    if (eng.StreamRope(stream) == mode) return;
    eng.Reset();
    eng.SetStreamRope(stream, mode);
  } else if (mode != RopeMode::Plain) {
    throw std::runtime_error("this engine serves no YaRN requests");
  }
}
}  // namespace crucible
