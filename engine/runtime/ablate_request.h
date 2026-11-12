// Abliteration per request. CRUCIBLE_ABLATE=FILE CRUCIBLE_ABLATE_MODE=request loads the
// sidecar once; a request with "ablate":1 runs ablated, every other request runs stock. Without the mode switch the
// sidecar ablates every request (the process-wide profile).
#pragma once
#include "../switches.h"
#include "sha256.h"
#include "tenant.h"
#include <string>

namespace crucible {
// True when the process serves stock and ablated requests side by side. The engine checks at start that
// CRUCIBLE_ABLATE names a sidecar and that the mode has no other value.
inline bool AblateRequestMode() {
  const char* mode = SwitchText("CRUCIBLE_ABLATE_MODE");
  return mode && std::string(mode) == "request";
}
// The state tenant of an ablated request: sha256 of a fixed tag and the request's tenant. Every slot, checkpoint,
// frontier and disk match compares the tenant, so a state of one mode never restores into a request of the other.
inline StateTenant AblatedTenant(const StateTenant& tenant) {
  static constexpr char kTag[] = "crucible-ablate-request-v1";
  Sha256 sha;
  sha.Update(reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
  sha.Update(tenant.data(), tenant.size());
  return sha.Digest();
}
}  // namespace crucible
