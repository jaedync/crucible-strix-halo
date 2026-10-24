// The tenant key: 32 bytes that a gateway in front of the server derives from a
// tenant ID. Every cache and state match compares it, so a request never restores a state of another tenant. A
// request that names no tenant is the local tenant (all zero): the owner's requests, and every request before W2.
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace crucible {
using StateTenant = std::array<uint8_t, 32>;
inline constexpr StateTenant kLocalTenant{};
// 64 hex characters (either case), or none. The caller never logs the text (section 9.4).
inline std::optional<StateTenant> ParseTenantHex(std::string_view text) {
  if (text.size() != 2 * StateTenant{}.size()) return std::nullopt;
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  StateTenant tenant{};
  for (size_t i = 0; i < tenant.size(); ++i) {
    const int high = nibble(text[2 * i]), low = nibble(text[2 * i + 1]);
    if (high < 0 || low < 0) return std::nullopt;
    tenant[i] = uint8_t(high << 4 | low);
  }
  return tenant;
}
}  // namespace crucible
