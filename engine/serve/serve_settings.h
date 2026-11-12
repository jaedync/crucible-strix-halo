// Parse resident startup switches before the model allocates device memory.
#pragma once
#include "serve_protocol.h"
#include "../runtime/rope_mode.h"
#include <charconv>
#include <span>
#include <string_view>

namespace crucible {
inline constexpr int kServePrefillAuto = -1, kServeMaxPrefillRows = 4096;
// The host's memory (the test host: 128 GiB), not a fit: batch startup admission checks the cache budget against free
// memory, and serial serve keeps the 8 GiB host reserve at each save.
inline constexpr int kServeMaxCacheMib = 131072;
// The MemAvailable that each conversation cache save keeps free (--host-reserve-mib). The engine's own startup
// admission keeps a separate, smaller reserve.
inline constexpr int kServeHostReserveMib = 8192, kServeMinHostReserveMib = 1024;
inline void CheckHostReserve(int mib) {
  if (mib < kServeMinHostReserveMib || mib > kServeMaxCacheMib)
    throw std::runtime_error("host-reserve-mib must be an integer in [1024,131072]");
}
struct ServeSettings {
  int ctx = 32768;
  int cache_mib = 0, prefill_rows = kServePrefillAuto;
  int host_reserve_mib = kServeHostReserveMib;
  DecodePolicy policy;
  bool exact_retry = false;
  // G15 (opt-in A/B): a later turn resumes from the previous turn's decode-time state when its prompt extends it.
  bool session_replay = false;
  bool vision = false;  // load the vision tower and accept request images
};

template <class T> inline T ServeNumber(std::string_view value) {
  T result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::runtime_error("serve options must contain complete numeric values");
  return result;
}

inline ServeSettings ParseServeSettings(std::span<const std::string_view> args) {
  if (args.size() % 2) throw std::runtime_error("serve options need a value after each switch");
  ServeSettings settings;
  std::set<std::string_view> seen;
  for (size_t index = 0; index < args.size(); index += 2) {
    const auto key = args[index], value = args[index + 1];
    if (!seen.insert(key).second) throw std::runtime_error("duplicate serve option");
    if (key == "--ctx") settings.ctx = ServeNumber<int>(value);
    else if (key == "--conversation-cache-mib") settings.cache_mib = ServeNumber<int>(value);
    else if (key == "--prefill") settings.prefill_rows = ServeNumber<int>(value);
    else if (key == "--host-reserve-mib") settings.host_reserve_mib = ServeNumber<int>(value);
    else if (key == "--conversation-cache-exact-retry") {
      const int parsed = ServeNumber<int>(value);
      if (parsed < 0 || parsed > 1) throw std::runtime_error("conversation-cache-exact-retry must be 0 or 1");
      settings.exact_retry = parsed != 0;
    }
    else if (key == "--session-replay") {
      const int parsed = ServeNumber<int>(value);
      if (parsed < 0 || parsed > 1) throw std::runtime_error("session-replay must be 0 or 1");
      settings.session_replay = parsed != 0;
    }
    else if (key == "--vision") {
      const int parsed = ServeNumber<int>(value);
      if (parsed < 0 || parsed > 1) throw std::runtime_error("vision must be 0 or 1");
      settings.vision = parsed != 0;
    }
    else if (key == "--spec") settings.policy.spec = ServeNumber<int>(value);
    else if (key == "--draft-q") settings.policy.draft_q = ServeNumber<double>(value);
    else if (key == "--draft-head") settings.policy.draft_head = value;
    else if (key == "--mtp-prefix") settings.policy.mtp_prefix = ServeNumber<int>(value);
    else if (key == "--draft-adapt") {
      settings.policy.draft_adapt = ServeNumber<int>(value);
      if (settings.policy.draft_adapt < 0) throw std::runtime_error("draft-adapt must be 0 or 1");
    } else throw std::runtime_error("unknown serve option");
  }
  CheckRopeContext(settings.ctx, RopeSettingFromEnv(), "serve context");
  if (settings.cache_mib < 0 || settings.cache_mib > kServeMaxCacheMib ||
      settings.prefill_rows < kServePrefillAuto || settings.prefill_rows > kServeMaxPrefillRows)
    throw std::runtime_error("serve cache and prefill settings exceed their supported ranges");
  CheckHostReserve(settings.host_reserve_mib);
  CheckDecodePolicy(settings.policy);
  return settings;
}
}  // namespace crucible
