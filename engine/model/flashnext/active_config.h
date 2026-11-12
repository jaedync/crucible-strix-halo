// One stderr line at engine start that names the active switches (the CRUCIBLE_* names of the registry that the
// environment sets, with their values) and the MTP sidecar by size and sha256 prefix. A switch script or a gate reads
// it to see what the engine really runs with. A path names the user, so a value with a '/' prints as "(path)", and the
// sidecar never prints its path.
#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../runtime/sha256.h"
#include "../../switches.h"

namespace crucible {
inline std::string ConfigValue(std::string_view value) {
  if (value.empty()) return "(empty)";
  if (value.find('/') != std::string_view::npos) return "(path)";
  return std::string(value.substr(0, 48));
}

inline std::string ActiveConfigLine(const std::vector<std::pair<std::string, std::string>>& set) {
  std::string line = "active config:";
  if (set.empty()) return line + " none";
  for (const auto& [name, value] : set) line += " " + name + "=" + value;
  return line;
}

// "<bytes>B-sha256:<8 hex>" of the file, or "unreadable".
inline std::string SidecarId(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "unreadable";
  Sha256 hash;
  std::vector<char> buffer(1 << 20);
  uint64_t bytes = 0;
  while (in.read(buffer.data(), std::streamsize(buffer.size())) || in.gcount() > 0) {
    hash.Update(reinterpret_cast<const uint8_t*>(buffer.data()), size_t(in.gcount()));
    bytes += uint64_t(in.gcount());
  }
  if (in.bad()) return "unreadable";
  return std::to_string(bytes) + "B-sha256:" + hash.HexDigest().substr(0, 8);
}

// The line for the process environment: the registry rows in name order, a build macro never.
inline std::string ActiveConfig() {
  std::vector<std::pair<std::string, std::string>> set;
  for (const SwitchInfo& info : kSwitches) {
    if (info.kind == SwitchKind::kBuild) continue;
    const char* text = SwitchText(info);
    if (!text) continue;
    const bool sidecar = std::string_view(info.name) == "CRUCIBLE_MTP_SIDECAR" && *text;
    set.emplace_back(info.name, sidecar ? SidecarId(text) : ConfigValue(text));
  }
  return ActiveConfigLine(set);
}
}  // namespace crucible
