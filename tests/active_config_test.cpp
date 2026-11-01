// The one-line "active config" at engine start: the CRUCIBLE_* switches that are set (registry names only) and the
// MTP sidecar as size plus sha256 prefix. A path names the user, so a path value never appears in the line.
#include "../engine/model/flashnext/active_config.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

int main() {
  using namespace crucible;
  assert(ConfigValue("1") == "1" && ConfigValue("8192+128") == "8192+128");
  assert(ConfigValue("~/dir/file.bin") == "(path)" && ConfigValue("rel/dir/f") == "(path)");
  assert(ConfigValue("") == "(empty)");
  assert(ConfigValue(std::string(200, 'x')).size() <= 48);  // a long value is cut, never the whole text

  assert(ActiveConfigLine({}) == "active config: none");
  assert(ActiveConfigLine({{"CRUCIBLE_EXACT_RESUME", "1"}, {"CRUCIBLE_PF_LAUNCH_ROWS", "8192+128"}}) ==
         "active config: CRUCIBLE_EXACT_RESUME=1 CRUCIBLE_PF_LAUNCH_ROWS=8192+128");

  char dir[] = "/tmp/crucible-active-config-XXXXXX";
  assert(mkdtemp(dir));
  const std::string side = std::string(dir) + "/side.safetensors";
  std::ofstream(side, std::ios::binary) << "abc";
  // sha256("abc") = ba7816bf...
  assert(SidecarId(side) == "3B-sha256:ba7816bf");
  assert(SidecarId(std::string(dir) + "/none") == "unreadable");

  unsetenv("CRUCIBLE_EXACT_RESUME");
  setenv("CRUCIBLE_EXACT_RESUME", "1", 1);
  setenv("CRUCIBLE_MTP_SIDECAR", side.c_str(), 1);
  setenv("CRUCIBLE_NOT_A_SWITCH", "1", 1);  // not in the registry: not listed
  const std::string line = ActiveConfig();
  assert(line == "active config: CRUCIBLE_EXACT_RESUME=1 CRUCIBLE_MTP_SIDECAR=3B-sha256:ba7816bf");
  assert(line.find(dir) == std::string::npos && line.find('/') == std::string::npos);
  unsetenv("CRUCIBLE_EXACT_RESUME");
  unsetenv("CRUCIBLE_MTP_SIDECAR");
  assert(ActiveConfig() == "active config: none");
  std::filesystem::remove_all(dir);
  std::puts("PASS: active config line names switches and the sidecar by size and sha, never a path");
}
