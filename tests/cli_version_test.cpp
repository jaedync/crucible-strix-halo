// crucible --version: the text, and the request check, without a GPU. The version text is in engine/cli/version.h.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "../engine/cli/version.h"

namespace {
bool Request(std::initializer_list<const char*> args) {
  std::string storage[4];
  char* argv[4] = {};
  int argc = 0;
  for (const char* arg : args) {
    storage[argc] = arg;
    argv[argc] = storage[argc].data();
    ++argc;
  }
  return crucible::IsVersionRequest(argc, argv);
}
}  // namespace

int main() {
  assert(Request({"crucible", "--version"}));
  assert(!Request({"crucible"}));
  assert(!Request({"crucible", "gen"}));
  assert(!Request({"crucible", "--version", "x"}));
  assert(!Request({"crucible", "serve", "--version"}));
  assert(!Request({"crucible", "--versions"}));
  assert(std::strlen(crucible::BuildVersion()) > 0);
  assert(std::string(crucible::BuildVersion()).find_first_not_of("0123456789.") == std::string::npos);
  assert(crucible::VersionLine() == std::string("crucible ") + crucible::BuildVersion());
  std::puts("cli_version_test ok");
  return 0;
}
