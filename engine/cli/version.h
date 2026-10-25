#pragma once
// `crucible --version`. The version text is the line below; the release step rewrites it
// together with the VERSION file, and tests/test_version.py checks that the two agree. The text is not a compiler
// flag on purpose: a flag reaches the device compile, where it would change the cache key of every kernel part
// (tools/devsplit.py). As plain host text it cannot change a device code object. A build may still override it with
// -DENGINE_BUILD_VERSION.
#include <cstring>
#include <string>

#ifndef ENGINE_BUILD_VERSION
#define ENGINE_BUILD_VERSION "0.26.1"
#endif

namespace crucible {

inline const char* BuildVersion() { return ENGINE_BUILD_VERSION; }

inline std::string VersionLine() { return std::string("crucible ") + BuildVersion(); }

// Only the exact command `crucible --version`: no other command takes this flag, so no engine option can hide it.
inline bool IsVersionRequest(int argc, char** argv) { return argc == 2 && std::strcmp(argv[1], "--version") == 0; }

}  // namespace crucible
