// RoPE mode words and the context cap (engine/runtime/rope_mode.h).
#include "../engine/runtime/rope_mode.h"
#include "../engine/serve/batched_serve_settings.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace crucible;
namespace {
template <class F> bool Throws(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
BatchedServeSettings Batched(std::vector<std::string_view> args) { return ParseBatchedServeSettings(args); }
ServeSettings Serial(std::vector<std::string_view> args) { return ParseServeSettings(args); }
}  // namespace

int main() {
  // The words.
  for (const char* plain : {(const char*)nullptr, "", "plain"}) {
    const RopeSetting s = ParseRopeModeWord(plain);
    assert(s.start == RopeMode::Plain && s.allow == RopeMode::Plain && !s.Forced() && s.ContextCap() == 262144);
  }
  assert(ParseRopeModeWord("yarn:2").start == RopeMode::Yarn2 && ParseRopeModeWord("yarn:2.0").ContextCap() == 524288);
  assert(ParseRopeModeWord("yarn:4").start == RopeMode::Yarn4 && ParseRopeModeWord("yarn:4.0").ContextCap() == 1048576);
  assert(ParseRopeModeWord("yarn:2").Forced() && !ParseRopeModeWord("auto:2").Forced());
  assert(ParseRopeModeWord("auto:2").start == RopeMode::Plain && ParseRopeModeWord("auto:2").allow == RopeMode::Yarn2);
  assert(ParseRopeModeWord("auto:4").ContextCap() == 1048576);
  for (const char* bad : {"yarn", "yarn:", "yarn:3", "yarn:8", "auto:1", "YARN:2", "on", "1", "yarn:2 "})
    assert(Throws([&] { ParseRopeModeWord(bad); }));
  assert(RopeMaxPositions(RopeMode::Plain) == kRopeNativePositions && kRopeMaxPositions == 1048576);
  assert(std::string(RopeModeName(RopeMode::Yarn2)) == "yarn:2");
  // Review M1: the entry check of a forward (before any kernel writes state).
  assert(RopeRowsFit(0, 1, RopeMode::Plain) && RopeRowsFit(262143, 1, RopeMode::Plain));
  assert(!RopeRowsFit(262144, 1, RopeMode::Plain) && !RopeRowsFit(262140, 8, RopeMode::Plain));
  assert(RopeRowsFit(262144, 1, RopeMode::Yarn2) && RopeRowsFit(524280, 8, RopeMode::Yarn2));
  assert(!RopeRowsFit(524288, 1, RopeMode::Yarn2) && RopeRowsFit(1048575, 1, RopeMode::Yarn4));
  assert(!RopeRowsFit(0, -1, RopeMode::Plain));
  assert(Throws([] { CheckRopeRows(262144, 1, RopeMode::Plain); }));
  CheckRopeRows(262144, 1, RopeMode::Yarn2);

  // The cap: plain keeps the old error text; YaRN allows up to 262144*F.
  try { CheckRopeContext(262145, {}, "serve context"); assert(false); }
  catch (const std::runtime_error& e) {
    assert(std::string(e.what()).starts_with("serve context must be an integer in [1,262144]"));
  }
  CheckRopeContext(262144, {}, "x");
  CheckRopeContext(524288, ParseRopeModeWord("yarn:2"), "x");
  CheckRopeContext(524288, ParseRopeModeWord("auto:2"), "x");
  assert(Throws([] { CheckRopeContext(524289, ParseRopeModeWord("yarn:2"), "x"); }));
  assert(Throws([] { CheckRopeContext(0, ParseRopeModeWord("yarn:4"), "x"); }));
  CheckRopeContext(1048576, ParseRopeModeWord("yarn:4"), "x");

  // The serve parsers read CRUCIBLE_ROPE_MODE.
  unsetenv("CRUCIBLE_ROPE_MODE");
  assert(Batched({"--streams", "4", "--ctx", "262144"}).ctx == 262144);
  assert(Throws([] { Batched({"--streams", "1", "--ctx", "262145"}); }));
  assert(Throws([] { Batched({"--streams", "1", "--ctx", "524288"}); }));
  assert(Serial({"--ctx", "262144"}).ctx == 262144);
  assert(Throws([] { Serial({"--ctx", "262145"}); }));
  setenv("CRUCIBLE_ROPE_MODE", "yarn:2", 1);
  assert(Batched({"--streams", "1", "--ctx", "524288"}).ctx == 524288);
  assert(Throws([] { Batched({"--streams", "1", "--ctx", "524289"}); }));
  assert(Serial({"--ctx", "524288"}).ctx == 524288);
  // Review M2: any YaRN word keeps states out of the disk tier until the tier records the mode.
  setenv("CRUCIBLE_ROPE_MODE", "auto:2", 1);
  assert(Throws([] {
    Batched({"--streams", "1", "--ctx", "262144", "--conversation-cache-mib", "64", "--session-replay", "1",
             "--state-dir", "/x"});
  }));
  setenv("CRUCIBLE_ROPE_MODE", "plain", 1);
  assert(Batched({"--streams", "1", "--ctx", "262144", "--conversation-cache-mib", "64", "--session-replay", "1",
                  "--state-dir", "/x"}).state_dir == "/x");
  setenv("CRUCIBLE_ROPE_MODE", "yarn:2", 1);
  // A forced YaRN process keeps its states out of the disk tier.
  assert(Throws([] {
    Batched({"--streams", "1", "--ctx", "524288", "--conversation-cache-mib", "64", "--session-replay", "1",
             "--state-dir", "/x"});
  }));
  setenv("CRUCIBLE_ROPE_MODE", "auto:4", 1);
  assert(Batched({"--streams", "1", "--ctx", "1048576"}).ctx == 1048576);
  setenv("CRUCIBLE_ROPE_MODE", "bad", 1);
  assert(Throws([] { Batched({"--streams", "1"}); }));
  unsetenv("CRUCIBLE_ROPE_MODE");
  std::printf("PASS: rope_mode_test\n");
}
