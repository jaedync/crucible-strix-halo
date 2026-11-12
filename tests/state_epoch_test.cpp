// The state epoch (engine/runtime/state_epoch.h) and the switch rule table (engine/runtime/state_switches.h):
// the state epoch design. CPU only.
#include "../engine/runtime/state_epoch.h"
#include "../engine/runtime/state_switches.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

using namespace crucible;

// The epoch is the 64 hex digits that the build stamped into the binary. An unstamped binary (the slot is 64 zeros),
// or a slot that is not 64 lower-case hex digits, falls back to the binary hash: the old identity rule.
void SlotRules() {
  const std::string binary(64, 'b'), zeros(64, '0');
  const std::string epoch = std::string("0123456789abcdef") + "fedcba9876543210" + "0011223344556677" + "8899aabbccddeeff";
  bool stamped = true;
  assert(StateEpochFromSlot(zeros, binary, &stamped) == binary && !stamped);
  assert(StateEpochFromSlot(epoch, binary, &stamped) == epoch && stamped);
  // Another binary with the same stamp has the same epoch: this is what keeps the folder over a daily switch.
  assert(StateEpochFromSlot(epoch, std::string(64, 'c')) == epoch);
  // Two stamps give two epochs.
  std::string other = epoch;
  other[63] = '0';
  assert(StateEpochFromSlot(other, binary) == other);
  // A damaged slot never gives a stamped epoch.
  for (const std::string& bad : {std::string(), epoch.substr(0, 63), epoch + "0", std::string(64, 'G'),
                                 std::string(64, 'A'), std::string("0123456789abcdef") + std::string(47, '0') + "\n"}) {
    stamped = true;
    assert(StateEpochFromSlot(bad, binary, &stamped) == binary && !stamped);
  }
  // The slot in this test binary is not stamped.
  stamped = true;
  assert(StateEpoch(binary, &stamped) == binary && !stamped);
}

// Every switch of kind kExactSpeed or kDiagnostic has exactly one rule with a reason, so a new switch must be
// classified (an unclassified one stays in the identity). Every other kind stays in the identity, except kBuild.
void RuleTable() {
  std::set<std::string> ruled;
  for (const auto& rule : kStateSwitchRules) {
    assert(ruled.insert(rule.name).second);  // once
    assert(rule.reason && std::strlen(rule.reason) > 20);
  }
  size_t classed = 0;
  for (const SwitchInfo& info : kSwitches) {
    const bool speed = info.kind == SwitchKind::kExactSpeed || info.kind == SwitchKind::kDiagnostic;
    assert(speed == ruled.count(info.name) && "rule table and registry kinds agree");
    classed += speed;
    switch (info.kind) {
      case SwitchKind::kNumerics: case SwitchKind::kLayout: case SwitchKind::kProfile:
        assert(StateSwitchInIdentity(info.name));
        break;
      case SwitchKind::kBuild:
        assert(!StateSwitchInIdentity(info.name));
        break;
      default: break;
    }
  }
  assert(classed == ruled.size());
  // Names outside the registry stay in the identity.
  assert(StateSwitchInIdentity("CRUCIBLE_NOT_A_SWITCH"));
  // A verify span above 8 rows stores 16 pending GDN rows a head and up to 13 replay rows (runtime/spec_span.h).
  assert(StateSwitchInIdentity("CRUCIBLE_SPEC_SPAN"));
  // Spot checks of both sides.
  for (const char* in : {"CRUCIBLE_EXACT_RESUME", "CRUCIBLE_PF_LAUNCH_ROWS", "CRUCIBLE_DRAFT_CHAIN", "CRUCIBLE_DRAFT_HEAD",
                         "CRUCIBLE_MTP_SIDECAR"})
    assert(StateSwitchInIdentity(in));
  for (const char* out : {"CRUCIBLE_PLE_FETCH", "CRUCIBLE_PROFILE", "CRUCIBLE_MOE_B128", "CRUCIBLE_WARMUP",
                          "CRUCIBLE_LAZY_SHADOW"})
    assert(!StateSwitchInIdentity(out));
}

int main() {
  SlotRules();
  RuleTable();
  std::puts("state_epoch_test passed");
}
