// Numerics mode. CRUCIBLE_EXACT=1 selects option A: every
// switch that selects a non-bitwise path reads as unset, so the output is
// bitwise equal to the named baseline whatever the per-change switches say.
// Unset or 0 is option B: the switches apply, and a non-bitwise change that
// passed the guard may be on by default (NumericsDefaultOn).
// Every numerics switch must be read through these functions; the classified
// list is the kNumerics kind in engine/switches.h, and tools/gates.py
// NUMERICS_SWITCHES must match it (tests/test_numerics_switch.py and
// tests/test_switch_lists.py).
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "../exl3/row_policy.h"

namespace crucible {
inline bool NumericsExact() {
  const char* v = crucible::SwitchText("CRUCIBLE_EXACT");
  if (!v || std::string(v) == "0") return false;
  if (std::string(v) == "1") return true;
  // A typo must not silently leave option B on.
  throw std::invalid_argument("CRUCIBLE_EXACT needs 0 or 1");
}

// The switch's environment value, or null in option A.
inline const char* NumericsEnv(SwitchName name) {
  return NumericsExact() ? nullptr : SwitchText(name);
}

// An integer switch; option A keeps the default (the baseline's path).
inline int NumericsInt(SwitchName name, int fallback, int low, int high) {
  return NumericsExact() ? fallback : RowPolicyInt(name, fallback, low, high);
}

// A "1" opt-in switch; unset, empty and "0" are off, and option A keeps it off. Any other text is a typo that
// would leave the candidate path silently off, so it fails like CRUCIBLE_EXACT does.
inline bool NumericsOn(SwitchName name) {
  const char* v = NumericsEnv(name);
  if (!v || !*v || std::string(v) == "0") return false;
  if (std::string(v) == "1") return true;
  throw std::invalid_argument(std::string(name.text()) + " needs 0 or 1");
}

// Checks the text of every opt-in switch (registry kind kNumerics, default "off") once at startup. Two of them are
// read inside function-local statics at the first long prompt, so without this a bad value would fail a request in
// progress (and again at each later long prompt). It also checks the text in option A, where NumericsOn never
// reads it. Each entry point calls it before it builds an engine.
inline void ValidateNumericsSwitches() {
  for (const SwitchInfo& info : kSwitches) {
    if (info.kind != SwitchKind::kNumerics || std::string(info.default_value) != "off") continue;
    const char* v = SwitchText(info);
    if (!v || !*v || std::string(v) == "0" || std::string(v) == "1") continue;
    throw std::invalid_argument(std::string(info.name) + " needs 0 or 1");
  }
}

// A landed option-B change: on unless the switch is 0, off in option A.
inline bool NumericsDefaultOn(SwitchName name) {
  if (NumericsExact()) return false;
  const char* v = SwitchText(name);
  return !(v && std::string(v) == "0");
}

// A landed option-B integer default: `fallback` when unset, the set value in range otherwise, and `exact` (the
// baseline's value) in option A without a read of the switch.
inline int NumericsDefaultInt(SwitchName name, int fallback, int exact, int low, int high) {
  return NumericsExact() ? exact : RowPolicyInt(name, fallback, low, high);
}

inline const char* NumericsModeText() {
  return NumericsExact() ? "A (CRUCIBLE_EXACT=1: numerics switches ignored, bitwise to the baseline)"
                         : "B (numerics switches and approved non-bitwise defaults apply)";
}
}  // namespace crucible
