// The state epoch: the part of a state's identity that names the code which wrote its bytes (cause 3 of the cache thrash,
// the state epoch design). It replaces the sha256 of the engine binary in StateDiskIdentityFor.
//
// A new binary used to be a new identity, so every daily switch emptied the SSD cache. Now the build stamps a
// 64-digit epoch into the binary (tools/state_epoch.py, run by the Makefile after the link). The epoch is a hash of
//   1. the state-format sources (the files that decide the layout of a stored state; comments and blanks do not count),
//   2. the name, kind and default of every kNumerics, kLayout and kProfile switch in engine/switches.h,
//   3. the gfx1151 machine code of every device function, the kernel descriptors and the device data objects,
//   4. the pins of tools/state_epoch_pins.tsv: a binary whose machine code differs may take the epoch of an older one
//      when the state-equality gate (tools/state_equality_gate.py) showed equal state bytes. Pins are evidence-gated.
// A change that alters state bytes changes at least one of these, so it gets a new epoch and a new folder. A change in
// the server, in logging, in a comment or in host code outside the listed sources keeps the epoch.
//
// A binary without a stamp (a plain `make` of another target, a test) uses its own sha256 as the epoch: the old rule.
#pragma once
#include <cstddef>
#include <string>
#include <string_view>

namespace crucible {
// The slot that tools/state_epoch.py finds in the binary and overwrites. The 20-character prefix is the search key and
// appears only here; the engine reads the 64 digits behind it. `volatile`: the compiler must read the bytes in the
// file, not the zeros of the initializer.
inline constexpr size_t kStateEpochPrefix = 20, kStateEpochDigits = 64;
inline const volatile char kStateEpochSlot[] =
    "STATE_EPOCH_SLOT_V1:0000000000000000000000000000000000000000000000000000000000000000";
static_assert(sizeof(kStateEpochSlot) == kStateEpochPrefix + kStateEpochDigits + 1, "the slot has a fixed size");

// The epoch for slot text `digits` (the 64 characters behind the prefix) and a binary hash. A stamp is 64 lower-case
// hex digits and not all zeros; anything else gives the binary hash. `stamped`: which one was used.
inline std::string StateEpochFromSlot(std::string_view digits, const std::string& binary_sha, bool* stamped = nullptr) {
  bool valid = digits.size() == kStateEpochDigits, zeros = true;
  for (const char c : digits) {
    valid = valid && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    zeros = zeros && c == '0';
  }
  const bool use = valid && !zeros;
  if (stamped) *stamped = use;
  return use ? std::string(digits) : binary_sha;
}
inline std::string StateEpoch(const std::string& binary_sha, bool* stamped = nullptr) {
  std::string digits(kStateEpochDigits, '0');
  for (size_t i = 0; i < kStateEpochDigits; ++i) digits[i] = kStateEpochSlot[kStateEpochPrefix + i];
  return StateEpochFromSlot(digits, binary_sha, stamped);
}
}  // namespace crucible
