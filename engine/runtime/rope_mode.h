// RoPE mode of a sequence. Plain is the model's native RoPE, up to 262,144
// positions. YaRN (factor 2 or 4) extends it to 524,288 or 1,048,576 positions with the tables of
// engine/layers/rope_tables.h. A sequence keeps one mode from token 0: a change is a full prefill.
// Host code only; the kernels read the table that Engine::QArgs gives each QsaArgs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../layers/rope_tables.h"
#include "../switches.h"

namespace crucible {
enum class RopeMode : uint8_t { Plain = 0, Yarn2 = 1, Yarn4 = 2 };
inline constexpr int kRopeModes = 3;
static_assert(kRopeModes == kRopeTableModes, "one engine table per RopeMode");
inline constexpr int kRopeNativePositions = 262144;
inline constexpr int kRopeMaxPositions = 4 * kRopeNativePositions;  // 1,048,576: the largest YaRN factor

constexpr int RopeFactor(RopeMode mode) {
  return mode == RopeMode::Yarn4 ? 4 : mode == RopeMode::Yarn2 ? 2 : 1;
}
// The positions that a sequence in this mode may hold.
constexpr int RopeMaxPositions(RopeMode mode) { return kRopeNativePositions * RopeFactor(mode); }
// Whether `rows` more positions after `length` fit the mode (review M1: checked at the entry of every forward,
// before a kernel writes state).
constexpr bool RopeRowsFit(size_t length, int rows, RopeMode mode) {
  return rows >= 0 && length + size_t(rows) <= size_t(RopeMaxPositions(mode));
}
inline const char* RopeModeName(RopeMode mode);
inline void CheckRopeRows(size_t length, int rows, RopeMode mode) {
  if (!RopeRowsFit(length, rows, mode))
    throw std::invalid_argument(std::string("a ") + RopeModeName(mode) + " sequence holds at most " +
                                std::to_string(RopeMaxPositions(mode)) + " positions");
}
inline const char* RopeModeName(RopeMode mode) {
  return mode == RopeMode::Yarn4 ? "yarn:4" : mode == RopeMode::Yarn2 ? "yarn:2" : "plain";
}

// The process setting from CRUCIBLE_ROPE_MODE. start: the mode of every new (or reset) sequence. allow: the
// largest mode that a sequence of this process may use; it sets the context cap.
//   unset, "" or "plain": start = allow = Plain (today's engine).
//   "yarn:2", "yarn:4" (also "yarn:2.0", "yarn:4.0"): TEST ONLY, every sequence is YaRN from token 0.
//   "auto:2", "auto:4": sequences start plain; the host sets YaRN per request (Engine::SetStreamRope).
struct RopeSetting {
  RopeMode start = RopeMode::Plain, allow = RopeMode::Plain;
  bool Forced() const { return start != RopeMode::Plain; }
  int ContextCap() const { return RopeMaxPositions(allow); }
};
inline RopeSetting ParseRopeModeWord(const char* value) {
  if (!value) return {};
  const std::string_view word(value);
  if (word.empty() || word == "plain") return {};
  const auto factor = [](std::string_view f) -> RopeMode {
    if (f == "2" || f == "2.0") return RopeMode::Yarn2;
    if (f == "4" || f == "4.0") return RopeMode::Yarn4;
    throw std::invalid_argument("CRUCIBLE_ROPE_MODE must be plain, yarn:2, yarn:4, auto:2 or auto:4");
  };
  if (word.starts_with("yarn:")) { const RopeMode m = factor(word.substr(5)); return {m, m}; }
  if (word.starts_with("auto:")) return {RopeMode::Plain, factor(word.substr(5))};
  throw std::invalid_argument("CRUCIBLE_ROPE_MODE must be plain, yarn:2, yarn:4, auto:2 or auto:4");
}
inline RopeSetting RopeSettingFromEnv() { return ParseRopeModeWord(SwitchText("CRUCIBLE_ROPE_MODE")); }

// A context above 262,144 needs a YaRN word that allows it. `what` names the switch in the error, which keeps
// the old text for plain: "<what> must be an integer in [1,262144]".
inline void CheckRopeContext(int ctx, const RopeSetting& rope, const char* what) {
  if (ctx >= 1 && ctx <= rope.ContextCap()) return;
  std::string message = std::string(what) + " must be an integer in [1," + std::to_string(rope.ContextCap()) + "]";
  if (ctx > kRopeNativePositions && ctx <= kRopeMaxPositions && rope.ContextCap() < ctx)
    message += " (a context above 262144 needs CRUCIBLE_ROPE_MODE=yarn:F or auto:F with 262144*F >= ctx)";
  throw std::runtime_error(message);
}
}  // namespace crucible
