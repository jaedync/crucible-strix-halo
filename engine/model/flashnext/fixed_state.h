// The fixed-size part of one sequence's device state, in one list (gdn-ckpt). SequenceState::Spans,
// ConversationStateIO::Spans and SlotCheckpointIO::Spans visit it. SequenceState checks its total against
// FixedBytes() at construction, so a buffer added to the list or to the formula alone stops the engine at start.
// CPU-visible: tests/fixed_state_test.cpp.
#pragma once
#include <cstddef>
#include "geometry.h"

namespace crucible {
constexpr size_t kGdnStateBytes = sizeof(float) * (kGdnStateFloats + kGdnScaleFloats);
constexpr size_t kGdnHistBytes = sizeof(float) * 16 * kGdnQkvWidth;
constexpr size_t kPleRingBytes = sizeof(float) * 16 * kWide;

// In snapshot order: GDN states, conv rings, pending GDN rows, then the PLE ring. `pend_bytes` is one layer's
// pending-row buffer (kPend lives in a device header).
template <class State, class Add>
void VisitFixedState(const State& state, size_t pend_bytes, Add&& add) {
  for (const auto& [layer, pointer] : state.gstate) add(pointer, kGdnStateBytes);
  for (const auto& [layer, pointer] : state.ghist) add(pointer, kGdnHistBytes);
  for (const auto& [layer, pointer] : state.gpend) add(pointer, pend_bytes);
  add(state.ple_ring, kPleRingBytes);
}

template <class State>
size_t FixedStateBytes(const State& state, size_t pend_bytes) {
  size_t bytes = 0;
  VisitFixedState(state, pend_bytes, [&](const void*, size_t size) { bytes += size; });
  return bytes;
}

template <class State>
bool FixedStateMatches(const State& state, size_t pend_bytes, size_t expected) {
  return FixedStateBytes(state, pend_bytes) == expected;
}
}  // namespace crucible
