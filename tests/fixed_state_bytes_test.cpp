// gateway-fsb: the engine's compact-state start line reports the fixed part of one slot's state
// (SequenceState::FixedBytes) beside the logical total, so the gateway draws "KV cache" and "Recurrent state" as two bands.
// This pins that number for the Flash-Next config: 36 layers of GDN state, conv ring and pending rows, then the
// PLE ring. The start-line format string and its argument are pinned in the start-line test.
#include "../engine/model/flashnext/fixed_state.h"
#include <cassert>
#include <cstddef>
#include <map>

using namespace crucible;

// As engine/layers/ops.h (the device header that owns kPend); tests/batch_snapshot_test.cpp mirrors it the same way.
constexpr int kPend = 258;
constexpr size_t kPendBytes = sizeof(float) * 48 * kMaxRows * kPend;
// SequenceState::FixedBytes() (engine/runtime/sequence_state.h) for the trunk: the 48-layer model holds GDN state in
// 36 layers, each with its state, conv ring and pending rows, plus the PLE ring of 16 rows.
constexpr size_t kFixedBytes = 36ull * (kGdnStateBytes + kGdnHistBytes + kPendBytes) + kPleRingBytes;

struct FakeState {
  std::map<int, float*> gstate, ghist, gpend;
  float* ple_ring = nullptr;
};

int main() {
  // 151770112 bytes of the slot's 6996972544 (2.2%): the gateway shows this as the "Recurrent state" band.
  static_assert(kFixedBytes == 151770112, "the fixed state size of one slot changed");
  assert(kFixedBytes == 151770112);
  // The same total through the one list (fixed_state.h), so a buffer added to the list alone is caught here too.
  static float cells[4 * 36 + 1];
  FakeState state;
  for (int layer = 0; layer < 36; ++layer) {
    state.gstate[layer] = cells + layer;
    state.ghist[layer] = cells + 36 + layer;
    state.gpend[layer] = cells + 72 + layer;
  }
  state.ple_ring = cells + 144;
  assert(FixedStateBytes(state, kPendBytes) == kFixedBytes);
  assert(FixedStateMatches(state, kPendBytes, kFixedBytes));
  return 0;
}
