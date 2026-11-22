// Rows of the pinned PLE host stage (engine.hip ple_stage_, ple_stage8_, ple_stage16_).
// Every row kind holds a whole prefill launch, so the host gather of a launch starts at its
// row upload (Engine::PleRowsUploaded) and overlaps the embedding and layer 0. A larger launch would be gathered in
// pieces while the device waits (Engine::GatherPle). The BF16 rows (CRUCIBLE_PLE_BF16) kept 4,096 rows before, and
// a 16384+128 launch then waited 103 ms per cold 32K prefill (the test machine, Nov 18). Host code can include this file.
#pragma once
#include "geometry.h"
#include "prefill_launch.h"
#include <cstddef>

namespace crucible {
constexpr int PleStageRowsFor(bool /*bf16*/) { return kMaxPrefillLaunchRows; }
// Pinned bytes of the BF16 stage (16 rows of kPleRowDim 16-bit words per token); the K-bit stages hold 61 or 81 words.
constexpr size_t PleStageBytes(bool bf16) {
  return (size_t)PleStageRowsFor(bf16) * 16 * (bf16 ? kPleRowDim : 61) * sizeof(uint16_t);
}
}  // namespace crucible
