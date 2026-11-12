// Keep a coefficient load behind its fixed-order accumulator dependency.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace crucible {
// The coefficient stays unchanged between compute barriers. The probe that read it with an inline-asm ds_load_b32
// behind `dependency` (a build macro that no build set) is retired; it is in the git history before the cache-direct merge.
__device__ __forceinline__ float PersistentCoefficient(const float* p, float /*dependency*/) { return *p; }
}  // namespace crucible
