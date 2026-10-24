// Shared descriptors and allocation helpers for engine linear plans.
#pragma once

#include <vector>

#include "exl3.h"
#include "decode_split.h"

namespace crucible {

struct Group {
  std::vector<LinJob> jobs;
  LinJob* d_jobs = nullptr;
  LinJob* d_front = nullptr;  // same jobs, tiles in front order (exl3_ring.h); null: none
  GemvItem* d_items = nullptr;
  XfItem* d_in = nullptr;
  XfItem* d_out = nullptr;
  int nitems = 0, nin = 0, nout = 0, R = 1;
  bool has_k6 = false;
  // A K7 or K8 job (kWideDecodeK, the h6_ng8 packs): the MaxK 8 kernels, never the ring kernels.
  bool wide = false;
  bool aligned = false;  // every job's split span is a multiple of kPrefetch
  bool aligned_chunk = false;  // ... and of the ring chunk (lin_ring::kChunk)
  int fuse = 0, cache = 0;
  size_t lds = 0;
};

template <typename T>
T* Upload(const std::vector<T>& v) {
  if (v.empty()) return nullptr;
  T* d;
  HIP_OK(hipMalloc(&d, sizeof(T) * v.size()));
#ifdef CRUCIBLE_QSA_TRACE
  TraceBuffer(d, sizeof(T) * v.size());
#endif
  HIP_OK(hipMemcpy(d, v.data(), sizeof(T) * v.size(), hipMemcpyHostToDevice));
  return d;
}

template <typename T>
T* Zeros(size_t n) {
  T* d;
  HIP_OK(hipMalloc(&d, sizeof(T) * n));
#ifdef CRUCIBLE_QSA_TRACE
  TraceBuffer(d, sizeof(T) * n);
#endif
  HIP_OK(hipMemset(d, 0, sizeof(T) * n));
  return d;
}

// Row and scratch buffers: a step writes each value before a later step reads it, so their first contents never
// reach an output. The default build fills them with zero (Zeros). CRUCIBLE_DEBUG_NONFINITE & 2 (an earlier change, a
// debug build) fills them with 0xFF bytes, a NaN in float, half and bf16, so a read of a value that no step wrote
// changes the gate's dumps or ends the turn with the nonfinite error. Buffers whose zero is a contract (counters,
// flags, indices, padded weights, sequence state) stay on Zeros.
template <typename T>
T* Scratch(size_t n) {
#if CRUCIBLE_DEBUG_NONFINITE & 2
  T* d = Zeros<T>(n);
  HIP_OK(hipMemset(d, 0xFF, sizeof(T) * n));
  return d;
#else
  return Zeros<T>(n);
#endif
}


}  // namespace crucible
