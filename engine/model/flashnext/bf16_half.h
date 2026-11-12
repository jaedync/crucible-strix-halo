// BF16 to F16 conversion for tensors that the kernels read as F16 (CPU only, no HIP).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../exl3/safetensors.h"

namespace crucible {

// Converts one BF16 word to an F16 word, round to nearest even. Returns false when the F16 result
// would not be finite (source inf, NaN or |x| > 65504). *exact is true when the F16 value equals the
// BF16 value. BF16 has 7 mantissa bits and F16 has 10, so every value with 2^-14 <= |x| <= 65504 is
// exact; a smaller value becomes an F16 subnormal (multiples of 2^-24) and may round.
inline bool Bf16ToHalf(uint16_t bf, uint16_t* half, bool* exact) {
  const uint32_t word = uint32_t(bf) << 16;
  float f;
  std::memcpy(&f, &word, sizeof(f));
  if (!std::isfinite(f)) return false;
  const _Float16 h = static_cast<_Float16>(f);  // round to nearest even
  const float back = static_cast<float>(h);
  if (!std::isfinite(back)) return false;
  std::memcpy(half, &h, sizeof(*half));
  *exact = back == f;
  return true;
}

struct HalfTensor {
  std::vector<uint16_t> bits;  // F16 words
  size_t inexact = 0;          // converted values that differ from the source
};

// Reads an F16 or BF16 tensor as F16 words (one pread, no mapped read). A BF16 value that does not fit in
// F16 (inf, NaN, |x| > 65504) stops with an error that names the tensor.
inline HalfTensor ReadHalfTensor(st::Checkpoint& ck, const std::string& name) {
  const st::TensorInfo& ti = ck.Info(name);
  if (ti.dtype != "F16" && ti.dtype != "BF16") ck.Error(name, "expected dtype F16/BF16, got " + ti.dtype);
  const size_t n = size_t(ti.end - ti.begin) / 2;
  HalfTensor out;
  out.bits.resize(n);
  ck.ReadBytes(name, out.bits.data(), n * 2);
  if (ti.dtype == "F16") return out;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t bf = out.bits[i];
    bool exact = false;
    if (!Bf16ToHalf(bf, &out.bits[i], &exact)) {
      const uint32_t word = uint32_t(bf) << 16;
      float f;
      std::memcpy(&f, &word, sizeof(f));
      char text[64];
      std::snprintf(text, sizeof(text), "%g", double(f));
      ck.Error(name, "element " + std::to_string(i) + " is BF16 " + text +
                         ", which does not fit in F16 (finite values up to 65504)");
    }
    out.inexact += !exact;
  }
  return out;
}

}  // namespace crucible
