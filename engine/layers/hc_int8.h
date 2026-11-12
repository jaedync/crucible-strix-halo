// Int8 HC weights (CRUCIBLE_HC_WMMA=2): the layout and the host quantizer.
//
// The decode HC read (hc_wmma.h) is bandwidth bound, and fn and up in fp16
// are 1,281 MB of the 5,595 MB that a 4.05 decode token reads. With the
// switch on, the loader quantizes each HC matrix once, deterministically:
//   s = fp16(absmax / 127), rounded up, for each group of G values along K,
//   q = clamp(rint(w / s), -127, 127),
//   W' = fp16(q * s).
// The fp16 fn and up then hold W', so every path that reads them (prefill,
// the VALU kernels, the fragment copies) runs the same code on W'. The WMMA
// decode read takes q and s and rebuilds W' bit for bit (q * s is exact in
// fp32, so one rounding to fp16 gives the same bits as fp16 q * fp16 s).
// So a row's x and post are the same for the int8 read and the fp16 read of
// W', and prefill and decode agree.
//
// Layout of one [N][K] matrix (one allocation):
//   q: 16-row tiles, 16-value K steps; the step j of tile t is 256 bytes at
//      (t * K / 16 + j) * 256, row r of the tile at r * 16. A byte is q + 128.
//      Rows past N (to the tile end) are zero (byte 128).
//   s: at Hc8ScaleOffset, fp16 [16 * tiles][K / G], row-major; padded rows 0.
// This header has no device code, so CPU tests include it.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../exl3/hc8_layout.h"

namespace crucible {

// Int8 blocks of one site, or null: fn [M][kWide] and up [kWide][kRank].
struct HcW8 {
  const uint8_t* fn = nullptr;
  const uint8_t* up = nullptr;
};

constexpr size_t Hc8QIndex(int n, int k, int K) {
  return ((size_t)(n / 16) * (K / 16) + k / 16) * 256 + (size_t)(n % 16) * 16 + k % 16;
}

// One group: the scale, and the codes q in [-127, 127]. A zero group gets s = 0.
inline _Float16 Hc8Group(const _Float16* w, int g, int8_t* q) {
  float amax = 0.f;
  for (int i = 0; i < g; ++i) amax = std::fmax(amax, std::fabs((float)w[i]));
  _Float16 s = (_Float16)(amax / 127.f);
  // Round the scale up, so no code clamps (subnormal scales round coarsely).
  if ((float)s * 127.f < amax) {
    uint16_t bits;
    std::memcpy(&bits, &s, 2);
    ++bits;
    std::memcpy(&s, &bits, 2);
  }
  const float sf = (float)s;
  for (int i = 0; i < g; ++i) {
    float v = sf > 0.f ? std::nearbyint((float)w[i] / sf) : 0.f;
    v = std::fmin(127.f, std::fmax(-127.f, v));
    q[i] = (int8_t)v;
  }
  return s;
}

inline _Float16 Hc8Value(int8_t q, _Float16 s) { return (_Float16)((float)q * (float)s); }

// w: fp16 [n][k] row-major. Writes the int8 block (Hc8Bytes) to `block` if it
// is non-null and W' (fp16 [n][k]) to `wq` if it is non-null; wq may be w.
inline void Hc8Quantize(const _Float16* w, int n, int k, int g, uint8_t* block, _Float16* wq) {
  _Float16* scales = block ? reinterpret_cast<_Float16*>(block + Hc8ScaleOffset(n, k)) : nullptr;
  const int rows = (int)Hc8Tiles(n) * 16;
  int8_t q[256];
  for (int r = 0; r < rows; ++r)
    for (int k0 = 0; k0 < k; k0 += g) {
      _Float16 s = 0;
      if (r < n) s = Hc8Group(w + (size_t)r * k + k0, g, q);
      else
        for (int i = 0; i < g; ++i) q[i] = 0;
      if (scales) scales[(size_t)r * (k / g) + k0 / g] = s;
      for (int i = 0; i < g; ++i) {
        if (block) block[Hc8QIndex(r, k0 + i, k)] = (uint8_t)(q[i] + 128);
        if (wq && r < n) wq[(size_t)r * k + k0 + i] = Hc8Value(q[i], s);
      }
    }
}

}  // namespace crucible
