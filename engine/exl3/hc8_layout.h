// Byte layout of one int8 HC block (CRUCIBLE_HC_WMMA=2; quantizer and format in
// layers/hc_int8.h). Here so that the static arena plan (static_layout.h) can
// size the blocks without a layers/ include.
#pragma once

#include <cstddef>

namespace crucible {

constexpr int kHc8FnGroup = 64;  // fn: K = 10240 (rel. error 0.72% on 4.05 L12)
constexpr int kHc8UpGroup = 32;  // up: K = 320 (0.67%; tools/hc8_qerr.py)

constexpr size_t Hc8Tiles(int n) { return (size_t)(n + 15) / 16; }
constexpr size_t Hc8QBytes(int n, int k) { return Hc8Tiles(n) * 16 * (size_t)k; }
constexpr size_t Hc8ScaleOffset(int n, int k) { return (Hc8QBytes(n, k) + 255) & ~size_t(255); }
constexpr size_t Hc8Bytes(int n, int k, int g) {
  return Hc8ScaleOffset(n, k) + Hc8Tiles(n) * 16 * (size_t)(k / g) * 2;
}

}  // namespace crucible
