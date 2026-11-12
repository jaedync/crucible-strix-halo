// CPU checks of the int8 HC quantizer and layout (engine/layers/hc_int8.h).
#include "../engine/layers/hc_int8.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

using namespace crucible;

namespace {

std::vector<_Float16> Fill(int n, int k, unsigned seed) {
  std::vector<_Float16> w((size_t)n * k);
  for (size_t i = 0; i < w.size(); ++i) {
    unsigned z = (unsigned)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    float v = ((z & 0xffff) / 65536.f - 0.5f) * 0.1f;
    if ((i / 64) % 7 == 3) v *= 1e-3f;  // subnormal scales
    w[i] = (_Float16)v;
  }
  for (int i = 0; i < 64; ++i) w[(size_t)2 * k + i] = 0;  // one zero group
  return w;
}

}  // namespace

int main() {
  int checks = 0;
  // Sizes of the engine sites.
  static_assert(Hc8Bytes(324, 10240, 64) == 3440640 + 336 * 160 * 2);
  static_assert(Hc8Bytes(320, 10240, 64) == 3276800 + 320 * 160 * 2);
  static_assert(Hc8Bytes(10240, 320, 32) == 3276800 + 10240 * 10 * 2);
  ++checks;

  // The tiled index is a bijection onto [0, QBytes) for a padded shape.
  {
    const int n = 37, k = 64;
    std::set<size_t> seen;
    for (int r = 0; r < (int)Hc8Tiles(n) * 16; ++r)
      for (int c = 0; c < k; ++c) seen.insert(Hc8QIndex(r, c, k));
    assert(seen.size() == Hc8QBytes(n, k) && *seen.rbegin() == Hc8QBytes(n, k) - 1);
    ++checks;
  }

  const int n = 37, k = 256, g = 64;
  const std::vector<_Float16> w = Fill(n, k, 9);
  std::vector<uint8_t> blk(Hc8Bytes(n, k, g), 0xee);
  std::vector<_Float16> wq(w.size());
  Hc8Quantize(w.data(), n, k, g, blk.data(), wq.data());
  const auto* sc = reinterpret_cast<const _Float16*>(blk.data() + Hc8ScaleOffset(n, k));

  // W' is q * s from the block, codes within [-127, 127], and every value
  // within half a step of w.
  {
    bool sub = false;
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < k; ++c) {
        const int q = (int)blk[Hc8QIndex(r, c, k)] - 128;
        const _Float16 s = sc[(size_t)r * (k / g) + c / g];
        assert(q >= -127 && q <= 127);
        sub |= s != 0 && std::fabs((float)s) < 6.1035e-5f;
        const _Float16 v = Hc8Value((int8_t)q, s);
        assert(std::memcmp(&v, &wq[(size_t)r * k + c], 2) == 0);
        assert(std::fabs((float)v - (float)w[(size_t)r * k + c]) <= 0.5f * (float)s + std::fabs((float)v) * 1e-3f + 1e-7f);
      }
    assert(sub);
    checks += 3;
  }
  // The zero group: scale 0, codes 0, W' 0.
  assert(sc[2 * (k / g)] == 0 && blk[Hc8QIndex(2, 5, k)] == 128 && wq[(size_t)2 * k + 5] == 0);
  ++checks;
  // Padded rows: codes 128, scales 0.
  for (int r = n; r < (int)Hc8Tiles(n) * 16; ++r) {
    for (int c = 0; c < k; ++c) assert(blk[Hc8QIndex(r, c, k)] == 128);
    for (int j = 0; j < k / g; ++j) assert(sc[(size_t)r * (k / g) + j] == 0);
  }
  ++checks;
  // In place (wq == w, the loader's W' task) gives the same W', and a second
  // run gives the same block (determinism).
  {
    std::vector<_Float16> inplace = w;
    Hc8Quantize(inplace.data(), n, k, g, nullptr, inplace.data());
    assert(std::memcmp(inplace.data(), wq.data(), wq.size() * 2) == 0);
    std::vector<uint8_t> again(blk.size(), 0xee);
    Hc8Quantize(w.data(), n, k, g, again.data(), nullptr);
    assert(again == blk);
    checks += 2;
  }
  std::cout << "hc_int8: " << checks << " CPU checks PASS\n";
}
