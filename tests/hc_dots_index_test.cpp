// The bulk prefill HC dots layout (engine/layers/hc_dots_index.h): stream-major [kStreams][R][M].
// A measurement of the old [R][M][kStreams] order showed: the four stream blocks of a row group each
// stored 4 bytes of every 16, so the dots kernel sent 2.3 times its buffer to DRAM. These checks hold the new order:
// the index is a bijection onto the buffer, and one dots block (one stream, a run of rows, every column) writes one
// contiguous range that no other stream touches.
#include "../engine/layers/hc_dots_index.h"
#include <cassert>
#include <cstdio>
#include <vector>

using namespace crucible;

namespace {
void Bijection(int R, int M) {
  std::vector<int> hit((size_t)kHcDotsStreams * R * M, 0);
  for (int s = 0; s < kHcDotsStreams; ++s)
    for (int r = 0; r < R; ++r)
      for (int m = 0; m < M; ++m) {
        const size_t i = HcDotsAt(s, r, m, R, M);
        assert(i < hit.size());
        ++hit[i];
      }
  for (const int h : hit) assert(h == 1);
}

// A dots block of stream s owns rows [r0, r0 + rows) and every column: its outputs are one run of rows * M floats.
void BlockRun(int R, int M, int r0, int rows, int s) {
  const size_t first = HcDotsAt(s, r0, 0, R, M);
  for (int r = r0; r < r0 + rows; ++r)
    for (int m = 0; m < M; ++m) assert(HcDotsAt(s, r, m, R, M) == first + (size_t)(r - r0) * M + m);
  // No other stream writes inside the run.
  for (int t = 0; t < kHcDotsStreams; ++t) {
    if (t == s) continue;
    for (int r = 0; r < R; ++r) {
      const size_t a = HcDotsAt(t, r, 0, R, M), b = HcDotsAt(t, r, M - 1, R, M);
      assert(b < first || a >= first + (size_t)rows * M);
    }
  }
}
}  // namespace

int main() {
  static_assert(kHcDotsStreams == 4);
  constexpr int kM = 324;  // kRank + kStreams of the trunk HC (prefill_hc.h)
  Bijection(1, kM);
  Bijection(37, kM);
  Bijection(64, 7);
  // The engine launch: 64-row blocks (PrefillHcDotsWsKernel RT 4), and a short last block.
  for (int s = 0; s < kHcDotsStreams; ++s) {
    BlockRun(200, kM, 0, 64, s);
    BlockRun(200, kM, 192, 8, s);
  }
  // The MidPack read of row r: the kStreams values of column m sit R * M floats apart.
  assert(HcDotsAt(1, 5, 9, 200, kM) - HcDotsAt(0, 5, 9, 200, kM) == (size_t)200 * kM);
  std::printf("hc_dots_index_test: PASS\n");
  return 0;
}
