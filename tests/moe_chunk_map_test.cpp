// CPU checks of the compact expert map for packed verify passes up to
// kPackedRows rows: jobs hold at most kMaxRows compact rows, and the shared
// expert splits into kMaxRows-row chunks.
#include "../engine/layers/moe_chunk_map.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>

using namespace crucible;
int main() {
  static_assert(kPackedRows == 16 && kMaxRows == 8);
  static_assert(MoeSharedChunks(1) == 1 && MoeSharedChunks(8) == 1 && MoeSharedChunks(9) == 2);
  static_assert(MoeExpertJobs(8) == 81 && MoeExpertJobs(16) == 162);
  static_assert(MoeGateUpHalves(8) == 3317760 && MoeGateUpHalves(16) == 6635520);
  for (int R = 1; R <= kPackedRows; ++R)
    for (int pattern = 0; pattern < 4; ++pattern) {
      std::array<int, kPackedRows * kTopK> sel{}, map{}, mapj{}, list{}, count{};
      std::array<int, kExperts> seen{};
      seen.fill(-1);
      std::array<int, kPackedRows * kTopK * kMaxRows> rows{};
      rows.fill(-1);
      for (int r = 0; r < R; ++r)
        for (int k = 0; k < kTopK; ++k)
          sel[r * kTopK + k] = pattern == 0 ? k : pattern == 1 ? r * kTopK + k
                             : pattern == 2 ? (r + k) % 32 : (r * 17 + k * 41) % kExperts;
      const int n = BuildMoeChunkMap(sel.data(), R, seen.data(), list.data(), count.data(), rows.data(),
                                     map.data(), mapj.data());
      assert(n > 0 && n <= R * kTopK);
      for (int u = 0; u < n; ++u) {
        assert(count[u] > 0 && count[u] <= kMaxRows);
        for (int j = 1; j < count[u]; ++j) assert(rows[u * kMaxRows + j] > rows[u * kMaxRows + j - 1]);
      }
      for (int r = 0; r < R; ++r)
        for (int k = 0; k < kTopK; ++k) {
          const int i = r * kTopK + k, u = map[i], j = mapj[i];
          assert(u >= 0 && u < n && j >= 0 && j < count[u]);
          assert(list[u] == sel[i] && rows[u * kMaxRows + j] == r);
        }
      // The per-slot form (one GPU thread per slot) gives the same map.
      std::array<int, kPackedRows * kTopK> occ{};
      for (int i = 0; i < R * kTopK; ++i) occ[i] = MoeSlotOcc(sel.data(), i);
      int created = 0;
      for (int i = 0; i < R * kTopK; ++i) {
        const int u = MoeSlotJob(sel.data(), occ.data(), i);
        assert(u == map[i] && occ[i] % kMaxRows == mapj[i]);
        if (occ[i] % kMaxRows == 0) {
          ++created;
          assert(list[u] == sel[i] && MoeSlotCount(sel.data(), occ.data(), R, i) == count[u]);
        }
      }
      assert(created == n);
      // Up to 8 rows the map is the one MoeRowsUniqueKernel always built.
      if (pattern == 0) assert(n == kTopK * ((R + kMaxRows - 1) / kMaxRows));
      for (int r = 0; r < R; ++r) {
        const int u = MoeSharedJob(R, r), j = MoeSharedRow(r);
        assert(u >= R * kTopK && u < MoeExpertJobs(R));
        assert(j >= 0 && j < kMaxRows && (u - R * kTopK) * kMaxRows + j == r);
      }
    }
  // The down group's xh (the stream-grid mid) never overlaps the gate/up inputs of any pass.
  for (int R = 1; R <= kPackedRows; ++R) {
    assert(MoeGateUpHalves(R) <= MoeDownXhOffset());
    assert(MoeDownHalves(R) == std::size_t(MoeExpertJobs(R)) * std::min(R, kMaxRows) * kMoeInter);
  }
  std::array<int, kExperts> seen{};
  seen.fill(-1);
  std::array<int, kPackedRows * kTopK> a{};
  std::array<int, kPackedRows * kTopK * kMaxRows> b{};
  assert(BuildMoeChunkMap(a.data(), kPackedRows + 1, seen.data(), a.data(), a.data(), b.data(), a.data(),
                          a.data()) == -1);
  std::cout << "moe chunk map CPU checks PASS\n";
}
