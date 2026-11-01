// PrefillInItemAt against the per-row list that PrefillLinear::Build wrote before 5c207006.
#include "../engine/exl3/prefill_items.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
  using namespace crucible;
  const int Rs[] = {1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000, 2047, 2048, 2049, 4095, 4096, 16384};
  const int Is[] = {128, 129, 255, 256, 640, 2560, 5120, 6144, 10240, 12800};
  long shapes = 0, items = 0;
  for (int jobs = 1; jobs <= 4; ++jobs)
    for (int R : Rs)
      for (int I : Is) {
        long w = 0;
        for (int j = 0; j < jobs; ++j)
          for (int r = 0; r < R; ++r)
            for (int b = 0; b < I / 128; ++b, ++w) {
              const PrefillInIndex d = PrefillInItemAt(R, I, (int)w);
              assert(d.job == j && d.r == r && d.blk == b);
            }
        assert(w == (long)jobs * R * (I / 128));
        ++shapes; items += w;
      }
  std::printf("shapes %ld items %ld mismatches 0\n", shapes, items);
  // Fused output classes (PrefillLinear::Build): K3..5 and K6 never share a
  // fused launch; a mixed group runs as two fused launches.
  const auto is = [](PrefillFusedClasses c, bool fused, bool k6, bool split) {
    return c.fused == fused && c.k6 == k6 && c.split == split;
  };
  const int k5[] = {5, 5, 5, 3}, k6[] = {6, 6}, mixed[] = {6, 6, 6, 4}, mixed2[] = {4, 6};
  assert(is(PrefillFusedClassesOf(k5, 4, true), true, false, false));
  assert(is(PrefillFusedClassesOf(k6, 2, true), true, true, false));
  assert(is(PrefillFusedClassesOf(mixed, 4, true), true, false, true));
  assert(is(PrefillFusedClassesOf(mixed2, 2, true), true, false, true));
  for (const int* k : {k5, k6, mixed}) assert(is(PrefillFusedClassesOf(k, 2, false), false, false, false));
  std::printf("fused classes ok\n");
  // K7 and K8 groups (h6_ng8 packs) run their own fused kernel; every job must have that width.
  const int k7[] = {7, 7, 7, 7}, k8[] = {8, 8}, wide_mixed[] = {7, 7, 6}, wide_two[] = {7, 8}, k3[] = {3};
  assert(PrefillWideKOf(k7, 4) == 7);
  assert(PrefillWideKOf(k8, 2) == 8);
  assert(PrefillWideKOf(wide_mixed, 3) == -1);
  assert(PrefillWideKOf(wide_two, 2) == -1);
  for (const int* k : {k5, k6, mixed, mixed2, k3}) assert(PrefillWideKOf(k, 1) == 0);
  assert(PrefillWideKOf(k5, 4) == 0 && PrefillWideKOf(mixed, 4) == 0);
  // A wide group is one fused class: no K6 launch and no split.
  assert(is(PrefillFusedClassesOf(k7, 4, true), true, false, false));
  std::printf("wide classes ok\n");
  // Half path items: every 128 x 128 block of every job exactly once, group by group.
  long half_items = 0;
  for (int G : {1, 3, 16})
    for (const std::vector<int>& Ms : std::vector<std::vector<int>>{{16}, {97, 97}, {4096, 4096}, {8192, 8192, 8192, 8192}, {300, 2048}})
      for (const std::vector<int>& Os : std::vector<std::vector<int>>{{128}, {10240, 6144}, {12288, 512, 512, 640}}) {
        const int n = (int)std::min(Ms.size(), Os.size());
        const auto v = PrefillHalfItemsOf(Ms.data(), Os.data(), n, G);
        std::vector<std::vector<int>> seen(n);
        for (int j = 0; j < n; ++j) seen[j].assign((size_t)((Ms[j] + 127) / 128) * (Os[j] / 128), 0);
        int last_group = 0;
        for (const auto& it : v) {
          assert(it.job >= 0 && it.job < n && it.m0 % 128 == 0 && it.m0 < Ms[it.job] && it.b0 % 8 == 0 && it.b0 < Os[it.job] / 16);
          assert(it.m0 / (128 * G) >= last_group);
          last_group = it.m0 / (128 * G);
          ++seen[it.job][(size_t)(it.m0 / 128) * (Os[it.job] / 128) + it.b0 / 8];
        }
        for (const auto& s : seen) for (int c : s) assert(c == 1);
        half_items += (long)v.size();
      }
  std::printf("half items ok (%ld)\n", half_items);
  return 0;
}
