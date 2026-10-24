// CPU check: every dense decode group of both packs loads in front order,
// whatever its K mix, and each front split equals the decode split. The
// decode split S sets the partial-sum order, so S must not move. The prefill
// GEMM must accept the mixed 4.05 group in front order (fused and split).
#include "../engine/exl3/decode_split.h"
#include "../engine/exl3/prefill_items.h"
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

using namespace crucible;

namespace {

// (I, O) from the pack headers (trellis row tiles x 16, column tiles x 16).
constexpr int kIn = 2560, kQ = 12288, kKv = 512, kIdx = 640, kGdnQkv = 10240, kGdnZ = 6144, kWideIn = 6144;

std::vector<DenseMatrix> QsaIn(int k, int k_idx) {
  return {{kIn, kQ, k}, {kIn, kKv, k}, {kIn, kKv, k}, {kIn, kIdx, k_idx}};
}

std::vector<std::pair<int, int>> Io(const std::vector<DenseMatrix>& group) {
  std::vector<std::pair<int, int>> io;
  for (const DenseMatrix& m : group) io.push_back({m.I, m.O});
  return io;
}

}  // namespace

int main() {
  // 4.05 QSA input (trunk and MTP layer): q, k, v K6 and the indexer K4.
  // Engine order sent it through LinPolicyKernel and LinGemvKernel, not the ring.
  assert((FrontGroupSplits(QsaIn(6, 4)) == std::vector<int>{2, 2, 2, 2}));
  // 3.05 (K5, K3) has the same S: the split does not depend on K.
  assert((FrontGroupSplits(QsaIn(5, 3)) == std::vector<int>{2, 2, 2, 2}));
  assert((FrontGroupSplits(QsaIn(6, 4)) == FrontGroupSplits(QsaIn(5, 3))));

  // Each front split equals the decode split of the same group (Engine::Build).
  const int packs[2][2] = {{5, 3}, {6, 4}};
  for (const auto& k : packs) {
    const std::vector<std::vector<DenseMatrix>> groups = {
        QsaIn(k[0], k[1]),
        {{kWideIn, kIn, k[0]}},                          // QSA o
        {{kIn, kGdnQkv, k[0]}, {kIn, kGdnZ, k[0]}},      // GDN qkv, z
        {{kGdnZ, kIn, k[0]}},                            // GDN out
    };
    for (const auto& group : groups) {
      const std::vector<int> front = FrontGroupSplits(group);
      assert(front == DecodeSplits(Io(group)));
      for (int s : front) assert(s > 0);
    }
  }

  // A width that the decode kernels cannot read keeps the engine order.
  assert((FrontGroupSplits({{kIn, kKv, 7}, {kIn, kKv, 6}}) == std::vector<int>{0, 0}));

  // Loader::FrontSplits: the trellis shapes of the pack headers (row tiles,
  // column tiles, 16 K), 4.05 and 3.05 QSA input.
  const FrontLoad l405 = FrontLoadSplits({{160, 768, 96}, {160, 32, 96}, {160, 32, 96}, {160, 40, 64}});
  assert((l405.splits == std::vector<int>{2, 2, 2, 2}) && l405.bad == -1);
  const FrontLoad l305 = FrontLoadSplits({{160, 768, 80}, {160, 32, 80}, {160, 32, 80}, {160, 40, 48}});
  assert(l305.splits == l405.splits && l305.bad == -1);
  // GDN qkv and z, the outputs: each fits.
  assert(FrontLoadSplits({{160, 640, 96}, {160, 384, 96}}).bad == -1);
  assert(FrontLoadSplits({{384, 160, 96}}).bad == -1);
  // A column count that is not a multiple of 8 blocks does not fit: the loader stops on matrix 1.
  assert(FrontLoadSplits({{160, 768, 96}, {160, 36, 96}}).bad == 1);

  // PrefillLinear::Build: a front-order group needs the fused GEMM. The fused
  // kernels need every O % 128 == 0 (16 * kFusedWaves); then the 4.05 group is
  // fused and split (one launch per width class), and the 3.05 group is fused only.
  for (int o : {kQ, kKv, kKv, kIdx}) assert(o % 128 == 0);
  const int k405[] = {6, 6, 6, 4}, k305[] = {5, 5, 5, 3};
  const PrefillFusedClasses c405 = PrefillFusedClassesOf(k405, 4, true);
  assert(c405.fused && !c405.k6 && c405.split);
  const PrefillFusedClasses c305 = PrefillFusedClassesOf(k305, 4, true);
  assert(c305.fused && !c305.k6 && !c305.split);
  return 0;
}
