// The prefill stage arena plan (engine/runtime/stage_plan.h): the sizes per view and part for the two packs at the
// launch rows 8192+128 and 16384+128, and the live-range rules that let the parts share bytes.
// The old sizes are the layout of trunk 10721d0b (AllocateStageArena before arena-trim). They reproduce the engine
// start lines "stage arena: 2346.1 MiB (shared; attention 2310.3 MiB, MoE 2346.1 MiB)" at 8192+128 and
// "4656.0 MiB" at 16384+128.
#include "../engine/runtime/stage_plan.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace crucible;
namespace {
constexpr double kMiB = 1048576.0;
constexpr size_t kPartial = size_t(16) << 20;  // floats, the decode floor of the engine
constexpr size_t kLaunchRowsAtCap = 16;

size_t OldMoeBytes(size_t room) {
  const size_t s = room * kSlots;
  return StageAlignUp(s * kMoeInter * 4) * 2 + StageAlignUp(s * kHidden * 4) + StageAlignUp(s * kHidden * 2) * 2 +
         StageAlignUp(s * kMoeInter * 2);
}
size_t OldAttentionBytes(size_t room, size_t wcopy_halves) {
  const size_t c = std::max<size_t>(room, kLaunchRowsAtCap);
  const auto f = [](size_t floats) { return StageAlignUp(floats * 4); };
  size_t b = f(c * kGdnQkvWidth) + f(c * kGdnValueWidth) + f(c * kQsaQWidth) + f(c * kQsaKvWidth) * 2 +
             f(c * kQsaIdxWidth) + f(c * kHidden) + f(c * 12800) + f(std::max(kPartial, c * 16384));
  b += StageAlignUp(room * kMixerOutWidth * 2) + StageAlignUp(room * kWide * 2) + StageAlignUp(wcopy_halves * 2);
  return b;
}

struct Range { const char* name; size_t offset, bytes; };
bool Disjoint(const std::vector<Range>& r, size_t limit, const char* phase) {
  bool ok = true;
  for (size_t i = 0; i < r.size(); ++i) {
    if (r[i].offset + r[i].bytes > limit) {
      std::fprintf(stderr, "%s: %s ends at %zu, past the view (%zu)\n", phase, r[i].name, r[i].offset + r[i].bytes, limit);
      ok = false;
    }
    for (size_t j = i + 1; j < r.size(); ++j)
      if (r[i].offset < r[j].offset + r[j].bytes && r[j].offset < r[i].offset + r[i].bytes) {
        std::fprintf(stderr, "%s: %s and %s overlap\n", phase, r[i].name, r[j].name);
        ok = false;
      }
  }
  return ok;
}

size_t Wcopy() {
  return std::max({size_t(kHidden) * (kGdnQkvWidth + kGdnValueWidth),
                   size_t(kHidden) * (kQsaQWidth + 2 * kQsaKvWidth + kQsaIdxWidth),
                   size_t(kMixerOutWidth) * kHidden, size_t(kQsaOutWidth) * kHidden});
}

// Every live set of the attention view, at full capacity.
void CheckAttention(const StageInputs& in, const AttentionPlan& a) {
  const size_t c = in.capacity, l = in.launch;
  const auto B = [](size_t floats) { return floats * 4; };
  const Range qkv{"qkv", a.qkv, B(c * kGdnQkvWidth)}, z{"z", a.z, B(c * kGdnValueWidth)};
  const Range q{"q", a.q, B(c * kQsaQWidth)}, k{"k", a.k, B(c * kQsaKvWidth)}, v{"v", a.v, B(c * kQsaKvWidth)},
      idx{"idx", a.idx, B(c * kQsaIdxWidth)};
  const Range xpack{"xpack", a.xpack, l * kMixerOutWidth * 2}, wcopy{"wcopy", a.wcopy, in.wcopy_halves * 2};
  const Range partial{"partial", a.partial, B(StagePartialFloats(in))};
  std::vector<Range> tail{partial};
  if (in.wcopy_halves) tail.push_back(wcopy);
  auto with = [&](std::vector<Range> head) { head.insert(head.end(), tail.begin(), tail.end()); return head; };
  // A GDN layer and a QSA layer, with the out-projection input and the GEMM workspaces.
  assert(Disjoint(with({qkv, z, xpack}), a.bytes, "GDN layer"));
  assert(Disjoint(with({q, k, v, idx, xpack}), a.bytes, "QSA layer"));
  // The PLE hook: eple -> kv -> plenv (layer 0 is done, layer 1 has not started).
  assert(Disjoint({{"eple", a.eple, B(c * kHidden)}, {"kv", a.kv, B(c * kStagePleWidth)},
                   {"plenv", a.plenv, l * kWide * 2}}, a.bytes, "PLE hook"));
  // The MTP warm-up: eple and qkv in, z (fe) and kv (fh) out, then the cache-only QSA block.
  assert(Disjoint(with({{"eple", a.eple, B(c * kHidden)}, {"qkv", a.qkv, B(c * kGdnQkvWidth)},
                        {"z", a.z, B(c * kHidden)}, {"kv", a.kv, B(c * kWide)}}), a.bytes, "MTP prep"));
  assert(Disjoint(with({q, k, v, idx}), a.bytes, "MTP cache block"));
  // The decode pass between layers works on the first kPackedRows rows of each buffer, with the workspaces.
  assert(Disjoint(with({{"qkv16", a.qkv, B(size_t(kPackedRows) * kGdnQkvWidth)},
                        {"z16", a.z, B(size_t(kPackedRows) * kGdnValueWidth)}}), a.bytes, "decode GDN"));
  assert(Disjoint(with({{"q16", a.q, B(size_t(kPackedRows) * kQsaQWidth)},
                        {"k16", a.k, B(size_t(kPackedRows) * kQsaKvWidth)},
                        {"v16", a.v, B(size_t(kPackedRows) * kQsaKvWidth)},
                        {"idx16", a.idx, B(size_t(kPackedRows) * kQsaIdxWidth)}}), a.bytes, "decode QSA"));
}

void CheckMoe(const StageInputs& in, const MoePlan& m) {
  const size_t s = m.slots, n = m.plain_slots;
  // Tile and pair paths, every slot: Fx writes xg, xu; the kernel writes xd; down reads xd, writes pd; combine.
  assert(Disjoint({{"xg", m.xg, s * kHidden * 2}, {"xu", m.xu, s * kHidden * 2}, {"xd", m.xd, s * kMoeInter * 2}},
                  m.bytes, "tile gate/up"));
  assert(Disjoint({{"xd", m.xd, s * kMoeInter * 2}, {"pd", m.pd, s * kHidden * 4}}, m.bytes, "tile down"));
  // Plain path, up to plain_slots: gate/up reads xg, xu and writes pg, pu; Mid reads pg, pu, writes xd.
  assert(Disjoint({{"xg", m.xg, n * kHidden * 2}, {"xu", m.xu, n * kHidden * 2}, {"pg", m.pg, n * kMoeInter * 4},
                   {"pu", m.pu, n * kMoeInter * 4}}, m.bytes, "plain gate/up"));
  assert(Disjoint({{"pg", m.pg, n * kMoeInter * 4}, {"pu", m.pu, n * kMoeInter * 4}, {"xd", m.xd, n * kMoeInter * 2}},
                  m.bytes, "plain mid"));
  assert(Disjoint({{"xd", m.xd, n * kMoeInter * 2}, {"pd", m.pd, n * kHidden * 4}}, m.bytes, "plain down"));
  (void)in;
}

struct Pack { const char* name; std::vector<int> widths; };
}  // namespace

int main() {
  // The half path's decode copy: 80 MiB.
  assert(Wcopy() * 2 == (80u << 20));
  // The old plan reproduces the engine logs.
  assert(std::fabs(OldMoeBytes(8320) / kMiB - 2346.1) < 0.1);
  assert(std::fabs(OldAttentionBytes(8320, Wcopy()) / kMiB - 2310.3) < 0.1);
  assert(std::fabs(OldMoeBytes(16512) / kMiB - 4656.0) < 0.1);

  // Widths of the projections: shared sg, su, sd, then the routed experts (MoeLayerWidths).
  const Pack packs[] = {{"4.05", {6, 6, 6, 4, 4, 4}}, {"3.05", {5, 5, 5, 3, 3, 3}}};
  for (const Pack& pack : packs) {
    const int kmax = PrefillMoeKmax(pack.widths);
    assert(PrefillMoeTileKmax(kmax));  // so every layer of the pack takes the tile path at 8192 rows or more
    for (const size_t room : {size_t(8320), size_t(16512)}) {
      StageInputs in;
      in.capacity = in.launch = room; in.bulk = true; in.wcopy_halves = Wcopy();
      in.decode_partial = kPartial; in.moe_all_tile = true;
      const AttentionPlan a = PlanAttention(in);
      const MoePlan m = PlanMoe(in);
      CheckAttention(in, a);
      CheckMoe(in, m);
      const size_t arena = std::max(a.bytes, m.bytes);
      const size_t old_arena = std::max(OldAttentionBytes(room, Wcopy()), OldMoeBytes(room));
      std::printf("%s %zu+128 rows: attention %.1f MiB (was %.1f), MoE %.1f MiB (was %.1f), arena %.1f MiB (was %.1f), "
                  "saved %.1f MiB\n",
                  pack.name, room - 128, a.bytes / kMiB, OldAttentionBytes(room, Wcopy()) / kMiB, m.bytes / kMiB,
                  OldMoeBytes(room) / kMiB, arena / kMiB, old_arena / kMiB, (old_arena - arena) / kMiB);
      // 16384+128 saves at least 1.8 GB (2.1 GB planned); 8192+128 at least 0.8 GB.
      if (room == 16512) assert(old_arena - arena >= size_t(2.1e9));
      if (room == 8320) assert(old_arena - arena >= size_t(0.8e9));
      // Both views shrink.
      assert(a.bytes < OldAttentionBytes(room, Wcopy()) && m.bytes < OldMoeBytes(room));
    }
  }

  // A pack with a layer that has no tile kernel set (kmax 4: routed K3/K4 with a K4 shared expert) keeps the plain
  // buffers at full size; the live-range sharing still holds.
  assert(!PrefillMoeTileKmax(PrefillMoeKmax(std::vector<int>{4, 4, 4, 3, 3, 3})));
  for (const size_t room : {size_t(4224), size_t(8320), size_t(16512)}) {
    StageInputs in;
    in.capacity = in.launch = room; in.bulk = true; in.wcopy_halves = Wcopy();
    in.decode_partial = kPartial; in.moe_all_tile = false;
    const MoePlan m = PlanMoe(in);
    assert(m.plain_slots == m.slots);
    CheckMoe(in, m);
    CheckAttention(in, PlanAttention(in));
    assert(m.bytes < OldMoeBytes(room));
  }

  // The default launch (4096+128) and a tile pack: the plain buffers are the whole launch (fewer than 8192 rows).
  {
    StageInputs in;
    in.capacity = in.launch = 4224; in.bulk = true; in.wcopy_halves = Wcopy();
    in.decode_partial = kPartial; in.moe_all_tile = true;
    const MoePlan m = PlanMoe(in);
    assert(m.plain_slots == m.slots);
    CheckMoe(in, m);
    CheckAttention(in, PlanAttention(in));
  }
  // Join rows: the launch is smaller than the capacity never; capacity is at least a packed pass.
  {
    StageInputs in;
    in.capacity = kPackedRows; in.launch = 8; in.bulk = false;
    in.decode_partial = kPartial;
    CheckAttention(in, PlanAttention(in));
    assert(PlanMoe(in).bytes == 0);
  }
  // An unfused group (CRUCIBLE_PF_TILES) keeps the full workspace.
  {
    StageInputs in;
    in.capacity = in.launch = 16512; in.bulk = true; in.unfused = true; in.decode_partial = kPartial;
    assert(StagePartialFloats(in) == StagePlanFloats(in));
    in.unfused = false;
    assert(StagePartialFloats(in) == kStageVisionRows * kStagePartialWidth);
  }
  return 0;
}
