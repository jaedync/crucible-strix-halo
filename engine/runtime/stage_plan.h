// The byte plan of the prefill stage arena (Engine::AllocateStageArena): which row buffers share which bytes.
// Host code only, no HIP, so tests/stage_plan_test.cpp checks the engine's own plan, live range by live range.
//
// Two views share one allocation (stage_arena.h): the attention view and the MoE view; the arena is the larger.
// Inside a view, buffers share bytes only when their live ranges never meet. One stream, so launch order:
//   GDN layer     qkv, z; xpack (the packed out-projection input)
//   QSA layer     q, k, v, idx; xpack
//   PLE hook      eple -> kv -> plenv. Runs between layer 0 and layer 1, with every layer buffer dead.
//                 PrefillCutPle reads kv and plenv at once, in the same hook.
//   MTP warm-up   eple and qkv in; fe: eple -> z (R x 2560); fh: qkv -> kv (R x 10240); the stride kernel reads z
//                 and kv; then the cache-only QSA block writes q, k, v, idx (qkv, z and kv are dead by then).
//                 The MTP block never writes xpack (cache-only returns before the out-projection).
//   decode pass   between prefill layers (InterleaveAfterLayer): the first 16 rows of the layer buffers, plus
//                 partial. Nothing of the prefill layer survives a layer boundary.
//   MoE layer     plain path: Fx writes xg, xu; gate/up reads them and writes pg, pu; Mid reads pg, pu and writes
//                 xd; down reads xd and writes pd; combine reads pd.
//                 pair and tile paths: Fx writes xg, xu; the pair or tile kernel reads them and writes xd; down
//                 reads xd and writes pd; combine reads pd. They never touch pg and pu.
//
// Shared bytes, attention view (c rows of capacity):
//   layer region L = c * 16384 floats: qkv then z (GDN), or q, k, v, idx (QSA). plenv sits at its start and eple
//   at the second quarter of z (the PLE hook and the MTP block see no layer buffer live there).
//   B = c * 12800 floats: kv, and xpack at its start (kv is dead while a layer runs).
//   partial: the fp32 split-K workspace. The fused packed GEMM never writes it (every prefill group has
//   O % 128 == 0), so only the decode plans, the vision encoder and an unfused CRUCIBLE_PF_TILES run use it.
// Shared bytes, MoE view (S = launch rows * 11 slots): xd, then xg, then xu. pd takes the bytes of xg and xu
// (it is written after both are dead). pg and pu are read only by the plain path, which runs only on
// launches of fewer than kPrefillMoeTileRows rows when every layer takes the tile path; they sit in the tail
// of xu that those launches never reach (and past the view when the tail is too short).
#pragma once
#include <algorithm>
#include <cstddef>
#include "stage_arena.h"
#include "../model/flashnext/geometry.h"
#include "../layers/prefill_moe_kset.h"

namespace crucible {
// Widths of every prefill linear group output (columns). PrefillLinear::Build fuses a packed group when each
// width is a multiple of 16 * kFusedWaves = 128; a fused group never touches its fp32 partial.
inline constexpr int kStageFusedWidths[] = {kGdnQkvWidth, kGdnValueWidth, kQsaQWidth,
                                            kQsaKvWidth,  kQsaIdxWidth,   kHidden, kWide};
constexpr bool StageWidthsFuse() {
  for (const int w : kStageFusedWidths)
    if (w % 128) return false;
  return true;
}
static_assert(StageWidthsFuse(), "every prefill linear group must take the fused packed GEMM (no fp32 partial)");

constexpr int kStagePleWidth = 12800;       // PLE projection output floats per row (kv)
constexpr size_t kStageHalfBytes = 2;       // sizeof(__half)
constexpr size_t kStageVisionRows = 4096;   // VisionEncoder::kChunk: the unfused vision groups' row limit
constexpr size_t kStagePartialWidth = 16384;  // floats per row of the unfused workspace (PrefillLinear::Build)
constexpr size_t kStagePlainRows = kPrefillMoeTileRows - 1;  // the most rows of a plain-path launch

constexpr size_t StageAlignUp(size_t n) { return (n + kStageAlign - 1) / kStageAlign * kStageAlign; }

struct StageInputs {
  size_t capacity = 0;         // rows of the attention row buffers: max(kPackedRows, launch)
  size_t launch = 0;           // pf_room_rows_: launch rows plus join rows
  bool bulk = false;           // the prefill GEMM path exists (prefill_rows_ > 0)
  size_t wcopy_halves = 0;     // the half path's decode copy, 0 when there is none
  size_t decode_partial = 0;   // the floor of the split-K workspace (decode plans)
  bool unfused = false;        // an unfused prefill group is possible (CRUCIBLE_PF_TILES): keep the full workspace
  bool moe_all_tile = false;   // every layer takes the MoE tile path on launches of kPrefillMoeTileRows rows or more
};
// The floats of the split-K workspace: the old size was c * 16384; the fused prefill groups need none, the vision
// encoder needs at most kStageVisionRows rows of it.
constexpr size_t StagePartialFloats(const StageInputs& in) {
  return std::max(in.decode_partial,
                  (in.unfused ? in.capacity : std::min(in.capacity, kStageVisionRows)) * kStagePartialWidth);
}
// The old size, which PackJobs of the MTP bulk plan still uses as its group limit (the plans do not change).
constexpr size_t StagePlanFloats(const StageInputs& in) {
  return std::max(in.decode_partial, in.capacity * kStagePartialWidth);
}

// Byte offsets inside the attention view (256-byte aligned) and the view size.
struct AttentionPlan {
  size_t qkv = 0, z = 0, q = 0, k = 0, v = 0, idx = 0, eple = 0, plenv = 0, kv = 0, xpack = 0, partial = 0, wcopy = 0;
  size_t layer_bytes = 0, bytes = 0;
};
// Byte offsets inside the MoE view and the view size. plain_slots: row slots that the plain path can use.
struct MoePlan {
  size_t xd = 0, xg = 0, xu = 0, pd = 0, pg = 0, pu = 0, bytes = 0, plain_slots = 0, slots = 0;
};

inline AttentionPlan PlanAttention(const StageInputs& in) {
  AttentionPlan p;
  const size_t c = in.capacity;
  const auto f = [](size_t floats) { return StageAlignUp(floats * sizeof(float)); };
  p.qkv = 0;
  p.z = f(c * kGdnQkvWidth);
  p.q = 0;
  p.k = f(c * kQsaQWidth);
  p.v = p.k + f(c * kQsaKvWidth);
  p.idx = p.v + f(c * kQsaKvWidth);
  p.layer_bytes = StageAlignUp(std::max(p.z + f(c * kGdnValueWidth), p.idx + f(c * kQsaIdxWidth)));
  p.eple = p.z + f(c * kHidden);  // inside z (c * 6144 floats hold the MTP z output and eple)
  p.plenv = 0;                    // inside qkv (c * 10240 floats)
  p.kv = p.layer_bytes;
  p.xpack = p.kv;
  size_t at = p.kv + f(c * kStagePleWidth);
  p.partial = at;
  at += f(StagePartialFloats(in));
  if (in.wcopy_halves) {
    p.wcopy = at;
    at += StageAlignUp(in.wcopy_halves * kStageHalfBytes);
  }
  p.bytes = at;
  return p;
}

inline MoePlan PlanMoe(const StageInputs& in) {
  MoePlan p;
  if (!in.bulk) return p;
  const size_t s = in.launch * kSlots;
  p.slots = s;
  p.xd = 0;
  p.xg = StageAlignUp(s * kMoeInter * kStageHalfBytes);
  p.xu = p.xg + StageAlignUp(s * kHidden * kStageHalfBytes);
  p.pd = p.xg;  // s * kHidden floats = the bytes of xg and xu together
  const size_t end = p.xu + StageAlignUp(s * kHidden * kStageHalfBytes);
  p.plain_slots = in.moe_all_tile ? std::min(s, kStagePlainRows * kSlots) : s;
  p.pg = p.xu + StageAlignUp(p.plain_slots * kHidden * kStageHalfBytes);
  p.pu = p.pg + StageAlignUp(p.plain_slots * kMoeInter * sizeof(float));
  p.bytes = std::max(end, p.pu + StageAlignUp(p.plain_slots * kMoeInter * sizeof(float)));
  return p;
}
}  // namespace crucible
