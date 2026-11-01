// Compact expert jobs for forwards of up to kPackedRows rows (from
// the speculative-decoding work). An expert job holds at most kMaxRows compact
// rows: an expert routed from more rows gets a second job, and the shared
// expert splits into kMaxRows-row chunks (the last chunk of a packed pass
// also runs kMaxRows rows; rows from R on are scratch). Up to kMaxRows rows
// the map is the
// one MoeRowsUniqueKernel always built, so those forwards do not change.
#pragma once
#include <cstddef>
#include "../model/flashnext/geometry.h"
#ifdef __HIPCC__
#define MOE_MAP_HD __host__ __device__
#else
#define MOE_MAP_HD
#endif
namespace crucible {
MOE_MAP_HD constexpr int MoeSharedChunks(int rows) { return (rows + kMaxRows - 1) / kMaxRows; }
// Routed job slots [0, R * kTopK), then the shared chunks.
MOE_MAP_HD constexpr int MoeExpertJobs(int rows) { return rows * kTopK + MoeSharedChunks(rows); }
MOE_MAP_HD constexpr int MoeSharedJob(int rows, int row) { return rows * kTopK + row / kMaxRows; }
MOE_MAP_HD constexpr int MoeSharedRow(int row) { return row % kMaxRows; }
// Gate/up input halves: every job slot (gate and up) holds up to kMaxRows rows.
MOE_MAP_HD constexpr std::size_t MoeGateUpHalves(int rows) {
  return std::size_t(MoeExpertJobs(rows)) * 2 * (rows < kMaxRows ? rows : kMaxRows) * kHidden;
}
// Down input halves (the stream-grid kernel's mid, moe_sg.h): every down job slot holds up to kMaxRows rows.
MOE_MAP_HD constexpr std::size_t MoeDownHalves(int rows) {
  return std::size_t(MoeExpertJobs(rows)) * (rows < kMaxRows ? rows : kMaxRows) * kMoeInter;
}
// The down group's xh (halves from the xh base): after the gate/up inputs of the largest pass.
MOE_MAP_HD constexpr std::size_t MoeDownXhOffset() { return MoeGateUpHalves(kPackedRows); }

// seen starts at -1; its entry names the latest job of each expert. Returns
// the routed job count, or -1 for an invalid row count or expert.
MOE_MAP_HD inline int BuildMoeChunkMap(const int* sel, int R, int* seen, int* list, int* count, int* rows,
                                       int* map, int* mapj) {
  if (R < 1 || R > kPackedRows) return -1;
  int n = 0;
  for (int i = 0; i < R * kTopK; ++i) {
    const int e = sel[i];
    if (e < 0 || e >= kExperts) return -1;
    int u = seen[e];
    if (u < 0 || count[u] == kMaxRows) {
      u = n++;
      seen[e] = u;
      list[u] = e;
      count[u] = 0;
    }
    map[i] = u;
    // Row order inside a job is ascending because i scans rows in order.
    mapj[i] = count[u];
    rows[u * kMaxRows + count[u]++] = i / kTopK;
  }
  return n;
}

// BuildMoeChunkMap per slot, for one GPU thread per slot. occ: earlier slots
// with the same expert. A slot starts a job when occ % kMaxRows == 0; job
// numbers count those slots in slot order, as BuildMoeChunkMap creates them.
MOE_MAP_HD inline int MoeSlotOcc(const int* sel, int i) {
  int c = 0;
  for (int j = 0; j < i; ++j) c += sel[j] == sel[i];
  return c;
}
MOE_MAP_HD inline int MoeSlotJob(const int* sel, const int* occ, int i) {
  int start = i;  // the slot that started this slot's job
  if (occ[i] % kMaxRows) {
    const int want = occ[i] - occ[i] % kMaxRows;
    for (int j = 0; j < i; ++j)
      if (sel[j] == sel[i] && occ[j] == want) start = j;
  }
  int u = 0;
  for (int j = 0; j < start; ++j) u += occ[j] % kMaxRows == 0;
  return u;
}
// Rows of the job that slot i starts (occ[i] % kMaxRows == 0).
MOE_MAP_HD inline int MoeSlotCount(const int* sel, const int* occ, int R, int i) {
  int total = occ[i];
  for (int j = i; j < R * kTopK; ++j) total += sel[j] == sel[i];
  const int left = total - occ[i];
  return left < kMaxRows ? left : kMaxRows;
}
}  // namespace crucible
#undef MOE_MAP_HD
