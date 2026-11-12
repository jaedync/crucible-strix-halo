// Resolution of CRUCIBLE_MOE_B128, the decode expert body: 0 the v0 MoeFusedKernel (rollback), 1 the B64 body,
// 2 the Q4 body (moe_b128.h), 3 the stream-grid kernel (moe_sg.h; the default of t28), 12 the grid-capped
// stream-grid kernel that holds the lookahead helper between its ramp and its tail (moe-r35b). Unset means 12.
// A pure function so a CPU test can check it (tests/moe_b128_switch_test.cpp).
#pragma once
#include <algorithm>
#include <cstring>
#include <string>

namespace crucible {
inline constexpr int kMoeB128Off = 0;
inline constexpr int kMoeB128Stream = 3;
inline constexpr int kMoeB128Hold = 12;
inline constexpr int kMoeB128Default = kMoeB128Hold;

// nullptr (unset) gives the default; "0".."3" and "12" give their value; any other text gives -1 (the caller stops).
inline int ResolveMoeB128(const char* v) {
  if (!v) return kMoeB128Default;
  if (std::strlen(v) == 1 && v[0] >= '0' && v[0] <= '3') return v[0] - '0';
  if (std::strcmp(v, "12") == 0) return kMoeB128Hold;
  return -1;
}

// The stream-grid kernel serves the passes of 2..16 rows. A pass of 1 row keeps MoeStageKernel (MoeStageOn) or,
// when the stage is off (route dump, expert counts, ablation), the v0 body.
inline constexpr bool MoeB128UsesStreamGrid(int mode, int pass_rows) {
  return (mode == kMoeB128Stream || mode == kMoeB128Hold) && pass_rows >= 2 && pass_rows <= 16;
}

// The modes that launch MoeSgKernel need its control words.
inline constexpr bool MoeSgCtlNeeded(int mode) { return mode == kMoeB128Stream || mode == kMoeB128Hold; }

// MoeSgKernel shape of a stream-grid mode: K4 steps in flight (depth), the K4 body set (MoeSgKernel's NP), waves per
// SIMD (0 = the default VGPR cap kSgWaves), MoeSgKernel's OPT and the grid cap (0 = all resident blocks).
// 3: D2, bodies for 1, 2, 4 and R rows at two call sites. 12 (moe-r35b): D5, bodies for 1 and R rows at one call site,
// 6 waves per SIMD, OPT 20 << 8 | 27 (bit 0 the shared jobs' down units last, bit 1 the mid loads first, bits 3 and 4
// hold the lookahead helper of exl3/lookahead.h between the ramp and the tail, late claim lead 20 steps), and a grid
// of at most 50 blocks (400 waves: 5 per SIMD on the 40 CUs of gfx1151) for passes of 2..8 rows. Bitwise equal to 3.
// The research shapes of moe-r35 and moe-r35b (modes 4 to 11 and 13 to 24 of those branches) are bench variants
// (bench/moe_sg_engine.hip BenchShape).
struct MoeSgShape {
  int depth, np, waves = 0, opt = 0, grid = 0;
};
inline constexpr MoeSgShape MoeSgModeShape(int mode) {
  return mode == kMoeB128Hold ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 50} : MoeSgShape{2, 1};
}

// The modes whose MoeSgKernel holds the lookahead helper (OPT bit 3): the engine starts the helper's hold kernel.
inline constexpr bool MoeSgHoldsLookahead(int mode) {
  return MoeSgCtlNeeded(mode) && (MoeSgModeShape(mode).opt & 8) != 0;
}

// The grid cap of a mode for a pass (0: none). The cap is for the verify passes (2..8 rows); the packed passes
// (9..16 rows, R = 8) keep the full grid, where the cap did not help (bench R 12 and 16).
inline constexpr int MoeSgGridCap(int mode, int pass_rows) {
  return pass_rows <= 8 ? MoeSgModeShape(mode).grid : 0;
}

// The grid of the stream-grid launch (moe_sg.h): `cap` persistent blocks, or fewer if not all are resident (`per`
// blocks per CU from the occupancy query, `cus` CUs). 0 means that no block fits; the caller must stop (review L2:
// a launch of 0 blocks would fail with no error check and the combine would read old partials).
inline constexpr int MoeSgGridBlocks(int cap, int per, int cus) {
  return per < 1 || cus < 1 ? 0 : std::min(cap, per * cus);
}

// The stream-grid kernel serves passes of 2 rows or more (MoeB128UsesStreamGrid), so its instances are R = 2 to kMaxRows.
inline constexpr int MoeSgFirstR = 2;

// The error text when no block of the instance for R fits on a CU (per = blocks per CU from the occupancy query).
inline std::string MoeSgNoBlockMessage(int r, int per) {
  return "moe_sg: R=" + std::to_string(r) + ": no block of the stream-grid kernel fits on a CU (occupancy " +
         std::to_string(per) + "); set CRUCIBLE_MOE_B128=0 to run without it\n";
}
}  // namespace crucible
