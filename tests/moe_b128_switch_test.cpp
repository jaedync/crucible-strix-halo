// CPU checks of the CRUCIBLE_MOE_B128 switch resolution (moe_b128_switch.h): 12 (the grid-capped stream-grid kernel
// that holds the lookahead helper between its ramp and its tail, moe-r35b) is the default, 3 is the stream-grid
// shape of t28 (the old default, an explicit A/B value), 0 is the rollback to the v0 MoeFusedKernel path, 1 and 2 stay
// explicit A/B values. The moe-r35 and moe-r35b research shapes 4 to 11 and 13 to 24 are bench variants only.
#include "../engine/layers/moe_b128_switch.h"
#include "../engine/switches.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

using namespace crucible;

int main() {
  assert(ResolveMoeB128(nullptr) == 12);  // unset: the default is mode 12
  assert(kMoeB128Default == kMoeB128Hold && kMoeB128Hold == 12 && kMoeB128Stream == 3);
  assert(ResolveMoeB128("0") == 0);       // rollback
  assert(ResolveMoeB128("1") == 1);
  assert(ResolveMoeB128("2") == 2);
  assert(ResolveMoeB128("3") == 3);       // the old default (t28) stays reachable
  assert(ResolveMoeB128("12") == 12);     // explicit default
  for (const char* bad : {"", "4", "5", "6", "7", "8", "9", "10", "11", "13", "20", "24", "25", "-1", "01", "012",
                          "on", "off", "3 ", " 3", "12 ", "30", "120"})
    assert(ResolveMoeB128(bad) == -1);
  // The stream-grid kernel serves passes of 2..16 rows; R = 1 keeps MoeStageKernel (MoeStageOn) unless it is off.
  for (int m : {3, 12})
    assert(MoeB128UsesStreamGrid(m, 2) && MoeB128UsesStreamGrid(m, 8) && MoeB128UsesStreamGrid(m, 16) &&
           !MoeB128UsesStreamGrid(m, 1) && !MoeB128UsesStreamGrid(m, 17) && MoeSgCtlNeeded(m));
  assert(!MoeB128UsesStreamGrid(0, 2) && !MoeB128UsesStreamGrid(0, 16));
  assert(!MoeB128UsesStreamGrid(1, 4) && !MoeB128UsesStreamGrid(2, 4));
  // The stream-grid grid: all 120 blocks when they fit, fewer when not, 0 when the occupancy query gives 0 blocks.
  static_assert(MoeSgGridBlocks(120, 3, 20) == 60 && MoeSgGridBlocks(120, 6, 20) == 120 && MoeSgGridBlocks(120, 1, 40) == 40);
  static_assert(MoeSgGridBlocks(120, 0, 20) == 0 && MoeSgGridBlocks(120, 3, 0) == 0 && MoeSgGridBlocks(120, -1, 20) == 0);
  // t29 review L1: the engine checks the grid of every R at start-up; the message names R and the rollback switch.
  const std::string msg = MoeSgNoBlockMessage(5, 0);
  assert(msg.find("R=5") != std::string::npos && msg.find("occupancy 0") != std::string::npos);
  assert(msg.find("set CRUCIBLE_MOE_B128=0") != std::string::npos);
  static_assert(MoeSgFirstR == 2, "R = 1 is the stage kernel; the stream grid serves passes of 2 rows or more");
  assert(!MoeB128UsesStreamGrid(7, 4) && !MoeB128UsesStreamGrid(20, 4));
  assert(!MoeSgCtlNeeded(0) && !MoeSgCtlNeeded(1) && !MoeSgCtlNeeded(2) && !MoeSgCtlNeeded(7));
  // 3: D2, bodies 1, 2, 4 and R at two call sites, all resident blocks, no OPT, no hold.
  static_assert(MoeSgModeShape(3).depth == 2 && MoeSgModeShape(3).np == 1 && MoeSgModeShape(3).waves == 0);
  static_assert(MoeSgModeShape(3).opt == 0 && MoeSgModeShape(3).grid == 0 && !MoeSgHoldsLookahead(3));
  // 12: D5, bodies 1 and R at one call site, 6 waves per SIMD, OPT 20 << 8 | 27 (the shared downs last, the mid loads
  // first, the hold between the ramp and the tail, late claim lead 20), at most 50 blocks for the passes of 2..8 rows;
  // the packed passes (9..16 rows) keep the full grid.
  static_assert(MoeSgModeShape(12).depth == 5 && MoeSgModeShape(12).np == 3 && MoeSgModeShape(12).waves == 6);
  static_assert(MoeSgModeShape(12).opt == (20 << 8 | 27) && MoeSgModeShape(12).grid == 50);
  static_assert(MoeSgGridCap(12, 2) == 50 && MoeSgGridCap(12, 8) == 50 && MoeSgGridCap(12, 9) == 0);
  static_assert(MoeSgGridCap(12, 16) == 0 && MoeSgGridCap(3, 4) == 0);
  // Only mode 12 starts the helper's hold kernel (LookaheadHoldKernel).
  static_assert(MoeSgHoldsLookahead(12) && !MoeSgHoldsLookahead(0) && !MoeSgHoldsLookahead(2));
  // The registry's default text (engine/switches.h) is the resolved default.
  const SwitchInfo* entry = nullptr;
  for (const SwitchInfo& i : kSwitches)
    if (std::strcmp(i.name, "CRUCIBLE_MOE_B128") == 0) entry = &i;
  assert(entry != nullptr && std::to_string(kMoeB128Default) == entry->default_value);
  // The start-up check (MoeSgCheckGrids, moe_sg.h) queries the occupancy of the instances of the active mode only:
  // MoeSgCtlNeeded(mode) modes, shape MoeSgModeShape(mode), R = MoeSgFirstR to kMaxRows. Mode 12 checks the mode 12
  // instance (D5, 6 waves, OPT 20 << 8 | 27), not the default one; moe_sg.h ties both launch and check to these shapes.
  static_assert(MoeSgCtlNeeded(12) && MoeSgCtlNeeded(3) && !MoeSgCtlNeeded(0));
  static_assert(MoeSgModeShape(12).waves != MoeSgModeShape(3).waves && MoeSgModeShape(12).depth != MoeSgModeShape(3).depth);
  std::cout << "moe_b128_switch_test: ok\n";
  return 0;
}
