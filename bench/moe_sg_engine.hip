// moe-sg-engine: engine-path exactness test of the stream-grid expert kernel (CRUCIBLE_MOE_B128=3, moe_sg.h).
//
//   moe_sg_engine ROUTES RLIST VARIANTS [DNXH] [CHECKS] [ITERS] [LAYERS]
//
// RLIST: pass rows, for example "1,2,3,4,5,6,7,8" or "9,12,16". A pass of R rows gets the engine's plan for R rows
// (MakePlan: MoeExpertJobs(R) slots of min(R, 8) rows, BuildLayer's item order) and the engine's kernel at
// min(R, kMaxRows) rows, as Engine::MoeRows launches it. DNXH: "engine" (the down group's xh where
// Engine::BuildLayer puts it) or "separate" (8 Mi halves up, the verify_moe plan).
// Each call runs the engine sequence after the router: MoeRowsUniqueKernel, MoeRowsInSlotsKernel, the variant,
// MoeRowsCombineKernel. The first CHECKS calls of each (R, variant) are checked on the host after the call:
//   - H after the combine, byte for byte against variant 0 (the reference);
//   - xh and partial: the whole buffers (with 1 MiB guards on each side) are poisoned with the NaN word
//     0x7f800001 before the call; after it, a word may differ only inside the gate/up xh and the partial pools
//     and inside the rows [0, d.R) of a live down job's xh;
//   - the stream-grid control words are all zero (the kernel resets them for the next launch).
// Then ITERS timed calls (median of the variant launch, events).
// Variants: 0 the default engine launch (MoeFusedLaunch: MoeFusedKernel<min(R, 8), 4, kMoePf, 4>), 170 the
// stream-grid engine launch (Engine::MoeSgLaunch). Mutants, each must fail: 171 shuffle sums (a0 + a2) + (a1 + a3),
// 172 the mid of slot u + 1, 173 U, J and NS from the template rows, 174 the mid writes one row too many (the
// poison check only). Launch shapes (timing, must stay exact): 175 D2 WPE 12, 176 D2 WPE 12 NP 1 (= 170 since
// moe-sg-fix), 177 D2 at SgWaves(R), 178 the D5 shape of moe-b128 (170 before the change). A GPU fault
// (__builtin_trap) ends the process: run fault-prone shapes in their own process.
// moe-r35 (one call site for gate/up and down units, MoeSgRowsSet body sets; must stay exact): 180, 181 and 182 the
// launches of the moe-r35 shapes 4, 5 and 6 (BenchModeLaunch: D2 bodies 1, 2, 4, R; D2 bodies 1, R; D5 bodies
// 1, R), 183 D2 bodies R only, 184 D2 bodies 1, 2, R, 185 D5 bodies 1, 2, 4, R. Mutants of the new paths, each must
// fail: 186 = 171 on 181, 187 = 172 on 181, 188 tiles 0 and 1 swapped on 182, 189 a 2-row job runs the 1-row body on
// 181. A variant G * 1000 + v (G >= 1) runs v with the grid capped at G blocks (timing of the grid size).
// moe-r35b timeline trace (timing and bytes as their base; MoeSgKernel MUT 40): 190 = 170, 191 = 181, 192 = 182,
// 193 = 201, 194 = 200. After the timed calls, 16 more calls of a trace variant write their trace (moe_sg.h
// kSgTr*) to TRACE_DIR/tr-R<R>-v<v>.bin (argument 8, default "."): per call 16 u64 (R, v, call, layer, bytes,
// wall clock kHz, units n, wave words m, event time of the launch in ns, 7 zeros), then the trace words
// [0, kSgTrUnit + 6 n) and the m wave words at kSgTrWave.
// moe-r35b launch shapes (D K4 steps in flight, WPE, NP; must stay exact): 200 D5 W6 NP3, 201 D10 W6 NP3, 202 D10 W8
// NP3, 203 D5 W8 NP3, 204 D10 W6 NP2, 205 D5 W6 NP2, 206 D10 W6 NP4, 207 D2 W6 NP3, 208 D10 W6 NP1. Mutants: 209 = 171
// on 201, 210 = 172 on 201, 211 = 188 on 201, 212 = 189 on 201.
// moe-r35b MoeSgKernel OPT (1 shared down units last, 2 the mid with the loads first, lead << 8 late claim) and K6
// depth DS (default 1); W = WPE; must stay exact: 220 D5 W6 NP3 OPT 1, 221 the same OPT 2, 222 OPT 3, 223 OPT 3 lead
// 10, 224 OPT 3 lead 20, 225 = 223 DS 2, 226 D10 W6 NP3 OPT 3, 227 the same lead 10, 228 = 227 DS 2, 229 D2 W12 NP3
// OPT 3, 230 the same lead 10, 231 D2 W12 NP1 OPT 3 lead 10, 232 D5 at SgWaves(R) NP3 OPT 3 lead 10, 233 D5 W6 NP2
// OPT 3 lead 10, 234 = 230 DS 2, 235 = 223 lead 5. Traces: 195 = 223, 196 = 227, 197 = 230, 198 = 225. Mutants on 223:
// 240 = 171, 241 = 172, 242 = 188, 243 = 189, 244 = 174.
// 236 = the moe-r35b shape 7 (BenchModeLaunch: its grid cap at 2..8 rows unless the variant gives one); mutants on 236:
// 245 = 171, 246 = 172, 247 = 174, 248 = 189.
// OPT bit 2 (deferred unit completion, SgDefer): 237 = mode 7 + bit 2, 238 = 237 without bit 1; trace 199 = 237;
// mutants on 237: 249 = 172, 250 = 171, 251 = 174.
// Two column blocks per unit (CBG = CBD = 2; no late claim): 252 = D5 W6 NP3 OPT 3, 253 = the same OPT 0, 254 = D2
// W12 NP3 OPT 3; mutant 255 = 171 on 252.
// Lookahead hold (OPT bit 3; the bench has no helper, so only the bytes): 256 = mode 8, 257 = mode 9, 258 to 262 =
// modes 10 to 14, 263 to 267 = modes 15 to 19.
// Per (R, variant) the line also gives the mean bytes of a call (48 layers' worth / 48: unique experts x
// kSgExpertBytes + kSgSharedBytes, estimated from the routes) and the rate bytes / median.
// Last line: "SUMMARY PASS" when every non-mutant variant was bitwise equal and clean, else "SUMMARY FAIL".
#include <chrono>
#include <cstring>
#include <map>
#include <string>

#include "../engine/layers/moe_rows.h"
#include "../engine/layers/moe_sg.h"
#include "verify_moe_plan.h"

namespace {

constexpr uint32_t kPoison = 0x7f800001u;
constexpr size_t kGuardWords = (size_t(1) << 20) / 4;
// Down-group xh offset (halves from the xh base) of Engine::BuildLayer.
constexpr size_t kEngineDnXh = MoeDownXhOffset();
// Scratch past the base: the separate plan's down pool at 16 rows is the largest.
constexpr size_t kXhHalves = kBenchDnXh + MoeDownHalves(kPackedRows);
static_assert(kEngineDnXh + MoeDownHalves(kPackedRows) <= kXhHalves);
constexpr size_t kPartialFloats = size_t(MoeExpertJobs(kPackedRows)) * kMaxRows * (2 * kMoeInter + kHidden);

__global__ void PoisonKernel(uint32_t* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = kPoison;
}

unsigned* g_ctl = nullptr;
uint64_t* g_trace = nullptr;  // kSgTrWords words; its pointer sits at g_ctl + kSgCtlWords
bool TraceVariant(int v) { return v % 1000 >= 190 && v % 1000 <= 199; }

// Engine::MoeSgLaunch<R> with the mutant MUT.
int g_grid_cap = 0;  // G of a variant G * 1000 + v

template <int R, int MUT>
void SgLaunch(const Group& gu, const Group& dn, int rows) {
  MoeSgEngineLaunch<R, MUT>(gu.d_jobs, dn.d_jobs, g_ctl, rows, g_grid_cap);
}

// The research shapes of moe-r35 and moe-r35b (their CRUCIBLE_MOE_B128 values 4 to 24; the engine keeps 3 and 12,
// moe_b128_switch.h): depth, NP, waves (0 = kSgWaves, -1 = SgWaves(R)), OPT, grid cap. 4: D2, bodies 1, 2, 4, R at
// one call site; 5: D2, bodies 1, R; 6: D5 at SgWaves(R); 7: D5 W6 OPT 20 << 8 | 3, at most 50 blocks; 8, 9: 3 and 7
// holding the lookahead helper for the launch; 10 to 19: hold variants (MoeSgKernel OPT bits 3, 4 and 16..26); 12
// = the engine's mode 12 (MoeSgModeLaunch); 20 = 12 (in the engine branch, GdnKernel held the helper too); 21 to 24:
// 12 with a grid cap of 40, 60, 80 and none.
constexpr MoeSgShape BenchShape(int mode) {
  return mode == 4   ? MoeSgShape{2, 2}
         : mode == 5 ? MoeSgShape{2, 3}
         : mode == 6 ? MoeSgShape{5, 3, -1}
         : mode == 7 ? MoeSgShape{5, 3, 6, 20 << 8 | 3, 50}
         : mode == 8 ? MoeSgShape{2, 1, 0, 8}
         : mode == 9 ? MoeSgShape{5, 3, 6, 20 << 8 | 11, 50}
         : mode == 10 ? MoeSgShape{2, 1, 0, 24}
         : mode == 11 ? MoeSgShape{2, 1, 0, 8 | 2 << 16}
         : mode == 12 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 50}
         : mode == 13 ? MoeSgShape{2, 1, 0, 24 | 2 << 16}
         : mode == 14 ? MoeSgShape{2, 1, 0, 8}
         : mode == 15 ? MoeSgShape{5, 3, 6, 20 << 8 | 27 | 4 << 19, 50}
         : mode == 16 ? MoeSgShape{5, 3, 6, 20 << 8 | 27 | 8 << 19, 50}
         : mode == 17 ? MoeSgShape{5, 3, 6, 20 << 8 | 27 | 8 << 23, 50}
         : mode == 18 ? MoeSgShape{5, 3, 6, 20 << 8 | 27 | 2 << 16, 50}
         : mode == 19 ? MoeSgShape{2, 1, 0, 24 | 4 << 19}
         : mode == 20 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 50}
         : mode == 21 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 40}
         : mode == 22 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 60}
         : mode == 23 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 80}
         : mode == 24 ? MoeSgShape{5, 3, 6, 20 << 8 | 27, 0}
                     : MoeSgShape{2, 1};
}

constexpr int BenchGridCap(int mode, int pass_rows) { return pass_rows <= 8 ? BenchShape(mode).grid : 0; }
static_assert(BenchShape(12).depth == MoeSgModeShape(12).depth && BenchShape(12).np == MoeSgModeShape(12).np &&
              BenchShape(12).waves == MoeSgModeShape(12).waves && BenchShape(12).opt == MoeSgModeShape(12).opt &&
              BenchShape(12).grid == MoeSgModeShape(12).grid);

// The launch of a research shape (MUT: the bench mutants; grid_cap > 0 replaces the shape's cap).
template <int R, int MUT = 0>
void BenchModeLaunch(int mode, const LinJob* gu, const LinJob* dn, unsigned* ctl, int pass_rows, int grid_cap = 0) {
  constexpr MoeSgShape s4 = BenchShape(4), s5 = BenchShape(5), s6 = BenchShape(6), s7 = BenchShape(7);
  constexpr MoeSgShape s8 = BenchShape(8), s9 = BenchShape(9);
  static_assert(s4.depth == kSgDepth && s5.depth == kSgDepth && s6.waves == -1 && s7.waves > 0);
  static_assert(s8.depth == kSgDepth && s8.np == kSgNp && s8.waves == 0 && s9.waves > 0);
  switch (mode) {
    case 7:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, s7.opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(7, pass_rows));
      return;
    case 9:
      MoeSgEngineLaunch<R, MUT, s9.depth, s9.waves, s9.np, kSgDepthK6, s9.opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(9, pass_rows));
      return;
    case 8: MoeSgEngineLaunch<R, MUT, s8.depth, kSgWaves, s8.np, kSgDepthK6, s8.opt>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 10: MoeSgEngineLaunch<R, MUT, kSgDepth, kSgWaves, kSgNp, kSgDepthK6, BenchShape(10).opt>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 11: MoeSgEngineLaunch<R, MUT, kSgDepth, kSgWaves, kSgNp, kSgDepthK6, BenchShape(11).opt>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 12: MoeSgModeLaunch<R, MUT>(12, gu, dn, ctl, pass_rows, grid_cap); return;
    case 21: case 22: case 23: case 24:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, BenchShape(12).opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(mode, pass_rows));
      return;
    case 15:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, BenchShape(15).opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(15, pass_rows));
      return;
    case 16:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, BenchShape(16).opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(16, pass_rows));
      return;
    case 17:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, BenchShape(17).opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(17, pass_rows));
      return;
    case 18:
      MoeSgEngineLaunch<R, MUT, s7.depth, s7.waves, s7.np, kSgDepthK6, BenchShape(18).opt>(
          gu, dn, ctl, pass_rows, grid_cap > 0 ? grid_cap : BenchGridCap(18, pass_rows));
      return;
    case 20: BenchModeLaunch<R, MUT>(12, gu, dn, ctl, pass_rows, grid_cap); return;
    case 19: MoeSgEngineLaunch<R, MUT, kSgDepth, kSgWaves, kSgNp, kSgDepthK6, BenchShape(19).opt>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 13: MoeSgEngineLaunch<R, MUT, kSgDepth, kSgWaves, kSgNp, kSgDepthK6, BenchShape(13).opt>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 14:
      if constexpr (R >= 6) MoeSgEngineLaunch<R, MUT, s8.depth, kSgWaves, s8.np, kSgDepthK6, s8.opt>(gu, dn, ctl, pass_rows, grid_cap);
      else MoeSgEngineLaunch<R, MUT>(gu, dn, ctl, pass_rows, grid_cap);
      return;
    case 4: MoeSgEngineLaunch<R, MUT, s4.depth, kSgWaves, s4.np>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 5: MoeSgEngineLaunch<R, MUT, s5.depth, kSgWaves, s5.np>(gu, dn, ctl, pass_rows, grid_cap); return;
    case 6: MoeSgEngineLaunch<R, MUT, s6.depth, SgWaves(R), s6.np>(gu, dn, ctl, pass_rows, grid_cap); return;
    default: MoeSgEngineLaunch<R, MUT>(gu, dn, ctl, pass_rows, grid_cap);
  }
}

// Engine::MoeFusedLaunch<R, 4> at cache policy 4 (full staging; the chunked stage is K3 only).
template <int R>
void DefaultLaunch(const Group& gu, const Group& dn, unsigned* done) {
  hipLaunchKernelGGL((MoeFusedKernel<R, 4, kMoePf, 4>), dim3(gu.nitems + dn.nitems), dim3(256),
                     MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, gu.nitems, dn.d_jobs, dn.d_items,
                     dn.nitems, done);
}

template <int R>
bool LaunchR(int v, const Group& gu, const Group& dn, unsigned* done, int rows) {
  const int gc = g_grid_cap;
  switch (v) {
    case 0: DefaultLaunch<R>(gu, dn, done); return true;
    case 170: SgLaunch<R, 0>(gu, dn, rows); return true;
    case 171: SgLaunch<R, 1>(gu, dn, rows); return true;
    case 172: SgLaunch<R, 21>(gu, dn, rows); return true;
    case 173: SgLaunch<R, 20>(gu, dn, rows); return true;
    case 174: SgLaunch<R, 22>(gu, dn, rows); return true;
    // Launch shapes (timing; each must stay bitwise equal): D2 at 12 waves per SIMD (bench 133), the same with K4
    // bodies for 1, 2, 4 and R rows (bench 161, the engine shape since moe-sg-fix), D2 at the D5 shape's VGPR cap,
    // the D5 shape of moe-b128 (bench 130).
    case 175: MoeSgEngineLaunch<R, 0, 2, 12, 0>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 176: MoeSgEngineLaunch<R, 0, 2, 12, 1>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 177: MoeSgEngineLaunch<R, 0, 2, SgWaves(R), 0>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 178: MoeSgEngineLaunch<R, 0, 5, SgWaves(R), 0>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 180: BenchModeLaunch<R>(4, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 181: BenchModeLaunch<R>(5, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 182: BenchModeLaunch<R>(6, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 183: MoeSgEngineLaunch<R, 0, 2, 12, 5>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 184: MoeSgEngineLaunch<R, 0, 2, 12, 4>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 185: MoeSgEngineLaunch<R, 0, 5, SgWaves(R), 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 186: BenchModeLaunch<R, 1>(5, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 187: BenchModeLaunch<R, 21>(5, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 188: BenchModeLaunch<R, 3>(6, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 189: BenchModeLaunch<R, 23>(5, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 190: MoeSgEngineLaunch<R, 40, 2, 12, 1>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 191: MoeSgEngineLaunch<R, 40, 2, 12, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 192: MoeSgEngineLaunch<R, 40, 5, SgWaves(R), 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 193: MoeSgEngineLaunch<R, 40, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 194: MoeSgEngineLaunch<R, 40, 5, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 195: MoeSgEngineLaunch<R, 40, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 196: MoeSgEngineLaunch<R, 40, 10, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 197: MoeSgEngineLaunch<R, 40, 2, 12, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 198: MoeSgEngineLaunch<R, 40, 5, 6, 3, 2, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 199: MoeSgEngineLaunch<R, 40, 5, 6, 3, 1, (20 << 8 | 7)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 200: MoeSgEngineLaunch<R, 0, 5, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 201: MoeSgEngineLaunch<R, 0, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 202: MoeSgEngineLaunch<R, 0, 10, 8, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 203: MoeSgEngineLaunch<R, 0, 5, 8, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 204: MoeSgEngineLaunch<R, 0, 10, 6, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 205: MoeSgEngineLaunch<R, 0, 5, 6, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 206: MoeSgEngineLaunch<R, 0, 10, 6, 4>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 207: MoeSgEngineLaunch<R, 0, 2, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 208: MoeSgEngineLaunch<R, 0, 10, 6, 1>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 209: MoeSgEngineLaunch<R, 1, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 210: MoeSgEngineLaunch<R, 21, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 211: MoeSgEngineLaunch<R, 3, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 212: MoeSgEngineLaunch<R, 23, 10, 6, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 220: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, 1>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 221: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 222: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 223: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 224: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, (20 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 225: MoeSgEngineLaunch<R, 0, 5, 6, 3, 2, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 226: MoeSgEngineLaunch<R, 0, 10, 6, 3, 1, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 227: MoeSgEngineLaunch<R, 0, 10, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 228: MoeSgEngineLaunch<R, 0, 10, 6, 3, 2, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 229: MoeSgEngineLaunch<R, 0, 2, 12, 3, 1, 3>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 230: MoeSgEngineLaunch<R, 0, 2, 12, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 231: MoeSgEngineLaunch<R, 0, 2, 12, 1, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 232: MoeSgEngineLaunch<R, 0, 5, SgWaves(R), 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 233: MoeSgEngineLaunch<R, 0, 5, 6, 2, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 234: MoeSgEngineLaunch<R, 0, 2, 12, 3, 2, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 235: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, (5 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 236: BenchModeLaunch<R>(7, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 237: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, (20 << 8 | 7)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 238: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, (20 << 8 | 5)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 256: BenchModeLaunch<R>(8, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 257: BenchModeLaunch<R>(9, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 258: case 259: case 260: case 261: case 262: case 263: case 264: case 265: case 266: case 267:
      BenchModeLaunch<R>(v - 248, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc);
      return true;
    case 252: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, 3, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 253: MoeSgEngineLaunch<R, 0, 5, 6, 3, 1, 0, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 254: MoeSgEngineLaunch<R, 0, 2, 12, 3, 1, 3, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 255: MoeSgEngineLaunch<R, 1, 5, 6, 3, 1, 3, 2>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 249: MoeSgEngineLaunch<R, 21, 5, 6, 3, 1, (20 << 8 | 7)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 250: MoeSgEngineLaunch<R, 1, 5, 6, 3, 1, (20 << 8 | 7)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 251: MoeSgEngineLaunch<R, 22, 5, 6, 3, 1, (20 << 8 | 7)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 245: BenchModeLaunch<R, 1>(7, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 246: BenchModeLaunch<R, 21>(7, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 247: BenchModeLaunch<R, 22>(7, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 248: BenchModeLaunch<R, 23>(7, gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 240: MoeSgEngineLaunch<R, 1, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 241: MoeSgEngineLaunch<R, 21, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 242: MoeSgEngineLaunch<R, 3, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 243: MoeSgEngineLaunch<R, 23, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    case 244: MoeSgEngineLaunch<R, 22, 5, 6, 3, 1, (10 << 8 | 3)>(gu.d_jobs, dn.d_jobs, g_ctl, rows, gc); return true;
    default: return false;
  }
}

// Engine::MoeFused(G, min(rows, kMaxRows)).
bool Launch(int v, int rows, const Group& gu, const Group& dn, unsigned* done) {
  switch (std::min(rows, kMaxRows)) {
    case 1: return LaunchR<1>(v, gu, dn, done, rows);
    case 2: return LaunchR<2>(v, gu, dn, done, rows);
    case 3: return LaunchR<3>(v, gu, dn, done, rows);
    case 4: return LaunchR<4>(v, gu, dn, done, rows);
    case 5: return LaunchR<5>(v, gu, dn, done, rows);
    case 6: return LaunchR<6>(v, gu, dn, done, rows);
    case 7: return LaunchR<7>(v, gu, dn, done, rows);
    default: return LaunchR<8>(v, gu, dn, done, rows);
  }
}

// 173 equals 170 up to kMaxRows rows (the template rows are the pass rows there): a mutant only in a packed pass.
// 189 needs a job of 2 rows (none at R = 1).
bool Mutant(int v, int R) {
  v %= 1000;
  return v == 171 || v == 172 || v == 174 || (v == 173 && R > kMaxRows) || v == 186 || v == 187 || v == 188 ||
         (v == 189 && R > 1) || v == 209 || v == 210 || v == 211 || (v == 212 && R > 1) || v == 240 || v == 241 ||
         v == 242 || (v == 243 && R > 1) || (v == 244 && R > 1) || v == 245 || v == 246 || (v == 247 && R > 1) ||
         (v == 248 && R > 1) || v == 249 || v == 250 || (v == 251 && R > 1) || v == 255;
}

// Bytes of one routed expert (gate, up and down with scales) and of the K6 shared expert per layer, 4.05
// the floor rate is 245.5 GB/s (docs/CALIBRATION.md).
constexpr double kSgExpertBytes = 2.4768e6, kSgSharedBytes = 3.6864e6;

std::vector<int> Ints(const char* p) {
  std::vector<int> out;
  while (*p) {
    out.push_back(std::atoi(p));
    const char* q = std::strchr(p, ',');
    p = q ? q + 1 : p + std::strlen(p);
  }
  return out;
}

// Words of buf (guards included) that differ from kPoison outside the allowed ranges [a, b) (word indices).
size_t Stray(const std::vector<uint32_t>& buf, std::vector<std::pair<size_t, size_t>> ok, size_t* first) {
  std::sort(ok.begin(), ok.end());
  size_t n = 0, k = 0;
  for (size_t i = 0; i < buf.size(); ++i) {
    while (k < ok.size() && ok[k].second <= i) ++k;
    if (k < ok.size() && ok[k].first <= i) {
      i = ok[k].second - 1;
      continue;
    }
    if (buf[i] != kPoison) {
      if (n == 0) *first = i;
      ++n;
    }
  }
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: moe_sg_engine ROUTES RLIST VARIANTS [engine|separate] [checks] [iters] [layers] [trace dir]\n");
    return 2;
  }
  const auto toks = Tokens(argv[1]);
  const std::vector<int> rlist = Ints(argv[2]), variants = Ints(argv[3]);
  const std::string mode = argc > 4 ? argv[4] : "engine";
  if (mode != "engine" && mode != "separate") {
    std::fprintf(stderr, "DNXH must be engine or separate\n");
    return 2;
  }
  const size_t dn_xh = mode == "engine" ? kEngineDnXh : kBenchDnXh;
  const int checks = argc > 5 ? std::atoi(argv[5]) : 32, iters = argc > 6 ? std::atoi(argv[6]) : 96;
  const int nl = argc > 7 ? std::atoi(argv[7]) : 8;
  if (variants.empty() || variants[0] != 0) {
    std::fprintf(stderr, "variant 0 (the reference) must come first\n");
    return 2;
  }
  std::printf("%zu route tokens, %d layers, dn xh %s (+%zu halves), %d checked and %d timed calls per R\n",
              toks.size(), nl, mode.c_str(), dn_xh, checks, iters);
  const std::vector<LayerW> layers = MakeLayers(nl);
  // xh and partial with guards: base pointers kGuardWords words in.
  const size_t xh_words = kXhHalves / 2 + 2 * kGuardWords, pa_words = kPartialFloats + 2 * kGuardWords;
  uint32_t* xh_raw = Alloc<uint32_t>(xh_words);
  uint32_t* pa_raw = Alloc<uint32_t>(pa_words);
  VmScratch s{};
  s.xh = reinterpret_cast<__half*>(xh_raw + kGuardWords);
  s.partial = reinterpret_cast<float*>(pa_raw + kGuardWords);
  s.x = Alloc<float>(kPackedRows * kHidden);
  FillFloats<<<64, 256>>>(s.x, kPackedRows * kHidden, 7, 2.f);
  s.H = Alloc<float>(kPackedRows * kWide);
  s.post = Alloc<float>(kPackedRows * kStreams);
  FillFloats<<<4, 64>>>(s.post, kPackedRows * kStreams, 9, 1.f);
  s.wts = Alloc<float>(kPackedRows * kSlots);
  FillFloats<<<4, 64>>>(s.wts, kPackedRows * kSlots, 11, 0.2f);
  s.map = Alloc<int>(kPackedRows * kTopK);
  s.mapj = Alloc<int>(kPackedRows * kTopK);
  s.rowl = Alloc<int>(MoeExpertJobs(kPackedRows) * kMaxRows);
  s.done = Alloc<unsigned>(MoeExpertJobs(kPackedRows));
  HIP_OK(hipMemset(s.done, 0, sizeof(unsigned) * MoeExpertJobs(kPackedRows)));
  static_assert(kSgCtlWords % 2 == 0);
  HIP_OK(hipMalloc(&g_ctl, kSgCtlWords * 4 + 8));
  HIP_OK(hipMemset(g_ctl, 0, kSgCtlWords * 4));
  HIP_OK(hipMalloc(&g_trace, kSgTrWords * 8));
  HIP_OK(hipMemcpy(g_ctl + kSgCtlWords, &g_trace, 8, hipMemcpyHostToDevice));
  int wclk = 0;
  HIP_OK(hipDeviceGetAttribute(&wclk, hipDeviceAttributeWallClockRate, 0));
  const char* trdir = argc > 8 ? argv[8] : ".";
  std::vector<uint64_t> trb(kSgTrWords);
  float* H0 = Alloc<float>(kPackedRows * kWide);
  FillFloats<<<64, 256>>>(H0, kPackedRows * kWide, 13, 1.f);
  HIP_OK(hipDeviceSynchronize());
  hipEvent_t ev[2];
  for (auto& e : ev) HIP_OK(hipEventCreate(&e));
  bool all_ok = true;
  std::vector<uint32_t> xb(xh_words), pb(pa_words), ctl(kSgCtlWords);
  std::map<std::pair<int, int>, std::vector<float>> ref;
  for (const int R : rlist) {
    if (R < 1 || R > kPackedRows || (size_t)R >= toks.size()) {
      std::fprintf(stderr, "bad R %d\n", R);
      return 2;
    }
    // Call i: bench layer i % nl, model layer (7 i) % 48, the routes of tokens [w, w + R).
    const int ncalls = std::max(checks, iters);
    std::vector<int*> sels;
    std::vector<double> bytes;  // estimated bytes of call i
    for (int i = 0; i < ncalls; ++i) {
      const int ml = (7 * i) % kLayers, w = (int)((size_t)i * 13 % (toks.size() - R));
      std::vector<int> sel;
      for (int r = 0; r < R; ++r)
        sel.insert(sel.end(), toks[w + r].begin() + ml * kTopK, toks[w + r].begin() + (ml + 1) * kTopK);
      sels.push_back(Upload(sel));
      std::vector<int> u(sel);
      std::sort(u.begin(), u.end());
      bytes.push_back((double)(std::unique(u.begin(), u.end()) - u.begin()) * kSgExpertBytes + kSgSharedBytes);
    }
    for (const int v : variants) {
      std::vector<Plan> plans;
      for (int l = 0; l < nl; ++l) plans.push_back(MakePlan(layers[l], R, s, kMoeItemBlocks<kMoeCols>, dn_xh));
      auto call = [&](int i, bool timed) {
        const Plan& p = plans[i % nl];
        MoeRowsUniqueKernel<<<1, 512>>>(sels[i], R, layers[i % nl].d_experts, p.gu.d_jobs, p.dn.d_jobs, s.map,
                                        s.mapj, s.rowl, s.done);
        MoeRowsInSlotsKernel<<<MoeInSlotsBlocks(R), 256>>>(p.gu.d_jobs, s.map, s.mapj, R);
        if (timed) HIP_OK(hipEventRecord(ev[0], 0));
        g_grid_cap = v / 1000;
        if (!Launch(v % 1000, R, p.gu, p.dn, s.done)) {
          std::fprintf(stderr, "unknown variant %d\n", v);
          std::exit(2);
        }
        if (timed) HIP_OK(hipEventRecord(ev[1], 0));
        MoeRowsCombineKernel<<<(R * 20 + 7) / 8, 256>>>(p.dn.d_jobs, s.map, s.mapj, s.wts, s.post, s.H, R);
      };
      // Pools of this plan (word indices of the raw buffers).
      const Plan& p0 = plans[0];
      size_t gu_x = 0, gu_p = 0, dn_p = 0;
      for (const LinJob& j : p0.gu.jobs) gu_x += (size_t)j.R * j.I, gu_p += (size_t)j.S * j.R * j.O;
      for (const LinJob& j : p0.dn.jobs) dn_p += (size_t)j.S * j.R * j.O;
      int ndiff = 0, nstray_x = 0, nstray_p = 0, nctl = 0, nnan = 0;
      size_t fx = 0, fp = 0;
      for (int i = 0; i < checks; ++i) {
        HIP_OK(hipMemcpy(s.H, H0, sizeof(float) * kPackedRows * kWide, hipMemcpyDeviceToDevice));
        PoisonKernel<<<512, 256>>>(xh_raw, xh_words);
        PoisonKernel<<<512, 256>>>(pa_raw, pa_words);
        call(i, false);
        HIP_OK(hipDeviceSynchronize());
        std::vector<float> h((size_t)R * kWide);
        HIP_OK(hipMemcpy(h.data(), s.H, h.size() * 4, hipMemcpyDeviceToHost));
        for (float f : h) nnan += !(f == f);
        if (v == 0) ref[{R, i}] = h;
        else ndiff += std::memcmp(ref[{R, i}].data(), h.data(), h.size() * 4) != 0;
        const Plan& p = plans[i % nl];
        std::vector<LinJob> dj(p.dn.jobs.size());
        HIP_OK(hipMemcpy(dj.data(), p.dn.d_jobs, dj.size() * sizeof(LinJob), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(xb.data(), xh_raw, xh_words * 4, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(pb.data(), pa_raw, pa_words * 4, hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(ctl.data(), g_ctl, kSgCtlWords * 4, hipMemcpyDeviceToHost));
        // xh words: the gate/up pool, and rows [0, d.R) of every live down job.
        std::vector<std::pair<size_t, size_t>> okx{{kGuardWords, kGuardWords + gu_x / 2}};
        for (const LinJob& d : dj) {
          if (d.K == 0) continue;
          const size_t a = kGuardWords + (size_t)(d.xh - s.xh) / 2;
          okx.push_back({a, a + (size_t)d.R * d.I / 2});
        }
        size_t f = 0;
        const size_t sx = Stray(xb, okx, &f);
        if (sx && !nstray_x) fx = f;
        nstray_x += sx != 0;
        const size_t sp = Stray(pb, {{kGuardWords, kGuardWords + gu_p + dn_p}}, &f);
        if (sp && !nstray_p) fp = f;
        nstray_p += sp != 0;
        nctl += std::any_of(ctl.begin(), ctl.end(), [](uint32_t c) { return c != 0; });
        if (nctl) HIP_OK(hipMemset(g_ctl, 0, kSgCtlWords * 4));  // a failed reset must not chain into the next call
      }
      for (int i = 0; i < nl; ++i) call(i, false);
      HIP_OK(hipDeviceSynchronize());
      std::vector<double> us;
      double bsum = 0;
      for (int i = 0; i < iters; ++i) {
        bsum += bytes[i];
        call(i, true);
        HIP_OK(hipEventSynchronize(ev[1]));
        float t;
        HIP_OK(hipEventElapsedTime(&t, ev[0], ev[1]));
        us.push_back(t * 1e3);
      }
      if (TraceVariant(v)) {
        const std::string fn = std::string(trdir) + "/tr-R" + std::to_string(R) + "-v" + std::to_string(v) + ".bin";
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) {
          std::fprintf(stderr, "cannot write %s\n", fn.c_str());
          std::exit(2);
        }
        for (int i = 0; i < 16; ++i) {
          HIP_OK(hipMemset(g_trace, 0, kSgTrWords * 8));
          call(i, true);
          HIP_OK(hipDeviceSynchronize());
          float et = 0;
          HIP_OK(hipEventElapsedTime(&et, ev[0], ev[1]));
          HIP_OK(hipMemcpy(trb.data(), g_trace, kSgTrWords * 8, hipMemcpyDeviceToHost));
          const uint64_t nu = std::min<uint64_t>(trb[0], kSgTrMaxUnits);
          const uint64_t hdr[16] = {(uint64_t)R, (uint64_t)v, (uint64_t)i, (uint64_t)(i % nl), (uint64_t)bytes[i],
                                    (uint64_t)wclk, nu, 2 * 8 * kSgBlocks, (uint64_t)(et * 1e6)};
          std::fwrite(hdr, 8, 16, fp);
          std::fwrite(trb.data(), 8, kSgTrUnit + 6 * nu, fp);
          std::fwrite(trb.data() + kSgTrWave, 8, 2 * 8 * kSgBlocks, fp);
        }
        std::fclose(fp);
      }
      std::sort(us.begin(), us.end());
      const bool clean = (v == 0 || ndiff == 0) && nnan == 0 && nstray_x == 0 && nstray_p == 0 && nctl == 0;
      if (!Mutant(v, R) && !clean) all_ok = false;
      if (Mutant(v, R) && clean) all_ok = false;  // a mutant that passes means the check is blind
      const double med = us.empty() ? 0.0 : us[us.size() / 2], mb = iters ? bsum / iters : 0.0;
      std::printf("R=%d v%d med %.1f us p10 %.1f p90 %.1f | %.2f MB %.1f GB/s | H %s (%d of %d calls differ, NaN %d) | xh stray %d calls"
                  " (first word %lld) | partial stray %d calls (first word %lld) | ctl not reset %d calls | %s\n",
                  R, v, med, us.empty() ? 0.0 : us[us.size() / 10], us.empty() ? 0.0 : us[us.size() * 9 / 10], mb / 1e6,
                  med > 0 ? mb / med / 1e3 : 0.0, v == 0 ? "reference" : ndiff ? "DIFFERENT" : "bitwise equal",
                  ndiff, checks, nnan, nstray_x, nstray_x ? (long long)fx - (long long)kGuardWords : 0, nstray_p,
                  nstray_p ? (long long)fp - (long long)kGuardWords : 0, nctl,
                  clean ? (Mutant(v, R) ? "MUTANT PASSED (blind check)" : "clean") : (Mutant(v, R) ? "mutant caught" : "FAIL"));
      std::fflush(stdout);
    }
  }
  std::printf("SUMMARY %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
