// LaneWindow and LaneStates (the state extraction of DecodeLane) against a bit-by-bit reference of
// the EXL3 tile codec (ref/exl3_ref.py decode_tile_symbols): K2..K8, random tiles, every lane.
// K7 and K8 (the h6_ng8 packs) need a third aligned word; the K2..K6 path must not change.
// Half-integer widths (exllamav3 1.6, codes 15, 25, 35 = K1.5, 2.5, 3.5) against a port of exllamav3's
// frac.cu (frac_s, frac_window, pack_trellis_frac with MASK 0xAAAA), which this test checks first.
#include "../engine/exl3/kernels/lane_window.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

// State i of a tile of 8K dwords: the 16 ring bits that end at stream bit (i + 1) K, MSB first.
static uint32_t RefState(const std::vector<uint32_t>& words, int K, int i) {
  const int bits = 256 * K;
  uint32_t state = 0;
  for (int m = 0; m < 16; ++m) {
    const int x = (((i + 1) * K - 16 + m) % bits + bits) % bits;
    state = state << 1 | ((words[x / 32] >> (31 - x % 32)) & 1u);
  }
  return state;
}

template <int K>
static long CheckWidth(std::mt19937& rng, int tiles) {
  long checked = 0;
  constexpr int T = 8 * K;
  for (int tile = 0; tile < tiles; ++tile) {
    std::vector<uint32_t> words(T);
    for (uint32_t& w : words) w = rng();
    if (tile == 0) for (uint32_t& w : words) w = 0xffffffffu;  // every bit set: no lost high bits
    for (int t = 0; t < 32; ++t) {
      const crucible::LaneWindow<K> win(t);
      assert(win.d0 >= 0 && win.d0 < T && win.shift >= 1 && win.shift <= 32);
      // The lane's span must fit the three loaded words.
      assert(win.shift + 7 * K + 16 <= 96);
      uint32_t st[8];
      crucible::LaneStates<K>(words[win.d0], words[(win.d0 + 1) % T], words[(win.d0 + 2) % T], win.shift, st);
      for (int j = 0; j < 8; ++j) {
        const uint32_t want = RefState(words, K, 8 * t + j);
        if ((st[j] & 0xffffu) != want) {
          std::fprintf(stderr, "K%d tile %d lane %d state %d: got %04x want %04x\n", K, tile, t, j,
                       st[j] & 0xffffu, want);
          return -1;
        }
        ++checked;
      }
    }
  }
  return checked;
}

// exllamav3 frac.cu, MASK 0xAAAA: position i takes D(i) = KA + (i odd) new bits; window i ends at S(i).
constexpr uint32_t kHalfMask = 0xAAAAu;
static int FracD(int i, int ka) { return ka + ((kHalfMask >> (i & 15)) & 1); }
static int FracS(int i, int ka) {
  const int bpb = 16 * ka + 8;
  int s = (i >> 4) * bpb;
  for (int j = 0; j <= (i & 15); ++j) s += FracD(j, ka);
  return s;
}
static uint32_t FracWindow(const std::vector<uint32_t>& words, int start) {
  const int nw = (int)words.size(), R = nw * 32;
  start %= R;
  if (start < 0) start += R;
  const int wi = start >> 5, o = start & 31;
  const uint64_t v = (uint64_t)words[wi] << 32 | words[(wi + 1) % nw];
  return (uint32_t)((v >> (48 - o)) & 0xffffu);
}
static std::vector<uint32_t> FracPack(const std::vector<uint32_t>& states, int ka, int nw) {
  std::vector<uint32_t> words(nw, 0);
  int pos = 0;
  for (int i = 0; i < 256; ++i) {
    const int d = FracD(i, ka);
    const uint32_t v = states[i] & ((1u << d) - 1);
    for (int b = d - 1; b >= 0; --b, ++pos)
      if ((v >> b) & 1) words[pos >> 5] |= 1u << (31 - (pos & 31));
  }
  return words;
}

template <int KC>
static long CheckHalf(std::mt19937& rng, int tiles) {
  static_assert(crucible::IsHalfK(KC));
  constexpr int KA = KC / 10, T = crucible::TileDwords(KC);
  static_assert(T == 4 * (2 * KA + 1));
  long checked = 0;
  for (int tile = 0; tile < tiles; ++tile) {
    std::vector<uint32_t> words(T);
    for (uint32_t& w : words) w = rng();
    if (tile == 0) for (uint32_t& w : words) w = 0xffffffffu;
    std::vector<uint32_t> states(256);
    for (int i = 0; i < 256; ++i) states[i] = FracWindow(words, FracS(i, KA) - 16);
    // The port is exllamav3's own invariant: pack(unpack(words)) == words.
    if (FracPack(states, KA, T) != words) {
      std::fprintf(stderr, "K%d.5 tile %d: the frac port does not round-trip\n", KA, tile);
      return -1;
    }
    for (int t = 0; t < 32; ++t) {
      const crucible::LaneWindow<KC> win(t);
      assert(win.d0 >= 0 && win.d0 < T && win.shift >= 1 && win.shift <= 32);
      uint32_t st[8];
      crucible::LaneStates<KC>(words[win.d0], words[(win.d0 + 1) % T], words[(win.d0 + 2) % T], win.shift, st);
      for (int j = 0; j < 8; ++j) {
        const uint32_t want = states[8 * t + j];
        if ((st[j] & 0xffffu) != want) {
          std::fprintf(stderr, "K%d.5 tile %d lane %d state %d: got %04x want %04x\n", KA, tile, t, j,
                       st[j] & 0xffffu, want);
          return -1;
        }
        ++checked;
      }
    }
  }
  return checked;
}

// The kernels load the window's third word only for K >= 5 (`K >= 5` guards in exl3_gemm.h,
// prefill_linear.h, exl3_moe_window.h): every width whose lanes reach past bit 64 must pass it.
// K3.5 needs it (its lanes end at bit 72); K1.5 and K2.5 do not.
template <int K>
static bool ThirdWordGuardHolds() {
  int end = 0;
  for (int t = 0; t < 32; ++t) {
    const crucible::LaneWindow<K> win(t);
    const int span = crucible::IsHalfK(K) ? 7 * (K / 10) + 20 : 7 * K + 16;
    end = win.shift + span > end ? win.shift + span : end;
  }
  const bool ok = end <= 96 && (end <= 64 || K >= 5);
  if (!ok) std::fprintf(stderr, "K code %d: lanes end at bit %d, but the third-word guard is K >= 5\n", K, end);
  return ok;
}

// T = TileDwords(K) replaced T = 8 * K in the kernels (exl3_gemm.h, exl3_moe_window.h, moe_rows.h,
// prefill_linear.h): for every integer K it is the same constant, so the integer-K code cannot change.
static_assert([] {
  for (int k = 1; k <= 8; ++k)
    if (crucible::TileDwords(k) != 8 * k) return false;
  return crucible::TileDwords(15) == 12 && crucible::TileDwords(25) == 20 && crucible::TileDwords(35) == 28;
}());

int main() {
  if (!(ThirdWordGuardHolds<2>() && ThirdWordGuardHolds<3>() && ThirdWordGuardHolds<4>() && ThirdWordGuardHolds<5>() &&
        ThirdWordGuardHolds<6>() && ThirdWordGuardHolds<7>() && ThirdWordGuardHolds<8>() && ThirdWordGuardHolds<15>() &&
        ThirdWordGuardHolds<25>() && ThirdWordGuardHolds<35>()))
    return 1;
  std::mt19937 rng(461);
  long total = 0;
  for (const long n : {CheckWidth<2>(rng, 64), CheckWidth<3>(rng, 64), CheckWidth<4>(rng, 64),
                       CheckWidth<5>(rng, 64), CheckWidth<6>(rng, 64), CheckWidth<7>(rng, 64),
                       CheckWidth<8>(rng, 64)}) {
    if (n < 0) return 1;
    total += n;
  }
  long half = 0;
  for (const long n : {CheckHalf<15>(rng, 64), CheckHalf<25>(rng, 64), CheckHalf<35>(rng, 64)}) {
    if (n < 0) return 1;
    half += n;
  }
  std::printf("lane_window_test: %ld states of K2..K8 and %ld of K1.5, K2.5, K3.5 match the reference\n", total,
              half);
}
