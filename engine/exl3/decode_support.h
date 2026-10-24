// CPU-visible decode capabilities. Dispatch and checkpoint validation share this list.
#pragma once
#include <array>
#include <string>
namespace crucible {
inline constexpr std::array kSupportedDecodeK{2, 3, 4, 5, 6};
// K7 and K8 (the h6_ng8 packs) have their own kernel instantiations (MaxK = kMaxWideDecodeK).
// They are not in kSupportedDecodeK: the ring kernels walk that whole list, and they hold
// K <= kMaxNarrowDecodeK only (lin_ring::kMaxK).
inline constexpr std::array kWideDecodeK{7, 8};
inline constexpr int kMaxNarrowDecodeK = 6, kMaxWideDecodeK = 8;
constexpr bool SupportedDecodeK(int k) {
  for (const int supported : kSupportedDecodeK) if (k == supported) return true;
  return false;
}
constexpr bool WideDecodeK(int k) {
  for (const int wide : kWideDecodeK) if (k == wide) return true;
  return false;
}
// Every width that a dense or expert matrix may have on this engine.
constexpr bool LoadableDecodeK(int k) { return SupportedDecodeK(k) || WideDecodeK(k); }
// Half-integer widths (exllamav3 1.6) by their code 10 KA + 5 (lane_window.h): 25 = K2.5 (routed
// experts of the 2.18 and 2.68 h6_ng8 packs), 35 = K3.5 (3.18 and 3.68). Only the MoE kernels
// (MoeFusedKernelWide, the kPrefillMoeK7Half and kPrefillMoeK6Low prefill MoE kernels) hold them, so
// only routed experts may have them.
inline constexpr std::array kHalfDecodeK{25, 35};
constexpr bool HalfDecodeK(int k) {
  for (const int half : kHalfDecodeK) if (k == half) return true;
  return false;
}
// Every width that a routed expert may have.
constexpr bool LoadableExpertK(int k) { return LoadableDecodeK(k) || HalfDecodeK(k); }
// "3.5" for the code 35 (a half width), else the integer.
inline std::string KName(int k) {
  return HalfDecodeK(k) ? std::to_string(k / 10) + ".5" : std::to_string(k);
}

// The MoE widths that the decode kernels hold (MoeFusedKernel, MoeFusedKernelWide; the engine's
// MoeFusedR): the routed experts of a layer have width kr, its shared expert ks.
struct MoeWidths {
  int kr, ks;
  constexpr bool operator==(const MoeWidths&) const = default;
};
inline constexpr std::array kSupportedMoeWidths{
    MoeWidths{3, 5},  // 3.05
    MoeWidths{4, 6},  // 4.05
    MoeWidths{4, 8},  // 4.18 (h6_ng8): 46 layers
    MoeWidths{5, 8},  // 4.18: 3 layers
    MoeWidths{3, 7},  // 3.18 (h6_ng8): 41 layers with the MTP layer
    MoeWidths{35, 7},  // 3.18: 8 layers; 3.68: 41 layers with the MTP layer
    MoeWidths{4, 7},  // 3.68: 8 layers
    MoeWidths{2, 6},   // 2.18 (h6_ng8): 41 layers with the MTP layer
    MoeWidths{25, 6},  // 2.18: 8 layers; 2.68: 41 layers with the MTP layer
    MoeWidths{3, 6},   // 2.68: 8 layers
};
constexpr bool SupportedMoeWidths(MoeWidths w) {
  for (const MoeWidths& supported : kSupportedMoeWidths) if (w == supported) return true;
  return false;
}
// Twice the width, so widths compare and sort by value (the code 35 = K3.5 between K3 and K4).
constexpr int WidthHalves(int k) { return HalfDecodeK(k) ? 2 * (k / 10) + 1 : 2 * k; }
// MoeFusedKernel<R, kr> (moe_rows.h): K3 or K4 routed experts with a shared expert of kr + 2.
constexpr bool NarrowMoeWidths(MoeWidths w) { return (w.kr == 3 || w.kr == 4) && w.ks == w.kr + 2; }
// MoeFusedKernelWide<R, kr, ks>: the engine instantiates one launch for each pair (MoeFusedWideR).
inline constexpr std::array kWideMoeWidths{MoeWidths{4, 8}, MoeWidths{5, 8}, MoeWidths{3, 7}, MoeWidths{35, 7},
                                           MoeWidths{4, 7}, MoeWidths{2, 6}, MoeWidths{25, 6}, MoeWidths{3, 6}};

// KMAX code of the prefill MoE kernels (prefill_moe.h) that hold K3.5 routed experts next to a K7
// shared expert (3.18, 3.68). They are separate instantiations, so the KMAX 7 kernels keep their code.
inline constexpr int kPrefillMoeK7Half = 75;
// KMAX code of the prefill MoE kernels that hold K2, K2.5 and K3 routed experts next to a K6 shared
// expert (2.18, 2.68). Separate instantiations too: the K6 kernels (4.05) keep their code.
inline constexpr int kPrefillMoeK6Low = 65;
// The KMAX of a layer's prefill MoE kernels from the widths of its projections: the widest K of
// K3..8 (5 or less, 6, 7 and 8 select the kernel sets), kPrefillMoeK7Half for K3.5 when the widest
// integer K is 7, kPrefillMoeK6Low for K2 or K2.5 routed experts (with K3) next to K6, 0 when no
// kernel set holds the widths.
template <class Widths>
constexpr int PrefillMoeKmax(const Widths& widths) {
  int kmax = 0;
  bool half35 = false, low = false;
  for (const int k : widths) {
    if (k == 35) half35 = true;
    else if (k == 25 || k == 2) low = true;
    else if (k < 3 || k > kMaxWideDecodeK) return 0;
    else kmax = k > kmax ? k : kmax;
  }
  if (half35 && low) return 0;
  if (half35) return kmax == 7 ? kPrefillMoeK7Half : 0;
  if (low) return kmax == 6 ? kPrefillMoeK6Low : 0;
  return kmax;
}
}  // namespace crucible
