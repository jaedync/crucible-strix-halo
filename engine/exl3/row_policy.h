// With the R > 1 dense ring (ring_rows.h), 4 lookahead blocks and 24 MiB win
// at every R on both packs, R=7 included (measured
// 2026-10-06). The head-K6 h6_ng8 packs use the same table. CRUCIBLE_TUNE_ROWS=0 selects the untuned policy.
#pragma once
#include <array>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "../switches.h"

namespace crucible {
struct RowPolicy {
  int fuse=0, cache=0, state=0, blocks=0, mib=16;
  bool operator==(const RowPolicy&) const = default;
};
inline int RowPolicyInt(SwitchName name, int fallback, int low, int high) {
  const char* text=SwitchText(name);
  if (!text) return fallback;
  char* end; errno=0;
  const long value=std::strtol(text,&end,10);
  if (errno || end==text || *end || value<low || value>high)
    throw std::invalid_argument(std::string(name.text())+" needs an integer within "+
                                std::to_string(low)+".."+std::to_string(high));
  return int(value);
}
inline RowPolicy ResolveRowPolicy(int rows, int headK, int expertK) {
  constexpr RowPolicy tuned{3,4,4,4,24};
  constexpr std::array<RowPolicy,8> regular{tuned,tuned,tuned,tuned,tuned,tuned,tuned,tuned};
  constexpr std::array<RowPolicy,8> quality=regular;
  RowPolicy policy;
  // Packed verify passes (9..16 rows) take the 8-row entry.
  if (RowPolicyInt("CRUCIBLE_TUNE_ROWS",1,0,1) && rows>=1 && rows<=16) {
    const int row=rows<8 ? rows : 8;
    if (headK==5 && expertK==3) policy=regular[row-1];
    // Every head-K6 pack takes the quality table: the h6_ng8 packs (routed K2..K5) keep the
    // same token IDs with it, and their layer-0 routed width is not 4.
    else if (headK==6) policy=quality[row-1];
  }
  policy.fuse=RowPolicyInt("CRUCIBLE_FUSE",policy.fuse,0,3);
  policy.cache=RowPolicyInt("CRUCIBLE_STATIC_CACHE",policy.cache,0,4);
  policy.state=RowPolicyInt("CRUCIBLE_STATE_CACHE",policy.state,0,6);
  policy.blocks=RowPolicyInt("CRUCIBLE_LOOKAHEAD",policy.blocks,0,8);
  policy.mib=RowPolicyInt("CRUCIBLE_LOOKAHEAD_MIB",policy.mib,1,32);
  if (policy.cache!=0 && policy.cache!=4)
    throw std::invalid_argument("CRUCIBLE_STATIC_CACHE needs 0 or 4");
  if (policy.state!=0 && policy.state!=4 && policy.state!=6)
    throw std::invalid_argument("CRUCIBLE_STATE_CACHE needs 0, 4, or 6");
  return policy;
}
}  // namespace crucible
