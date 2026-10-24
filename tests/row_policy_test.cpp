// CPU-only policy checks keep environment overrides separate from measured tables.
#include "../engine/exl3/row_policy.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace crucible;
void Clear() {
  for (const char* name : {"CRUCIBLE_TUNE_ROWS", "CRUCIBLE_LOOKAHEAD", "CRUCIBLE_LOOKAHEAD_MIB",
       "CRUCIBLE_FUSE", "CRUCIBLE_STATIC_CACHE", "CRUCIBLE_STATE_CACHE"}) unsetenv(name);
}
template<class F> void Reject(F run) {
  bool rejected=false;
  try { run(); } catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
}
int main() {
  Clear();
  const RowPolicy base{};
  // With the R > 1 ring rows, 4 lookahead blocks and 24 MiB win at every R
  // on both packs, and R=7 runs its table.
  assert(ResolveRowPolicy(1,5,3)==(RowPolicy{3,4,4,4,24}));
  assert(ResolveRowPolicy(8,6,4)==(RowPolicy{3,4,4,4,24}));
  assert(ResolveRowPolicy(7,5,3)==(RowPolicy{3,4,4,4,24}));
  assert(ResolveRowPolicy(7,6,4)==(RowPolicy{3,4,4,4,24}));
  setenv("CRUCIBLE_TUNE_ROWS","0",1);
  for (int row=1;row<=8;++row) {
    assert(ResolveRowPolicy(row,5,3)==base);
    assert(ResolveRowPolicy(row,6,4)==base);
  }
  setenv("CRUCIBLE_TUNE_ROWS","1",1);
  const RowPolicy tuned{3,4,4,4,24};
  const std::array<RowPolicy,8> regular{tuned,tuned,tuned,tuned,tuned,tuned,tuned,tuned};
  const std::array<RowPolicy,8> quality=regular;
  unsetenv("CRUCIBLE_TUNE_ROWS");
  for (int row=1;row<=8;++row) {
    assert(ResolveRowPolicy(row,5,3)==regular[row-1]);
    assert(ResolveRowPolicy(row,6,4)==quality[row-1]);
  }
  setenv("CRUCIBLE_TUNE_ROWS","1",1);
  for (int row=1;row<=8;++row) {
    assert(ResolveRowPolicy(row,5,3)==regular[row-1]);
    assert(ResolveRowPolicy(row,6,4)==quality[row-1]);
  }
  // The h6_ng8 packs (head K6; layer-0 routed K2, 2.5, 3, 3.5 or 4) take the quality table.
  for (int expertK : {2, 25, 3, 35, 4, 5})
    for (int row=1;row<=16;++row) assert(ResolveRowPolicy(row,6,expertK)==tuned);
  assert(ResolveRowPolicy(8,4,3)==base);
  assert(ResolveRowPolicy(0,5,3)==base);
  // Packed verify passes (9..16 rows) run the same policy; no table above.
  for (int row=9;row<=16;++row) {
    assert(ResolveRowPolicy(row,5,3)==tuned);
    assert(ResolveRowPolicy(row,6,4)==tuned);
  }
  assert(ResolveRowPolicy(17,5,3)==base);
  setenv("CRUCIBLE_LOOKAHEAD","0",1); setenv("CRUCIBLE_LOOKAHEAD_MIB","12",1);
  setenv("CRUCIBLE_FUSE","1",1); setenv("CRUCIBLE_STATIC_CACHE","0",1);
  setenv("CRUCIBLE_STATE_CACHE","6",1);
  assert(ResolveRowPolicy(8,5,3)==(RowPolicy{1,0,6,0,12}));
  assert(ResolveRowPolicy(8,4,3)==(RowPolicy{1,0,6,0,12}));
  Clear(); setenv("CRUCIBLE_LOOKAHEAD","2",1);
  assert(ResolveRowPolicy(8,5,3)==(RowPolicy{3,4,4,2,24}));
  assert(ResolveRowPolicy(7,5,3)==(RowPolicy{3,4,4,2,24}));
  for (const char* bad : {"", "x", "1x", "-1", "2", "999999999999999999999999"}) {
    Clear(); setenv("CRUCIBLE_TUNE_ROWS",bad,1);
    Reject([] { ResolveRowPolicy(1,5,3); });
  }
  Clear(); setenv("CRUCIBLE_STATIC_CACHE","6",1);
  Reject([] { ResolveRowPolicy(1,5,3); });
  Clear(); setenv("CRUCIBLE_STATE_CACHE","2",1);
  Reject([] { ResolveRowPolicy(1,5,3); });
  Clear(); setenv("CRUCIBLE_FUSE","4",1);
  Reject([] { ResolveRowPolicy(1,5,3); });
  Clear(); setenv("CRUCIBLE_LOOKAHEAD_MIB","0",1);
  Reject([] { ResolveRowPolicy(1,5,3); });
  Clear();
  std::cout << "row policy CPU checks PASS\n";
}
