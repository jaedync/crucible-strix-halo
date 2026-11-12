// CPU checks of the CRUCIBLE_GDN_VERIFY switch resolution (gdn_verify_switch.h): unset or "0" is off (GdnKernel),
// "1".."8" is the smallest row count that runs GdnVerifyKernel, every other text is an error.
#include "../engine/layers/gdn_verify_switch.h"
#include <cassert>
#include <iostream>

using namespace crucible;
int main() {
  static_assert(kGdnVerifyOff == 0 && kGdnVerifyMaxRows == 8);
  assert(ResolveGdnVerify(nullptr) == 0);  // unset: GdnKernel
  assert(ResolveGdnVerify("") == 0);       // empty is unset
  assert(ResolveGdnVerify("0") == 0);
  for (int n = 1; n <= 8; ++n) {
    const char text[2] = {char('0' + n), 0};
    assert(ResolveGdnVerify(text) == n);
  }
  for (const char* bad : {"9", "10", "-1", "+1", " 1", "1 ", "01", "00", "1x", "on", "off", "true", "1.0", "0x1"})
    assert(ResolveGdnVerify(bad) == -1);
  // The row threshold: off never runs the kernel; N runs it for N or more rows.
  assert(!GdnVerifyUses(0, 1) && !GdnVerifyUses(0, 8) && !GdnVerifyUses(0, 16));
  assert(GdnVerifyUses(1, 1) && GdnVerifyUses(1, 8));
  assert(!GdnVerifyUses(4, 1) && !GdnVerifyUses(4, 3) && GdnVerifyUses(4, 4) && GdnVerifyUses(4, 16));
  assert(!GdnVerifyUses(8, 7) && GdnVerifyUses(8, 8) && GdnVerifyUses(8, 9));
  std::cout << "gdn_verify_switch_test: ok\n";
  return 0;
}
