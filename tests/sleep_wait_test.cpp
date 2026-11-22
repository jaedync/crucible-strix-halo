// The host sleeps while it waits for the device (engine/gfx1151/sleep_wait.h): SleepUntil tests the work, sleeps once between two tests, and stops at the first test that is true.
#include "../engine/gfx1151/sleep_wait.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>

using namespace crucible;

int main() {
  // Ready at the first test: no sleep.
  int tests = 0, sleeps = 0;
  assert(SleepUntil([&] { ++tests; return true; }, [&] { ++sleeps; }) == 0);
  assert(tests == 1 && sleeps == 0);
  // Ready at the fourth test: three sleeps, one between each two tests.
  tests = sleeps = 0;
  assert(SleepUntil([&] { return ++tests == 4; }, [&] { ++sleeps; }) == 3);
  assert(tests == 4 && sleeps == 3);
  // An error in the test stops the wait and goes to the caller.
  tests = sleeps = 0;
  bool threw = false;
  try {
    SleepUntil([&]() -> bool { if (++tests == 2) throw std::runtime_error("hip error"); return false; }, [&] { ++sleeps; });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw && tests == 2 && sleeps == 1);
  std::printf("sleep_wait_test: PASS\n");
  return 0;
}
