// Release of a slot tail that the host cache refused (cause 1 of the cache thrash,
// design A): the lazy plan serves every release, and a handover (the next request overwrites the slot) waits a bounded
// time for the disk writer to read the device rows. CPU only.
#include "../engine/serve/handover_spill.h"
#include <cassert>
#include <chrono>
#include <cstdio>

using namespace crucible;
constexpr size_t kMin = size_t(64) << 20, kBig = size_t(4) << 30;

// The old rule took the lazy plan for a resumed request only. A handover (resumed = false) copied the whole 3 to 5 GB
// state to the host first and was refused "memory" when it did not fit the 2 GiB of room (41 times on Nov 2).
void LazyForEveryRelease() {
  const auto handover = PlanReleaseSpill(false, true, kBig, kMin);
  assert(handover.lazy);
  const auto resumed = PlanReleaseSpill(true, true, kBig, kMin);
  assert(resumed.lazy);
  // No disk tier, or a tail under the minimum: the eager save, as before.
  assert(!PlanReleaseSpill(false, false, kBig, kMin).lazy);
  assert(!PlanReleaseSpill(false, true, kMin - 1, kMin).lazy);
  assert(PlanReleaseSpill(false, true, kMin, kMin).lazy);
}

// A resumed request keeps the rows while it runs: no wait. A handover overwrites the rows at once: wait up to 8 s.
void OnlyAHandoverWaits() {
  assert(PlanReleaseSpill(true, true, kBig, kMin).drain_seconds == 0);
  assert(PlanReleaseSpill(false, true, kBig, kMin).drain_seconds == kHandoverDrainSeconds);
  assert(kHandoverDrainSeconds == 8);
  assert(PlanReleaseSpill(false, false, kBig, kMin).drain_seconds == 0);
  assert(PlanReleaseSpill(false, true, 1, kMin).drain_seconds == 0);
}

void DrainRules() {
  double asked = -1;
  int calls = 0;
  const auto drain = [&](double seconds) { asked = seconds; ++calls; return true; };
  const auto handover = PlanReleaseSpill(false, true, kBig, kMin);
  assert(DrainHandover(handover, true, drain) && calls == 1 && asked == kHandoverDrainSeconds);
  // A spill that was not queued has nothing to wait for.
  calls = 0;
  assert(DrainHandover(handover, false, drain) && calls == 0);
  // A resumed release does not wait.
  assert(DrainHandover(PlanReleaseSpill(true, true, kBig, kMin), true, drain) && calls == 0);
  // The writer did not finish inside the bound: the result says so (the slot is overwritten and the write is lost).
  assert(!DrainHandover(handover, true, [](double) { return false; }));
}

int main() {
  LazyForEveryRelease();
  OnlyAHandoverWaits();
  DrainRules();
  std::puts("handover_spill_test ok");
}
