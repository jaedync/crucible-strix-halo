// How Release saves a slot tail that the host cache refused (cause 1 of the cache thrash,
// design A).
//
// The tail goes straight to the SSD (Direct). The eager save copies the whole state (3 to 5 GB at 128K to 190K tokens)
// to host memory first, and the direct spill is refused "memory" when that does not fit the room above the reserve (about
// 2 GiB on the daily layout): the state is lost. The lazy plan leaves the full blocks on the device and the
// disk writer reads them in 8 MiB pieces, so the host holds about 0.3 GiB only.
//
// A resumed request keeps the slot rows while it runs, so the writer can read them later. A handover is different: the
// next request overwrites the slot right after Release (LazyEnd ends the lazy sources, and a write that walks an ended
// source is refused "lost"). So a handover waits, for at most kHandoverDrainSeconds, until the writer has read the
// device rows. When the bound passes, the overwrite goes on and the state is lost as before. Nothing is corrupted: the
// end of a source waits for the one read in progress.
#pragma once
#include <cstddef>

namespace crucible {
// Writing 4.5 GB takes about 0.7 s at 6.9 GB/s. The bound leaves room for jobs that are in the queue already.
inline constexpr double kHandoverDrainSeconds = 8;

struct ReleaseSpill {
  bool lazy = false;          // use the lazy plan (the eager save copies the whole state to the host)
  double drain_seconds = 0;   // wait this long for the writer after the spill is queued (0: no wait)
};

// `resumed`: the next request resumes from the slot's own rows. `disk`: the disk tier is on. `lazy_bytes`: the bytes
// that the lazy plan leaves on the device. `min_lazy_bytes`: below this the eager save is cheap.
inline ReleaseSpill PlanReleaseSpill(bool resumed, bool disk, size_t lazy_bytes, size_t min_lazy_bytes) {
  if (!disk || lazy_bytes < min_lazy_bytes) return {};
  return {true, resumed ? 0 : kHandoverDrainSeconds};
}

// After the spill is queued: waits for the writer when the plan says so. `drain(seconds)` returns true when the writer
// queue is empty. False: the bound passed, and the caller overwrites the slot (the write is lost).
template<class Drain>
bool DrainHandover(const ReleaseSpill& plan, bool queued, Drain&& drain) {
  if (!queued || plan.drain_seconds <= 0) return true;
  return drain(plan.drain_seconds);
}
}  // namespace crucible
