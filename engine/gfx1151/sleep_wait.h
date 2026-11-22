// The host sleeps while it waits for the device (prefill-resid host lever H1).
// In the default HIP mode (hipDeviceScheduleAuto: active wait on this host) a hipDeviceSynchronize or a blocking
// device-to-host hipMemcpy spins the calling thread at 100% of a core for the whole wait, and a 16384-row prefill
// launch waits about 8 s. The blocking mode is no fix: it is safe only before the first HIP call, and then
// the ROCr async event thread spins in place of the main thread (the test machine, Nov 18). The engine tests an event with a sleep
// between the tests and then makes its old sync call, which returns at once. No HIP flag changes.
// Host code can include this file (tests/sleep_wait_test.cpp).
#pragma once

namespace crucible {
// ready(): true when the work is done (it throws on an error). pause(): one sleep. Returns the number of sleeps.
template <class Ready, class Pause>
long SleepUntil(Ready ready, Pause pause) {
  long sleeps = 0;
  while (!ready()) {
    pause();
    ++sleeps;
  }
  return sleeps;
}
}  // namespace crucible
