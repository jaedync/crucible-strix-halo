// The disk tier's stop rule (engine/serve/batched_serve.h BatchedServe): a stopping engine flushes its states for this
// long at most. SIGTERM (a service stop that signals the whole group) and SIGINT (Ctrl-C, which the terminal sends to the
// whole foreground group) do not end the engine; the server then closes stdin, the engine flushes and exits.
// Without a server, the input can stay open (a pipe or a terminal that nobody closes): kStateInputWait seconds after the
// first such signal, a read that would wait for input ends the input (serve_line.h g_serve_input_stop), so the engine
// flushes and exits 0. A server closes stdin before that wait ends (server/sd_notify.py joins its watchdog for 5 s at
// most), so its stop does not change. The engine exits kStateTermGrace seconds after the first signal in any case.
// The stop budget: the flush starts only after the command in flight ends, and an open can hold a handover drain
// of 8 s (handover_spill.h kHandoverDrainSeconds) plus a prefill chunk. The server waits 85 s after it closes stdin
// (server/batch_backend.py STATE_FLUSH_SECONDS), then kills the engine; a 75 s flush left about 10 s for that command and
// the ledger write. 60 s leaves 25 s. All of it fits inside Kiln's TimeoutStopSec=120 (server/tests/test_state_disk.py).
// Host code only, so a CPU test includes it (tests/state_stop_test.cpp).
#pragma once
#include "serve_line.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <thread>

inline constexpr double kStateFlushSeconds = 60, kStateTermGrace = 90;
inline constexpr double kStateInputWait = 8;
inline volatile std::sig_atomic_t g_state_term = 0;
// CLOCK_MONOTONIC nanoseconds of the first stop signal (0: none). The handler writes it: clock_gettime is
// async-signal-safe and the atomic is lock-free.
inline std::atomic<int64_t> g_state_term_ns{0};
static_assert(std::atomic<int64_t>::is_always_lock_free);
inline int64_t StateStopClockNs() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return int64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
}
// True kStateInputWait seconds after the first stop signal: an input that is still open and has no command ends.
inline bool StateStopInputEnded() {
  const int64_t at = g_state_term_ns.load();
  return at != 0 && double(StateStopClockNs() - at) >= kStateInputWait * 1e9;
}
inline void KeepRunningOnTerm() {
  struct sigaction action{};
  action.sa_handler = [](int) {
    int64_t none = 0;
    g_state_term_ns.compare_exchange_strong(none, StateStopClockNs());
    g_state_term = 1;
  };
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESTART;
  sigaction(SIGTERM, &action, nullptr);
  sigaction(SIGINT, &action, nullptr);
  crucible::g_serve_input_stop = StateStopInputEnded;
  std::thread([] {
    while (!g_state_term) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::this_thread::sleep_for(std::chrono::duration<double>(kStateTermGrace));
    std::fprintf(stderr, "state_disk stop: no end of input %.0f s after a stop signal (SIGTERM or SIGINT)\n", kStateTermGrace);
    std::_Exit(143);
  }).detach();
}
