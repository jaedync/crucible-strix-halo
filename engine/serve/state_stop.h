// The disk tier's stop rule (engine/serve/batched_serve.h BatchedServe): a stopping engine flushes its states for this
// long at most. SIGTERM (a service stop that signals the whole group) and SIGINT (Ctrl-C, which the terminal sends to the
// whole foreground group, A6 F2) do not end the engine; the server then closes stdin, the engine flushes and exits.
// Without a server, the engine exits kStateTermGrace seconds after the first such signal.
// The stop budget (A6 F3): the flush starts only after the command in flight ends, and an open can hold a handover drain
// of 8 s (handover_spill.h kHandoverDrainSeconds) plus a prefill chunk. The server waits 85 s after it closes stdin
// (server/batch_backend.py STATE_FLUSH_SECONDS), then kills the engine; a 75 s flush left about 10 s for that command and
// the ledger write. 60 s leaves 25 s. All of it fits inside Kiln's TimeoutStopSec=120 (server/tests/test_state_disk.py).
// Host code only, so a CPU test includes it (tests/state_stop_test.cpp).
#pragma once
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>

inline constexpr double kStateFlushSeconds = 60, kStateTermGrace = 90;
inline volatile std::sig_atomic_t g_state_term = 0;
inline void KeepRunningOnTerm() {
  struct sigaction action{};
  action.sa_handler = [](int) { g_state_term = 1; };
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESTART;
  sigaction(SIGTERM, &action, nullptr);
  sigaction(SIGINT, &action, nullptr);
  std::thread([] {
    while (!g_state_term) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::this_thread::sleep_for(std::chrono::duration<double>(kStateTermGrace));
    std::fprintf(stderr, "state_disk stop: no end of input %.0f s after a stop signal (SIGTERM or SIGINT)\n", kStateTermGrace);
    std::_Exit(143);
  }).detach();
}
