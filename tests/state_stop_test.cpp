// The disk tier's stop rule (engine/serve/state_stop.h). A tier engine outlives a stop signal to its process group, so
// the server's stop can close its stdin and the engine flushes its states. Ctrl-C sends SIGINT to the whole
// foreground group; the engine died on it (default action) before its flush. Each case runs in a child process.
#include "../engine/serve/state_stop.h"
#include "../engine/serve/serve_line.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
// The child installs the rule, gets SIGNUM, and exits 0 when it is still alive with the stop noted.
int Survives(int signum) {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    KeepRunningOnTerm();
    raise(signum);
    for (int i = 0; i < 100 && !g_state_term; ++i) usleep(10000);
    _exit(g_state_term ? 0 : 3);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  if (WIFSIGNALED(status)) return -WTERMSIG(status);
  return WEXITSTATUS(status);
}

// A run without a server (review stop-fit L1): the engine reads its commands from an input that nobody closes, as
// batched_serve.h does (KeepRunningOnTerm, then ServeLineReader until the end of input, then the flush). Ctrl-C (or
// SIGTERM) must end the read, so the flush runs and the engine exits 0. Before the fix the read never ended: the
// engine was alive after 3 signals and exited 143 at kStateTermGrace (90 s) with no flush.
struct OpenInputRun {
  pid_t child = -1;
  int input = -1, result = -1;
  std::chrono::steady_clock::time_point signalled;
};
OpenInputRun StartOpenInput(int signum) {
  int in[2], out[2];
  assert(pipe(in) == 0 && pipe(out) == 0);
  OpenInputRun run;
  run.child = fork();
  assert(run.child >= 0);
  if (run.child == 0) {
    close(in[1]);
    close(out[0]);
    KeepRunningOnTerm();
    crucible::ServeLineReader reader(in[0], 1 << 20);
    assert(write(out[1], "r", 1) == 1);  // the rule is on and the read starts
    std::string line;
    bool oversized = false;
    int lines = 0;
    while (reader.Next(line, oversized)) lines += line == "ping";
    // The flush of batched_serve.h runs here; the parent reads the number of commands that the engine read.
    const std::string done = "flushed " + std::to_string(lines) + "\n";
    assert(write(out[1], done.data(), done.size()) == ssize_t(done.size()));
    _exit(0);
  }
  close(in[0]);
  close(out[1]);
  run.input = in[1];
  run.result = out[0];
  char ready = 0;
  assert(read(run.result, &ready, 1) == 1 && ready == 'r');
  // The review's probe: 3 signals, as from 3 presses of Ctrl-C. Only the first one counts.
  for (int i = 0; i < 3; ++i) {
    assert(kill(run.child, signum) == 0);
    if (i == 0) run.signalled = std::chrono::steady_clock::now();
    usleep(50000);
  }
  // A command after the signal is still read: a server keeps the engine busy until it closes the input.
  assert(write(run.input, "ping\n", 5) == 5);
  return run;
}
// The seconds from the first signal to the exit, or -1 when the child was still alive after `within` seconds.
double FinishOpenInput(OpenInputRun& run, double within, int& status, std::string& said) {
  double seconds = -1;
  for (;;) {
    const pid_t got = waitpid(run.child, &status, WNOHANG);
    assert(got >= 0);
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - run.signalled).count();
    if (got == run.child) {
      seconds = now;
      break;
    }
    if (now > within) {
      kill(run.child, SIGKILL);
      assert(waitpid(run.child, &status, 0) == run.child);
      break;
    }
    usleep(20000);
  }
  char text[64] = {};
  const ssize_t n = read(run.result, text, sizeof text - 1);
  said.assign(text, n > 0 ? size_t(n) : 0);
  close(run.input);
  close(run.result);
  return seconds;
}
}  // namespace

int main() {
  assert(Survives(SIGTERM) == 0);
  // Ctrl-C to the group is a stop like SIGTERM (it gave -SIGINT: the engine died before its flush).
  const int interrupted = Survives(SIGINT);
  if (interrupted != 0) std::fprintf(stderr, "state_stop_test: SIGINT gave %d\n", interrupted);
  assert(interrupted == 0);
  // The two runs with an open input go at the same time, so the test waits for the stop time once.
  constexpr double kStopWithin = 20;  // far below the 90 s grace, above the wait for a server's close
  static_assert(kStateInputWait + 2 < kStopWithin);  OpenInputRun runs[] = {StartOpenInput(SIGINT), StartOpenInput(SIGTERM)};
  int status[2] = {};
  std::string said[2];
  double seconds[2] = {};
  for (int i = 0; i < 2; ++i) {
    seconds[i] = FinishOpenInput(runs[i], kStopWithin, status[i], said[i]);
    std::fprintf(stderr, "state_stop_test: open input, 3 x %s: seconds=%.1f exit=%d said=%s", i ? "SIGTERM" : "SIGINT",
                 seconds[i], WIFEXITED(status[i]) ? WEXITSTATUS(status[i]) : -WTERMSIG(status[i]),
                 said[i].empty() ? "-\n" : said[i].c_str());
  }
  for (int i = 0; i < 2; ++i) {
    assert(seconds[i] >= 0);
    assert(WIFEXITED(status[i]) && WEXITSTATUS(status[i]) == 0);
    assert(said[i] == "flushed 1\n");
  }
  // The flush and the command in flight before it fit inside the server's wait (server/batch_backend.py
  // STATE_FLUSH_SECONDS = 85; server/tests/test_state_disk.py checks the sum with the handover drain).
  static_assert(kStateFlushSeconds <= 60);
  static_assert(kStateTermGrace > kStateFlushSeconds);
  // A server closes stdin after it joins its watchdog (5 s at most, server/sd_notify.py), so its stop ends the input
  // first; server/tests/test_state_disk.py checks the budget of a run without a server.
  static_assert(kStateInputWait > 5 && kStateInputWait < kStateTermGrace - kStateFlushSeconds);
  // Each run printed its stop time; it must be the input wait (plus the poll step), not the grace.
  for (const double s : seconds) assert(s >= kStateInputWait && s < kStateInputWait + 2);
  std::puts("state_stop_test ok");
}
