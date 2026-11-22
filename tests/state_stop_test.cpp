// The disk tier's stop rule (engine/serve/state_stop.h). A tier engine outlives a stop signal to its process group, so
// the server's stop can close its stdin and the engine flushes its states. Ctrl-C sends SIGINT to the whole
// foreground group; the engine died on it (default action) before its flush. Each case runs in a child process.
#include "../engine/serve/state_stop.h"
#include <cassert>
#include <cstdio>
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
}  // namespace

int main() {
  assert(Survives(SIGTERM) == 0);
  // Ctrl-C to the group is a stop like SIGTERM (it gave -SIGINT: the engine died before its flush).
  const int interrupted = Survives(SIGINT);
  if (interrupted != 0) std::fprintf(stderr, "state_stop_test: SIGINT gave %d\n", interrupted);
  assert(interrupted == 0);
  // The flush and the command in flight before it fit inside the server's wait (server/batch_backend.py
  // STATE_FLUSH_SECONDS = 85; server/tests/test_state_disk.py checks the sum with the handover drain).
  static_assert(kStateFlushSeconds <= 60);
  static_assert(kStateTermGrace > kStateFlushSeconds);
  std::puts("state_stop_test ok");
}
