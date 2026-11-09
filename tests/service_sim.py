"""A virtual machine for the tests of tools/service_faulttest.py: a clock, one user unit and one engine.
It models what the design (s1, s2, s5) says systemd and the server do, so that the driver's pass rules can be
checked with good and with broken behavior. No process, no socket, no real time."""
import signal
from dataclasses import dataclass, field


@dataclass
class Behavior:
    load: float = 4.0            # seconds from start to READY
    restart: float = 5.0         # RestartSec
    flush: float = 6.0           # engine flush after the server is gone or stopped
    found: float = 70.0          # seconds from SIGSTOP to the watchdog kill
    gap: float = 3.2             # largest busy engine gap that the journal reports
    gap_line: bool = True        # the server logs the gap line
    stop_sigkill: bool = False   # a stop ends with SIGKILL (flaw)
    lose_states: bool = False    # the flush writes nothing (flaw)
    idle_restart: bool = False   # a restart happens while idle (flaw)
    heal_extra: float = 0.0      # extra seconds before a kill -9 of the engine heals (flaw)
    bad_setting_hangs: bool = False  # a bad setting does not end in "failed" (flaw)
    userns: bool = True
    leftover: bool = False       # a process stays after uninstall (flaw)
    full_disk_crash: bool = False  # the server exits on a full disk (flaw)
    prefill_per_token: float = 0.0004
    mem_back: bool = True
    status_json: bool = True
    core_dump: bool = False
    nofile: int = 65536  # the unit's open files limit (the unit text sets LimitNOFILE=65536)
    locked_same: bool = True  # the unit's locked memory limit equals a login shell's
    old_guard: bool = False  # a guard timer of a dead run is still loaded (the Nov 6 fake run on the test machine)


class Handle:
    def __init__(self, sim, conv, kind):
        self.sim, self.conv, self.kind, self.epoch = sim, conv, kind, sim.epoch

    def first_token(self, timeout):
        return self.sim.state == "active"

    def result(self, timeout):
        if self.sim.epoch != self.epoch:
            return dict(ok=False, status=0, error="connection closed")
        self.sim.finish(self.conv)
        return dict(ok=True, status=200, cached_tokens=0, prompt_tokens=self.conv.tokens)


class Sim:
    def __init__(self, behavior=None, userns=True):
        self.b = behavior or Behavior()
        self.t = 5000.0
        self.state = "inactive"
        self.nrestarts = 0
        self.epoch = 0
        self.result = "success"
        self.events = []
        self.jlog = []
        self.pid = 1000
        self.server_pid = self.engine_pid = None
        self.unit_text = None
        self.serve_text = None
        self.enabled = None
        self.memory = set()
        self.disk = set()
        self.finished = []
        self.calls = []
        self.guard = self.b.old_guard
        self.direct = None
        self.removed = False
        self.gpu = 170 * 2**20
        self.env = {}
        self.status_text = ""
        self.failed_starts = 0
        self.last_stop_began = None
        self.nstarts = 0
        self.start_times = []  # StartLimitBurst=4 in 15 min: every start counts, manual or by restart

    # clock -------------------------------------------------------------------------------------
    def now(self):
        return self.t

    def sleep(self, s):
        end = self.t + s
        while self.events and min(e[0] for e in self.events) <= end:
            self.events.sort(key=lambda e: e[0])
            when, fn = self.events.pop(0)
            self.t = max(self.t, when)
            fn()
        self.t = end

    def at(self, delay, fn):
        self.events.append((self.t + delay, fn))

    def log(self, text):
        self.jlog.append((self.t, text))

    def newpid(self):
        self.pid += 1
        return self.pid

    # unit lifecycle ----------------------------------------------------------------------------
    def go_active(self):
        self.state = "active"
        self.result = "success"
        self.server_pid, self.engine_pid = self.newpid(), self.newpid()
        self.status_text = "ready: 0 of 2 sessions busy; engine answered 1 s ago"
        self.log("Started crucible-ft.service.")

    def die(self, why, heal_after, result="exit-code"):
        """The unit leaves active; the restart makes it active again after heal_after."""
        self.epoch += 1
        self.state = "activating"
        self.result = result
        self.server_pid = self.engine_pid = None
        self.memory.clear()
        self.nrestarts += 1
        self.log(why)
        if self.count_start():
            return
        self.at(heal_after, self.go_active)

    def count_start(self):
        """True when the start limit stops the unit (more than 4 starts in 900 s since the last reset-failed)."""
        self.start_times = [x for x in self.start_times if self.t - x < 900] + [self.t]
        if len(self.start_times) > 4:
            self.state, self.result = "failed", "start-limit-hit"
            self.log("crucible-ft.service: Start request repeated too quickly.")
            return True
        return False

    def flush_to_disk(self):
        if not self.b.lose_states:
            self.disk |= {c for c in self.finished}

    def systemctl(self, *args):
        self.calls.append(args)
        verb = args[0] if args else ""
        if verb == "start":
            self.nstarts += 1
            if self.unit_text is None:
                return 5, "Unit crucible-ft.service not found."
            if "no-such-pack-folder" in self.unit_text:
                self.state = "activating"
                if self.b.bad_setting_hangs:
                    return 0, ""
                self.failed_starts = 0
                for i in range(4):
                    self.at(5 * (i + 1) + 3 * i, self.bad_start)
                return 1, "failed"
            if self.count_start():
                return 1, "start-limit-hit"
            self.nrestarts = 0  # a manual start resets the counter
            self.sleep(self.b.load)
            self.go_active()
            if self.b.idle_restart:
                self.at(100, lambda: self.die("spurious restart", 5 + self.b.load))
            return 0, ""
        if verb == "stop":
            unit = args[-1]
            if unit.startswith("crucible-ft-guard"):
                self.guard = False
                return 0, ""
            if self.state == "active":
                self.last_stop_began = self.t
                self.state = "deactivating"
                self.flush_to_disk()
                self.log("Stopping crucible-ft.service...")
                if self.b.stop_sigkill:
                    self.log("crucible-ft.service: Killing process 1 (python) with signal SIGKILL.")
                self.epoch += 1
                self.sleep(self.b.flush)
            self.state, self.server_pid, self.engine_pid = "inactive", None, None
            self.memory.clear()
            return 0, ""
        if verb == "restart":
            self.systemctl("stop", "crucible-ft.service")
            return self.systemctl("start", "crucible-ft.service")
        if verb == "reset-failed":
            self.start_times = []
            if self.state == "failed":
                self.state = "inactive"
            return 0, ""
        if verb == "enable":
            self.enabled = "enabled-runtime"
            return 0, ""
        if verb == "disable":
            self.enabled = None
            return 0, ""
        if verb == "is-enabled":
            return (0, self.enabled) if self.enabled else (1, "disabled")
        return 0, ""

    def bad_start(self):
        self.failed_starts += 1
        self.nrestarts = max(self.nrestarts, self.failed_starts - 1)
        if self.failed_starts >= 4:
            self.state, self.result = "failed", "start-limit-hit"
            self.log("crucible-ft.service: Start request repeated too quickly.")
        self.log("doctor: FAIL pack-files: the pack folder no-such-pack-folder is missing")

    def run(self, argv, timeout=60):
        self.calls.append(("run",) + tuple(argv))
        if "systemd-run" in argv[0]:
            if self.guard:
                return 1, "Failed to start transient timer unit: Unit crucible-ft-guard.timer was already loaded or has a fragment file."
            self.guard = True
        return 0, ""

    def show(self):
        return dict(ActiveState=self.state, SubState=self.state, NRestarts=self.nrestarts, Result=self.result,
                    MainPID=self.server_pid or 0, StatusText=self.status_text, Type="notify")

    def pids(self):
        return self.server_pid, self.engine_pid

    # files -------------------------------------------------------------------------------------
    def write_unit(self, text):
        self.unit_text, self.removed = text, False

    def write_serve(self, text):
        self.serve_text = text

    def remove_files(self):
        self.unit_text = self.serve_text = None
        self.removed = True

    # faults ------------------------------------------------------------------------------------
    def kill(self, pid, sig):
        self.calls.append(("kill", pid, sig))
        if self.state != "active":
            return
        if pid == self.engine_pid and sig == signal.SIGKILL:
            self.die("engine stopped: killed", self.b.restart + self.b.load + 1 + self.b.heal_extra)
        elif pid == self.server_pid and sig == signal.SIGKILL:
            self.flush_to_disk()
            self.die("main process killed", self.b.flush + self.b.restart + self.b.load)
        elif pid == self.engine_pid and sig == signal.SIGSTOP:
            self.log("engine silent")
            self.at(self.b.found, lambda: self.die("Watchdog timeout (limit 30s)!", self.b.restart + self.b.load, "watchdog"))

    # requests ----------------------------------------------------------------------------------
    def health(self):
        return 200 if self.state == "active" else 0

    def tokenize_len(self, text):
        return len(text.split())

    def finish(self, conv):
        self.finished.append(conv.key)
        self.memory.add(conv.key)

    def chat(self, conv, max_tokens=8, timeout=60):
        if self.state != "active":
            return dict(ok=False, status=0, error="connection refused")
        p = conv.prefix
        have = lambda c: c.key in self.memory or c.key in self.disk
        cached = p.tokens if p is not None and have(p) else conv.tokens if have(conv) else 0
        took = conv.tokens * self.b.prefill_per_token
        if conv.tokens >= 60000:
            self.log("engine busy")
            if self.b.gap_line:
                self.log(f"watchdog: largest busy engine gap {self.b.gap:.1f} s")
        self.sleep(took)
        self.finish(conv)
        return dict(ok=True, status=200, cached_tokens=cached, prompt_tokens=conv.tokens, total_s=took)

    def start_chat(self, conv, max_tokens=8, timeout=60):
        return Handle(self, conv, "chat")

    def journal(self, since):
        return "\n".join(text for when, text in self.jlog if when >= since)

    def limits(self, pid):
        return dict(locked=8388608 if self.b.locked_same else 64, nofile=self.b.nofile)

    def login_limits(self):
        return dict(locked=8388608, nofile=524288)

    def coredumps(self):
        return self.nrestarts if self.b.core_dump else 0

    def gpu_mem(self):
        leaked = self.nstarts > 0 and not self.b.mem_back
        return self.gpu + (100 * 2**30 if leaked or self.state == "active" else 0)

    def port_free(self):
        return self.state in ("inactive", "failed")

    def leftovers(self):
        return [4242] if self.b.leftover else []

    def commands_logged(self):
        return None

    def prepare(self):
        pass

    def set_fake_env(self, env):
        self.env = dict(env)

    # N9 ----------------------------------------------------------------------------------------
    def userns_ok(self):
        return self.b.userns

    def direct_start(self):
        self.direct = dict(alive=True, requests=0)
        self.log("direct server up")
        return True

    def direct_alive(self):
        if self.direct and self.b.full_disk_crash and self.direct["requests"] >= 5:
            self.direct["alive"] = False
        return bool(self.direct and self.direct["alive"])

    def direct_chat(self, conv, max_tokens=8, timeout=60):
        if not self.direct_alive():
            return dict(ok=False, status=0, error="connection refused")
        self.direct["requests"] += 1
        return dict(ok=True, status=200, cached_tokens=0, prompt_tokens=conv.tokens)

    def direct_log(self):
        return "state disk full: write failed (full)\n" if self.direct else ""

    def direct_stop(self):
        if self.direct:
            self.direct["alive"] = False

    def service_status(self):
        if not self.b.status_json:
            return None
        return dict(state=self.state, reason="Stopped after 4 failed starts. Last start check: the pack folder no-such-pack-folder is missing.")
