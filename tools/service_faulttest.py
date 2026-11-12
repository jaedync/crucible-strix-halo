#!/usr/bin/env python3
"""Night fault driver for the Crucible service (design s5 C, steps N1 to N11).

It installs its own runtime unit (crucible-ft.service, port 8791, state folder ~/.cache/crucible-ft/states),
breaks it on purpose and checks that it heals. A person does nothing during the run. It writes a JSON report.

  python3 tools/service_faulttest.py --real          the GPU night run (hold a gpu-run lease around it)
  python3 tools/service_faulttest.py --fake          the CPU dry run: the same steps, the fake engine, scaled times
  python3 tools/service_faulttest.py --real --plan   print the steps and change nothing
  python3 tools/service_faulttest.py --snapshot F    save units and timers (before a reboot test)
  python3 tools/service_faulttest.py --compare F     compare units and timers now with the snapshot F

Steps and pass rules (L = the load time that N1 measured):
  N1  install and first start     READY; /health 200; locked memory equals a login shell (or is unlimited); open files >= 65536
  N2  long prefill                64K prompt, then 4 turns; largest busy engine gap < 25 s
  N3  kill -9 the engine          healthy again within L + 15 s; one restart
  N4  kill -9 the server          healthy within flush + L + 15 s; a resent conversation restores from the disk
  N5  SIGSTOP the engine, decoding   found within 90 s; no new core file; healthy again
  N6  SIGSTOP the engine, idle    the same rule
  N7  idle 10 min                 the restart count does not change
  N8  stop during a 64K prefill   stop within 120 s, no SIGKILL; every finished conversation restores; the cut request ends
  N9  full state disk             20 turns of 20K tokens, all 200, the server stays up, "full" in its log (skip without user namespaces)
  N10 bad setting                 the unit is failed within 60 s; at most 4 restarts; the reason names the check
  N11 uninstall                   no process, the port free, GPU memory back to the start value

Guards: a runtime unit (gone after a reboot), a systemd-run timer that stops the unit after 80 min, and a
`finally` that uninstalls. The standard library only. The pass numbers are the design's; --fake scales the times.
"""
import argparse
import http.client
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEP_IDS = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11")
STEP_NAMES = {
    "N1": "install and first start", "N2": "long prefill and the largest engine gap", "N3": "kill -9 the engine",
    "N4": "kill -9 the server", "N5": "SIGSTOP the engine while it decodes", "N6": "SIGSTOP the engine while idle",
    "N7": "idle", "N8": "stop during a long prefill", "N9": "full state disk", "N10": "bad setting", "N11": "uninstall",
}
STEP_MINUTES = {"N1": 5, "N2": 5, "N3": 3, "N4": 4, "N5": 5, "N6": 3, "N7": 10, "N8": 6, "N9": 6, "N10": 1, "N11": 1}
PATH_OK = re.compile(r"^[A-Za-z0-9._/@+ -]+$")
GAP_LINE = re.compile(r"largest busy engine gap ([0-9]+(?:\.[0-9]+)?) s")
TURN_LINE = re.compile(r"crucible_turn (\{.*\})")
KILLED = re.compile(r"Killing process|with signal SIGKILL|State 'stop-\w+' timed out")
NOFILE_MIN = 65536  # LimitNOFILE of the unit text; N1 needs at least this many open files
LOGIN_LIMITS = "ulimit -l; ulimit -n"


@dataclass
class Config:
    repo: Path
    python: str
    ckpt: Path
    engine: Path
    state_dir: Path
    work: Path
    report: Path
    unit: str = "crucible-ft.service"
    guard: str = "crucible-ft-guard"
    port: int = 8791
    bind: str = "127.0.0.1"
    pack: str = "4.05"
    models_dir: Path = None
    streams: int = 4
    ctx: int = 262144
    fake: bool = False
    only: list = None
    max_minutes: float = 75
    guard_minutes: int = 80
    watchdog_sec: int = 30
    stall_sec: int = 50
    found_limit: float = 90
    flush_max: float = 85
    stop_limit: float = 120
    heal_margin: float = 15
    bad_limit: float = 60
    gap_limit: float = 25
    long_tokens: int = 65536
    disk_tokens: int = 20000  # the engine writes a state to the disk tier only from 16384 tokens (server/batch_backend.py STAGE_MIN_TOKENS)
    idle_seconds: int = 600
    n8_wait: float = 10
    n9_turns: int = 20
    n9_tmpfs: str = "2g"
    n9_limit: int = 4500  # the life limit of the N9 server in seconds (timeout): it ends even when the driver is killed
    n9_kill_after: int = 130  # seconds between the stop signal and the kill; the flush of a tier engine needs time
    decode_tokens: int = 4000
    start_timeout: int = 1800
    default_load: float = 300
    restore_fraction: float = 0.5
    default_pings: bool = False
    limits_strict: bool = True  # False in the dry run: the driver's own shell sets the login limits there, not a login
    fake_env: dict = field(default_factory=dict)
    git_sha: str = ""

    def __post_init__(self):
        for name in ("repo", "ckpt", "engine", "state_dir", "work", "report"):
            setattr(self, name, Path(getattr(self, name)))
        if self.models_dir is None:
            self.models_dir = self.ckpt.parent

    @property
    def serve(self):
        return self.work / "ft-serve.sh"

    @property
    def check(self):
        return self.work / "ft-check.sh"

    @property
    def unit_path(self):
        return Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "systemd/user" / self.unit

    @property
    def check_file(self):
        """The start check result of the fault unit. The default file belongs to the real service; the test must not write it."""
        return self.work / "last-start-check.json"

    @property
    def envfile(self):
        return self.work / "ft.env"


def scaled(cfg):
    """The dry-run values: the same steps with short times (watchdog 3 s, stall 20 s: twice the idle ping interval of 10 s, so a quiet engine is not a stall)."""
    cfg.fake, cfg.limits_strict = True, False
    cfg.watchdog_sec, cfg.stall_sec = 3, 20
    cfg.found_limit = cfg.stall_sec + cfg.watchdog_sec + 12
    cfg.idle_seconds, cfg.n8_wait, cfg.decode_tokens = 30, 2, 600
    cfg.n9_tmpfs, cfg.start_timeout, cfg.default_load = "64m", 60, 20
    cfg.n9_limit, cfg.n9_kill_after = 600, 30
    cfg.max_minutes = 12
    cfg.fake_env = {
        "FAKE_STATE_DISK": "1", "FAKE_DISK_FILE": str(cfg.state_dir / "fake-disk.json"), "FAKE_PING": "1",
        "FAKE_DELAY": "0.4", "FAKE_COMMAND_SECONDS": "0.4", "FAKE_PREFILL_ROWS": "4096",
        "FAKE_SLOT_CONTEXTS": "131072,131072", "FAKE_COMMAND_LOG": str(cfg.work / "commands.log"),
        "FAKE_FLUSH_DELAY": "1", "FAKE_IGNORE_TERM": "1",  # a tier engine flushes after SIGTERM
        # FAKE_STAGE_FULL is set for step N9 only (direct_start)
    }
    return cfg


# ---- the unit text -------------------------------------------------------------------------------------------------

def _path(value):
    text = os.fspath(value)
    if not PATH_OK.match(text):
        raise ValueError(f"The path {text!r} has a character outside [A-Za-z0-9._/@+ -]. Use another folder.")
    return text


def render_unit(cfg, missing_pack=False):
    """The unit of design s1 for the fault test. missing_pack points the start check at a folder that is not there (N10)."""
    py, repo, serve = _path(cfg.python), _path(cfg.repo), _path(cfg.serve)
    state, models = _path(cfg.state_dir), _path(cfg.models_dir if not missing_pack else cfg.work / "no-such-pack-folder")
    if cfg.fake:
        pre = f'"{_path(cfg.check)}" "{_path(cfg.ckpt) if not missing_pack else models}" "{_path(cfg.check_file)}"'
    else:
        pre = (f'"{py}" -I "{repo}/crucible" doctor --for start --quiet --pack {cfg.pack} --models-dir "{models}" '
               f'--state-dir "{state}" --port {cfg.port} --check-file "{_path(cfg.check_file)}"')
    return f"""# Written by tools/service_faulttest.py for the fault test. It is a runtime unit and is removed at the end of the run.
[Unit]
Description=Crucible fault test server on {cfg.bind}:{cfg.port}
StartLimitIntervalSec=15min
StartLimitBurst=4

[Service]
Type=notify
NotifyAccess=main
WorkingDirectory={repo}
Environment=PYTHONUNBUFFERED=1
ExecStartPre={pre}
ExecStart="{serve}" --systemd --watchdog-stall-seconds {cfg.stall_sec}
TimeoutStartSec={cfg.start_timeout}s
TimeoutStopSec=120s
KillMode=control-group
WatchdogSec={cfg.watchdog_sec}s
WatchdogSignal=SIGKILL
Restart=on-failure
RestartSec=5s
RestartPreventExitStatus=2 78
LimitCORE=0
LimitNOFILE={NOFILE_MIN}
UMask=0077
NoNewPrivileges=yes
SyslogIdentifier=crucible-ft

[Install]
WantedBy=default.target
"""


def check_script(cfg):
    """Fake mode only: stands in for `doctor --for start` (the dry run has no GPU). Exit 78 when the pack folder is missing.
    It writes the check file in the format of doctor: {time, ok, code, failed: [{id, detail, fix}]}."""
    return ('#!/bin/sh\n# Written by tools/service_faulttest.py (dry run).\n'
            'if [ ! -f "$1/tokenizer.json" ]; then\n'
            '  echo "FAIL  pack-files: the pack folder $1 is missing" >&2\n'
            '  printf \'{"time": 0, "ok": false, "code": 78, "failed": [{"id": "pack-files", "detail": "the pack folder %s is missing", '
            '"fix": "none"}]}\\n\' "$1" > "$2"\n  exit 78\nfi\n'
            'printf \'{"time": 0, "ok": true, "code": 0, "failed": []}\\n\' > "$2"\nexit 0\n')


def serve_script(cfg):
    """The server command on the fault-test port, with the fault-test state folder. Extra arguments come last and win."""
    from crucible_cli import launch, packs
    from crucible_cli.settings import Settings, load_saved
    saved = load_saved(cfg.repo)
    s = Settings(repo=cfg.repo, pack=cfg.pack, models_dir=cfg.models_dir, port=cfg.port, bind=cfg.bind,
                 streams=cfg.streams, ctx=cfg.ctx, state_dir=cfg.state_dir,
                 state_disk_gib=int(saved.get("state_disk_gib", 200)))
    env, argv = launch.server_command(s, cfg.python)
    argv[argv.index("--engine") + 1] = os.fspath(cfg.engine)
    argv[3] = os.fspath(cfg.ckpt)
    env = {k: v for k, v in env.items() if v is not None}
    q = shlex.quote
    words = " ".join(f"{k}={q(v)}" for k, v in env.items())
    cmd = " ".join(q(str(a)) for a in argv)
    fake = ""
    if cfg.fake:
        fake = f"[ -f {q(os.fspath(cfg.envfile))} ] && {{ set -a; . {q(os.fspath(cfg.envfile))}; set +a; }}\n"
    return (f"#!/bin/sh\n# Written by tools/service_faulttest.py (fault test only). Removed at the end of the run.\n"
            f"cd {q(os.fspath(cfg.repo))} || exit 1\n{fake}exec env {words} {cmd} \"$@\"\n")


# ---- conversations ---------------------------------------------------------------------------------------------------

@dataclass
class Conv:
    """A completion prompt. An extension adds words at the end, so its token IDs start with the IDs of the prefix."""
    key: str
    text: str
    tokens: int
    prefix: "Conv" = None

    def extended(self, tokens, tokenize_len=None):
        extra = _words(f"{self.key}x", tokens)
        text = self.text + " " + extra
        return Conv(f"{self.key}+{tokens}", text, self.tokens + tokens, self)


def _words(seed, n):
    return " ".join(f"{seed}{i % 211}" if i % 5 == 0 else f"w{(i * 7 + len(seed)) % 997}" for i in range(n))


def make_conv(m, key, tokens):
    """A prompt of about TOKENS tokens. The machine counts tokens (/tokenize), so the size is right for any tokenizer."""
    words = tokens
    for _ in range(3):
        text = _words(key, words)
        got = m.tokenize_len(text)
        if abs(got - tokens) <= max(8, tokens // 200):
            return Conv(key, text, got)
        words = max(1, int(words * tokens / max(got, 1)))
    return Conv(key, text, got)


# ---- the report ------------------------------------------------------------------------------------------------------

class StepRec:
    def __init__(self, sid):
        self.id, self.name = sid, STEP_NAMES[sid]
        self.facts, self.notes, self.failed, self.skipped, self.incomplete, self.seconds = {}, [], False, False, False, 0.0

    def check(self, ok, note):
        if not ok:
            self.failed = True
            self.notes.append(note)
        return ok

    def note(self, text):
        self.notes.append(text)

    def skip(self, why, incomplete=False):
        self.skipped, self.incomplete = True, incomplete
        self.notes.append(why)

    @property
    def status(self):
        return "skip" if self.skipped else "fail" if self.failed else "pass"

    def to_json(self):
        out = dict(id=self.id, name=self.name, status=self.status, seconds=round(self.seconds, 1), facts=self.facts,
                   notes="; ".join(self.notes))
        if self.incomplete:
            out["incomplete"] = True
        return out


class Abort(Exception):
    """A step cannot go on (for example, the unit does not come up)."""


# ---- the driver ------------------------------------------------------------------------------------------------------

class Driver:
    def __init__(self, cfg, machine, say=None):
        self.cfg, self.m = cfg, machine
        self.say = say or (lambda text: None)
        self.recs = []
        self.up = False
        self.load = None
        self.gpu_start = None
        self.guard_on = False
        self.installed = False
        self.started_at = None

    # helpers ------------------------------------------------------------------------------------------------------
    def show(self):
        return self.m.show()

    def healthy(self):
        s = self.show()
        return s.get("ActiveState") == "active" and self.m.health() == 200

    def wait_healthy(self, restarts_at_least, limit, t0=None):
        """Seconds from t0 until the unit is active, /health is 200 and the restart count reached the target. None on timeout."""
        t0 = self.m.now() if t0 is None else t0
        while self.m.now() - t0 <= limit:
            s = self.show()
            if int(s.get("NRestarts", 0)) >= restarts_at_least and s.get("ActiveState") == "active" and self.m.health() == 200:
                return self.m.now() - t0
            self.m.sleep(1)
        return None

    def restarts(self):
        return int(self.show().get("NRestarts", 0))

    @property
    def L(self):
        return self.load if self.load is not None else self.cfg.default_load

    def heal_limit(self, extra=0.0):
        return extra + self.L + self.cfg.heal_margin

    def ensure_up(self, st):
        # StartLimitBurst=4 counts every start of 15 min, the restarts that a fault causes too. Clear it before each fault,
        # or step 5 of the night would meet the limit that the unit is meant to have only for a bad setting (N10).
        self.m.systemctl("reset-failed", self.cfg.unit)
        if self.healthy():
            return
        st.note("the unit was not healthy at the start of the step; started it again")
        self.m.systemctl("start", self.cfg.unit)
        if self.wait_healthy(0, self.cfg.start_timeout) is None:
            raise Abort("the unit does not come up")

    def resend(self, conv):
        """Send the conversation again with 64 more tokens. Returns (reply, tokens restored from the cache or the disk).

        A disk restore does not set usage cached_tokens. The server writes it in its crucible_turn journal line
        (state_disk_tokens, from the stage reply of the engine), so the driver reads both."""
        since = self.m.now()
        ext = conv.extended(64)
        r = self.m.chat(ext, max_tokens=8, timeout=self.cfg.start_timeout)
        got = int(r.get("cached_tokens") or 0)
        if r.get("ok"):
            self.m.sleep(1)
            for line in self.m.journal(since).splitlines():
                m = TURN_LINE.search(line)
                if m:
                    try:
                        turn = json.loads(m.group(1))
                    except ValueError:
                        continue
                    if turn.get("prompt_tokens") == r.get("prompt_tokens", ext.tokens):
                        got = max(got, *(v for v in (turn.get("state_disk_tokens"), turn.get("cached_tokens")) if type(v) is int))
        return r, got

    def decode(self, conv):
        return self.m.start_chat(conv, max_tokens=self.cfg.decode_tokens, timeout=self.cfg.start_timeout)

    # install and uninstall ----------------------------------------------------------------------------------------
    def install(self, missing_pack=False):
        cfg, m = self.cfg, self.m
        m.write_serve(serve_script(cfg))
        m.write_unit(render_unit(cfg, missing_pack))
        self.installed = True
        m.systemctl("daemon-reload")
        if not missing_pack:
            m.systemctl("enable", "--runtime", cfg.unit)

    def uninstall(self):
        cfg, m = self.cfg, self.m
        if self.guard_on:
            m.systemctl("stop", f"{cfg.guard}.timer")
            m.systemctl("reset-failed", f"{cfg.guard}.service")
            self.guard_on = False
        if self.installed:
            m.systemctl("stop", cfg.unit)
            m.systemctl("disable", "--runtime", cfg.unit)
            m.remove_files()
            m.systemctl("daemon-reload")
            m.systemctl("reset-failed", cfg.unit)
            self.installed = False

    # the steps ----------------------------------------------------------------------------------------------------
    def n1(self, st):
        cfg, m = self.cfg, self.m
        self.install()
        t0 = m.now()
        rc, out = m.systemctl("start", cfg.unit)
        self.load = m.now() - t0
        st.facts["load_seconds"] = round(self.load, 1)
        if not st.check(rc == 0, f"systemctl start failed: {out.strip()[-200:]}"):
            return
        s = self.show()
        st.check(s.get("ActiveState") == "active", f"state is {s.get('ActiveState')}, not active")
        st.check(s.get("Type") == "notify", "the unit is not Type=notify")
        st.check(m.health() == 200, "/health is not 200")
        self.up = True
        st.facts["nrestarts"] = self.restarts()
        pid = int(s.get("MainPID", 0) or 0)
        unit_limits, login = (m.limits(pid) if pid else {}), m.login_limits()
        st.facts["limits"] = dict(unit=unit_limits, login=login)
        locked = unit_limits.get("locked")
        if locked != "unlimited" and locked != login.get("locked"):
            self._limit_problem(st, f"the unit limit locked {locked} differs from a login shell {login.get('locked')}")
        nofile = unit_limits.get("nofile")
        if nofile != "unlimited" and not (isinstance(nofile, int) and nofile >= NOFILE_MIN):
            self._limit_problem(st, f"the unit limit nofile {nofile} is below {NOFILE_MIN}")

    def _limit_problem(self, st, text):
        if self.cfg.limits_strict:
            st.check(False, text)
        else:
            st.note(text + " (dry run: noted, not failed)")

    def n2(self, st):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        r0 = self.restarts()
        conv = make_conv(m, "n2long", cfg.long_tokens)
        since = m.now()
        r = m.chat(conv, max_tokens=8, timeout=cfg.start_timeout)
        took = m.now() - since
        st.facts.update(prompt_tokens=r.get("prompt_tokens", conv.tokens), prefill_seconds=round(took, 1))
        st.check(r.get("ok"), f"the long request failed: {r.get('error')}")
        st.check(conv.tokens >= 0.95 * cfg.long_tokens, "the prompt is shorter than 95% of the target")
        turn = conv
        for i in range(4):
            turn = turn.extended(256)
            r = m.chat(turn, max_tokens=16, timeout=cfg.start_timeout)
            st.check(r.get("ok"), f"turn {i + 1} failed: {r.get('error')}")
        m.sleep(2)
        gaps = [float(g) for g in GAP_LINE.findall(m.journal(since))]
        gap = max(gaps) if gaps else 0.0
        st.facts["largest_gap_s"] = gap
        if not gaps:
            st.facts["largest_gap_s"] = "below 1"
            st.check(took <= cfg.gap_limit, f"the journal has no engine gap line, but the prefill took {took:.0f} s: no evidence of the gap")
        else:
            st.check(gap < cfg.gap_limit, f"the largest engine gap {gap:.1f} s is not below {cfg.gap_limit:g} s: raise the stall to twice the gap")
        st.check(self.restarts() == r0, "the unit restarted during the prefill")

    def n3(self, st):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        n0 = self.restarts()
        h = self.decode(make_conv(m, "n3", 2048))
        st.check(h.first_token(120), "the decode did not start")
        _, engine = m.pids()
        if not st.check(engine, "no engine process found"):
            return
        m.kill(engine, signal.SIGKILL)
        t0 = m.now()
        took = self.wait_healthy(n0 + 1, self.heal_limit() + 30, t0)
        st.facts.update(heal_seconds=None if took is None else round(took, 1), limit=round(self.heal_limit(), 1))
        st.check(took is not None and took <= self.heal_limit(), f"healthy again after {took} s, limit {self.heal_limit():.0f} s")
        st.check(self.restarts() == n0 + 1, f"restart count is {self.restarts()}, expected {n0 + 1}")
        res = h.result(30)
        st.facts["cut_request_ended"] = not res.get("ok")

    def n4(self, st):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        n0 = self.restarts()
        convs = [make_conv(m, f"n4c{i}", cfg.disk_tokens) for i in range(2)]
        for c in convs:
            r = m.chat(c, max_tokens=8, timeout=cfg.start_timeout)
            st.check(r.get("ok"), f"the setup request failed: {r.get('error')}")
        server, _ = m.pids()
        if not st.check(server, "no server process found"):
            return
        m.kill(server, signal.SIGKILL)
        t0 = m.now()
        limit = self.heal_limit(cfg.flush_max)
        took = self.wait_healthy(n0 + 1, limit + 30, t0)
        st.facts.update(heal_seconds=None if took is None else round(took, 1), limit=round(limit, 1))
        st.check(took is not None and took <= limit, f"healthy again after {took} s, limit {limit:.0f} s")
        restored = 0
        for c in convs:
            r, got = self.resend(c)
            st.check(r.get("ok"), f"the resent request failed: {r.get('error')}")
            restored = max(restored, got)
        st.facts["restored_tokens"] = restored
        st.check(restored >= cfg.restore_fraction * convs[0].tokens, f"a resent conversation restored {restored} tokens, expected at least {cfg.restore_fraction * convs[0].tokens:.0f}")

    def _stop_fault(self, st, decoding):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        n0 = self.restarts()
        cores0 = m.coredumps()
        h = None
        if decoding:
            h = self.decode(make_conv(m, "n5", 2048))
            st.check(h.first_token(120), "the decode did not start")
        _, engine = m.pids()
        if not st.check(engine, "no engine process found"):
            return
        m.kill(engine, signal.SIGSTOP)
        t0 = m.now()
        found = None
        while m.now() - t0 <= cfg.found_limit + 60:
            s = self.show()
            if int(s.get("NRestarts", 0)) > n0 or s.get("ActiveState") != "active":
                found = m.now() - t0
                break
            m.sleep(1)
        st.facts.update(found_seconds=None if found is None else round(found, 1), limit=cfg.found_limit)
        st.check(found is not None and found <= cfg.found_limit, f"the hang was found after {found} s, limit {cfg.found_limit:g} s")
        took = self.wait_healthy(n0 + 1, cfg.found_limit + self.heal_limit() + 60, t0)
        limit = cfg.found_limit + self.heal_limit()
        st.facts["healthy_seconds"] = None if took is None else round(took, 1)
        st.check(took is not None and took <= limit, f"healthy again after {took} s, limit {limit:.0f} s")
        st.check(re.search(r"Watchdog timeout", m.journal(t0)) is not None, "the journal has no watchdog timeout line")
        cores1 = m.coredumps()
        if cores0 is None or cores1 is None:
            st.note("coredumpctl is not usable here: the core file check did not run")
        else:
            st.facts["new_core_files"] = cores1 - cores0
            st.check(cores1 == cores0, f"{cores1 - cores0} new core file(s)")
        if h:
            h.result(30)

    def n5(self, st):
        self._stop_fault(st, True)

    def n6(self, st):
        self._stop_fault(st, False)

    def n7(self, st):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        n0 = self.restarts()
        pings0 = m.commands_logged()
        t0 = m.now()
        last_health = t0
        while m.now() - t0 < cfg.idle_seconds:
            m.sleep(5)
            s = self.show()
            if s.get("ActiveState") != "active" or int(s.get("NRestarts", 0)) != n0:
                st.check(False, f"the unit left the idle state after {m.now() - t0:.0f} s: {s.get('ActiveState')}, restarts {s.get('NRestarts')}")
                break
            if m.now() - last_health >= 30:
                last_health = m.now()
                if not st.check(m.health() == 200, "/health was not 200 while idle"):
                    break
        st.facts.update(idle_seconds=round(m.now() - t0), restarts=self.restarts() - n0)
        pings1 = m.commands_logged()
        if pings0 is not None and pings1 is not None:
            want = int(cfg.idle_seconds / 10) - 3
            st.facts["pings"] = pings1 - pings0
            st.check(pings1 - pings0 >= want, f"{pings1 - pings0} pings in {cfg.idle_seconds} s, expected at least {want}")

    def n8(self, st):
        cfg, m = self.cfg, self.m
        self.ensure_up(st)
        convs = [make_conv(m, f"n8c{i}", cfg.disk_tokens) for i in range(3)]
        for c in convs:
            r = m.chat(c, max_tokens=8, timeout=cfg.start_timeout)
            st.check(r.get("ok"), f"the setup request failed: {r.get('error')}")
        h = m.start_chat(make_conv(m, "n8long", cfg.long_tokens), max_tokens=8, timeout=cfg.start_timeout)
        m.sleep(cfg.n8_wait)
        t0 = m.now()
        m.systemctl("stop", cfg.unit)
        took = m.now() - t0
        st.facts["stop_seconds"] = round(took, 1)
        st.check(took <= cfg.stop_limit, f"the stop took {took:.0f} s, limit {cfg.stop_limit:g} s")
        st.check(KILLED.search(m.journal(t0)) is None, "the journal shows a SIGKILL or a stop timeout")
        cut = h.result(15)
        st.facts["cut_request"] = "error" if not cut.get("ok") else "completed"
        st.check(not cut.get("ok"), "the interrupted request ended without an error")
        st.facts["cut_request_error_event"] = bool(cut.get("error_event"))
        m.systemctl("start", cfg.unit)
        if self.wait_healthy(0, cfg.start_timeout) is None:
            st.check(False, "the unit did not come up after the stop")
            return
        lost = 0
        for c in convs:
            r, got = self.resend(c)
            if not r.get("ok") or got < cfg.restore_fraction * c.tokens:
                lost += 1
        st.facts["lost_states"] = lost
        st.check(lost == 0, f"{lost} of {len(convs)} finished conversations did not resume from the cache")

    def n9(self, st):
        cfg, m = self.cfg, self.m
        if not m.userns_ok():
            st.skip("user namespaces are blocked (unshare -rm failed): the full disk step did not run")
            return
        m.systemctl("stop", cfg.unit)
        ok = 0
        try:
            if not st.check(m.direct_start(), "the server did not start in the namespace"):
                return
            for i in range(cfg.n9_turns):
                r = m.direct_chat(make_conv(m, f"n9t{i}", cfg.disk_tokens), max_tokens=8, timeout=cfg.start_timeout)
                if r.get("status") == 200:
                    ok += 1
            alive = m.direct_alive()
            m.direct_stop()  # the log is read after the stop: the last flush writes its refusal then
            log = m.direct_log()
            full = len(re.findall(r"\bfull\b", log))
            st.facts.update(ok=ok, turns=cfg.n9_turns, server_alive=alive, full_lines=full)
            st.check(ok == cfg.n9_turns, f"{ok} of {cfg.n9_turns} requests returned 200")
            st.check(alive, "the server exited")
            st.check(full > 0, "the server log has no line with 'full'")
        finally:
            m.direct_stop()

    def n10(self, st):
        cfg, m = self.cfg, self.m
        m.systemctl("stop", cfg.unit)
        m.write_unit(render_unit(cfg, missing_pack=True))
        m.systemctl("daemon-reload")
        m.systemctl("reset-failed", cfg.unit)
        n0 = self.restarts()
        t0 = m.now()
        m.systemctl("start", cfg.unit)
        failed = None
        while m.now() - t0 <= cfg.bad_limit:
            if self.show().get("ActiveState") == "failed":
                failed = m.now() - t0
                break
            m.sleep(1)
        restarts = self.restarts()
        st.facts.update(failed_seconds=None if failed is None else round(failed, 1), restarts=restarts)
        st.check(failed is not None, f"the unit was not failed after {cfg.bad_limit:g} s")
        st.check(restarts <= 4, f"{restarts} restarts, limit 4")
        status = m.service_status()
        reason = (status or {}).get("reason")
        if not reason:
            lines = [ln for ln in m.journal(t0).splitlines() if "FAIL" in ln]
            reason = lines[-1].strip() if lines else ""
            st.facts["reason_from"] = "journal"
        else:
            st.facts["reason_from"] = "service status"
        st.facts["reason"] = reason
        st.check("pack" in reason.lower(), "the reason does not name the failed check (the pack folder)")

    def n11(self, st):
        cfg, m = self.cfg, self.m
        self.uninstall()
        st.check(m.leftovers() == [], f"processes are left: {m.leftovers()}")
        st.check(m.port_free(), f"port {cfg.port} is not free")
        gpu = m.gpu_mem()
        if gpu is None or self.gpu_start is None:
            st.note("GPU memory was not readable: the memory check did not run")
        else:
            end = m.now() + 30
            while gpu > self.gpu_start + (256 << 20) and m.now() < end:
                m.sleep(2)
                gpu = m.gpu_mem()
            st.facts.update(gpu_start_mib=self.gpu_start >> 20, gpu_end_mib=gpu >> 20)
            st.check(gpu <= self.gpu_start + (256 << 20), f"GPU memory is {gpu >> 20} MiB, it was {self.gpu_start >> 20} MiB at the start")
        st.check(self.show().get("ActiveState") in ("inactive", "failed", None, ""), "the unit is still loaded")

    # the run ------------------------------------------------------------------------------------------------------
    def plan(self):
        cfg = self.cfg
        picked = set(cfg.only or STEP_IDS)
        return [s for s in STEP_IDS if s in picked or s in ("N1", "N11")]

    def run(self):
        cfg, m = self.cfg, self.m
        self.started_at = time.time()
        t_begin = m.now()
        verdict = "aborted"
        try:
            m.prepare()
            if cfg.fake:
                m.set_fake_env(cfg.fake_env)
            self.gpu_start = m.gpu_mem()
            # A guard of a dead run (killed driver) would make systemd-run fail with "already loaded". The unit name is shared, so
            # only one run lives at a time: clear any old guard first.
            m.systemctl("stop", f"{cfg.guard}.timer")
            m.systemctl("reset-failed", f"{cfg.guard}.timer", f"{cfg.guard}.service")
            rc, out = m.run(["systemd-run", "--user", "--unit", cfg.guard, f"--on-active={cfg.guard_minutes}min", "--collect",
                             "systemctl", "--user", "stop", cfg.unit])
            if rc != 0:
                raise RuntimeError(f"the guard timer did not start: {out.strip()[-200:]}")
            self.guard_on = True
            for sid in self.plan():
                st = StepRec(sid)
                self.recs.append(st)
                began = m.now()
                try:
                    if sid not in ("N1", "N10", "N11") and not self.up:
                        st.skip("N1 did not bring the unit up", incomplete=True)
                    elif sid != "N11" and m.now() - t_begin > cfg.max_minutes * 60:
                        st.skip(f"out of time ({cfg.max_minutes:g} min)", incomplete=True)
                    else:
                        getattr(self, sid.lower())(st)
                except Abort as exc:
                    st.check(False, str(exc))
                except Exception as exc:  # a step error is a fail; the next steps still run
                    st.check(False, f"error: {type(exc).__name__}: {exc}")
                st.seconds = m.now() - began
                self.say(f"{sid} {st.status} {st.name}")
            verdict = None
        finally:
            try:
                self.uninstall()
            except Exception as exc:  # keep the first error
                self.say(f"uninstall error: {exc}")
            report = self.report(verdict, m.now() - t_begin)
            self.write_report(report)
        return report

    def report(self, verdict, seconds):
        cfg = self.cfg
        bad = any(r.failed or r.incomplete for r in self.recs)
        verdict = verdict or ("fail" if bad else "pass")
        return dict(
            tool="service_faulttest", mode="fake" if cfg.fake else "real", verdict=verdict, git=cfg.git_sha,
            started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at or time.time())),
            total_seconds=round(seconds, 1), unit=cfg.unit, port=cfg.port,
            limits=dict(watchdog_sec=cfg.watchdog_sec, stall_sec=cfg.stall_sec, found_limit=cfg.found_limit,
                        stop_limit=cfg.stop_limit, gap_limit=cfg.gap_limit, bad_limit=cfg.bad_limit),
            steps=[r.to_json() for r in self.recs])

    def write_report(self, report):
        text = json.dumps(report, indent=2) + "\n"
        text = text.replace(str(Path.home()), "~")
        try:
            self.cfg.report.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.report.with_suffix(".tmp")
            tmp.write_text(text)
            os.replace(tmp, self.cfg.report)
        except OSError as exc:
            self.say(f"cannot write the report: {exc}")


# ---- the real machine ------------------------------------------------------------------------------------------------

class Stream:
    """A streaming completion in a thread. first_token() waits for the first piece of text."""

    def __init__(self, machine, conv, max_tokens, timeout):
        self.m, self.first, self.out = machine, threading.Event(), {}
        body = dict(model="crucible", prompt=conv.text, max_tokens=max_tokens, temperature=0, stream=True)
        self.thread = threading.Thread(target=self._run, args=(body, timeout), daemon=True)
        self.thread.start()

    def _run(self, body, timeout):
        conn = http.client.HTTPConnection(self.m.cfg.bind, self.m.cfg.port, timeout=timeout)
        out = dict(ok=False, status=0, error="", error_event=False)
        try:
            conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            out["status"] = resp.status
            if resp.status != 200:
                out["error"] = f"HTTP {resp.status}"
            else:
                done = False
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    if line == "data: [DONE]":
                        done = True
                        break
                    try:
                        data = json.loads(line[5:])
                    except ValueError:
                        continue
                    if "error" in data:
                        out.update(error=str(data["error"])[:200], error_event=True)
                        break
                    if (data.get("choices") or [{}])[0].get("text"):
                        self.first.set()
                out["ok"] = done and not out["error"]
                if not done and not out["error"]:
                    out["error"] = "the stream ended without [DONE]"
        except (OSError, http.client.HTTPException) as exc:
            out["error"] = str(exc)[:200]
        finally:
            conn.close()
        self.out = out

    def first_token(self, timeout):
        return self.first.wait(timeout)

    def result(self, timeout):
        self.thread.join(timeout)
        return self.out or dict(ok=False, status=0, error="no end within the wait")


class RealMachine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.direct = None
        self.direct_log_path = cfg.work / "n9-server.log"

    # time and processes
    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)

    def run(self, argv, timeout=120):
        try:
            p = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, str(exc)
        return p.returncode, p.stdout + p.stderr

    def systemctl(self, *args):
        limit = self.cfg.start_timeout + 60 if args and args[0] in ("start", "restart") else self.cfg.stop_limit + 60
        return self.run(["systemctl", "--user", *args], timeout=limit)

    def show(self):
        props = ("ActiveState", "SubState", "NRestarts", "Result", "MainPID", "StatusText", "Type", "ControlGroup")
        rc, out = self.run(["systemctl", "--user", "show", self.cfg.unit] + [f"-p{p}" for p in props], timeout=30)
        return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

    def pids(self):
        s = self.show()
        main = int(s.get("MainPID", 0) or 0)
        if not main:
            return None, None
        procs = Path("/sys/fs/cgroup") / s.get("ControlGroup", "").lstrip("/") / "cgroup.procs"
        try:
            members = [int(x) for x in procs.read_text().split()]
        except (OSError, ValueError):
            members = []
        engine = None
        for pid in members:
            if pid == main:
                continue
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
                words = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError, IndexError):
                continue
            if ppid == main and b"server" not in words[:3]:
                engine = pid
        return main, engine

    def kill(self, pid, sig):
        os.kill(int(pid), sig)

    # files
    def prepare(self):
        self.cfg.work.mkdir(parents=True, exist_ok=True)
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.fake:
            prepare_fake(self.cfg)

    def _write(self, path, text, mode):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)

    def write_unit(self, text):
        self._write(self.cfg.unit_path, text, 0o644)

    def write_serve(self, text):
        self._write(self.cfg.serve, text, 0o755)
        if self.cfg.fake:
            self._write(self.cfg.check, check_script(self.cfg), 0o755)

    def remove_files(self):
        for p in (self.cfg.unit_path, self.cfg.serve, self.cfg.check, self.cfg.envfile):
            try:
                p.unlink()
            except OSError:
                pass

    def set_fake_env(self, env):
        lines = "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items())
        self.cfg.work.mkdir(parents=True, exist_ok=True)
        self._write(self.cfg.envfile, lines, 0o600)
        Path(env.get("FAKE_COMMAND_LOG", os.devnull)).unlink(missing_ok=True)
        Path(env.get("FAKE_DISK_FILE", os.devnull)).unlink(missing_ok=True) if "FAKE_DISK_FILE" in env else None

    def commands_logged(self):
        log = self.cfg.fake_env.get("FAKE_COMMAND_LOG")
        if not log:
            return None
        try:
            return sum(1 for line in Path(log).read_text().splitlines() if '"ping"' in line)
        except OSError:
            return 0

    # http
    def _http(self, method, path, body=None, timeout=30):
        conn = http.client.HTTPConnection(self.cfg.bind, self.cfg.port, timeout=timeout)
        try:
            conn.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, r.read()
        except (OSError, http.client.HTTPException):
            return 0, b""
        finally:
            conn.close()

    def health(self):
        return self._http("GET", "/health", timeout=5)[0]

    def tokenize_len(self, text):
        status, data = self._http("POST", "/tokenize", dict(content=text), timeout=120)
        if status != 200:
            raise RuntimeError(f"/tokenize returned {status}")
        return len(json.loads(data)["tokens"])

    def chat(self, conv, max_tokens=8, timeout=60):
        began = time.time()
        status, data = self._http("POST", "/v1/completions",
                                  dict(model="crucible", prompt=conv.text, max_tokens=max_tokens, temperature=0), timeout)
        out = dict(ok=status == 200, status=status, total_s=time.time() - began, error="" if status == 200 else f"HTTP {status}")
        if status == 200:
            try:
                usage = json.loads(data).get("usage", {})
                out["prompt_tokens"] = usage.get("prompt_tokens")
                out["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            except ValueError:
                out.update(ok=False, error="bad JSON")
        return out

    def start_chat(self, conv, max_tokens=8, timeout=60):
        return Stream(self, conv, max_tokens, timeout)

    # journal, limits, cores, GPU, ports
    def journal(self, since):
        rc, out = self.run(["journalctl", "--user", "-u", self.cfg.unit, f"--since=@{int(since) - 1}", "--no-pager", "-o", "short-iso"], 60)
        return out

    @staticmethod
    def _limit(value):
        return value if value == "unlimited" else int(value)

    def limits(self, pid):
        out = {}
        for line in Path(f"/proc/{pid}/limits").read_text().splitlines():
            if line.startswith("Max locked memory"):
                out["locked"] = self._limit(line.split()[3])
            elif line.startswith("Max open files"):
                out["nofile"] = self._limit(line.split()[3])
        return out

    def login_limits(self):
        rc, out = self.run(["bash", "-lc", LOGIN_LIMITS], 30)
        words = out.split()
        if rc != 0 or len(words) < 2:
            return {}
        lock = words[-2]
        return dict(locked=lock if lock == "unlimited" else int(lock) * 1024, nofile=self._limit(words[-1]))

    def coredumps(self):
        rc, out = self.run(["coredumpctl", "list", "--no-legend", "--no-pager"], 30)
        if rc == 127 or rc > 1:
            return None
        return len([ln for ln in out.splitlines() if ln.strip() and "No coredumps" not in ln])

    def gpu_mem(self):
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            try:
                if (card / "vendor").read_text().strip() != "0x1002":
                    continue
                return sum(int((card / name).read_text()) for name in ("mem_info_vram_used", "mem_info_gtt_used"))
            except (OSError, ValueError):
                continue
        return None

    def port_free(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the server does: closed connections in TIME_WAIT are not a user
            try:
                s.bind((self.cfg.bind, self.cfg.port))
            except OSError:
                return False
        return True

    def leftovers(self):
        """Processes that still belong to the unit's control group, and the N9 server when it still runs."""
        found = []
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                groups = (p / "cgroup").read_text()
            except OSError:
                continue
            if self.cfg.unit in groups:
                found.append(int(p.name))
        if self.direct is not None and self.direct.poll() is None:
            found.append(self.direct.pid)
        return found

    def service_status(self):
        rc, out = self.run([self.cfg.python, "-I", self.cfg.repo / "crucible", "service", "status", "--unit", self.cfg.unit, "--json",
                          "--check-file", self.cfg.check_file], 60)
        try:
            data = json.loads(out)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    # N9: the server in a mount namespace with a small tmpfs as the state folder
    def userns_ok(self):
        return self.run(["unshare", "-rm", "true"], 30)[0] == 0

    def direct_start(self):
        cfg = self.cfg
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        # The server runs under `timeout` in the session of the driver (the lease session), not in a session of its own.
        # So the session kill of the lease reaches it, and it ends at the life limit even when the driver is killed.
        argv = ["timeout", "--foreground", "-k", str(cfg.n9_kill_after), str(cfg.n9_limit),
                "unshare", "-rm", "sh", "-c", 'mount -t tmpfs -o size="$1" tmpfs "$2" && shift 2 && exec "$@"', "sh",
                cfg.n9_tmpfs, os.fspath(cfg.state_dir), os.fspath(cfg.serve), "--state-dir", os.fspath(cfg.state_dir),
                "--state-disk-gib", "200"]
        log = open(self.direct_log_path, "wb")
        env = dict(os.environ, FAKE_STAGE_FULL="1") if cfg.fake else None
        self.direct = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        end = time.time() + cfg.start_timeout
        while time.time() < end:
            if self.direct.poll() is not None:
                return False
            if self.health() == 200:
                return True
            time.sleep(1)
        return False

    def direct_alive(self):
        return self.direct is not None and self.direct.poll() is None

    def direct_chat(self, conv, max_tokens=8, timeout=60):
        return self.chat(conv, max_tokens, timeout)

    def direct_log(self):
        try:
            return self.direct_log_path.read_text(errors="replace")
        except OSError:
            return ""

    def direct_stop(self):
        p, self.direct = self.direct, None
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(self.cfg.n9_kill_after + 20)  # timeout sends the kill after n9_kill_after seconds
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(30)


def prepare_fake(cfg):
    """The dry run needs a checkpoint folder with a tokenizer, and an engine that is the fake batch engine."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[UNK]": 0}
    for i in range(1, 1500):
        vocab[f"w{i}"] = i
    special = {"<|endoftext|>": 248044, "<|im_end|>": 248046, "<tool_call>": 248058, "</tool_call>": 248059}
    vocab.update(special)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    cfg.ckpt.mkdir(parents=True, exist_ok=True)
    tok.save(str(cfg.ckpt / "tokenizer.json"))
    # The server needs the four special tokens as added tokens (content encoding reads them), with their real IDs.
    path = cfg.ckpt / "tokenizer.json"
    data = json.loads(path.read_text())
    data["added_tokens"] = [dict(id=i, content=c, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
                            for c, i in special.items()]
    path.write_text(json.dumps(data))
    template = (cfg.repo / "server/tests/fixtures/qwen_chat_template.jinja").read_text()
    (cfg.ckpt / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    wrapper = f'#!/bin/sh\nexec "{cfg.python}" "{cfg.repo}/server/tests/fake_batch_engine.py" "$@"\n'
    cfg.engine.parent.mkdir(parents=True, exist_ok=True)
    cfg.engine.write_text(wrapper)
    cfg.engine.chmod(0o755)


# ---- units and timers before and after a reboot ------------------------------------------------------------------------

def take_snapshot(run=None):
    run = run or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout)
    def names(argv):
        return sorted(" ".join(line.split()[:4]) for line in run(argv).splitlines() if line.strip())
    runtime = []
    base = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "systemd"
    for sub in ("user", "transient"):
        d = base / sub
        if d.is_dir():
            runtime += sorted(f"{sub}/{p.name}" for p in d.iterdir())
    linger = run(["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"]).strip()
    return dict(
        units=names(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", "--no-pager"])
        + names(["systemctl", "list-units", "--all", "--no-legend", "--plain", "--no-pager"]),
        timers=names(["systemctl", "--user", "list-timers", "--all", "--no-legend", "--no-pager"])
        + names(["systemctl", "list-timers", "--all", "--no-legend", "--no-pager"]),
        runtime_units=runtime, linger=linger, taken=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def compare_snapshots(before, after):
    out, problems = {}, []
    for key in ("units", "timers", "runtime_units"):
        b, a = set(before.get(key, [])), set(after.get(key, []))
        out[f"{key}_added"], out[f"{key}_removed"] = sorted(a - b), sorted(b - a)
        problems += [f"{key}: {x} is still there" for x in sorted(a) if "crucible-ft" in x]
    if before.get("linger") != after.get("linger"):
        problems.append(f"linger changed from {before.get('linger')} to {after.get('linger')}")
    out.update(problems=problems, verdict="fail" if problems else "pass")
    return out


# ---- command line ----------------------------------------------------------------------------------------------------

def parser():
    ap = argparse.ArgumentParser(prog="service_faulttest", description=__doc__.split("\n\n")[0] + "\n\n" + "\n".join(
        f"  {sid:<4}{STEP_NAMES[sid]}" for sid in STEP_IDS), formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Full text: the head of tools/service_faulttest.py. Exit code 0 when every step passed, 1 when a step failed, 2 for a usage error.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="the GPU night run; hold a gpu-run lease around it")
    mode.add_argument("--fake", action="store_true", help="the CPU dry run with the fake engine and scaled times")
    ap.add_argument("--plan", action="store_true", help="print the steps and change nothing")
    ap.add_argument("--repo", type=Path, default=HERE.parent, help="the checkout to test (default: this one)")
    ap.add_argument("--python", default=None, help="Python of the server (default: .venv/bin/python of the repo, else this one)")
    ap.add_argument("--pack", default=None, help="model pack, 4.05 or 3.05 (default: the saved setup choice)")
    ap.add_argument("--models-dir", type=Path, default=None, help="folder of the pack (default: the saved setup choice)")
    ap.add_argument("--state-dir", type=Path, default=Path("~/.cache/crucible-ft/states").expanduser(),
                    help="state folder of the test server (default ~/.cache/crucible-ft/states)")
    ap.add_argument("--work", type=Path, default=Path("~/.cache/crucible-ft/work").expanduser(), help="folder for scripts and logs")
    ap.add_argument("--report", type=Path, default=None, help="JSON report (default: <work>/report.json)")
    ap.add_argument("--port", type=int, default=8791, help="port of the test server (default 8791)")
    ap.add_argument("--only", default=None, help="run only these steps, for example N3,N5 (N1 and N11 always run)")
    ap.add_argument("--max-minutes", type=float, default=None, help="skip the remaining steps after this time (default 75, dry run 12)")
    ap.add_argument("--snapshot", type=Path, default=None, help="save the units and timers to this file and exit")
    ap.add_argument("--compare", type=Path, default=None, help="compare the units and timers now with this snapshot and exit")
    return ap


def build_config(args):
    from crucible_cli import packs
    from crucible_cli.settings import load_saved
    repo = Path(args.repo).resolve()
    saved = load_saved(repo)
    python = args.python or (str(repo / ".venv/bin/python") if (repo / ".venv/bin/python").exists() else sys.executable)
    pack = args.pack or saved.get("pack") or packs.DEFAULT
    models = args.models_dir or (Path(saved["models_dir"]) if saved.get("models_dir") else repo / "models")
    work = Path(args.work)
    ckpt = work / "fixture-ckpt" if args.fake else Path(models) / packs.PACKS[pack].folder
    engine = work / "fake-engine.sh" if args.fake else repo / "build/engine/crucible"
    cfg = Config(repo=repo, python=python, ckpt=ckpt, engine=engine, state_dir=args.state_dir, work=work,
                 report=args.report or work / "report.json", port=args.port, pack=pack,
                 models_dir=ckpt.parent if args.fake else models, streams=2 if args.fake else int(saved.get("streams", 4)),
                 ctx=131072 if args.fake else int(saved.get("ctx", 262144)))
    if args.fake:
        scaled(cfg)
    if args.only:
        cfg.only = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    if args.max_minutes:
        cfg.max_minutes = args.max_minutes
    try:
        cfg.git_sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                                     timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cfg


def print_plan(cfg, out=print):
    out(f"Fault test plan ({'dry run, fake engine' if cfg.fake else 'GPU run'}): unit {cfg.unit}, port {cfg.port}")
    total = 0
    for sid in Driver(cfg, None).plan():
        out(f"  {sid:<4}{STEP_NAMES[sid]:<42} about {STEP_MINUTES[sid]} min")
        total += STEP_MINUTES[sid]
    out(f"Total about {total} min. Guard: a timer stops the unit after {cfg.guard_minutes} min; the unit is a runtime unit.")


def preflight(cfg, manager_ok=None):
    """Problems that stop a run before it changes anything. A real run needs the engine and the pack."""
    problems = []
    if manager_ok is None:
        manager_ok = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True, timeout=30).returncode == 0
    if not manager_ok:
        problems.append("There is no systemd user manager here (systemctl --user fails). Run this in a login session of the user.")
    if not cfg.fake:
        if not cfg.engine.exists():
            problems.append(f"The engine {cfg.engine} is missing. Run ./crucible setup first.")
        if not cfg.ckpt.is_dir():
            problems.append(f"The pack folder {cfg.ckpt} is missing. Run ./crucible setup first.")
    return problems


def _say(text):
    print(text, flush=True)


def main(argv=None, machine=None, out=_say):
    args = parser().parse_args(argv)
    if args.snapshot or args.compare:
        snap = take_snapshot()
        if args.snapshot:
            args.snapshot.write_text(json.dumps(snap, indent=2) + "\n")
            out(f"Saved {len(snap['units'])} units and {len(snap['timers'])} timers to {args.snapshot}")
            return 0
        result = compare_snapshots(json.loads(args.compare.read_text()), snap)
        out(json.dumps(result, indent=2))
        return 0 if result["verdict"] == "pass" else 1
    if not (args.real or args.fake):
        out("Choose --real (the GPU run) or --fake (the CPU dry run). Add --plan to see the steps first.")
        return 2
    cfg = build_config(args)
    if args.plan:
        print_plan(cfg, out)
        return 0
    if machine is None:
        problems = preflight(cfg)
        if problems:
            out("\n".join(problems))
            return 2
    m = machine or RealMachine(cfg)
    report = Driver(cfg, m, say=lambda t: out(t)).run()
    out(f"Verdict: {report['verdict']}. Report: {cfg.report}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. The unit was removed.", file=sys.stderr)
        sys.exit(130)
