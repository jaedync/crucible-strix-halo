"""One server in a systemd user unit with the fake batch engine.

The unit text comes from the generator of the service CLI (tools/crucible_cli/service.py render_unit) when it exists, so
the tests run the shipped unit. Only the paths and the timing values are changed. Without the generator, a copy of
the unit text in this file stands in."""
import http.client
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAKE = REPO / "server/tests/fake_batch_engine.py"
RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "systemd/user"

# The unit of the design (s1) with the placeholders the generator fills; used only when service.py is absent.
FALLBACK = """[Unit]
Description=Crucible test unit
StartLimitIntervalSec=15min
StartLimitBurst=4

[Service]
Type=notify
NotifyAccess=main
WorkingDirectory=@REPO@
Environment=PYTHONUNBUFFERED=1
ExecStart=@EXEC@
TimeoutStartSec=300
TimeoutStopSec=120s
KillMode=control-group
WatchdogSec=30s
WatchdogSignal=SIGKILL
Restart=on-failure
RestartSec=5s
RestartPreventExitStatus=2 78
LimitCORE=0
UMask=0077
NoNewPrivileges=yes
SyslogIdentifier=crucible
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def base_unit(exec_start):
    try:
        if os.environ.get("CRUCIBLE_UNIT_REPO"):        # a checkout that holds service.py (before it merges here)
            sys.path.insert(0, os.environ["CRUCIBLE_UNIT_REPO"])
            for name in [n for n in sys.modules if n == "tools" or n.startswith("tools.")]: del sys.modules[name]
        from tools.crucible_cli.service import render_unit  # the shipped unit
        return render_unit(repo=str(REPO), venv_py=sys.executable, exec_start=exec_start, bind="127.0.0.1", port=0,
                           start_timeout=300, version="test")
    except (ImportError, TypeError):
        return FALLBACK.replace("@REPO@", str(REPO)).replace("@EXEC@", exec_start)


def set_directives(text, **values):
    """Replace or add [Service] keys (a key may repeat in the text). A value None removes the key."""
    lines = text.splitlines()
    out, seen = [], set()
    for line in lines:
        key = line.split("=", 1)[0]
        if key in values and "=" in line and not line.startswith("#"):
            if key not in seen and values[key] is not None:
                out.append(f"{key}={values[key]}")
            seen.add(key)
            continue
        out.append(line)
    text = "\n".join(out)
    extra = [f"{key}={value}" for key, value in values.items() if key not in seen and value is not None]
    return text.replace("[Service]", "[Service]\n" + "\n".join(extra), 1) if extra else text


def systemctl(*args, check=False):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=check)


class Unit:
    """A server in a transient-name unit. `real` keeps the generator's timing; else the values are scaled."""

    def __init__(self, tmp_path, *, env=None, real=False, streams=2, ctx=512, extra_args=(), state=False, unit_keys=None,
                 stall=None, watchdog="3s", restart_sec="1s", start_timeout="30s", pre=None):
        self.name = f"crucible-t1-{uuid.uuid4().hex[:8]}"
        self.unit = f"{self.name}.service"
        self.port = free_port()
        self.tmp = tmp_path
        self.checkpoint = self.make_checkpoint(tmp_path / "ckpt")
        self.command_log = tmp_path / "commands.log"
        self.disk_file = tmp_path / "disk.json"
        self.state_dir = tmp_path / "states"
        engine = tmp_path / "engine.sh"
        engine.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        engine.chmod(0o755)
        args = [sys.executable, "-m", "server", str(self.checkpoint), "--engine", str(engine), "--ctx", str(ctx),
                "--streams", str(streams), "--port", str(self.port), "--systemd", "--no-update-check", "--lookup", "0"]
        if state:
            args += ["--conversation-cache-mib", "64", "--state-dir", str(self.state_dir)]
        if stall is not None:
            args += ["--watchdog-stall-seconds", str(stall)]
        args += list(extra_args)
        fake = {"FAKE_COMMAND_LOG": str(self.command_log), "FAKE_SLOT_CONTEXTS": ",".join([str(ctx)] * streams)}
        if state:
            fake |= {"FAKE_STATE_DISK": "1", "FAKE_DISK_FILE": str(self.disk_file), "FAKE_IGNORE_TERM": "1"}
        fake |= env or {}
        text = base_unit(" ".join(f'"{a}"' if " " in a else a for a in args))
        keys = {"ExecStart": " ".join(args), "ExecStartPre": pre, "WorkingDirectory": str(REPO)}
        if not real:
            keys |= {"WatchdogSec": watchdog, "RestartSec": restart_sec, "TimeoutStartSec": start_timeout}
        keys |= unit_keys or {}
        text = set_directives(text, **keys)
        environment = "\n".join(f'Environment="{k}={v}"' for k, v in fake.items())
        text = text.replace("[Service]", "[Service]\n" + environment + "\nEnvironment=CRUCIBLE_NO_UPDATE_CHECK=1", 1)
        text = re.split(r"^\[Install\]", text, flags=re.M)[0]  # transient: never enabled
        self.text = text
        RUNTIME.mkdir(parents=True, exist_ok=True)
        (RUNTIME / self.unit).write_text(text)
        systemctl("daemon-reload", check=True)

    @staticmethod
    def make_checkpoint(root):
        spec = importlib.util.spec_from_file_location("content_tokens_fixture", REPO / "server/tests/test_content_tokens.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(REPO))
        spec.loader.exec_module(module)
        root.mkdir(parents=True, exist_ok=True)
        return module.synthetic_checkpoint(root)

    # ----- systemd
    def show(self, *props):
        out = systemctl("show", self.unit, *[f"-p{p}" for p in props]).stdout
        return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

    def start(self, wait=True, timeout=60):
        result = systemctl("start", "--no-block", self.unit)
        assert result.returncode == 0, result.stderr
        return self.wait_ready(timeout) if wait else None

    def stop(self, timeout=150):
        began = time.monotonic()
        systemctl("stop", self.unit)
        return time.monotonic() - began

    def cleanup(self):
        systemctl("stop", self.unit)
        systemctl("reset-failed", self.unit)
        try: (RUNTIME / self.unit).unlink()
        except FileNotFoundError: pass
        systemctl("daemon-reload")

    def state(self):
        return self.show("ActiveState", "SubState", "Result", "NRestarts", "ExecMainStatus", "MainPID", "StatusText")

    def wait_ready(self, timeout=60):
        """The unit is active (READY=1 came) and /health answers 200. Returns the seconds it took."""
        began = time.monotonic()
        while time.monotonic() - began < timeout:
            if self.show("ActiveState")["ActiveState"] == "active" and self.health() == 200:
                return time.monotonic() - began
            time.sleep(0.1)
        raise AssertionError(f"not ready in {timeout} s: {self.state()}\n{self.journal()}")

    def wait_for(self, predicate, timeout, interval=0.05, what="condition"):
        began = time.monotonic()
        while time.monotonic() - began < timeout:
            seen = predicate()
            if seen:
                return seen, time.monotonic() - began
            time.sleep(interval)
        raise AssertionError(f"{what} not seen in {timeout} s: {self.state()}\n{self.journal()}")

    def journal(self, lines=40):
        out = subprocess.run(["journalctl", "--user", "-u", self.unit, "-n", str(lines), "--no-pager", "-o", "cat"],
                             capture_output=True, text=True).stdout
        return out

    def journal_has(self, text, timeout=15, lines=100):
        """Poll the journal until TEXT appears (journald writes the lines of a unit a moment after systemd reports its exit).
        Returns (found, last output); use it as `found, out = unit.journal_has(...)` and `assert found, out`."""
        end = time.monotonic() + timeout
        while True:
            out = self.journal(lines)
            if text in out or time.monotonic() >= end:
                return text in out, out
            time.sleep(0.5)

    # ----- processes
    def cgroup_pids(self):
        group = self.show("ControlGroup")["ControlGroup"]
        try:
            return [int(x) for x in Path(f"/sys/fs/cgroup{group}/cgroup.procs").read_text().split()]
        except OSError:
            return []

    def server_pid(self):
        return int(self.show("MainPID")["MainPID"])

    def engine_pid(self):
        for pid in self.cgroup_pids():
            try:
                if b"fake_batch_engine.py" in Path(f"/proc/{pid}/cmdline").read_bytes() and \
                        b"engine.sh" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                    return pid
            except OSError:
                pass
        raise AssertionError(f"no engine process in {self.cgroup_pids()}")

    def alive(self, pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] not in ("Z", "X")
        except OSError:
            return False

    # ----- http
    def health(self):
        try:
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
            connection.request("GET", "/health")
            return connection.getresponse().status
        except OSError:
            return None

    def post(self, ids, max_tokens=4, timeout=120, stream=False):
        """POST /v1/completions with a token-ID prompt. Returns (status, parsed body or text)."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        body = json.dumps(dict(prompt=list(ids), max_tokens=max_tokens, spec=0, stream=stream))
        connection.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode()
        connection.close()
        try: return response.status, json.loads(data)
        except ValueError: return response.status, data

    def stream_first_token(self, ids, max_tokens=200, timeout=30):
        """Open a streaming request and read until the first token; returns the open connection (the caller closes it)."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        connection.request("POST", "/v1/completions", json.dumps(dict(prompt=list(ids), max_tokens=max_tokens, spec=0, stream=True)),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        response.fp.readline()
        return connection, response

    def commands_clear(self):
        self.command_log.unlink(missing_ok=True)

    def commands(self):
        if not self.command_log.exists():
            return []
        return [json.loads(line) for line in self.command_log.read_text().splitlines() if line.strip()]
