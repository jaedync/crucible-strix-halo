"""A1: the unit's ExecStartPre (doctor --for start) waits while GPU memory is busy, then the server starts.

The hardware checks of the real doctor cannot pass on every machine, so a small wrapper runs the real settle loop
(doctor.for_start) with the check list replaced: it reports a busy GPU while a flag file exists. The unit, the
ExecStartPre exit codes and the timeouts are the real systemd ones. The tests skip until tools/crucible_cli/doctor.py
holds for_start (set CRUCIBLE_UNIT_REPO to a checkout that has it)."""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from svc_harness import REPO

WRAPPER = '''
import json, os, sys
sys.path.insert(0, {repo!r})
from tools.crucible_cli import doctor
from tools.crucible_cli.host import Host
flag = {flag!r}
def failures(host, s):
    if not os.path.exists(flag): return []
    return [doctor.CheckResult("gpu-free", "GPU memory is free", "fail", "9.0 GB of GPU memory is in use", "Stop the program.", ("start",))]
doctor.start_failures = failures
doctor.START_WAIT_SECONDS = {wait!r}
sys.exit(doctor.for_start(Host(), None, quiet=True, check_file={check!r}))
'''


def doctor_repo():
    for root in filter(None, (os.environ.get("CRUCIBLE_UNIT_REPO"), str(REPO))):
        path = Path(root) / "tools/crucible_cli/doctor.py"
        if path.exists() and "def for_start" in path.read_text():
            return root
    pytest.skip("doctor.for_start is not here yet (T4); set CRUCIBLE_UNIT_REPO")


def unit_with_wait(make_unit, tmp_path, wait, **options):
    flag, check = tmp_path / "gpu-busy", tmp_path / "check.json"
    flag.write_text("busy")
    script = tmp_path / "pre.py"
    script.write_text(WRAPPER.format(repo=doctor_repo(), flag=str(flag), wait=wait, check=str(check)))
    unit = make_unit(pre=f'{sys.executable} -I {script}', **options)
    return unit, flag, check


def free_after(flag, seconds):
    timer = threading.Timer(seconds, lambda: flag.unlink(missing_ok=True))
    timer.start()
    return timer


@pytest.mark.real_values
def test_a1_gpu_busy_for_60_seconds_the_service_comes_up(make_unit, tmp_path):
    unit, flag, check = unit_with_wait(make_unit, tmp_path, 120.0, real=True)
    timer = free_after(flag, 60)
    began = time.monotonic()
    try:
        unit.start(wait=False)
        unit.wait_ready(120)
    finally:
        timer.cancel()
    took = time.monotonic() - began
    assert 60 <= took <= 60 + 5 + 20, took           # one settle interval (5 s) after the GPU frees, plus the load
    state = unit.state()
    assert state["NRestarts"] == "0" and state["ActiveState"] == "active", state
    assert '"ok": true' in check.read_text().replace('":true', '": true')


def test_a1_busy_longer_than_the_wait_fails_with_75_then_a_restart_comes_up(make_unit, tmp_path):
    unit, flag, check = unit_with_wait(make_unit, tmp_path, 12.0, start_timeout="120s")
    timer = free_after(flag, 20)
    try:
        unit.start(wait=False)
        state, seconds = unit.wait_for(lambda: (s := unit.state())["Result"] == "exit-code" and s, 40, what="the 75 exit")
        assert 12 <= seconds <= 12 + 5 + 8, seconds     # the wait ends at 12 s (+ one interval)
        found, journal = unit.journal_has("gpu-free")    # --quiet prints the failed ids (journald lags the exit a moment)
        assert found, journal
        unit.wait_ready(60)                              # RestartSec 1 s, then the GPU is free
    finally:
        timer.cancel()
    assert int(unit.state()["NRestarts"]) >= 1
