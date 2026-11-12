"""Fault tests F1 to F6 and F9 to F12 of the service design: each fault has a pass rule.

Timing is scaled (WatchdogSec 3 s, stall 10 s) unless the test says real_values; the unit text is the shipped one."""
import json
import os
import signal
import threading
import time

import pytest

LONG = list(range(1000, 1000 + 16384 + 50))   # long enough for the disk tier (STAGE_MIN_TOKENS 16384)
SCALED_FOUND = 10 + 3 + 6      # stall + WatchdogSec + margin
REAL_FOUND = 90                # design target: stall 50 s + WatchdogSec 30 s + margin


def first_failure(unit, timeout):
    """The unit state at the first moment Result is not success (the kill is seen)."""
    state, seconds = unit.wait_for(lambda: (s := unit.state())["Result"] != "success" and s, timeout, what="a failed run")
    return state, seconds


def start_decode(unit, max_tokens=300):
    return unit.stream_first_token([100, 101], max_tokens)


def no_core_file(unit):
    from svc_harness import REPO
    assert not [p for p in REPO.glob("core*") if p.is_file() and p.stat().st_mtime > time.time() - 600]


# F1 -------------------------------------------------------------------------------------------------------------
def test_f1_kill_9_of_the_engine_gives_exit_3_and_a_restart(make_unit):
    unit = make_unit(env={"FAKE_LOAD_SECONDS": "1"})
    unit.start()
    engine = unit.engine_pid()
    os.kill(engine, signal.SIGKILL)
    began = time.monotonic()
    state, _ = first_failure(unit, 20)
    assert state["ExecMainStatus"] == "3" and state["Result"] == "exit-code", state
    ready = unit.wait_ready(30)
    assert time.monotonic() - began <= 1 + 1 + 15        # load + RestartSec + margin of the design
    assert unit.state()["NRestarts"] == "1"
    assert unit.engine_pid() != engine


# F2 -------------------------------------------------------------------------------------------------------------
def test_f2_kill_9_of_the_server_lets_the_engine_flush(make_unit):
    unit = make_unit(state=True, ctx=20000, env={"FAKE_FLUSH_DELAY": "1"})
    unit.start()
    first = LONG
    assert unit.post(first, max_tokens=2)[0] == 200
    os.kill(unit.server_pid(), signal.SIGKILL)
    state, _ = first_failure(unit, 20)
    assert state["Result"] == "signal", state
    unit.wait_ready(40)
    assert unit.state()["NRestarts"] == "1"
    assert first in json.loads(unit.disk_file.read_text())        # the flush file holds the finished prompt
    unit.commands_clear()
    status, body = unit.post(first + [102, 103], max_tokens=2)    # the next start reads it back from the disk file
    assert status == 200, body
    ops = [c["op"] for c in unit.commands()]
    assert "stage" in ops and ops.index("stage") < ops.index("open"), ops     # the restore ran before the open
    assert body["usage"].get("prompt_tokens_details", {}).get("cached_tokens", len(first)) == len(first), body["usage"]


# F3 / F4 --------------------------------------------------------------------------------------------------------
def stop_and_time(unit, engine, limit):
    began = time.monotonic()
    os.kill(engine, signal.SIGSTOP)
    state, _ = first_failure(unit, limit + 20)
    found = time.monotonic() - began
    assert state["Result"] == "watchdog", state
    return found


def test_f3_sigstop_of_the_engine_during_a_decode(make_unit):
    unit = make_unit(stall=10, env={"FAKE_DELAY": "0.05", "FAKE_PING": "1"})
    unit.start()
    engine = unit.engine_pid()
    connection, _ = start_decode(unit)
    found = stop_and_time(unit, engine, SCALED_FOUND)
    connection.close()
    assert found <= SCALED_FOUND, found
    unit.wait_ready(30)
    assert unit.state()["NRestarts"] == "1"
    unit.wait_for(lambda: not unit.alive(engine), 10, what="the killed engine process gone")  # SIGKILL: wait for the exit
    no_core_file(unit)


@pytest.mark.real_values
def test_f3_real_values(make_unit):
    unit = make_unit(real=True, env={"FAKE_DELAY": "0.05", "FAKE_PING": "1"})
    unit.start()
    engine = unit.engine_pid()
    connection, _ = start_decode(unit)
    found = stop_and_time(unit, engine, REAL_FOUND)
    connection.close()
    assert found <= REAL_FOUND, found
    unit.wait_for(lambda: not unit.alive(engine), 10, what="the killed engine process gone")  # SIGKILL: wait for the exit
    unit.wait_ready(60)
    no_core_file(unit)


@pytest.mark.parametrize("ping", ["1", "0"], ids=["with-ping", "without-ping"])
def test_f4_sigstop_of_an_idle_engine(make_unit, ping):
    unit = make_unit(stall=10, env={"FAKE_PING": ping})
    unit.start()
    time.sleep(3)
    engine = unit.engine_pid()
    found = stop_and_time(unit, engine, SCALED_FOUND)
    assert found <= SCALED_FOUND, found
    unit.wait_ready(30)
    assert unit.state()["NRestarts"] == "1"


@pytest.mark.real_values
def test_f4_real_values_without_ping(make_unit):
    unit = make_unit(real=True)
    unit.start()
    time.sleep(3)
    engine = unit.engine_pid()
    found = stop_and_time(unit, engine, REAL_FOUND)
    assert found <= REAL_FOUND, found
    unit.wait_ready(60)


# F5 -------------------------------------------------------------------------------------------------------------
@pytest.mark.real_values
def test_f5_idle_three_minutes_with_real_values(make_unit):
    unit = make_unit(real=True, env={"FAKE_PING": "1"})
    unit.start()
    time.sleep(180)
    assert unit.state()["NRestarts"] == "0" and unit.state()["ActiveState"] == "active"
    pings = [c for c in unit.commands() if c["op"] == "ping"]
    assert len(pings) >= 15, len(pings)


# F6 -------------------------------------------------------------------------------------------------------------
def test_f6_slow_commands_do_not_cause_a_false_restart(make_unit):
    # Every command takes stall - 10 s: a long prefill chunk must never look like a hang.
    unit = make_unit(stall=20, watchdog="6s", env={"FAKE_COMMAND_SECONDS": "10", "FAKE_PING": "1", "FAKE_PREFILL_ROWS": "1"})
    unit.start()
    status, body = unit.post([100, 101, 102], max_tokens=2, timeout=200)
    assert status == 200, body
    state = unit.state()
    assert state["NRestarts"] == "0" and state["ActiveState"] == "active", state


# F9 -------------------------------------------------------------------------------------------------------------
def test_f9_stop_during_a_long_prefill_loses_no_finished_prompt(make_unit):
    unit = make_unit(state=True, ctx=20000, env={"FAKE_COMMAND_SECONDS": "0.5", "FAKE_PREFILL_ROWS": "1", "FAKE_FLUSH_DELAY": "1"})
    unit.start()
    finished = [100, 101]
    assert unit.post(finished, max_tokens=2)[0] == 200
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("r", unit.post(list(range(100, 124)), max_tokens=2)), daemon=True)
    thread.start()
    time.sleep(2)
    took = unit.stop()
    thread.join(timeout=30)
    assert took <= 120, took
    state = unit.state()
    assert state["Result"] == "success" and state["ExecMainStatus"] == "0", state
    assert "SIGKILL" not in unit.journal(200)
    assert finished in json.loads(unit.disk_file.read_text())      # 0 lost states


# F10 ------------------------------------------------------------------------------------------------------------
def test_f10_crash_loop_ends_in_start_limit_hit(make_unit):
    unit = make_unit(env={"FAKE_EXIT_AFTER_READY": "1"})
    unit.start(wait=False)
    unit.wait_for(lambda: unit.state()["Result"] == "start-limit-hit", 90, interval=0.2, what="start-limit-hit")
    state = unit.state()
    assert state["ActiveState"] == "failed", state
    assert int(state["NRestarts"]) <= 4


# F11 ------------------------------------------------------------------------------------------------------------
def test_f11_hung_load_ends_in_timeout_and_counts_for_the_start_limit(make_unit):
    unit = make_unit(env={"FAKE_LOAD_SECONDS": "300"}, start_timeout="4s")
    unit.start(wait=False)
    unit.wait_for(lambda: unit.state()["Result"] == "timeout", 30, interval=0.1, what="timeout")
    unit.wait_for(lambda: unit.state()["Result"] == "start-limit-hit", 90, interval=0.2, what="start-limit-hit")


# F12 ------------------------------------------------------------------------------------------------------------
def test_f12_sigterm_to_the_whole_group_flushes_and_exits_without_a_restart(make_unit):
    unit = make_unit(state=True, ctx=20000, env={"FAKE_FLUSH_DELAY": "1"})     # FAKE_IGNORE_TERM=1 is set by state=True
    unit.start()
    finished = [100, 101]
    assert unit.post(finished, max_tokens=2)[0] == 200
    from svc_harness import systemctl
    systemctl("kill", "--kill-whom=all", "--signal=SIGTERM", unit.unit, check=True)
    unit.wait_for(lambda: unit.state()["ActiveState"] == "inactive", 60, interval=0.2, what="inactive")
    state = unit.state()
    assert state["Result"] == "success" and state["NRestarts"] == "0", state
    time.sleep(3)
    assert unit.state()["ActiveState"] == "inactive"                # no restart
    assert finished in json.loads(unit.disk_file.read_text())
