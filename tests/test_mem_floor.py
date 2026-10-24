"""CPU tests for tools/mem_floor.sh (the MemAvailable check inside a lease). Moved from the train chain tests."""
import os
from pathlib import Path
import subprocess
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/mem_floor.sh"


def mem_floor(tmp_path, floor, available_kb, wait="0"):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal:       131072000 kB\nMemAvailable:   {available_kb} kB\n")
    log = tmp_path / "d305-off.mem"
    run = subprocess.run(["bash", str(SCRIPT), str(floor), str(log)],
                         env={**os.environ, "MEM_FLOOR_MEMINFO": str(meminfo), "MEM_FLOOR_WAIT_S": wait},
                         capture_output=True, text=True)
    return run.returncode, log.read_text()


def test_mem_floor_is_executable_in_git():
    # the chain calls it with bash, but an operator runs it directly.
    if not (REPO / ".git").exists():  # a git archive (cpu_checks.sh) keeps the mode bits of the commit
        assert os.access(SCRIPT, os.X_OK)
        return
    mode = subprocess.run(["git", "ls-files", "-s", "tools/mem_floor.sh"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.split()
    assert mode[0] == "100755"


def test_mem_floor_polls_until_the_memory_is_free(tmp_path):
    # a check right after the previous soak's lease saw 109.9 GB; a few minutes later
    # 128.4 GB was free. The floor polls for up to MEM_FLOOR_WAIT_S before it gives rc 76.
    meminfo, log = tmp_path / "meminfo", tmp_path / "x.mem"
    meminfo.write_text("MemAvailable:   109922876 kB\n")
    proc = subprocess.Popen(["bash", str(SCRIPT), "126", str(log)],
                            env={**os.environ, "MEM_FLOOR_MEMINFO": str(meminfo), "MEM_FLOOR_WAIT_S": "30",
                                 "MEM_FLOOR_POLL_S": "0.1"}, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    assert proc.poll() is None
    meminfo.write_text("MemAvailable:   128400000 kB\n")
    assert proc.wait(timeout=20) == 0
    lines = log.read_text().splitlines()
    assert "MemAvailable 109922876 kB floor 126 GB below: wait up to 30 s" in lines[0]
    assert "MemAvailable 128400000 kB floor 126 GB start after " in lines[-1]


def test_mem_floor_gives_rc_76_when_the_wait_ends(tmp_path):
    meminfo, log = tmp_path / "meminfo", tmp_path / "x.mem"
    meminfo.write_text("MemAvailable:   100 kB\n")
    begin = time.monotonic()
    run = subprocess.run(["bash", str(SCRIPT), "126", str(log)],
                         env={**os.environ, "MEM_FLOOR_MEMINFO": str(meminfo), "MEM_FLOOR_WAIT_S": "1",
                              "MEM_FLOOR_POLL_S": "0.2"}, capture_output=True, text=True, timeout=20)
    assert run.returncode == 76 and time.monotonic() - begin >= 1
    assert "MemAvailable 100 kB floor 126 GB below: no start after " in log.read_text().splitlines()[-1]


@pytest.mark.parametrize("name,value", [("MEM_FLOOR_WAIT_S", "-1"), ("MEM_FLOOR_WAIT_S", "10m"),
                                        ("MEM_FLOOR_WAIT_S", "86401"), ("MEM_FLOOR_POLL_S", "0"),
                                        ("MEM_FLOOR_POLL_S", "x")])
def test_mem_floor_refuses_a_bad_wait(tmp_path, name, value):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   200000000 kB\n")
    run = subprocess.run(["bash", str(SCRIPT), "126", str(tmp_path / "x.mem")],
                         env={**os.environ, "MEM_FLOOR_MEMINFO": str(meminfo), name: value}, capture_output=True,
                         text=True)
    assert run.returncode == 2 and name in run.stderr, run


def test_mem_floor_logs_and_stops_below_the_floor(tmp_path):
    # a concurrent build left 33.06 GB at engine ready and the 4 short soaks did not start. The engine
    # load takes about 92.6 GB, so the recommended pre-start floor is 126 GB (126 x 10^6 kB; kB is KiB in /proc/meminfo).
    code, text = mem_floor(tmp_path, 126, 129_500_000)
    assert code == 0 and "MemAvailable 129500000 kB floor 126 GB start" in text
    code, text = mem_floor(tmp_path, 126, 125_000_000)
    assert code == 76 and "MemAvailable 125000000 kB floor 126 GB below: no start" in text
