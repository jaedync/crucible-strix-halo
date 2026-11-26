"""tests/testports.py: two test runs at the same time never get the same port (Nov 21 Errno 98 at fe79a075)."""
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testports  # noqa: E402

HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    import testports
    ports = [int(p) for p in sys.argv[3].split(",")]
    with testports.range_port(ports, wait_s=5, lock_dir=sys.argv[2]) as port:
        print(port, flush=True)
        sys.stdin.readline()
""")


def two_ports():
    a, b = socket.socket(), socket.socket()
    try:
        a.bind(("127.0.0.1", 0))
        b.bind(("127.0.0.1", 0))
        return [a.getsockname()[1], b.getsockname()[1]]
    finally:
        a.close()
        b.close()


def test_free_port_is_bindable():
    assert testports.bindable(testports.free_port())


def test_a_port_that_another_process_holds_is_not_given_again(tmp_path):
    ports = two_ports()
    holder = subprocess.Popen([sys.executable, "-I", "-c", HOLDER, str(Path(__file__).parent), str(tmp_path),
                               ",".join(map(str, ports))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        held = int(holder.stdout.readline())
        assert held == ports[0]
        with testports.range_port(ports, wait_s=1, lock_dir=str(tmp_path)) as mine:
            assert mine == ports[1]
    finally:
        holder.communicate("\n", timeout=10)
    assert holder.returncode == 0
    with testports.range_port(ports, wait_s=1, lock_dir=str(tmp_path)) as again:   # the holder ended: its lock is gone
        assert again == ports[0]


def test_a_port_with_a_listener_is_skipped(tmp_path):
    ports = two_ports()
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", ports[0]))
        busy.listen()
        assert not testports.bindable(ports[0])
        with testports.range_port(ports, wait_s=1, lock_dir=str(tmp_path)) as port:
            assert port == ports[1]


def test_every_port_taken_waits_then_raises(tmp_path):
    ports = two_ports()
    with testports.range_port(ports, wait_s=1, lock_dir=str(tmp_path)), \
            testports.range_port(ports, wait_s=1, lock_dir=str(tmp_path)):
        with pytest.raises(TimeoutError, match="no free port"):
            with testports.range_port(ports, wait_s=0.3, lock_dir=str(tmp_path)):
                pass


def test_lock_dir_is_one_folder_for_every_run_of_the_user():
    assert testports.LOCK_DIR == f"/tmp/crucible-test-ports-{os.getuid()}"
