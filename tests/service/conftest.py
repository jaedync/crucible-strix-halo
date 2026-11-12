"""Fault tests that run the server under a real systemd user manager (make test-service).

They are collected only when CRUCIBLE_SERVICE_TESTS=1 (the test-fast run does not see them). They need `systemctl --user`
and a running user manager: a sandboxed build worker has none, so run them in a normal user session."""
import os
import shutil
import subprocess

import pytest

if os.environ.get("CRUCIBLE_SERVICE_TESTS") != "1":
    collect_ignore_glob = ["test_*.py"]


def pytest_configure(config):
    config.addinivalue_line("markers", "real_values: runs with the unit's real timing (a few minutes)")


def user_manager_ready():
    if not shutil.which("systemctl") or not shutil.which("systemd-run"):
        return False
    result = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
    return result.stdout.strip() in ("running", "degraded")


@pytest.fixture(scope="session", autouse=True)
def need_user_manager():
    if not user_manager_ready():
        pytest.skip("no systemd user manager (a sandbox has none)")


@pytest.fixture
def make_unit(tmp_path):
    from svc_harness import Unit
    units = []

    def make(**options):
        unit = Unit(tmp_path, **options)
        units.append(unit)
        return unit
    yield make
    for unit in units:
        unit.cleanup()
