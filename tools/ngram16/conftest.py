import pytest

import common


@pytest.fixture(autouse=True)
def _test_host(monkeypatch):
    # tmpfs has no O_DIRECT, and the test host's free memory must not gate the tests.
    monkeypatch.setattr(common, "DIRECT", False)
    monkeypatch.setattr(common, "MEM_START_KB", 0)
    monkeypatch.setattr(common, "MEM_PAUSE_KB", 0)
