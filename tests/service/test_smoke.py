"""The harness itself: a server in a user unit reaches READY, serves a request, shows its status and stops with exit 0."""
from svc_harness import Unit


def test_ready_serves_and_stops_clean(make_unit):
    unit = make_unit(env={"FAKE_PING": "1"})
    seconds = unit.start()
    assert seconds < 20
    state = unit.state()
    assert state["ActiveState"] == "active" and state["NRestarts"] == "0"
    status, body = unit.post([100, 101], max_tokens=3)
    assert status == 200, body
    unit.wait_for(lambda: unit.state()["StatusText"].startswith("ready: "), 15, what="status text")
    assert "sessions busy" in unit.state()["StatusText"]
    assert "silent" not in unit.journal(100)         # no false watchdog alarm after READY
    took = unit.stop()
    assert took < 20
    state = unit.state()
    assert state["ActiveState"] == "inactive" and state["Result"] == "success"
