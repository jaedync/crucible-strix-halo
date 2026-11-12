"""The watchdog (server/sd_notify.py Watchdog) with a fake clock, and the batch ping and idle clock of the backends."""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.api import validate_request
from server.batch_backend import BatchedEngineBackend
from server.sd_notify import Watchdog, proc_state

FAKE = Path(__file__).parent / "fake_batch_engine.py"


class Clock:
    def __init__(self): self.now = 1000.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


class Notes:
    watchdog_seconds = 30.0
    def __init__(self): self.sent = []
    def send(self, text): self.sent.append(text); return True
    def feeds(self): return sum(1 for t in self.sent if t.startswith("WATCHDOG=1"))


class Server:
    def __init__(self, clock): self.http_alive_at = clock(); self.shut = 0
    def shutdown(self): self.shut += 1


class Backend:
    """What the watchdog reads from a backend."""
    def __init__(self, clock, ping_op=False):
        self.fatal = None
        self.last_event_at = clock()
        self.awaiting = False
        self.ping_op = ping_op
        self.process = SimpleNamespace(pid=4242)
    def session_counts(self): return (1, 4)


def build(ping_op=False, stall=50, state="S"):
    clock = Clock()
    notes, server, backend = Notes(), Server(clock), Backend(clock, ping_op)
    states = {"state": state}
    dog = Watchdog(notes, backend, server, stall_seconds=stall, clock=clock, proc_state=lambda pid: states["state"], log=lambda text: None)
    return clock, notes, server, backend, dog, states


def run(clock, dog, seconds, backend=None, events_every=None, http=True, server=None):
    """Tick once a second for `seconds`; optionally an engine event each `events_every` s and an HTTP loop tick."""
    for step in range(int(seconds)):
        clock.advance(1)
        if events_every and step % events_every == 0 and backend: backend.last_event_at = clock()
        if http and server: server.http_alive_at = clock()
        dog.tick()


class WatchdogTest(unittest.TestCase):
    def test_busy_engine_with_events_keeps_feeding_each_third_of_the_interval(self):
        clock, notes, server, backend, dog, _ = build()
        backend.awaiting = True
        run(clock, dog, 60, backend, events_every=3, server=server)
        self.assertGreaterEqual(notes.feeds(), 6)   # first tick + one each 10 s
        self.assertLessEqual(notes.feeds(), 7)
        self.assertFalse(dog.fatal_exit)

    def test_idle_ping_engine_that_answers_keeps_feeding(self):
        clock, notes, server, backend, dog, _ = build(ping_op=True)
        run(clock, dog, 120, backend, events_every=10, server=server)
        self.assertGreaterEqual(notes.feeds(), 11)

    def test_silent_busy_engine_stops_the_feed_after_the_stall_and_says_so_once(self):
        clock, notes, server, backend, dog, _ = build()
        backend.awaiting = True
        run(clock, dog, 20, backend, events_every=1, server=server)   # healthy
        before = notes.feeds()
        run(clock, dog, 120, None, server=server)                     # the engine goes silent
        # Feeds continue only while the silence is at most 50 s: at most 5 more (one each 10 s).
        self.assertLessEqual(notes.feeds() - before, 6)
        silent = [t for t in notes.sent if t.startswith("STATUS=engine silent for")]
        self.assertEqual(len(silent), 1, notes.sent)
        run(clock, dog, 60, None, server=server)
        count = notes.feeds()
        run(clock, dog, 30, None, server=server)
        self.assertEqual(notes.feeds(), count)                        # no feed any more

    def test_feed_resumes_when_events_return(self):
        clock, notes, server, backend, dog, _ = build()
        backend.awaiting = True
        run(clock, dog, 90, None, server=server)
        count = notes.feeds()
        backend.last_event_at = clock()
        run(clock, dog, 12, backend, events_every=2, server=server)
        self.assertGreater(notes.feeds(), count)

    def test_idle_old_engine_alive_by_proc_state_moves_the_clock(self):
        for state in ("R", "S", "D"):
            clock, notes, server, backend, dog, states = build(ping_op=False, state=state)
            run(clock, dog, 200, None, server=server)
            self.assertGreaterEqual(notes.feeds(), 19, state)
            self.assertFalse([t for t in notes.sent if "silent" in t], state)

    def test_idle_old_engine_stopped_or_zombie_stops_the_feed(self):
        for state in ("T", "t", "Z", "X", None):
            clock, notes, server, backend, dog, states = build(ping_op=False, state="S")
            run(clock, dog, 30, None, server=server)
            states["state"] = state
            before = notes.feeds()
            run(clock, dog, 150, None, server=server)
            self.assertLessEqual(notes.feeds() - before, 6, state)
            self.assertTrue([t for t in notes.sent if t.startswith("STATUS=engine silent")], state)

    def test_proc_state_does_not_count_while_a_command_is_in_flight(self):
        clock, notes, server, backend, dog, states = build(ping_op=False, state="S")
        backend.awaiting = True   # a deadlocked command loop is in state S too
        run(clock, dog, 150, None, server=server)
        self.assertTrue([t for t in notes.sent if t.startswith("STATUS=engine silent")])

    def test_ping_engine_does_not_use_proc_state(self):
        clock, notes, server, backend, dog, states = build(ping_op=True, state="S")
        run(clock, dog, 150, None, server=server)   # no event ever: the ping is not answered
        self.assertTrue([t for t in notes.sent if t.startswith("STATUS=engine silent")])

    def test_http_loop_stall_stops_the_feed(self):
        clock, notes, server, backend, dog, _ = build()
        run(clock, dog, 20, backend, events_every=1, server=server)
        before = notes.feeds()
        run(clock, dog, 60, backend, events_every=1, http=False, server=server)
        self.assertLessEqual(notes.feeds() - before, 2)

    def test_fatal_backend_shuts_the_server_down_and_asks_for_exit_3(self):
        clock, notes, server, backend, dog, _ = build()
        backend.fatal = "The engine stopped. Check its stderr log."
        self.assertFalse(dog.tick())
        self.assertEqual(server.shut, 1)
        self.assertTrue(dog.fatal_exit)
        self.assertTrue(any(t.startswith("STATUS=engine stopped: The engine stopped") for t in notes.sent))
        self.assertEqual(notes.feeds(), 0)

    def test_status_line_names_sessions_and_the_last_answer(self):
        clock, notes, server, backend, dog, _ = build(ping_op=True)
        clock.advance(3); server.http_alive_at = clock(); dog.tick()
        status = [t for t in notes.sent if t.startswith("STATUS=ready")]
        self.assertEqual(status, ["STATUS=ready: 1 of 4 sessions busy; engine answered 3 s ago"])

    def test_no_watchdog_interval_means_no_feed_but_fatal_still_works(self):
        clock, notes, server, backend, dog, _ = build()
        notes.watchdog_seconds = None
        dog = Watchdog(notes, backend, server, stall_seconds=50, clock=clock, proc_state=lambda pid: "S", log=lambda text: None)
        run(clock, dog, 30, backend, events_every=1, server=server)
        self.assertEqual(notes.feeds(), 0)
        backend.fatal = "x"
        self.assertFalse(dog.tick())

    def test_start_and_stop_run_the_loop_in_a_thread(self):
        notes, backend, server = Notes(), Backend(time.monotonic), Server(time.monotonic)
        backend.process = SimpleNamespace(pid=os.getpid())
        dog = Watchdog(notes, backend, server, stall_seconds=10, check_seconds=.01, log=lambda text: None)
        dog.start()
        time.sleep(.2)
        dog.stop()
        self.assertGreaterEqual(notes.feeds(), 1)
        self.assertFalse(dog.thread.is_alive())

    def test_proc_state_reads_the_third_field_even_with_a_bracket_in_the_name(self):
        self.assertEqual(proc_state(os.getpid()), "R")  # this process is running
        with patch("server.sd_notify.Path.read_text", return_value="12 (a) b (c) T 1 2 3"):
            self.assertEqual(proc_state(12), "T")
        self.assertIsNone(proc_state(2 ** 22 + 12345))


class PingTest(unittest.TestCase):
    def start(self, **environment):
        popen = subprocess.Popen
        env = dict(environment)
        with patch.dict(os.environ, env), \
                patch("server.backend.subprocess.Popen", side_effect=lambda *a, **k: popen([sys.executable, str(FAKE)], **k)):
            backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3)
        self.addCleanup(backend.close)
        return backend

    def log(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []

    def test_old_engine_gets_no_ping(self):
        with self.subTest(), __import__("tempfile").TemporaryDirectory() as folder:
            path = os.path.join(folder, "log")
            backend = self.start(FAKE_COMMAND_LOG=path)
            self.assertFalse(backend.ping_op)
            backend.ping_idle_seconds = .05
            time.sleep(.5)
            self.assertEqual([c for c in self.log(path) if c["op"] == "ping"], [])

    def test_ping_engine_is_pinged_when_idle(self):
        with __import__("tempfile").TemporaryDirectory() as folder:
            path = os.path.join(folder, "log")
            backend = self.start(FAKE_COMMAND_LOG=path, FAKE_PING="1")
            self.assertTrue(backend.ping_op)
            backend.ping_idle_seconds = .1
            before = backend.last_event_at
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and len([c for c in self.log(path) if c["op"] == "ping"]) < 3: time.sleep(.05)
            pings = [c for c in self.log(path) if c["op"] == "ping"]
            self.assertGreaterEqual(len(pings), 3)
            self.assertEqual(pings[0], {"op": "ping"})
            self.assertGreater(backend.last_event_at, before)
            self.assertIsNone(backend.fatal)

    def test_a_wrong_ping_reply_is_fatal(self):
        with __import__("tempfile").TemporaryDirectory() as folder:
            backend = self.start(FAKE_PING="1", FAKE_PING_REPLY="wrong")
            backend.ping_idle_seconds = .05
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not backend.fatal: time.sleep(.05)
            self.assertTrue(backend.fatal)

    def test_a_request_still_works_with_ping_on(self):
        backend = self.start(FAKE_PING="1")
        backend.ping_idle_seconds = .05
        request = validate_request(dict(prompt="x", max_tokens=4, spec=0), "completion", "crucible")
        time.sleep(.2)
        job = backend.submit([1], request)
        ids = []
        while True:
            event = job.events.get(timeout=5)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done": break
            ids.append(event["id"])
        self.assertEqual(ids, [2, 3, 4, 5])

    def test_last_event_at_moves_with_every_engine_line(self):
        backend = self.start()
        first = backend.last_event_at
        time.sleep(.05)
        request = validate_request(dict(prompt="x", max_tokens=2, spec=0), "completion", "crucible")
        job = backend.submit([1], request)
        while job.events.get(timeout=5)["event"] != "done": pass
        self.assertGreater(backend.last_event_at, first)

    def test_awaiting_is_false_when_idle_and_true_during_a_slow_command(self):
        backend = self.start(FAKE_COMMAND_SECONDS="1.0")
        self.assertFalse(backend.awaiting)
        request = validate_request(dict(prompt="x", max_tokens=2, spec=0), "completion", "crucible")
        job = backend.submit([1], request)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not backend.awaiting: time.sleep(.01)
        self.assertTrue(backend.awaiting)
        while job.events.get(timeout=10)["event"] != "done": pass
        time.sleep(.1)
        self.assertFalse(backend.awaiting)

    def test_largest_busy_gap_is_logged_once_per_new_maximum(self):
        import io
        from contextlib import redirect_stderr
        errors = io.StringIO()
        with redirect_stderr(errors):
            backend = self.start(FAKE_COMMAND_SECONDS="1.2")
            request = validate_request(dict(prompt="x", max_tokens=2, spec=0), "completion", "crucible")
            job = backend.submit([1], request)
            while job.events.get(timeout=10)["event"] != "done": pass
        lines = [l for l in errors.getvalue().splitlines() if l.startswith("watchdog: largest busy engine gap ")]
        self.assertTrue(lines, errors.getvalue())
        import re
        self.assertRegex(lines[-1], r"^watchdog: largest busy engine gap \d+\.\d s$")
        self.assertGreaterEqual(float(lines[-1].split()[-2]), 1.0)
        self.assertEqual(len(lines), len(set(lines)))

    def test_session_counts(self):
        backend = self.start()
        self.assertEqual(backend.session_counts(), (0, 2))


class HttpClockTest(unittest.TestCase):
    def test_a_new_server_starts_with_a_fresh_http_clock(self):
        # Before the first service_actions call the watchdog must not read "silent since boot" (T5a dry run).
        from server.__main__ import ThreadingHTTPServer
        server = ThreadingHTTPServer(("127.0.0.1", 0), SimpleNamespace)
        try:
            self.assertLess(time.monotonic() - server.http_alive_at, 5)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
