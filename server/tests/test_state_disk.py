"""The disk tier for conversation states: the server stages a disk read before an
open, and a stop lets the engine write its states, so a restarted server continues a session without a cold prefill."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from server.api import validate_request
from server.engine_messages import ProtocolError
from server import batch_backend
from server.batch_backend import (BatchedEngineBackend, STAGE_AHEAD_POLL_SECONDS, STAGE_MIN_TOKENS, STAGE_POLL_SECONDS,
                                  STAGE_REFRESH_SECONDS, STATE_FLUSH_SECONDS)
# The fix adds this constant; the tests of today's code use a stand-in, so they fail on behavior.
STAGE_NONE_POLL_SECONDS = getattr(batch_backend, "STAGE_NONE_POLL_SECONDS", 5.0)

FAKE = Path(__file__).parent / "fake_batch_engine.py"


class StateDiskTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.disk = Path(self.dir.name) / "disk.json"
        self.commands = Path(self.dir.name) / "commands.jsonl"
        self.request = validate_request(dict(prompt="x", max_tokens=4, spec=0), "completion", "crucible")
        self.backends = []

    def tearDown(self):
        for backend in self.backends: backend.close()
        self.dir.cleanup()

    def start(self, state_disk="1", flush_delay="0", ignore_term="0", **options):
        popen = subprocess.Popen
        env = dict(FAKE_STATE_DISK=state_disk, FAKE_DISK_FILE=str(self.disk), FAKE_COMMAND_LOG=str(self.commands),
                   FAKE_FLUSH_DELAY=flush_delay, FAKE_SLOT_CONTEXTS="40000,40000", FAKE_IGNORE_TERM=ignore_term)
        with patch.dict(os.environ, env), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            backend = BatchedEngineBackend("unused", "unused", context=40000, streams=2, conversation_cache_mib=64,
                                           session_replay=True, state_dir=str(Path(self.dir.name) / "states"),
                                           **options)
        self.backends.append(backend)
        return backend

    def run_job(self, backend, ids):
        job = backend.submit(ids, self.request)
        while True:
            event = job.events.get(timeout=10)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done": return job

    def ops(self):
        if not self.commands.exists(): return []
        return [json.loads(line)["op"] for line in self.commands.read_text().splitlines()]

    def test_restart_resumes_from_disk(self):
        ids = list(range(1, STAGE_MIN_TOKENS + 100))
        first = self.start(flush_delay="0.4")
        self.assertTrue(first.state_disk)
        self.run_job(first, ids)
        start = time.monotonic()
        first.close()
        # The server closed stdin and waited for the engine to write its states; it did not stop it with a signal.
        self.assertGreaterEqual(time.monotonic() - start, 0.4)
        self.assertEqual(first.process.returncode, 0)
        self.assertEqual(json.loads(self.disk.read_text()), [ids])
        self.commands.unlink()
        second = self.start()
        job = self.run_job(second, ids + [7, 8, 9])
        self.assertEqual(job.turn.get("opened_cached"), len(ids))
        ops = self.ops()
        # The read ran before the open: two polls while it read, then the ready reply.
        self.assertEqual(ops[:ops.index("open")], ["stage", "stage", "stage"])
        self.assertEqual(job.turn.get("state_disk"), "ready")

    def test_short_prompts_skip_the_stage_command(self):
        backend = self.start()
        self.run_job(backend, [1, 2, 3])
        self.assertNotIn("stage", self.ops())

    def test_an_engine_without_the_tier_gets_no_stage_command(self):
        backend = self.start(state_disk="0")
        self.assertFalse(backend.state_disk)
        self.run_job(backend, list(range(1, STAGE_MIN_TOKENS + 10)))
        self.assertNotIn("stage", self.ops())

    def test_options_are_forwarded(self):
        with patch("server.batch_backend.EngineBackend.__init__") as initialize:
            BatchedEngineBackend("unused", "unused", streams=2, conversation_cache_mib=64, session_replay=True,
                                 state_dir="/home/u/states", state_disk_gib=100, state_write_gib_per_day=0,
                                 state_write_delay_ms=3000)
        options = initialize.call_args.kwargs["engine_options"]
        # Fix 2: the same-second proof's write delay (0 never goes out: older engines see the same options).
        self.assertEqual(options[options.index("--state-disk-write-delay-ms") + 1], "3000")
        self.assertEqual(options[options.index("--state-disk-write-gib-per-day") + 1], "0")
        self.assertEqual(options[options.index("--state-dir") + 1], "/home/u/states")
        self.assertEqual(options[options.index("--state-disk-gib") + 1], "100")
        with patch("server.batch_backend.EngineBackend.__init__") as initialize:
            BatchedEngineBackend("unused", "unused", streams=2, conversation_cache_mib=64, cache_probation=0,
                                 cache_idle_minutes=5)
        options = initialize.call_args.kwargs["engine_options"]
        self.assertEqual(options[options.index("--cache-probation") + 1], "0")
        self.assertEqual(options[options.index("--cache-idle-minutes") + 1], "5")
        for bad in (dict(state_dir="relative"), dict(cache_probation=2), dict(cache_probation=1, conversation_cache_mib=0, session_replay=False), dict(state_dir="/x", session_replay=False),
                    dict(state_disk_gib=10), dict(state_dir="/x", state_disk_gib=0),
                    dict(state_write_gib_per_day=10), dict(state_dir="/x", state_write_gib_per_day=-1),
                    dict(state_write_delay_ms=10), dict(state_dir="/x", state_write_delay_ms=60001),
                    dict(cache_idle_minutes=0), dict(cache_idle_minutes=10081),
                    dict(cache_idle_minutes=5, conversation_cache_mib=0, session_replay=False)):
            arguments = dict(streams=2, conversation_cache_mib=64, session_replay=True) | bad
            with self.assertRaises(ValueError), patch("server.batch_backend.EngineBackend.__init__"):
                BatchedEngineBackend("unused", "unused", **arguments)


    def test_a_ready_read_is_kept_while_the_turn_waits_for_a_slot(self):
        """the engine drops a ready read that no poll touched for 60 s, so a turn that waits for a
        slot polls its ready read again; a read that the engine started again counts as a new wait."""
        backend = object.__new__(BatchedEngineBackend)
        backend.staging, backend.queued_polls = {}, {}
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto")
        job.cancelled = type("Event", (), {"is_set": lambda self: False})()
        job.turn = {}
        replies = iter(["ready", "ready", "reading", "ready"])
        backend._stage = lambda payload, *args: batch_backend.StageReply(next(replies))
        clock = [0.0]
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertTrue(backend._staged(job))
            clock[0] = STAGE_REFRESH_SECONDS / 2
            self.assertTrue(backend._staged(job))           # no poll yet
            clock[0] = STAGE_REFRESH_SECONDS + 1
            self.assertTrue(backend._staged(job))           # a poll keeps the read state ("ready")
            clock[0] = 2 * STAGE_REFRESH_SECONDS + 2
            self.assertFalse(backend._staged(job))          # the engine reads it again: the turn waits
            self.assertEqual(job.turn["state_disk_wait_s"], 0)
            clock[0] += 1
            self.assertTrue(backend._staged(job))
        self.assertLess(STAGE_REFRESH_SECONDS, 60 / 2)  # engine/runtime/state_disk.h stage_seconds

    def test_a_turn_that_waits_for_a_slot_keeps_its_read(self):
        """the exact path. A turn with no free slot never reaches _staged in _admit, so only
        _stage_ahead can touch its read. The model engine drops a ready read that no stage touched for 60 s, and a
        stage of a dropped read starts a new one. The turn waits 300 s for a slot: one read only, and it opens on it."""
        import queue, threading
        from collections import deque
        clock = [0.0]

        class Engine:
            def __init__(self): self.reads, self.started, self.touched = 0, None, None
            def state(self):
                if self.started is not None and clock[0] - self.started >= 1 and clock[0] - self.touched > 60:
                    self.started = None  # the engine dropped the ready read (state_disk.h Expire)
                if self.started is None:
                    self.reads += 1
                    self.started = clock[0]
                self.touched = clock[0]
                return "ready" if clock[0] - self.started >= 1 else "reading"

        engine = Engine()
        backend = object.__new__(BatchedEngineBackend)
        backend.state_disk, backend.staging, backend.kept, backend.queued_polls = True, {}, {}, {}
        backend.waiting, backend.jobs, backend.free_contexts = deque(), queue.Queue(), []
        backend.admission_lock, backend.pending_tokens = threading.Lock(), 0
        backend._stage = lambda payload, *args: batch_backend.StageReply(engine.state())
        opened = []
        backend._open = lambda job, slot_context: opened.append((clock[0], engine.state())) or True
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto", max_tokens=8)
        job.cancelled = threading.Event()
        job.after, job.turn = None, {}
        backend.jobs.put(job)
        backend.pending_tokens = len(job.payload["ids"])
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 300:  # the worker loop runs _admit all the time (here every 0.5 s); no slot is free
                self.assertFalse(backend._admit())
                clock[0] += 0.5
            backend.free_contexts.append(40000)
            self.assertTrue(backend._admit())
        self.assertEqual(engine.reads, 1)
        self.assertEqual(opened, [(300.0, "ready")])

    def waiting_backend(self, engine_state, clock):
        """A backend with one long turn in its queue and no free slot; `engine_state` answers each stage command."""
        import queue, threading
        from collections import deque
        backend = object.__new__(BatchedEngineBackend)
        backend.state_disk, backend.staging, backend.kept, backend.queued_polls = True, {}, {}, {}
        backend.waiting, backend.jobs, backend.free_contexts = deque(), queue.Queue(), []
        backend.admission_lock, backend.pending_tokens = threading.Lock(), 0
        backend._stage = lambda payload, *args: batch_backend.StageReply(engine_state(payload))
        opened = []
        backend._open = lambda job, slot_context: opened.append(clock[0]) or True
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto", max_tokens=8)
        job.cancelled = threading.Event()
        job.after, job.turn = None, {}
        backend.jobs.put(job)
        backend.pending_tokens = len(job.payload["ids"])
        return backend, job, opened

    def test_a_state_that_reaches_the_disk_while_the_turn_waits_is_read(self):
        """An earlier change: a returning turn arrives while the host cache still holds its
        state, so the first stage reply is "none" (the disk state is not longer than the host's). While the turn waits
        for a slot, other sessions evict that host entry and the tier writes it to disk. The turn must stage again and
        open on the disk read, not on the shared prefix. Before the fix a "none" reply was final: 0 in-run reads."""
        clock = [0.0]

        class Engine:
            def __init__(self): self.reads, self.started, self.polls = 0, None, 0
            def state(self, payload=None):
                self.polls += 1
                if clock[0] < 100: return "none"  # the host cache holds the state (floor), or no disk state yet
                if self.started is None: self.reads, self.started = self.reads + 1, clock[0]
                return "ready" if clock[0] - self.started >= 1 else "reading"

        engine = Engine()
        backend, job, opened = self.waiting_backend(engine.state, clock)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 300:
                self.assertFalse(backend._admit())
                clock[0] += 0.5
            backend.free_contexts.append(40000)
            while not opened and clock[0] < 400:
                backend._admit()
                clock[0] += 0.05
        self.assertEqual(engine.reads, 1)
        self.assertEqual(job.turn["state_disk"], "ready")
        self.assertLess(opened[0] - 300, 1)  # the read ran during the wait: the slot did not wait for it
        # A waiting turn whose reply is "none" is polled rarely (each poll costs the engine thread about 8 ms).
        self.assertLessEqual(engine.polls, 100 / STAGE_NONE_POLL_SECONDS + 1 / STAGE_AHEAD_POLL_SECONDS +
                             200 / STAGE_REFRESH_SECONDS + 4)

    def test_a_none_reply_is_checked_again_when_the_slot_frees(self):
        """The open path: the last "none" reply is older than the slot. The turn stages once more before the open,
        so a state that reached the disk after the last poll is read; a turn whose reply stays "none" opens at once."""
        clock = [0.0]
        replies = []

        def state(payload=None):
            replies.append(clock[0])
            return "none" if clock[0] < 3 else "reading" if clock[0] < 4 else "ready"

        backend, job, opened = self.waiting_backend(state, clock)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertFalse(backend._admit())  # the first poll: "none"
            clock[0] = 3.0                      # sooner than the next "none" poll of a waiting turn
            backend.free_contexts.append(40000)
            self.assertFalse(backend._admit())  # the open stages again: the read started, the turn waits
            while not opened and clock[0] < 10:
                backend._admit()
                clock[0] += 0.05
        self.assertGreaterEqual(opened[0], 4)
        self.assertEqual(job.turn["state_disk"], "ready")
        clock[0] = 0.0
        replies.clear()
        backend, job, opened = self.waiting_backend(lambda payload=None: replies.append(clock[0]) or "none", clock)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertFalse(backend._admit())
            clock[0] = 3.0
            backend.free_contexts.append(40000)
            self.assertTrue(backend._admit())
        self.assertEqual((replies, opened), ([0.0, 3.0], [3.0]))

    def test_only_the_next_turn_starts_a_read_and_the_others_say_they_wait(self):
        """Rule 2: a read holds host memory inside the cache budget until its turn opens, so only the
        turn next in line may start one. The turns behind it send read=0 every STAGE_REFRESH_SECONDS with their place
        in the queue: the engine's read-ahead rule takes a host state only from a session that waits behind the read."""
        clock = [0.0]
        calls = []
        backend, first, opened = self.waiting_backend(lambda payload: "none", clock)
        backend._stage = lambda payload, position=0, read=True, job=0: calls.append(
            (clock[0], len(payload["ids"]), position, read)) or batch_backend.StageReply(
                "none", "queued" if not read else "nostate", 0, 0 if read else 3)
        second = type("Job", (), {})()
        second.payload = dict(ids=list(range(STAGE_MIN_TOKENS + 1)), cache_control="auto", max_tokens=8)
        second.cancelled, second.after, second.turn = type(first.cancelled)(), None, {}
        short = type("Job", (), {})()
        short.payload = dict(ids=[1, 2, 3], cache_control="auto", max_tokens=8)
        short.cancelled, short.after, short.turn = type(first.cancelled)(), None, {}
        backend.jobs.put(short)
        backend.jobs.put(second)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 60:
                self.assertFalse(backend._admit())
                clock[0] += 0.5
        head = [call for call in calls if call[1] == STAGE_MIN_TOKENS]
        behind = [call for call in calls if call[1] == STAGE_MIN_TOKENS + 1]
        self.assertTrue(head and all(position == 0 and read for _, _, position, read in head))
        # The short turn between them does not count: the second long turn is next after the first.
        self.assertTrue(behind and all(position == 1 and not read for _, _, position, read in behind))
        self.assertEqual([t for t, *_ in behind], [0.0, STAGE_REFRESH_SECONDS, 2 * STAGE_REFRESH_SECONDS])
        self.assertNotIn("state_disk", second.turn)  # a wait report is not the turn's stage reply
        self.assertEqual(second.turn["state_disk_tag"], 3)  # but a read took its host state: the turn line says so
        self.assertEqual(backend._position(second), 1)

    def test_the_stage_reply_carries_the_read_ahead_tag(self):
        """An earlier change: the engine names the host_evict tag of a turn whose host state a read took; an older engine
        sends no reason and no tag. A bad tag is a protocol error, a bad reason becomes "other"."""
        backend = object.__new__(BatchedEngineBackend)
        sent = []
        backend.process = SimpleNamespace(stdin=SimpleNamespace(write=sent.append, flush=lambda: None))
        def replies(*events):
            items = iter(list(events) + [dict(event="batch_end", op="stage")])
            backend._next = lambda: next(items)
        replies(dict(event="staged", state="ready", tokens=40, reason="start", tag=7))
        self.assertEqual(backend._stage(dict(ids=[1, 2]), 2, read=False), ("ready", "start", 40, 7))
        self.assertEqual(json.loads(sent[-1])["position"], 2)
        self.assertEqual(json.loads(sent[-1])["read"], 0)
        replies(dict(event="staged", state="none", tokens=0))
        self.assertEqual(backend._stage(dict(ids=[1, 2])), ("none", "-", 0, 0))
        # Review HIGH-2: the default command (next turn, may read) is the 7b3106e8 one, byte for byte, so this server
        # also drives the train 23 engine, which refuses unknown fields.
        self.assertEqual(sent[-1], json.dumps(dict(op="stage", request=dict(ids=[1, 2])), separators=(",", ":"),
                                              allow_nan=False) + "\n")
        replies(dict(event="staged", state="none", tokens=0))
        backend._stage(dict(ids=[1, 2]), 0, read=False)
        self.assertEqual(json.loads(sent[-1]), dict(op="stage", request=dict(ids=[1, 2]), read=0))
        replies(dict(event="staged", state="none", tokens=0, reason="Bad Word"))
        self.assertEqual(backend._stage(dict(ids=[1, 2])).reason, "other")
        replies(dict(event="staged", state="none", tokens=0, tag=-1))
        with self.assertRaises(ProtocolError): backend._stage(dict(ids=[1, 2]))

    def test_the_stage_job_tag_goes_only_to_an_engine_that_takes_it(self):
        """Fix 2 (cause.md item 10): each long turn has one tag on the engine's stage and host_evict lines and on its
        turn line. The train 23 engine refuses unknown fields (HIGH-2), so the tag goes out only when the ready event
        says "stage_job"."""
        backend = object.__new__(BatchedEngineBackend)
        sent = []
        backend.process = SimpleNamespace(stdin=SimpleNamespace(write=sent.append, flush=lambda: None))
        def replies():
            items = iter([dict(event="staged", state="reading", tokens=40, reason="writing"),
                          dict(event="batch_end", op="stage")])
            backend._next = lambda: next(items)
        job, other = SimpleNamespace(turn={}), SimpleNamespace(turn={})
        self.assertEqual(backend._job_tag(job), 0)
        self.assertNotIn("state_disk_job", job.turn)
        backend.stage_job = True
        self.assertEqual((backend._job_tag(job), backend._job_tag(job), backend._job_tag(other)), (1, 1, 2))
        self.assertEqual(job.turn["state_disk_job"], 1)
        replies()
        self.assertEqual(backend._stage(dict(ids=[1, 2]), 0, True, 1), ("reading", "writing", 40, 0))
        self.assertEqual(json.loads(sent[-1]), dict(op="stage", request=dict(ids=[1, 2]), job=1))
        # The ready event decides it.
        backend = object.__new__(BatchedEngineBackend)
        backend.requested_state_dir = "states"
        for ready, expect in ((dict(state_disk=True, stage_job=True), True), (dict(state_disk=True), False),
                              (dict(stage_job=True), False)):
            with patch.object(batch_backend.BatchedEngineBackend.__mro__[1], "_validate_ready", lambda self, r: None):
                backend.streams, backend.requested_prefill, backend.expected_contexts = 1, None, [8]
                backend._validate_ready(dict(protocol="batch-v1", context=8, streams=1, prefill_rows=0,
                                             slot_contexts=[8], **ready))
            self.assertIs(backend.stage_job, expect)

    def test_a_writing_reply_waits_and_the_poll_after_the_write_reads(self):
        """Fix 2 (cause.md items 1, 2, 4, 5; the daily slot-3 turn of Oct 21): the turn's own state waits for the
        engine's writer, so the reply is Reading "writing". The turn waits as for a read; the poll after the write starts
        the read, and the turn opens on it, not on a cold prefill."""
        clock = [0.0]
        replies = []

        def state(payload=None):
            replies.append(clock[0])
            return "reading" if clock[0] < 2 else "ready"  # writing until 1.0 s, then the read

        backend, job, opened = self.waiting_backend(state, clock)
        backend.free_contexts.append(40000)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while not opened and clock[0] < 10:
                backend._admit()
                clock[0] += 0.05
        self.assertGreaterEqual(opened[0], 2)
        self.assertLess(opened[0], 2.2)
        self.assertEqual(job.turn["state_disk"], "ready")

    def test_turns_without_a_slot_are_polled_less_often(self):
        """each stage poll costs the engine command thread about 8 ms while other sessions
        decode. A reading turn that has no slot is polled every STAGE_AHEAD_POLL_SECONDS, not every 50 ms."""
        import queue, threading
        from collections import deque
        clock = [0.0]
        polls = []
        backend = object.__new__(BatchedEngineBackend)
        backend.state_disk, backend.staging, backend.kept, backend.queued_polls = True, {}, {}, {}
        backend.waiting, backend.jobs, backend.free_contexts = deque(), queue.Queue(), []
        backend.admission_lock, backend.pending_tokens = threading.Lock(), 0
        backend._stage = lambda payload, *args: batch_backend.StageReply(polls.append(clock[0]) or "reading")  # a slow read
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto", max_tokens=8)
        job.cancelled = threading.Event()
        job.after, job.turn = None, {}
        backend.jobs.put(job)
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            while clock[0] < 10:
                self.assertFalse(backend._admit())
                clock[0] += 0.01
            self.assertLessEqual(len(polls), 10 / STAGE_AHEAD_POLL_SECONDS + 2)
            # With a free slot the turn waits for its read and polls it every STAGE_POLL_SECONDS again.
            backend.free_contexts.append(40000)
            before = len(polls)
            while clock[0] < 11:
                self.assertFalse(backend._admit())
                clock[0] += 0.01
            self.assertGreaterEqual(len(polls) - before, 0.5 / STAGE_POLL_SECONDS)  # 0.01 s steps: about 17

    def test_a_new_read_after_a_gap_is_a_new_wait(self):
        """A "reading" reply after a long gap (the engine dropped the ready read and reads it again) restarts the
        wait, so the turn does not open beside an unused full read."""
        backend = object.__new__(BatchedEngineBackend)
        backend.staging, backend.queued_polls = {}, {}
        job = type("Job", (), {})()
        job.payload = dict(ids=list(range(STAGE_MIN_TOKENS)), cache_control="auto")
        job.cancelled = type("Event", (), {"is_set": lambda self: False})()
        job.turn = {}
        replies = iter(["reading", "reading", "ready"])
        backend._stage = lambda payload, *args: batch_backend.StageReply(next(replies))
        clock = [0.0]
        with patch.object(batch_backend.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertFalse(backend._staged(job))
            clock[0] = 300.0
            self.assertFalse(backend._staged(job))
            self.assertEqual(job.turn["state_disk_wait_s"], 0)
            clock[0] += 1
            self.assertTrue(backend._staged(job))

    def test_stop_fits_the_service_stop_time(self):
        """A service unit stops with TimeoutStopSec=120. The engine flushes for at most 75 s, the server
        waits a little longer, then kills it (a tier engine waits after SIGTERM)."""
        header = (Path(__file__).parents[2] / "engine/serve/batched_serve.h").read_text()
        import re
        flush, grace = map(float, re.search(r"kStateFlushSeconds = (\d+), kStateTermGrace = (\d+);", header).groups())
        self.assertLessEqual(flush, 75)
        self.assertLess(flush, STATE_FLUSH_SECONDS)
        self.assertLessEqual(STATE_FLUSH_SECONDS + 10, 120 - 15)
        self.assertLessEqual(grace, 120 - 15)
        backend = self.start(flush_delay="30", ignore_term="1")
        backend.flush_seconds = 0.5
        start = time.monotonic()
        backend.close()
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(backend.process.returncode, -9)


if __name__ == "__main__":
    unittest.main()
