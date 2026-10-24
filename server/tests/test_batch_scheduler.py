import io
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from server.api import validate_request
from server.backend import Job
from server.batch_backend import BatchedEngineBackend, LiveRequest
from server.batch_scheduler import PendingPrefill, choose, packed_mode

FAKE = Path(__file__).parent / "fake_batch_engine.py"

class PackedModeTest(unittest.TestCase):
    def test_packed_verify_packs_from_two(self):
        self.assertFalse(packed_mode(False, 1))
        self.assertTrue(packed_mode(False, 2))
        self.assertFalse(packed_mode(True, 1))

    def test_g6c_hysteresis(self):
        self.assertFalse(packed_mode(False, 2, enter=3))
        self.assertTrue(packed_mode(False, 3, enter=3))
        self.assertTrue(packed_mode(True, 2, enter=3))
        self.assertFalse(packed_mode(True, 1, enter=3))
        self.assertFalse(packed_mode(False, 8, enter=9))


class ChooseTest(unittest.TestCase):
    def test_idle_and_decode_only(self):
        self.assertIsNone(choose(10, [], [], 0, 1000))
        self.assertEqual(choose(10, [3, 4], [], 9.99, 1000), ("step", [3, 4]))

    def test_waiting_turn_goes_before_recent_decode(self):
        pending = [PendingPrefill(5, 4096, 9.0)]
        self.assertEqual(choose(10, [3], pending, 9.9, 1000), ("prefill", 5))

    def test_decode_bound_preempts_prefill(self):
        pending = [PendingPrefill(5, 4096, 9.0)]
        self.assertEqual(choose(10, [3], pending, 9.7, 1000), ("step", [3]))
        self.assertEqual(choose(10, [3], pending, 9.7, 1000, decode_max=.5), ("prefill", 5))

    def test_shortest_remaining_first(self):
        pending = [PendingPrefill(5, 16384, 1.0), PendingPrefill(6, 256, 1.5), PendingPrefill(7, 256, 1.2)]
        self.assertEqual(choose(2, [], pending, 0, 1000), ("prefill", 7))

    def test_aged_turn_moves_ahead(self):
        # 16384 tokens at 1000 tok/s: deadline = 1 + 1 + 2 * 16.384 = 34.768.
        pending = [PendingPrefill(5, 16384, 1.0), PendingPrefill(6, 256, 30.0)]
        self.assertEqual(choose(34.7, [], pending, 0, 1000), ("prefill", 6))
        self.assertEqual(choose(34.8, [], pending, 0, 1000), ("prefill", 5))

    def test_think_cap_continuation_goes_before_aged_turns(self):
        # A continuation's client already has its first token; aged peers waited up to 30 s for it.
        pending = [PendingPrefill(5, 16384, 1.0), PendingPrefill(6, 26, 40.0, continuation=True),
                   PendingPrefill(7, 256, 1.2)]
        self.assertEqual(choose(40.0, [], pending, 0, 1000), ("prefill", 6))
        pending.append(PendingPrefill(4, 26, 39.0, continuation=True))
        self.assertEqual(choose(40.0, [], pending, 0, 1000), ("prefill", 4))

    def test_decode_bound_still_preempts_a_continuation(self):
        pending = [PendingPrefill(6, 26, 9.0, continuation=True)]
        self.assertEqual(choose(10, [3], pending, 9.7, 1000), ("step", [3]))

    def test_backend_marks_think_cap_continuations(self):
        backend = BatchedEngineBackend.__new__(BatchedEngineBackend)
        first = Job(payload=dict(ids=[1, 2, 3]))
        backend.active = {1: LiveRequest(Job(payload=dict(ids=[1] * 40)), admitted_at=2.0),
                          2: LiveRequest(Job(payload=dict(ids=[1, 2, 3, 4]), after=first), admitted_at=3.0, prefilled=1)}
        self.assertEqual(backend._pending_prefills(), [PendingPrefill(1, 40, 2.0), PendingPrefill(2, 3, 3.0, True)])

    def test_a_capped_job_holds_new_chunks_until_it_closes(self):
        backend = BatchedEngineBackend.__new__(BatchedEngineBackend)
        first = LiveRequest(Job(payload=dict(ids=[1, 2, 3]), think_cap=1), capped_at=time.monotonic())
        backend.active = {1: first}
        self.assertTrue(backend._awaiting_continuation())
        # continue_job cancels the first job after it queues the continuation. Until _cancelled closes that job and
        # opens the continuation on its slot, a new turn's chunk must still wait.
        first.job.cancelled.set()
        self.assertTrue(backend._awaiting_continuation())
        backend.active = {1: LiveRequest(Job(payload=dict(ids=[1])), capped_at=time.monotonic() - 1.0)}
        self.assertFalse(backend._awaiting_continuation())
        # The server disarms the cap when the model closed its think block itself: no continuation comes.
        disarmed = LiveRequest(Job(payload=dict(ids=[1])), capped_at=time.monotonic())
        backend.active = {1: disarmed}
        self.assertFalse(backend._awaiting_continuation())
        backend.active = {}
        self.assertFalse(backend._awaiting_continuation())

class ChunkedBackendTest(unittest.TestCase):
    def setUp(self):
        popen = subprocess.Popen
        def start(*args, **kwargs):
            # The backend passes env= when it removes switches (for example CRUCIBLE_PI_AI_DIR in this suite).
            return popen([sys.executable, str(FAKE)], env=dict(kwargs.pop("env", os.environ), FAKE_PREFILL_ROWS="4"), **kwargs)
        with patch("server.backend.subprocess.Popen", side_effect=start):
            self.backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=5)

    def tearDown(self):
        self.backend.close()

    def collect(self, job):
        ids = []
        while True:
            event = job.events.get(timeout=5)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done": return ids
            ids.append(event["id"])

    def test_ready_reports_rows(self):
        self.assertEqual(self.backend.prefill_rows, 4)

    def test_long_prompt_and_peer_both_complete(self):
        short = validate_request(dict(prompt="x", max_tokens=6, spec=0), "completion", "crucible")
        jobs = [self.backend.submit(list(range(1, 41)), short), self.backend.submit([9], short)]
        self.assertEqual(self.collect(jobs[1]), list(range(10, 16)))
        self.assertEqual(self.collect(jobs[0]), list(range(41, 47)))
        self.assertFalse(self.backend.fatal)

    def test_progress_matches_reported_chunk_size(self):
        from server.batch_backend import LiveRequest
        from server.backend import Job
        for rows, expected, rate in ((None, 0, 1000.0), (8, 24, None), (0, 24, None), (16, 40, None)):
            with self.subTest(rows=rows):
                backend = object.__new__(BatchedEngineBackend)
                backend.prefill_rows = rows; backend.prefill_rate = 1000.0
                backend.active = {1: LiveRequest(Job(dict(ids=list(range(40)), max_tokens=4)))}
                backend._command = lambda payload, rows=None: (0.0, 0.01)
                for _ in range(3): backend._prefill(1)
                self.assertEqual(backend.active[1].prefilled, expected)
                if rate is not None: self.assertEqual(backend.prefill_rate, rate)
                self.assertGreater(backend.prefill_rate, 1.0)

    def test_turn_metrics_and_trace(self):
        short = validate_request(dict(prompt="x", max_tokens=3, spec=0, crucible_metrics=True), "completion", "crucible")
        job = self.backend.submit(list(range(1, 11)), short)
        events = []
        while not events or events[-1]["event"] != "done": events.append(job.events.get(timeout=5))
        times = [event["crucible_t"] for event in events[:-1]]
        self.assertEqual(times, sorted(times))
        m = events[-1]["crucible_metrics"]
        self.assertEqual([chunk[2] for chunk in m["prefill_chunks"]], [4, 4, 2])
        self.assertLessEqual(m["t_submit"], m["t_first_gpu"])
        self.assertLessEqual(m["t_first_gpu"], m["t_prefill_end"])
        self.assertLessEqual(m["t_prefill_end"], m["t_first_token"])
        self.assertEqual(m["t_first_token"], times[0])
        self.assertGreaterEqual(m["between_chunks_s"], 0)
        self.assertIsNone(m["restore_s"]); self.assertIsNone(m["memory"])
        ttft = m["queue_s"] + m["prefill_s"] + m["between_chunks_s"] + m["post_prefill_s"]
        self.assertAlmostEqual(ttft, m["t_first_token"] - m["t_submit"], places=9)
        self.assertEqual(m["prefill_chunk_count"], 3)
        plain = self.backend.submit([5], validate_request(dict(prompt="x", max_tokens=1, spec=0), "completion", "crucible"))
        done = plain.events.get(timeout=5)
        while done["event"] != "done": done = plain.events.get(timeout=5)
        self.assertNotIn("crucible_metrics", done)
        ops = [entry for entry in self.backend.trace.since(m["t_submit"]) if m["request_id"] in entry["request_ids"]]
        self.assertEqual([entry["op"] for entry in ops], ["open", "prefill", "prefill", "prefill", "step", "step"])

    def test_invalid_rows_rejected(self):
        backend = object.__new__(BatchedEngineBackend)
        backend.context = 1; backend.streams = 1; backend.prefill_rows = None; backend.requested_prefill = None
        for rows in (True, -1, 4097, "8", None):
            with self.assertRaises(RuntimeError):
                backend._validate_ready(dict(event="ready", context=1, streams=1, protocol="batch-v1", prefill_rows=rows))


class RestoredRemainingTest(unittest.TestCase):
    """SRPT and aging count only the rows that a request still prefills after its cache restore."""

    def backend(self, rows=4096):
        from server import metrics
        from server.backend import Job
        from server.batch_backend import LiveRequest
        backend = object.__new__(BatchedEngineBackend)
        backend.prefill_rows = rows; backend.prefill_rate = 1000.0; backend.fatal = None
        backend.trace = metrics.Trace(); backend.script = []
        backend.process = SimpleNamespace(stdin=io.StringIO())
        backend._next = lambda: backend.script.pop(0)
        def live(id, count, admitted):
            job = Job(dict(ids=list(range(count)), max_tokens=4, stop_ids=[]))
            return LiveRequest(job, admitted_at=admitted, slot_context=262144)
        # A is a 100K follow-up that restores 99K; B is a cold 30K turn admitted earlier.
        backend.active = {1: live(1, 100_000, 5.0), 2: live(2, 30_000, 0.0)}
        return backend

    def open(self, backend, id, cached):
        backend.script += [dict(event="opened", request_id=id, slot_generation=id, slot_context=262144,
                                cached_tokens=cached), dict(event="batch_end", op="open")]
        backend._command(dict(op="open", request_id=id, request={}))

    def test_cached_follow_up_goes_before_cold_prefill(self):
        backend = self.backend()
        self.open(backend, 1, 99_000); self.open(backend, 2, 0)
        remaining = {item.request_id: item.remaining for item in backend._pending_prefills()}
        self.assertEqual(remaining, {1: 1_000, 2: 30_000})
        self.assertEqual(choose(6.0, [], backend._pending_prefills(), 0, backend.prefill_rate), ("prefill", 1))

    def chunk(self, backend, boundary):
        backend.script += [boundary]
        with patch("server.batch_backend.metrics.now", side_effect=[10.0, 10.1]):
            backend._prefill(1)

    def test_reported_cursor_sets_progress_and_a_head_chunk_does_not_inflate_the_rate(self):
        backend = self.backend()
        self.open(backend, 1, 99_000)
        # A restored tail runs one 8-row head chunk; the engine names its cursor after the chunk.
        self.chunk(backend, dict(event="batch_end", op="prefill", cursor=99_008))
        self.assertEqual(backend.active[1].prefilled, 99_008)
        self.assertLess(backend.prefill_rate, 5000)
        self.assertEqual(backend.trace.since(0)[-1]["rows"], 8)
        self.chunk(backend, dict(event="batch_end", op="prefill", cursor=100_000))
        self.assertEqual(backend.active[1].prefilled, 100_000)
        self.assertEqual(backend.trace.since(0)[-1]["rows"], 992)
        self.assertAlmostEqual(backend.prefill_rate, .7 * 1000 + .3 * 9920)

    def test_engine_without_cursor_keeps_the_rate_after_a_restore(self):
        backend = self.backend()
        self.open(backend, 1, 99_000)
        self.chunk(backend, dict(event="batch_end", op="prefill"))
        self.assertEqual(backend.active[1].prefilled, 100_000)
        self.assertEqual(backend.prefill_rate, 1000.0)

    def test_invalid_cursor_is_fatal(self):
        for cursor in (99_000, 100_001, "99008", True):
            with self.subTest(cursor=cursor):
                backend = self.backend()
                self.open(backend, 1, 99_000)
                with self.assertRaises(RuntimeError):
                    self.chunk(backend, dict(event="batch_end", op="prefill", cursor=cursor))

    def test_interleaved_chunk_reports_its_cursor(self):
        # Train 13 merge (serve-api x decode-interleave): a chunk with decode passes inside it ends with the same
        # batch_end, which names the cursor and engine_s. The progress comes from the cursor.
        backend = self.backend()
        self.open(backend, 1, 99_000)
        backend.ownership_lock, backend.closed, backend.owned_jobs = __import__("threading").Lock(), SimpleNamespace(
            is_set=lambda: False), {}
        decoder = backend.active[2]
        decoder.generation, decoder.ready = 2, True
        backend.script += [dict(event="token", request_id=2, slot_generation=2, id=5),
                           dict(event="batch_part", op="verify_step"),
                           dict(event="batch_end", op="prefill", cursor=99_008, engine_s=0.25)]
        backend._command(dict(op="prefill", request_id=1, request_ids=[2], every=10), 4096)
        self.assertEqual(backend.prefill_cursor, 99_008)
        self.assertEqual(decoder.count, 1)
        entries = backend.trace.since(0)
        self.assertEqual([e["op"] for e in entries[-2:]], ["verify_step", "prefill"])
        self.assertEqual(entries[-1]["rows"], 8)

if __name__ == "__main__": unittest.main()


def test_interleave_passes_mirror_the_engine_rule():
    from server.batch_scheduler import interleave_passes
    assert interleave_passes(8) == 6 and interleave_passes(1) == 46 and interleave_passes(2) == 23
    assert interleave_passes(0) == 0 and interleave_passes(47) == 1 and interleave_passes(24) == 2


class EncoderRowsTest(unittest.TestCase):
    """SRPT and aging count the encoder slices that an image turn still owes, as prompt rows of equal time."""

    def test_encoder_rows_are_slices_times_slice_time_times_rate(self):
        from server.batch_scheduler import ENCODER_SLICE_SECONDS, encoder_rows
        self.assertEqual(encoder_rows(29, ENCODER_SLICE_SECONDS, 1500.0), round(29 * 0.076 * 1500))  # 3,306 rows
        self.assertEqual(encoder_rows(0, ENCODER_SLICE_SECONDS, 1500.0), 0)
        self.assertEqual(encoder_rows(-1, ENCODER_SLICE_SECONDS, 1500.0), 0)

    def backend(self):
        backend = RestoredRemainingTest().backend()
        backend.prefill_rate = 1500.0
        def live(count, admitted, slices=0):
            job = Job(dict(ids=list(range(count)), max_tokens=4, stop_ids=[], spec=0))
            return LiveRequest(job, admitted_at=admitted, slot_context=262144, vision_slices=slices)
        # A 2560x1440 screenshot turn (3,600 rows, 29 slices) and a text turn of 5,000 rows, admitted together.
        backend.active = {1: live(3_600, 1.0, slices=29), 2: live(5_000, 1.0)}
        return backend

    def test_a_longer_text_turn_goes_before_an_image_turn_of_more_work(self):
        backend = self.backend()
        remaining = {item.request_id: item.remaining for item in backend._pending_prefills()}
        self.assertEqual(remaining, {1: 3_600 + round(29 * 0.076 * 1500), 2: 5_000})
        self.assertEqual(choose(1.5, [], backend._pending_prefills(), 0, backend.prefill_rate), ("prefill", 2))
        # Before an earlier change the image turn ranked by its 3,600 prompt rows only and went first.
        backend.active[1].vision_slices = 0
        self.assertEqual(choose(1.5, [], backend._pending_prefills(), 0, backend.prefill_rate), ("prefill", 1))

    def test_a_slice_command_lowers_the_remaining_work_and_feeds_the_slice_time(self):
        backend = self.backend()
        backend.script += [dict(event="batch_end", op="prefill")]
        with patch("server.batch_backend.metrics.now", side_effect=[10.0, 10.1]):
            backend._prefill(1)
        entry = backend.active[1]
        self.assertEqual((entry.vision_slices, entry.prefilled), (28, 0))
        self.assertAlmostEqual(backend.encoder_slice_seconds, .7 * 0.076 + .3 * 0.1)
        self.assertEqual(backend.prefill_rate, 1500.0)  # a slice is no prompt row
        remaining = {item.request_id: item.remaining for item in backend._pending_prefills()}
        self.assertEqual(remaining[1], 3_600 + round(28 * backend.encoder_slice_seconds * 1500))

    def slice(self, backend, seconds, **patches):
        backend.script += [dict(event="batch_end", op="prefill")]
        with patch("server.batch_backend.metrics.now", side_effect=[10.0, 10.0 + seconds]):
            if patches:
                with patch.multiple(backend, **patches):
                    backend._prefill(1)
            else:
                backend._prefill(1)

    def test_each_slice_sample_is_held_to_the_band(self):
        from server.batch_scheduler import ENCODER_SLICE_BAND
        backend = self.backend()
        self.slice(backend, 2.5)
        self.assertAlmostEqual(backend.encoder_slice_seconds, .7 * 0.076 + .3 * ENCODER_SLICE_BAND[1])
        self.assertEqual(backend.active[1].vision_slices, 28)
        before = backend.encoder_slice_seconds
        self.slice(backend, 0.001)
        self.assertAlmostEqual(backend.encoder_slice_seconds, .7 * before + .3 * ENCODER_SLICE_BAND[0])

    def test_a_slice_command_with_decoders_does_not_feed_the_slice_time(self):
        # with prefill_rows set, a slice command takes ready decoders (rows >= INTERLEAVE_MIN_ROWS),
        # so its time is not one slice.
        backend = self.backend()
        backend.prefill_rows, backend.prefill_interleave = 4096, 10
        backend.active[2].ready = True
        backend.active[2].job.events = __import__("queue").Queue(maxsize=256)  # room for the passes of one chunk
        decoders = backend._interleave_decoders(1, 3_600)
        self.assertEqual(decoders, [2])
        backend.script += [dict(event="batch_end", op="prefill")]
        with patch("server.batch_backend.metrics.now", side_effect=[10.0, 10.9]):
            backend._prefill(1)
        self.assertEqual(backend.encoder_slice_seconds, 0.076)
        self.assertEqual(backend.active[1].vision_slices, 28)

    def test_a_failed_slice_does_not_feed_the_slice_time(self):
        backend = self.backend()
        original = backend._command
        def failing(payload, rows=None):
            result = original(payload, rows)
            del backend.active[1]  # the engine failed the request (for example a changed image file)
            return result
        backend._command = failing
        backend.script += [dict(event="batch_end", op="prefill")]
        with patch("server.batch_backend.metrics.now", side_effect=[10.0, 10.1]):
            backend._prefill(1)
        self.assertEqual(backend.encoder_slice_seconds, 0.076)

    def test_a_first_image_counts_at_the_default_slice_time_and_rate(self):
        backend = RestoredRemainingTest().backend()  # prefill_rate 1000.0, as at worker start
        job = Job(dict(ids=list(range(3_600)), max_tokens=4, stop_ids=[]))
        backend.active = {1: LiveRequest(job, admitted_at=1.0, slot_context=262144, vision_slices=29)}
        (item,) = backend._pending_prefills()
        self.assertEqual(item.remaining, 3_600 + round(29 * 0.076 * 1000.0))  # 2,204 encoder rows
        from server.batch_scheduler import ENCODER_SLICES_MAX, encoder_rows
        self.assertEqual(encoder_rows(1 << 20, 0.076, 1000.0), round(ENCODER_SLICES_MAX * 0.076 * 1000.0))

    def test_aging_counts_the_encoder_work(self):
        # Image turn: 3,600 rows + 29 slices (3,306 rows at 1,500 rows/s), admitted at 0. Text turn: 5,000 rows at 0.1.
        # Rows only, the image turn ages at 1 + 2 * 3,600 / 1,500 = 5.8 s; with its encoder work at 10.2 s.
        backend = self.backend()
        backend.active[1].admitted_at, backend.active[2].admitted_at = 0.0, 0.1
        self.assertEqual(choose(6.0, [], backend._pending_prefills(), 0, backend.prefill_rate), ("prefill", 2))
        backend.active[1].vision_slices = 0  # before an earlier change: aged at 6.0, so it went first
        self.assertEqual(choose(6.0, [], backend._pending_prefills(), 0, backend.prefill_rate), ("prefill", 1))
