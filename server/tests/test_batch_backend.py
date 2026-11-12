import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from server.api import ApiError, validate_request
from server.batch_backend import BatchedEngineBackend

FAKE = Path(__file__).parent / "fake_batch_engine.py"

class BatchBackendTest(unittest.TestCase):
    def setUp(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen", side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3)
        self.request = validate_request(dict(prompt="x", max_tokens=8, spec=0), "completion", "crucible")

    def tearDown(self):
        self.backend.close()
        self.assertIsNotNone(self.backend.process.poll())

    def collect(self, job):
        ids = []
        while True:
            event = job.events.get(timeout=5)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done":
                self.done_context = event.get("slot_context")
                self.assertEqual(event["completion_tokens"], len(ids)); return ids
            ids.append(event["id"])

    def test_concurrent_requests_preserve_identity(self):
        jobs = [self.backend.submit([id], self.request) for id in (1,9,1)]
        self.assertEqual(self.collect(jobs[0]), list(range(2,10)))
        self.assertEqual(self.collect(jobs[1]), list(range(10,18)))
        self.assertEqual(self.collect(jobs[2]), list(range(2,10)))

    def test_opened_event_reports_the_restored_tokens(self):
        """A closed think turn cancels its first job before done; its restore count comes from the opened event."""
        job = self.backend.submit([1], self.request)
        self.collect(job)
        self.assertEqual(job.turn.get("opened_cached"), 0)

    def test_slow_output_does_not_block_peer(self):
        long = validate_request(dict(prompt="x", max_tokens=80, spec=0), "completion", "crucible")
        slow = self.backend.submit([1], long)
        deadline = time.monotonic() + 3
        while slow.events.qsize() < 30 and time.monotonic() < deadline: time.sleep(.005)
        self.assertEqual(slow.events.qsize(), 30)
        peer = self.backend.submit([9], self.request)
        self.assertEqual(self.collect(peer), list(range(10,18)))
        self.assertFalse(self.backend.fatal)
        slow.cancelled.set()

    def test_cancelled_peer_and_recycled_slot(self):
        cancelled = self.backend.submit([1], self.request)
        cancelled.events.get(timeout=5); cancelled.cancelled.set()
        peer = self.backend.submit([9], self.request)
        self.assertEqual(self.collect(peer), list(range(10,18)))
        self.assertEqual(self.collect(self.backend.submit([1], self.request)), list(range(2,10)))

    def test_a_cancelled_prefill_counts_its_rows(self):
        """Start plan B1a: a turn cancelled in prefill names the prompt rows that it computed and nobody uses
        (cancel_prefill_tokens: its rows past the restored state). A cancel after the first token names none."""
        from server.backend import Job
        from server.batch_backend import LiveRequest
        backend = object.__new__(BatchedEngineBackend)
        commands = []
        backend._command = lambda payload, *args: commands.append(payload)
        backend.free_contexts, backend.kept, backend.state_disk = [], None, False
        backend.ownership_lock, backend.owned_jobs = __import__("threading").Lock(), {}
        in_prefill, decoding = Job(dict(ids=list(range(9000)), max_tokens=4)), Job(dict(ids=[1, 2], max_tokens=4))
        in_prefill.turn["opened_cached"] = 4096
        backend.active = {1: LiveRequest(in_prefill, prefilled=8192, slot_context=128),
                          2: LiveRequest(decoding, ready=True, prefilled=2, slot_context=128)}
        in_prefill.cancelled.set(), decoding.cancelled.set()
        self.assertTrue(backend._cancelled())
        self.assertEqual(in_prefill.turn["cancel_prefill_tokens"], 4096)
        self.assertNotIn("cancel_prefill_tokens", decoding.turn)
        self.assertEqual([command["op"] for command in commands], ["close", "close"])
        from server import metrics
        self.assertEqual(metrics.load_fields([in_prefill.turn])["cancel_prefill_tokens"], 4096)

    def test_ready_and_timeout_types_are_strict(self):
        backend = object.__new__(BatchedEngineBackend)
        backend.context = 1; backend.streams = 1
        for context, streams in ((True, 1), (1, True)):
            with self.assertRaises(RuntimeError):
                backend._validate_ready(dict(event="ready", context=context, streams=streams, protocol="batch-v1"))
        for timeout in (True, "1", None):
            with self.assertRaises(ValueError):
                BatchedEngineBackend("unused", "unused", output_timeout=timeout)

    def test_prefill_option_is_forwarded_including_zero(self):
        from types import SimpleNamespace
        for rows in (0, 1024, 4096):
            with patch('server.batch_backend.EngineBackend.__init__') as initialize:
                BatchedEngineBackend('unused', 'unused', streams=2, prefill=rows)
            options = initialize.call_args.kwargs['engine_options']
            self.assertIn('--prefill', options)
            self.assertEqual(options[options.index('--prefill') + 1], str(rows))
        for rows in (True, -1, 4097, '1024'):
            with self.assertRaises(ValueError): BatchedEngineBackend('unused', 'unused', prefill=rows)

    def test_ready_prefill_is_required_and_strict(self):
        backend = object.__new__(BatchedEngineBackend)
        backend.context = 128; backend.streams = 2; backend.requested_prefill = None
        backend.expected_contexts = [128, 128]
        ready = dict(event='ready', context=128, streams=2, protocol='batch-v1', slot_contexts=[128, 128])
        for rows in (None, True, -1, 4097, '1024'):
            with self.assertRaises(RuntimeError): backend._validate_ready({**ready, 'prefill_rows': rows})
        backend._validate_ready({**ready, 'prefill_rows': 0})
        self.assertEqual(backend.prefill_rows, 0)
        backend.requested_prefill = 1024
        with self.assertRaises(RuntimeError): backend._validate_ready({**ready, 'prefill_rows': 4096})
        backend.requested_prefill = None
        for contexts in (None, [128], [128, 16], [128, 128, 128]):
            with self.assertRaises(RuntimeError):
                backend._validate_ready({**ready, 'prefill_rows': 0, 'slot_contexts': contexts})
        self.assertEqual(backend.free_contexts, [128, 128])

    def test_slot_classes_are_forwarded_and_validated(self):
        with patch('server.batch_backend.EngineBackend.__init__') as initialize:
            backend = BatchedEngineBackend('unused', 'unused', 128, streams=3, short_streams=2, short_ctx=16)
        options = initialize.call_args.kwargs['engine_options']
        self.assertEqual(options[options.index('--short-streams'):][:4], ('--short-streams', '2', '--short-ctx', '16'))
        self.assertEqual(backend.expected_contexts, [128, 16, 16])
        with patch('server.batch_backend.EngineBackend.__init__') as initialize:
            BatchedEngineBackend('unused', 'unused', 128, streams=3)
        self.assertNotIn('--short-streams', initialize.call_args.kwargs['engine_options'])
        for short, size in ((3, 16), (-1, 16), (1, 0), (1, 128), (0, 16), (True, 16), (1, True)):
            with self.assertRaises(ValueError):
                BatchedEngineBackend('unused', 'unused', 128, streams=3, short_streams=short, short_ctx=size)

    def test_slot_classes_against_the_engine_protocol(self):
        import os, tempfile
        popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "opens"
            with patch.dict(os.environ, FAKE_SLOT_CONTEXTS="128,16,16", FAKE_OPEN_LOG=str(log)), \
                    patch("server.backend.subprocess.Popen",
                          side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
                backend = BatchedEngineBackend("unused", "unused", context=128, streams=3, output_timeout=.3,
                                               short_streams=2, short_ctx=16)
            try:
                long = validate_request(dict(prompt="x", max_tokens=40, spec=0), "completion", "crucible")
                jobs = [backend.submit([1], long), backend.submit([9], self.request), backend.submit([1], long)]
                self.assertEqual(self.collect(jobs[0]), list(range(2, 42)))
                self.assertEqual(self.collect(jobs[1]), list(range(10, 18)))
                self.assertEqual(self.collect(jobs[2]), list(range(2, 42)))
                self.assertEqual(self.done_context, 128)
                self.assertFalse(backend.fatal)
                self.assertEqual(sorted(backend.free_contexts), [16, 16, 128])
            finally:
                backend.close()
            self.assertEqual([line.split()[1] for line in log.read_text().split("\n") if line], ["128", "16", "128"])

    def test_admission_takes_the_smallest_fitting_slot_without_head_of_line_blocking(self):
        import threading
        from collections import deque
        from server.backend import Job
        backend = object.__new__(BatchedEngineBackend)
        backend.jobs, backend.waiting, backend.active = queue.Queue(), deque(), {}
        backend.free_contexts, backend.next_id = [128, 16, 16], 1
        backend.pending_tokens, backend.admission_lock = 0, threading.Lock()
        backend.ownership_lock, backend.owned_jobs = threading.Lock(), {}
        opened = []
        backend._command = lambda payload, rows=None: opened.append(payload["request"]["max_tokens"])
        def job(max_tokens):
            backend.pending_tokens += 1
            backend.jobs.put(Job(dict(ids=[1], max_tokens=max_tokens)))
        for count in (10, 40, 60, 16, 17): job(count)
        while backend._admit(): pass
        # 10 and 16 take the short slots, 40 takes the full slot; 60 and 17 wait for a slot that holds them.
        self.assertEqual(opened, [10, 40, 16])
        self.assertEqual(sorted(entry.slot_context for entry in backend.active.values()), [16, 16, 128])
        self.assertEqual([entry.payload["max_tokens"] for entry in backend.waiting], [60, 17])
        backend._release(3)  # a short slot frees: 60 still waits, and nothing else fits it
        self.assertFalse(backend._admit())
        backend._release(2)  # the full slot frees: the earlier 60 takes it before 17
        self.assertTrue(backend._admit()); self.assertEqual(opened[-1], 60)
        self.assertFalse(backend._admit())
        backend.waiting[0].cancelled.set()
        self.assertTrue(backend._admit()); self.assertEqual(len(backend.waiting), 0)
        self.assertEqual(backend.pending_tokens, 0)
        self.assertEqual(sorted(backend.free_contexts), [16])

    def test_unsupported_speculation_is_rejected_before_enqueue(self):
        for extra in (dict(draft_adapt=1), dict(draft_q=0), dict(draft_head="full")):
            request = validate_request(dict(prompt="x", spec=7, **extra), "completion", "crucible")
            with self.assertRaises(ApiError): self.backend.submit([1], request)
        self.assertEqual(self.backend.pending_tokens, 0)

    def test_full_mtp_window_speculates_for_clients(self):
        # mtp_prefix 0 keeps the MTP rows in the cache snapshots. At an equal
        # restore point the window changes only the drafts, not the IDs, so clients may send 0 (train 17 refused it).
        self.assertFalse(hasattr(self.backend, "full_mtp_window"))
        spec = validate_request(dict(prompt="x", max_tokens=8, spec=7, mtp_prefix=0), "completion", "crucible")
        job = self.backend.submit([1], spec)
        self.assertEqual(job.payload["mtp_prefix"], 0)
        self.assertEqual(self.collect(job), list(range(2, 10)))
        self.assertFalse(self.backend.fatal)

    def test_speculative_steps_publish_windows(self):
        spec = validate_request(dict(prompt="x", max_tokens=8, spec=7), "completion", "crucible")
        jobs = [self.backend.submit([1], spec), self.backend.submit([9], self.request)]
        self.assertEqual(jobs[0].payload["draft_adapt"], 0)
        self.assertEqual(self.collect(jobs[0]), list(range(2, 10)))
        self.assertEqual(self.collect(jobs[1]), list(range(10, 18)))
        sampled = validate_request(dict(prompt="x", max_tokens=5, spec=7, temperature=.6, seed=3), "completion", "crucible")
        self.assertEqual(self.collect(self.backend.submit([20], sampled)), list(range(21, 26)))
        self.assertFalse(self.backend.fatal)
        ops = [entry["op"] for entry in self.backend.trace.since(0)]
        self.assertIn("spec_step", ops)
        # A one-token request never speculates, as in serial generation.
        one = validate_request(dict(prompt="x", max_tokens=1, spec=7, temperature=.5), "completion", "crucible")
        self.assertEqual(self.collect(self.backend.submit([4], one)), [5])

    def test_speculative_requests_join_the_packed_pass_under_load(self):
        self.backend.pack_enter, self.backend.packed_op = 2, "step"
        spec = validate_request(dict(prompt="x", max_tokens=8, spec=7), "completion", "crucible")
        jobs = [self.backend.submit([1], spec), self.backend.submit([9], self.request)]
        self.assertEqual(self.collect(jobs[0]), list(range(2, 10)))
        self.assertEqual(self.collect(jobs[1]), list(range(10, 18)))
        entries = self.backend.trace.since(0)
        packed = [entry for entry in entries if entry["op"] == "step" and len(entry["request_ids"]) == 2]
        self.assertTrue(packed)
        # Once packed, the mode holds until one decoder remains.
        first = entries.index(packed[0])
        both = [entry for entry in entries[first:] if entry["op"] in ("step", "spec_step")]
        self.assertTrue(all(entry["op"] == "step" for entry in both[:3]))
        self.assertFalse(self.backend.fatal)

    def test_packed_verify_publishes_windows(self):
        self.assertEqual((self.backend.pack_enter, self.backend.packed_op), (2, "verify_step"))  # the defaults
        spec = validate_request(dict(prompt="x", max_tokens=9, spec=7), "completion", "crucible")
        jobs = [self.backend.submit([1], spec), self.backend.submit([20], self.request)]
        self.assertEqual(self.collect(jobs[0]), list(range(2, 11)))
        self.assertEqual(self.collect(jobs[1]), list(range(21, 29)))
        ops = [entry for entry in self.backend.trace.since(0) if entry["op"] == "verify_step"]
        self.assertTrue(any(len(entry["request_ids"]) == 2 for entry in ops))
        self.assertFalse(self.backend.fatal)

    def test_packed_op_switch(self):
        with self.assertRaises(ValueError):
            BatchedEngineBackend("unused", "unused", context=128, streams=2, packed_op="bogus")

    def test_spec_window_contract(self):
        from types import SimpleNamespace
        backend = object.__new__(BatchedEngineBackend)
        job = SimpleNamespace(payload=dict(max_tokens=4, stop_ids=[99], ids=[1]))
        backend.active = {1: SimpleNamespace(job=job, count=1, generation=5)}
        token = lambda id: dict(event="token", request_id=1, slot_generation=5, id=id)
        tokens = {}
        for id in (2, 3, 4): backend._validate_event(token(id), "spec_step", {1}, tokens)
        with self.assertRaises(RuntimeError): backend._validate_event(token(5), "spec_step", {1}, tokens)  # past max
        tokens = {}
        backend._validate_event(token(99), "spec_step", {1}, tokens)
        with self.assertRaises(RuntimeError): backend._validate_event(token(3), "spec_step", {1}, tokens)  # after stop
        tokens = {}
        backend._validate_event(token(2), "step", {1}, tokens)
        with self.assertRaises(RuntimeError): backend._validate_event(token(3), "step", {1}, tokens)

    def test_token_budget_rejects_before_queue(self):
        self.backend.pending_token_limit = 2
        with self.assertRaises(ApiError): self.backend.submit([1,2,3], self.request)
        self.assertEqual(self.backend.pending_tokens, 0)

    def test_terminal_contract_rejects_invalid_or_missing_done(self):
        from io import StringIO
        from types import SimpleNamespace
        from server.batch_backend import LiveRequest
        from server.backend import Job
        for max_tokens, stops, done in (
            (8, [99], dict(finish_reason="length", prompt_tokens=1, completion_tokens=1)),
            (8, [99], dict(finish_reason="stop", prompt_tokens=1, completion_tokens=1)),
            (1, [], dict(finish_reason="length", prompt_tokens=True, completion_tokens=1)),
            (1, [], dict(finish_reason="length", prompt_tokens=1, completion_tokens=True)),
            (1, [], None), (8, [2], None),
            (8, [2], dict(finish_reason="length", prompt_tokens=1, completion_tokens=1))):
            with self.subTest(max_tokens=max_tokens, stops=stops, done=done):
                backend = object.__new__(BatchedEngineBackend)
                backend.active = {900: LiveRequest(Job(dict(ids=[1], max_tokens=max_tokens, stop_ids=stops)), generation=2)}
                backend.process = SimpleNamespace(stdin=StringIO())
                events = [dict(event="token", request_id=900, slot_generation=2, id=2)]
                if done is not None: events.append(dict(event="done", request_id=900, slot_generation=2, **done))
                events.append(dict(event="batch_end", op="step"))
                source = iter(events); backend._next = lambda: next(source)
                delivered = []; backend._deliver = lambda job, event: delivered.append(event)
                with self.assertRaises(RuntimeError): backend._command(dict(op="step", request_ids=[900]))
                self.assertEqual(delivered, [])
                self.assertEqual(backend.active[900].count, 0)

    def test_stale_generation_is_fatal(self):
        from server.batch_backend import LiveRequest
        from server.backend import Job
        entry = LiveRequest(Job(dict(ids=[1], max_tokens=8)), generation=2)
        backend = object.__new__(BatchedEngineBackend)
        backend.active = {900: entry}
        with self.assertRaises(RuntimeError):
            backend._validate_event(dict(event="token", request_id=900, slot_generation=1, id=2), "step", {900}, set())

if __name__ == "__main__": unittest.main()


class InterleaveBackendTest(unittest.TestCase):
    """Decode passes inside a prompt chunk: each pass publishes before the chunk ends."""

    def start(self, **env):
        popen = subprocess.Popen
        self.log = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "commands.jsonl"
        with patch.dict(os.environ, dict(FAKE_PREFILL_ROWS="16", FAKE_COMMAND_LOG=str(self.log), **env)), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            # 16-row chunks: interleave_min_rows=0 keeps their passes (the served default skips chunks below 300).
            backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3,
                                           prefill_interleave=8, interleave_min_rows=0)
        self.addCleanup(backend.close)
        return backend

    def collect(self, job):
        ids = []
        while True:
            event = job.events.get(timeout=5)
            if event["event"] == "error": return ids + ["error"]
            if event["event"] == "done": return ids
            ids.append(event["id"])

    def run_pair(self, backend, decode_tokens):
        decode = validate_request(dict(prompt="x", max_tokens=decode_tokens, spec=0), "completion", "crucible")
        prompt = validate_request(dict(prompt="x", max_tokens=2, spec=0), "completion", "crucible")
        decoder = backend.submit([1], decode)
        first = decoder.events.get(timeout=5)
        self.assertEqual(first["id"], 2)
        long = backend.submit(list(range(40)), prompt)
        return [2] + self.collect(decoder), self.collect(long)

    def test_passes_publish_inside_the_chunk(self):
        backend = self.start()
        self.assertEqual(backend.event_capacity, 32 + 9 * 6)
        decoded, prompted = self.run_pair(backend, 30)
        self.assertEqual(decoded, list(range(2, 32)))
        self.assertEqual(prompted, [40, 41])
        self.assertFalse(backend.fatal)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        interleaved = [c for c in commands if c["op"] == "prefill" and "request_ids" in c]
        self.assertTrue(interleaved)
        self.assertTrue(all(c["every"] == 8 and c["request_ids"] == [1] for c in interleaved))
        parts = [e for e in backend.trace.since(0) if e["op"] == "verify_step"]
        self.assertGreaterEqual(len(parts), 5)

    def test_a_decoder_that_finishes_in_a_pass_is_released(self):
        backend = self.start()
        decoded, prompted = self.run_pair(backend, 3)
        self.assertEqual(decoded, [2, 3, 4])
        self.assertEqual(prompted, [40, 41])
        self.assertFalse(backend.fatal)
        self.assertEqual(backend.active, {})

    def test_a_pass_without_its_boundary_is_fatal(self):
        backend = self.start(FAKE_PART_MODE="missing-boundary")
        decoded, _ = self.run_pair(backend, 30)
        self.assertEqual(decoded[-1], "error")
        self.assertTrue(backend.fatal)

    def test_interleave_range(self):
        for bad in (-1, 48, 1.5):
            with self.assertRaises(ValueError):
                BatchedEngineBackend("unused", "unused", context=128, streams=2, prefill_interleave=bad)
