"""The server side of the verify span (CRUCIBLE_SPEC_SPAN, engine/runtime/spec_span.h).

A solo speculative step may keep up to kSpecSpanMax tokens (13). The batch backend must accept a step of that many
tokens and keep output room for all of them plus the done event. Older servers had both limits at 8 and 9
(the 9th token of a span 13 step raised "The engine returned an invalid batch token").
"""
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from server.batch_backend import SPEC_SPAN_MAX, BatchedEngineBackend

ROOT = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).parent / "fake_batch_engine.py"


def engine_span_max():
    text = (ROOT / "engine/runtime/spec_span.h").read_text()
    return int(re.search(r"inline constexpr int kSpecSpanMax = (\d+);", text).group(1))


class SpecSpanServerTests(unittest.TestCase):
    def test_limit_is_the_engine_one(self):
        self.assertEqual(SPEC_SPAN_MAX, engine_span_max())
        self.assertEqual(SPEC_SPAN_MAX, 13)

    def backend(self, max_tokens=64):
        backend = object.__new__(BatchedEngineBackend)
        job = SimpleNamespace(payload=dict(max_tokens=max_tokens, stop_ids=[99999], ids=[1], spec=7),
                              events=queue.Queue(maxsize=32), cancelled=threading.Event())
        backend.active = {1: SimpleNamespace(job=job, count=1, generation=5, ready=True, paused_since=None)}
        backend.output_timeout = 300.0
        return backend, job

    def test_a_spec_step_of_13_tokens_is_accepted(self):
        backend, _ = self.backend()
        token = lambda id: dict(event="token", request_id=1, slot_generation=5, id=id)
        for op in ("spec_step", "verify_step"):
            tokens = {}
            for id in range(2, 2 + SPEC_SPAN_MAX): backend._validate_event(token(id), op, {1}, tokens)
            self.assertEqual(len(tokens[1]), 13)
            with self.assertRaises(RuntimeError):  # the 14th token of one step
                backend._validate_event(token(50), op, {1}, tokens)

    def test_spec_entry_with_19_queued_events_stays_ready_on_default_engine(self):
        # Without CRUCIBLE_SPEC_SPAN a step publishes at most 8 tokens, so the room is 9.
        backend, job = self.backend()
        for _ in range(19): job.events.put_nowait(None)
        self.assertEqual(backend._ready(), [1])

    def room(self, span):
        """The most queued events that leave a speculative entry ready, on a backend started (fake engine) with
        CRUCIBLE_SPEC_SPAN set to span, or unset when span is None. The backend is closed first, so no worker thread
        reads the entry."""
        environ = {name: value for name, value in os.environ.items() if name != "CRUCIBLE_SPEC_SPAN"}
        if span is not None: environ["CRUCIBLE_SPEC_SPAN"] = span
        popen = subprocess.Popen
        with patch.dict(os.environ, environ, clear=True), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3)
        backend.close()
        _, job = self.backend()
        backend.active = {1: SimpleNamespace(job=job, count=1, generation=5, ready=True, paused_since=None)}
        backend.output_timeout = 300.0
        while job.events.qsize() < job.events.maxsize and backend._ready() == [1]: job.events.put_nowait(None)
        return job.events.qsize() - 1

    def test_output_room_follows_the_engine_span(self):
        # Room for one span and done: the default and 8 keep trunk's 9, 13 needs 14.
        self.assertEqual(self.room(None), 32 - 9)
        self.assertEqual(self.room(""), 32 - 9)
        self.assertEqual(self.room("8"), 32 - 9)
        self.assertEqual(self.room("10"), 32 - 11)
        self.assertEqual(self.room("13"), 32 - 14)
        self.assertEqual(self.room("99"), 32 - 14)  # the engine refuses to start; the room stays the widest


if __name__ == "__main__":
    unittest.main()
