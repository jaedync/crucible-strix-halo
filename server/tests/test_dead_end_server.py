"""A grammar dead end in the middle of an output ends only its own request.

The engine sends one recoverable constraint_dead_end error event in place of the request's next token and frees the
slot (engine/serve/batch_session.h EndDeadEnd). The worker delivers the tokens before it, then the error (503), releases
the request, and keeps every peer of the command running. FAKE_DEAD_END=7 marks the request whose prompt ends on 7.
"""
import io
import json
import os
import subprocess
import sys
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from server import structured
from server.api import validate_request
from server.batch_backend import BatchedEngineBackend
from server.http import Service, handler_for

FAKE = Path(__file__).parent / "fake_batch_engine.py"
DEAD_ERROR = dict(event="error", error=structured.DEAD_END, status=503)


def start_backend(test, after, op="", **options):
    popen = subprocess.Popen
    env = dict(FAKE_DEAD_END="7", FAKE_DEAD_END_AFTER=str(after), FAKE_DEAD_END_OP=op, FAKE_PREFILL_ROWS="16")
    with patch.dict(os.environ, env), patch(
            "server.backend.subprocess.Popen",
            side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
        backend = BatchedEngineBackend("unused", "unused", context=128, streams=2, output_timeout=.3, **options)
    test.addCleanup(backend.close)
    return backend


def request(max_tokens=8, spec=0):
    return validate_request(dict(prompt="x", max_tokens=max_tokens, spec=spec), "completion", "crucible")


def drain(job):
    """The token IDs of a job and its last event (done or error)."""
    ids = []
    while True:
        event = job.events.get(timeout=5)
        if event["event"] != "token": return ids, event
        ids.append(event["id"])


class DeadEndBackendTest(unittest.TestCase):
    def check_dead(self, job, ids):
        got, last = drain(job)
        self.assertEqual(got, ids)
        self.assertEqual({key: last.get(key) for key in DEAD_ERROR}, DEAD_ERROR)

    def check_peer(self, job, ids):
        got, last = drain(job)
        self.assertEqual((got, last["event"]), (ids, "done"))

    def check_alive(self, backend):
        """Not fatal, no lease left, every slot free, and a new request is served."""
        self.assertFalse(backend.fatal)
        self.check_peer(backend.submit([20], request(3)), [21, 22, 23])
        self.assertEqual(backend.active, {})
        self.assertEqual(sorted(backend.free_contexts), sorted(backend.expected_contexts))

    def last_command(self, backend, job):
        """The last command (or decode pass) that held the job's request."""
        return [entry for entry in backend.trace.since(0) if job.turn["request_id"] in entry["request_ids"]][-1]

    def test_a_plain_step_with_a_peer(self):
        backend = start_backend(self, after=2, op="step")
        backend.packed_op, backend.pack_enter = "step", 1  # every decoder in one step command
        dead, peer = backend.submit([7], request()), backend.submit([9], request())
        self.check_dead(dead, [8, 9])
        self.check_peer(peer, list(range(10, 18)))
        last = self.last_command(backend, dead)
        self.assertEqual((last["op"], len(last["request_ids"])), ("step", 2))
        self.check_alive(backend)

    def test_a_verify_span_with_a_peer(self):
        """A packed verify step: the dead request publishes the first token of its span (11), then the error."""
        backend = start_backend(self, after=4, op="verify_step")
        backend.pack_enter = 1
        dead, peer = backend.submit([7], request(spec=7)), backend.submit([9], request())
        self.check_dead(dead, [8, 9, 10, 11])
        self.check_peer(peer, list(range(10, 18)))
        last = self.last_command(backend, dead)
        self.assertEqual((last["op"], len(last["request_ids"])), ("verify_step", 2))
        self.check_alive(backend)

    def test_a_spec_step(self):
        """A solo speculative step: two tokens of its three, then the error; a plain peer goes on."""
        backend = start_backend(self, after=3, op="spec_step")
        backend.pack_enter = 99  # no packed pass: the speculative request runs spec_step
        dead, peer = backend.submit([7], request(spec=7)), backend.submit([9], request())
        self.check_dead(dead, [8, 9, 10])
        self.check_peer(peer, list(range(10, 18)))
        self.assertEqual(self.last_command(backend, dead)["op"], "spec_step")
        self.check_alive(backend)

    def test_the_first_token_of_a_prefill(self):
        backend = start_backend(self, after=0, op="prefill")
        peer = backend.submit([9], request())
        dead = backend.submit([7], request())
        self.check_dead(dead, [])
        self.check_peer(peer, list(range(10, 18)))
        self.assertEqual(self.last_command(backend, dead)["op"], "prefill")
        self.check_alive(backend)

    def test_a_decode_pass_inside_a_prompt_chunk(self):
        """batch_part: the decoder ends in a pass; the chunk's own request and the later passes go on."""
        backend = start_backend(self, after=1, op="part", prefill_interleave=8, interleave_min_rows=0)
        dead = backend.submit([7], request(30))
        self.assertEqual(dead.events.get(timeout=5)["id"], 8)
        long = backend.submit(list(range(40)), request(2))
        ids, last = drain(dead)
        self.assertEqual(ids, list(range(9, 9 + len(ids))))
        self.assertEqual({key: last.get(key) for key in DEAD_ERROR}, DEAD_ERROR)
        self.check_peer(long, [40, 41])
        self.assertEqual(self.last_command(backend, dead)["op"], "verify_step")
        self.check_alive(backend)


class DeadEndProtocolTest(unittest.TestCase):
    """A dead end outside the engine's contract stays a protocol error: the worker fails, as before."""

    def check_fatal(self, mode):
        with patch.dict(os.environ, dict(FAKE_DEAD_END_MODE=mode)):
            backend = start_backend(self, after=2, op="step")
        _, last = drain(backend.submit([7], request()))
        self.assertEqual(last["event"], "error")
        self.assertTrue(backend.fatal)

    def test_output_after_the_dead_end(self):
        self.check_fatal("output-after")

    def test_a_stale_generation(self):
        self.check_fatal("generation")


class Tokenizer:
    """The prompt is [7], the request that dead-ends."""
    def encode(self, request): return [7]
    def cache_prefix_tokens(self, request, ids): return 0
    def stream(self): return None
    def step(self, stream, token): return "a"
    def decode(self, ids): return "a" * len(ids)


class DeadEndHttpTest(unittest.TestCase):
    def call(self, after, stream):
        backend = start_backend(self, after=after, op="step")
        handler = handler_for(Service(backend, Tokenizer()))
        h = handler.__new__(handler)
        h.path = "/v1/chat/completions"
        body = json.dumps(dict(messages=[dict(role="user", content="hi")], max_tokens=8, spec=0, stream=stream)).encode()
        h.headers = Message()
        for key, value in {"Host": "127.0.0.1:8790", "Content-Length": str(len(body)),
                           "Content-Type": "application/json"}.items():
            h.headers[key] = value
        h.server = SimpleNamespace(server_address=("127.0.0.1", 8790))
        h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
        status = []
        h.send_response = status.append
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        h.do_POST()
        self.assertFalse(backend.fatal)
        self.assertEqual(backend.active, {})
        return status, h.wfile.getvalue()

    def test_a_response_is_a_503_with_the_api_error_shape(self):
        status, body = self.call(after=2, stream=False)
        self.assertEqual(status, [503])
        self.assertEqual(json.loads(body), dict(error=dict(message=structured.DEAD_END, type="server_error",
                                                           param=None, code=None)))

    def test_a_stream_gets_an_error_event_and_ends(self):
        status, body = self.call(after=2, stream=True)
        self.assertEqual(status, [200])
        chunks = [line[6:] for line in body.decode().split("\n\n") if line.startswith("data: ")]
        self.assertEqual(chunks[-1], "[DONE]")
        self.assertEqual(json.loads(chunks[-2]), dict(error=dict(message=structured.DEAD_END, type="server_error",
                                                                param=None, code=None)))
        text = "".join(choice["delta"].get("content", "") for chunk in chunks[:-2]
                       for choice in json.loads(chunk).get("choices", []))
        self.assertEqual(text, "aa")


if __name__ == "__main__": unittest.main()
