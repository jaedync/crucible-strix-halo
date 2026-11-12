"""One RoPE mode for each request (M1 and M2).

The server decides the mode of a request once (Service.fit_reserve, the reply reserve rule). Every engine job of that
request carries it in the payload ("rope": "plain" | "yarn2" | "yarn4"): the first job, a serial first job that is
bounded at the think cap, the think-cap continuation (close_thinking, also with the shorter canonical render), the
serial resume (resume_bounded) and the end-of-turn continuation (keep_end_of_turn). Without the key the engine picks the
mode from the job's own fed positions, which differ from the request's in those jobs. A server with no auto word sends
no key, so the default payload keeps its bytes.
"""
import queue
from pathlib import Path
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from server import engine_messages, metrics, rope
from server.api import ServerDefaults, validate_request
from server.backend import EngineBackend
from server.http import Service, handler_for
from server.think_budget import think_cap

NATIVE = rope.ROPE_NATIVE_POSITIONS


def engine_mode(payload):
    """The mode that the engine gives a job (engine/runtime/rope_policy.h): the payload word when present, else the
    job's fed positions with the auto:2 gate."""
    if "rope" in payload:
        return payload["rope"]
    fed = len(payload["ids"]) + payload["max_tokens"] - 1
    return "plain" if fed <= NATIVE else "yarn2"


def backend(serial=True, context=524288):
    """A real EngineBackend whose submit builds the real payloads; no worker runs the jobs."""
    b = EngineBackend.__new__(EngineBackend)
    for key, value in dict(context=context, lookup=None, draft_detail=0, cache_tail=False, bounds_think_cap=serial,
                           event_capacity=8, engine_line_bytes=1 << 30, engine_id="e", counts_stall=False,
                           owned_jobs={}, jobs=queue.Queue(), ownership_lock=threading.Lock(),
                           closed=threading.Event(), fatal=None).items():
        setattr(b, key, value)
    return b


class Tokenizer:
    """think_close_ids gives the close text; closed_think_ids gives the canonical render, SHORTER tokens shorter."""

    def __init__(self, shorter=0):
        self.shorter = shorter

    def think_close_ids(self):
        return (7, 8)

    def closed_think_ids(self, reasoning, ids):
        return list(ids)[:max(0, len(ids) - self.shorter)]

    def end_of_turn_text_ids(self, token):
        return (5, 6)


def service(b, tokenizer=None, reserve=rope.YARN_REPLY_RESERVE, word="auto:2"):
    return Service(b, tokenizer or Tokenizer(), defaults=ServerDefaults(), rope_reserve=reserve, rope_word=word)


def completion(max_tokens, **body):
    return validate_request(dict(prompt="x", max_tokens=max_tokens, **body), "completion", "crucible")


def payloads(b):
    out = []
    while not b.jobs.empty():
        out.append(b.jobs.get_nowait().payload)
    return out


class RequestModeTest(unittest.TestCase):
    def test_the_fit_decides_the_mode_once(self):
        s = service(backend())
        self.assertEqual(s.fit(completion(32768), 232000).rope_mode, "yarn2")
        self.assertEqual(s.fit(completion(200000), 100000).rope_mode, "plain")  # clamped
        self.assertEqual(s.fit(completion(8000), 250000).rope_mode, "plain")
        self.assertEqual(s.fit(completion(8000), 258000).rope_mode, "yarn2")
        # auto:4: the smallest factor that holds the request.
        s4 = Service(backend(context=1048576), Tokenizer(), rope_reserve=rope.YARN_REPLY_RESERVE, rope_word="auto:4")
        self.assertEqual(s4.fit(completion(64), 600000).rope_mode, "yarn4")
        self.assertEqual(s4.fit(completion(64), 300000).rope_mode, "yarn2")
        self.assertEqual(rope.request_mode("auto:2", 524288, 1), "yarn2")
        self.assertIsNone(rope.request_mode("auto:2", 524288, 2))

    def test_m1_a_serial_bounded_first_job_keeps_the_server_mode(self):
        # The review's repro: prompt 232,000 + max_tokens 32,768 is YaRN, but the first job feeds 259,852 positions.
        b = backend(serial=True)
        s = service(b)
        p, m = 232000, 32768
        self.assertIsNone(rope.plain_max_tokens(p, m, rope.YARN_REPLY_RESERVE))
        request = s.fit(completion(m), p)
        with patch.object(metrics, "emit"):
            job = b.submit([1] * p, request, think_cap=think_cap(m).tokens)
        self.assertLess(len(job.payload["ids"]) + job.payload["max_tokens"] - 1, NATIVE)  # the bounded count
        self.assertEqual(engine_mode(job.payload), "yarn2")

    def test_m2_a_shorter_canonical_continuation_keeps_yarn(self):
        # Original fed 262,199 (YaRN). After the cap, the canonical render is 60 tokens shorter, so the continuation
        # feeds 262,139 positions: without the word the engine would run it plain on the pinned slot.
        b = backend(serial=False)
        s = service(b, Tokenizer(shorter=60))
        p, m = 262000, 200
        request = s.fit(completion(m), p)
        self.assertEqual(request.rope_mode, "yarn2")
        handler = handler_for(s)
        first = SimpleNamespace(turn={}, cancelled=threading.Event(), successor=None)
        output = SimpleNamespace(ids=list(range(100, 200)), closed=None, think=SimpleNamespace(reasoning="r"))
        output.token = output.ids.append
        fake = SimpleNamespace(jobs=[first])
        with patch.object(metrics, "emit"):
            continuation = handler.close_thinking(fake, request, [1] * p, output, SimpleNamespace(tokens=100), {})
        job = fake.jobs[-1]
        self.assertEqual(job.payload["ids"], continuation)
        self.assertLessEqual(len(continuation) + job.payload["max_tokens"] - 1, NATIVE)
        self.assertEqual(engine_mode(job.payload), "yarn2")

    def test_the_serial_resume_keeps_the_mode(self):
        b = backend(serial=True)
        s = service(b)
        p, m = 232000, 32768
        request = s.fit(completion(m), p)
        handler = handler_for(s)
        bounded = SimpleNamespace(bounded_at=100)
        output = SimpleNamespace(ids=list(range(100)), think=SimpleNamespace(thinking=False))
        fake = SimpleNamespace(jobs=[bounded])
        with patch.object(metrics, "emit"):
            handler.resume_bounded(fake, request, [1] * p, output, bounded, 0, dict(finish_reason="length"), {})
        self.assertEqual(engine_mode(fake.jobs[-1].payload), "yarn2")

    def test_the_end_of_turn_continuation_keeps_the_mode(self):
        b = backend(serial=True)
        s = service(b)
        p, m = 232000, 32768
        request = s.fit(completion(m), p)
        request = replace(request, kind="chat")
        handler = handler_for(s)
        output = SimpleNamespace(ids=list(range(10)), think=SimpleNamespace(thinking=True), tools=None,
                                 end_of_turn_kept={"reasoning": 0, "tool_call": 0}, stop_filter=SimpleNamespace(stopped=False))
        output.token = lambda piece, t=None: output.ids.append(piece)
        job = SimpleNamespace(turn={})
        fake = SimpleNamespace(jobs=[job], wait_event=lambda j, first=False: dict(event="done"))
        with patch.object(s, "validate_event", return_value=None), patch.object(metrics, "emit"):
            result = handler.keep_end_of_turn(fake, request, [1] * p, [1] * p, output, job, 248046, 0, {},
                                              SimpleNamespace(tokens=4000))
        self.assertEqual(result, "continued")
        payload = fake.jobs[-1].payload
        self.assertLess(len(payload["ids"]) + payload["max_tokens"] - 1, NATIVE)  # bounded at the cap that is left
        self.assertEqual(engine_mode(payload), "yarn2")

    def test_a_plain_request_sends_plain(self):
        b = backend(serial=True)
        s = service(b)
        request = s.fit(completion(200000), 100000)
        with patch.object(metrics, "emit"):
            job = b.submit([1] * 100000, request)
        self.assertEqual(job.payload["rope"], "plain")

    def test_a_default_server_sends_no_mode_key(self):
        b = backend(serial=True, context=262144)
        s = Service(b, Tokenizer(), clamp_max_tokens=True)
        request = s.fit(completion(200000), 100000)
        self.assertIsNone(request.rope_mode)
        with patch.object(metrics, "emit"):
            job = b.submit([1] * 100000, request, think_cap=think_cap(request.max_tokens).tokens)
        self.assertNotIn("rope", job.payload)
        # The forced test word has no reserve and no word: the engine keeps every sequence in its mode.
        self.assertIsNone(Service(backend(), Tokenizer()).fit(completion(64), 300000).rope_mode)


class StartTextTest(unittest.TestCase):
    """Review L2: the context start error is built from the RoPE cap; every text that it can be is a fixed text."""

    def test_the_context_texts_are_known(self):
        hint = " (a context above 262144 needs CRUCIBLE_ROPE_MODE=yarn:F or auto:F with 262144*F >= ctx)"
        for cap in (262144, 524288, 1048576):
            for what in ("serve context", "context"):
                text = f"{what} must be an integer in [1,{cap}]"
                self.assertEqual(engine_messages.loggable(text), text)
                if cap < 1048576:
                    self.assertEqual(engine_messages.loggable(text + hint), text + hint)
        self.assertEqual(engine_messages.loggable("serve context must be an integer in [1,262145]"), "other")
        source = (Path(__file__).parents[2] / "engine/runtime/rope_mode.h").read_text()
        self.assertIn(hint.strip()[:-1], source)  # the engine's own hint (without the closing parenthesis)
        self.assertIn('" must be an integer in [1,"', source)


if __name__ == "__main__":
    unittest.main()
