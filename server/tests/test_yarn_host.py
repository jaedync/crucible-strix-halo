"""Y2 host side of YaRN: the long context flag, the "max" room that keeps a short prompt
plain, and the rope fields of the turn line."""
import io
import os
import unittest
from unittest.mock import MagicMock, patch

from server import metrics, rope
from server.api import ServerDefaults, validate_request
from server.http import Service
from server.tests.test_serial_flag import refused, run_main

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


class LongContextFlagTest(unittest.TestCase):
    def test_a_context_above_262144_turns_yarn_on(self):
        # One binary; the context option alone turns YaRN on for the sequences that need it.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CRUCIBLE_ROPE_MODE", None)
            seen = {}
            _, batched = run_main(["--ctx", "524288", "--streams", "2"],
                                  on_backend=lambda: seen.update(word=os.environ.get("CRUCIBLE_ROPE_MODE")))
            self.assertEqual(seen["word"], "auto:2")
            self.assertEqual(batched.call_args.args[2], 524288)
            # The default context never sets the word: the engine starts as today.
            os.environ.pop("CRUCIBLE_ROPE_MODE", None)
            run_main(["--ctx", "262144"], on_backend=lambda: seen.update(word=os.environ.get("CRUCIBLE_ROPE_MODE")))
            self.assertIsNone(seen["word"])
            # Factor 4 (1M) is not a default yet: it needs the explicit word.
            os.environ.pop("CRUCIBLE_ROPE_MODE", None)
            self.assertIn("CRUCIBLE_ROPE_MODE=auto:4", refused(["--ctx", "1048576"]))
            os.environ.pop("CRUCIBLE_ROPE_MODE", None)
            self.assertIn("--state-dir", refused(["--ctx", "524288", "--conversation-cache-mib", "64",
                                                  "--state-dir", "~/states"]))

    def test_the_word_must_hold_the_context(self):
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "auto:2"}):
            text = refused(["--ctx", "1048576"])
        self.assertIn("auto:4", text)
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "plain"}):
            self.assertIn("CRUCIBLE_ROPE_MODE", refused(["--ctx", "262145"]))
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "auto:4"}):
            self.assertIn("--ctx", refused(["--ctx", "1048577"]))

    def test_a_rope_word_refuses_the_disk_tier(self):
        # v1 (review M2): until the disk tier records the mode, no YaRN word runs with --state-dir.
        for word in ("auto:2", "yarn:2", "auto:4"):
            with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": word}):
                text = refused(["--conversation-cache-mib", "64", "--state-dir", "~/states"])
            self.assertIn("--state-dir", text)
            self.assertIn("CRUCIBLE_ROPE_MODE", text)

    def test_the_long_profile_starts(self):
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "auto:2"}):
            _, batched = run_main(["--ctx", "524288", "--streams", "3", "--short-streams", "2", "--short-ctx", "262144"])
        self.assertEqual(batched.call_args.args[2], 524288)
        self.assertEqual(batched.call_args.kwargs["short_streams"], 2)
        self.assertEqual(batched.call_args.kwargs["short_ctx"], 262144)
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "auto:2"}):
            _, batched = run_main(["--ctx", "524288", "--streams", "2"])
        self.assertEqual(batched.call_args.args[2], 524288)
        with patch.dict(os.environ, {"CRUCIBLE_ROPE_MODE": "yarn:2"}):
            _, batched = run_main(["--ctx", "524288", "--streams", "1"])
        self.assertEqual(batched.call_args.args[2], 524288)

    def test_the_daily_context_is_unchanged(self):
        os.environ.pop("CRUCIBLE_ROPE_MODE", None)
        _, batched = run_main(["--ctx", "262144", "--streams", "4"])
        self.assertEqual(batched.call_args.args[2], 262144)


class LongBackend:
    context = 524288


class MaxRoomTest(unittest.TestCase):
    def service(self, **options):
        return Service(LongBackend(), None, defaults=ServerDefaults(max_tokens="max"), **options)

    def request(self, **body):
        return validate_request(CHAT | body, "chat", "crucible", defaults=ServerDefaults(max_tokens="max"))

    def test_max_keeps_a_prompt_that_fits_262144_plain(self):
        # The forced word (no reply reserve): the room ends at the native limit, fed positions = 262,144.
        self.assertEqual(self.service().fit(self.request(), 1000).max_tokens, 262144 - 1000 + 1)
        self.assertEqual(self.service().fit(self.request(), 262144).max_tokens, 1)

    def test_max_gives_a_longer_prompt_the_long_room(self):
        self.assertEqual(self.service().fit(self.request(), 300000).max_tokens, 524288 - 300000 + 1)


NATIVE = 262144
RESERVE = 32768


def fed(prompt, request):
    return prompt + request.max_tokens - 1


class ReplyReserveTest(unittest.TestCase):
    """The reply reserve rule (server/rope.py plain_max_tokens): a request is plain
    when prompt + min(max_tokens, R) - 1 <= 262,144, else YaRN. A plain request's max_tokens is clamped to the plain
    limit. R = --yarn-reply-reserve (default 32,768). It applies only with an auto word (--ctx above 262,144)."""

    def service(self, default=128, **options):
        return Service(LongBackend(), None, defaults=ServerDefaults.checked(max_tokens=default),
                       rope_reserve=RESERVE, **options)

    def request(self, default=128, **body):
        return validate_request(CHAT | body, "chat", "crucible", defaults=ServerDefaults.checked(max_tokens=default))

    def test_240k_and_32k_is_yarn(self):
        out = self.service().fit(self.request(max_tokens=32000), 240000)
        self.assertEqual(out.max_tokens, 32000)
        self.assertGreater(fed(240000, out), NATIVE)
        self.assertEqual(out.rope_clamped_from, 0)

    def test_100k_and_200k_is_plain_with_a_clamp(self):
        for service in (self.service(), self.service(clamp_max_tokens=True)):
            out = service.fit(self.request(max_tokens=200000), 100000)
            self.assertEqual(out.max_tokens, NATIVE - 100000 + 1)
            self.assertEqual(fed(100000, out), NATIVE)
            self.assertEqual(out.rope_clamped_from, 200000)

    def test_100k_and_no_max_tokens_is_plain(self):
        # The server default is 128 (api.ServerDefaults): plain, no clamp.
        out = self.service().fit(self.request(), 100000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (128, 0))
        # --default-max-tokens max: the context room, then the clamp to the plain limit.
        out = self.service("max").fit(self.request("max"), 100000)
        self.assertEqual(out.max_tokens, NATIVE - 100000 + 1)
        self.assertEqual(out.rope_clamped_from, 524288 - 100000 + 1)
        self.assertFalse(out.max_tokens_all)

    def test_250k_and_8k_is_plain(self):
        out = self.service().fit(self.request(max_tokens=8000), 250000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (8000, 0))
        self.assertLessEqual(fed(250000, out), NATIVE)

    def test_258k_and_8k_is_yarn(self):
        out = self.service().fit(self.request(max_tokens=8000), 258000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (8000, 0))
        self.assertGreater(fed(258000, out), NATIVE)

    def test_the_edges_of_the_reserve(self):
        # prompt + R - 1 = 262,144 is plain (clamped); one more prompt token is YaRN with the full count.
        prompt = NATIVE - RESERVE + 1
        out = self.service().fit(self.request(max_tokens=40000), prompt)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (RESERVE, 40000))
        out = self.service().fit(self.request(max_tokens=40000), prompt + 1)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (40000, 0))
        # A prompt above 262,144 is always YaRN; max gives it the long room.
        out = self.service("max").fit(self.request("max"), 300000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (524288 - 300000 + 1, 0))
        # max with a prompt at the limit: R does not fit, so it runs YaRN with the long room (not 1 token of plain).
        out = self.service("max").fit(self.request("max"), NATIVE)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (524288 - NATIVE + 1, 0))

    def test_the_context_clamp_still_applies_to_yarn(self):
        out = self.service(clamp_max_tokens=True).fit(self.request(max_tokens=200000), 400000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (524288 - 400000 + 1, 0))

    def test_no_reserve_keeps_todays_fit(self):
        # A server with no auto word has no reserve: the fit is the one of today.
        class Daily:
            context = 262144
        service = Service(Daily(), None, clamp_max_tokens=True)
        out = service.fit(self.request(max_tokens=200000), 100000)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (262144 - 100000 + 1, 0))
        out = service.fit(self.request(max_tokens=100), 100000)
        self.assertIs(out, out)
        self.assertEqual((out.max_tokens, out.rope_clamped_from), (100, 0))

    def test_the_rule_function(self):
        # (prompt, max_tokens) -> the plain max_tokens, or None for YaRN.
        self.assertIsNone(rope.plain_max_tokens(240000, 32000, RESERVE))
        self.assertEqual(rope.plain_max_tokens(100000, 200000, RESERVE), 162145)
        self.assertEqual(rope.plain_max_tokens(100000, 128, RESERVE), 128)
        self.assertEqual(rope.plain_max_tokens(250000, 8000, RESERVE), 8000)
        self.assertIsNone(rope.plain_max_tokens(258000, 8000, RESERVE))
        self.assertIsNone(rope.plain_max_tokens(262145, 1, RESERVE))
        self.assertEqual(rope.plain_max_tokens(262144, 1, RESERVE), 1)


def service_kwargs(argv, environ):
    with patch.dict(os.environ, environ, clear=False):
        for name in ("CRUCIBLE_ROPE_MODE",):
            if name not in environ:
                os.environ.pop(name, None)
        with patch("server.__main__.Service") as service:
            run_main(argv)
    return service.call_args.kwargs


class ReplyReserveFlagTest(unittest.TestCase):
    def test_the_reserve_is_on_with_an_auto_word_only(self):
        self.assertEqual(service_kwargs(["--ctx", "524288", "--streams", "2"], {})["rope_reserve"], 32768)
        self.assertEqual(service_kwargs(["--ctx", "524288", "--streams", "2"], {})["rope_word"], "auto:2")
        self.assertEqual(service_kwargs(["--ctx", "262144"], {"CRUCIBLE_ROPE_MODE": "auto:4"})["rope_word"], "auto:4")
        self.assertIsNone(service_kwargs(["--ctx", "262144", "--streams", "4"], {})["rope_word"])
        self.assertEqual(service_kwargs(["--ctx", "524288", "--streams", "2", "--yarn-reply-reserve", "8192"], {})
                         ["rope_reserve"], 8192)
        self.assertEqual(service_kwargs(["--ctx", "262144"], {"CRUCIBLE_ROPE_MODE": "auto:2"})["rope_reserve"], 32768)
        # Today's server and the forced test word: no reserve, the fit of today.
        self.assertIsNone(service_kwargs(["--ctx", "262144", "--streams", "4"], {})["rope_reserve"])
        self.assertIsNone(service_kwargs(["--ctx", "524288"], {"CRUCIBLE_ROPE_MODE": "yarn:2"})["rope_reserve"])

    def test_the_flag_needs_an_auto_word_and_a_count(self):
        os.environ.pop("CRUCIBLE_ROPE_MODE", None)
        self.assertIn("--yarn-reply-reserve", refused(["--ctx", "262144", "--yarn-reply-reserve", "8192"]))
        self.assertIn("--yarn-reply-reserve", refused(["--ctx", "524288", "--yarn-reply-reserve", "0"]))
        self.assertIn("--yarn-reply-reserve", refused(["--ctx", "524288", "--yarn-reply-reserve", "262145"]))


class RopeCounterLineTest(unittest.TestCase):
    """One crucible_rope line with numbers for each clamp, each cut (a clamped reply that ends at the plain limit) and
    each crossing (the engine's rope_crossings count goes up)."""

    def setUp(self):
        self.counters = rope.RopeCounters()

    def test_a_clamp_line(self):
        line = self.counters.clamp_line(7, 100000, 200000, 162145)
        self.assertTrue(line.startswith("crucible_rope "))
        for text in ('"event": "clamp"', '"request_id": 7', '"prompt_tokens": 100000', '"max_tokens": 200000',
                     '"plain_max_tokens": 162145', '"clamps": 1'):
            self.assertIn(text, line)
        self.assertIn('"clamps": 2', self.counters.clamp_line(8, 1, 2, 1))

    def test_a_cut_line_only_for_a_clamped_reply_at_its_limit(self):
        turn = dict(request_id=3, rope_cap=162145)
        self.assertIsNone(self.counters.done_lines({}, dict(finish_reason="length", completion_tokens=162145)))
        self.assertIsNone(self.counters.done_lines(turn, dict(finish_reason="stop", completion_tokens=10)))
        self.assertIsNone(self.counters.done_lines(turn, dict(finish_reason="length", completion_tokens=10)))
        lines = self.counters.done_lines(turn, dict(finish_reason="length", completion_tokens=162145, prompt_tokens=100000))
        self.assertEqual(len(lines), 1)
        for text in ('"event": "cut"', '"request_id": 3', '"completion_tokens": 162145', '"cuts": 1'):
            self.assertIn(text, lines[0])

    def test_a_crossing_line_when_the_engine_count_goes_up(self):
        self.assertIsNone(self.counters.done_lines({}, dict(rope_crossings=0)))
        lines = self.counters.done_lines(dict(request_id=4), dict(rope_crossings=1, rope_crossing_tokens=262000))
        self.assertEqual(len(lines), 1)
        for text in ('"event": "crossing"', '"request_id": 4', '"crossings": 1', '"crossing_tokens": 262000'):
            self.assertIn(text, lines[0])
        self.assertIsNone(self.counters.done_lines({}, dict(rope_crossings=1)))  # the same count: no new line
        self.assertIsNone(self.counters.done_lines({}, dict(rope_crossings="x")))

    def test_a_plain_reply_line_above_200k_context_in_a_long_profile(self):
        # Review 0530 item 2: the reply length of each plain reply above 200K context, to check R = 32,768 later.
        turn = dict(request_id=11, rope_mode="plain")
        self.assertIsNone(self.counters.done_lines(turn, dict(prompt_tokens=200000, completion_tokens=900)))
        self.assertIsNone(self.counters.done_lines(dict(request_id=12, rope_mode="yarn2"),
                                                   dict(prompt_tokens=300000, completion_tokens=900)))
        self.assertIsNone(self.counters.done_lines(dict(request_id=13), dict(prompt_tokens=250000, completion_tokens=900)))
        self.assertIsNone(self.counters.done_lines(turn, dict(prompt_tokens=250000, completion_tokens="x")))
        lines = self.counters.done_lines(turn, dict(prompt_tokens=200001, completion_tokens=31000, finish_reason="stop"))
        self.assertEqual(len(lines), 1)
        for text in ('"event": "plain_reply"', '"request_id": 11', '"prompt_tokens": 200001', '"completion_tokens": 31000',
                     '"plain_replies": 1'):
            self.assertIn(text, lines[0])
        self.assertIn('"plain_replies": 2', self.counters.done_lines(turn, dict(prompt_tokens=260000, completion_tokens=1))[0])

    def test_the_backend_gives_the_turn_its_rope_mode(self):
        from server.backend import EngineBackend
        job = MagicMock()
        job.turn = dict(request_id=6, rope_mode="plain", enabled=False, log=False, t_first_token=None, t_submit=0.0)
        with patch.object(rope, "COUNTERS", rope.RopeCounters()), patch.object(metrics, "emit") as emit:
            EngineBackend._stamp(job, dict(event="done", finish_reason="stop", completion_tokens=40, prompt_tokens=230000))
        lines = [c.args[0] for c in emit.call_args_list if c.args[0].startswith("crucible_rope ")]
        self.assertEqual(len(lines), 1)
        self.assertIn('"event": "plain_reply"', lines[0])
        source = open(rope.__file__.replace("rope.py", "backend.py")).read()
        self.assertIn('job.turn["rope_mode"] = request.rope_mode', source)

    def test_the_backend_writes_the_lines(self):
        from server.backend import EngineBackend
        job = MagicMock()
        job.turn = dict(request_id=5, rope_cap=10, enabled=False, log=False, t_first_token=None, t_submit=0.0)
        with patch.object(rope, "COUNTERS", rope.RopeCounters()), patch.object(metrics, "emit") as emit:
            EngineBackend._stamp(job, dict(event="done", finish_reason="length", completion_tokens=10, prompt_tokens=1))
        lines = [c.args[0] for c in emit.call_args_list if c.args[0].startswith("crucible_rope ")]
        self.assertEqual(len(lines), 1)
        self.assertIn('"event": "cut"', lines[0])

    def test_the_clamp_line_comes_at_done_with_the_request_id(self):
        # The batch scheduler gives the turn its request_id at admission, after submit (the test machine smoke: request_id null).
        turn = dict(request_id=9, rope_clamp=[262104, 256, 41], rope_cap=41)
        lines = self.counters.done_lines(turn, dict(finish_reason="length", completion_tokens=41, prompt_tokens=262104))
        self.assertEqual(len(lines), 2)
        for text in ('"event": "clamp"', '"request_id": 9', '"prompt_tokens": 262104', '"max_tokens": 256',
                     '"plain_max_tokens": 41', '"clamps": 1'):
            self.assertIn(text, lines[0])
        self.assertIn('"event": "cut"', lines[1])
        self.assertIsNone(self.counters.done_lines(dict(rope_clamp="x"), dict(finish_reason="stop")))

    def test_submit_marks_the_turn_and_writes_no_line(self):
        from server.backend import EngineBackend
        from server.api import validate_request
        backend = EngineBackend.__new__(EngineBackend)
        for name, value in dict(context=524288, lookup=None, draft_detail=0, cache_tail=False, bounds_think_cap=False,
                                event_capacity=8, engine_line_bytes=None, engine_id="e", counts_stall=False,
                                owned_jobs={}).items():
            setattr(backend, name, value)
        import queue as q
        import threading
        backend.jobs, backend.ownership_lock, backend.closed, backend.fatal = q.Queue(), threading.Lock(), threading.Event(), None
        request = validate_request({"prompt": "x", "max_tokens": 41}, "completion", "crucible")
        from dataclasses import replace
        request = replace(request, rope_clamped_from=256)
        with patch.object(metrics, "emit") as emit:
            job = backend.submit([1] * 262104, request)
        self.assertEqual(job.turn["rope_clamp"], [262104, 256, 41])
        self.assertEqual(job.turn["rope_cap"], 41)
        self.assertFalse([c for c in emit.call_args_list if c.args[0].startswith("crucible_rope ")])

    def test_a_plain_process_writes_no_rope_line(self):
        from server.backend import EngineBackend
        job = MagicMock()
        job.turn = dict(request_id=5, enabled=False, log=False, t_first_token=None, t_submit=0.0)
        with patch.object(rope, "COUNTERS", rope.RopeCounters()), patch.object(metrics, "emit") as emit:
            EngineBackend._stamp(job, dict(event="done", finish_reason="length", completion_tokens=10, prompt_tokens=1))
        self.assertFalse([c for c in emit.call_args_list if c.args[0].startswith("crucible_rope ")])


class RopeMetricsTest(unittest.TestCase):
    def test_the_turn_line_keeps_the_rope_words_and_counts(self):
        event = dict(prompt_tokens=300000, completion_tokens=4, finish_reason="length", cached_tokens=0, rope="yarn2",
                     rope_yarn_opens=3, rope_crossings=1, rope_crossing_tokens=262000)
        line = metrics.turn_line({"request_id": 1}, event)
        for text in ('"rope": "yarn2"', '"rope_yarn_opens": 3', '"rope_crossings": 1', '"rope_crossing_tokens": 262000'):
            self.assertIn(text, line)

    def test_an_unknown_rope_word_is_dropped(self):
        line = metrics.turn_line({"request_id": 1}, dict(prompt_tokens=1, rope="yarn:2 secret", rope_crossings="x"))
        self.assertNotIn("rope", line)


if __name__ == "__main__":
    unittest.main()
