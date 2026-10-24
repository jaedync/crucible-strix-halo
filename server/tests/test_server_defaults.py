"""Server-level sampling defaults (--temperature, --top-k, --top-p, --min-p) and --default-max-tokens."""
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server.api import ApiError, Request, ServerDefaults, validate_request
from server.backend import EngineBackend
from server.http import Service
from server.tests.serve_fixtures import invoke

TESTS = Path(__file__).parent
CHAT = dict(messages=[dict(role="user", content="hi")])
VOCAB = 248320


def check(body=None, kind="chat", **defaults):
    body = (CHAT if kind == "chat" else dict(prompt=[1, 2, 3])) | (body or {})
    return validate_request(body, kind, "crucible", defaults=ServerDefaults(**defaults))


class SamplingDefaultsTest(unittest.TestCase):
    def test_no_flag_gives_the_request_that_the_server_always_made(self):
        # Exactness: with no new flag, every field is the value that validate_request used before this feature.
        expected = Request("chat", "crucible", 128, 7, False, False, False, "xhigh", False, None,
                           validate_request(CHAT, "chat", "crucible").messages, temperature=0, top_k=0, top_p=1,
                           min_p=0, seed=5)
        body = CHAT | dict(seed=5)
        self.assertEqual(validate_request(body, "chat", "crucible"), expected)
        self.assertEqual(validate_request(body, "chat", "crucible", defaults=ServerDefaults()), expected)
        self.assertFalse(expected.max_tokens_all)

    def test_defaults_fill_the_missing_fields(self):
        request = check(dict(seed=1), temperature=0.6, top_k=20, top_p=0.95, min_p=0.05)
        self.assertEqual((request.temperature, request.top_k, request.top_p, request.min_p), (0.6, 20, 0.95, 0.05))

    def test_a_request_field_always_wins(self):
        body = dict(temperature=0, top_k=0, top_p=1, min_p=0)
        request = check(body, temperature=0.6, top_k=20, top_p=0.95, min_p=0.05)
        self.assertEqual((request.temperature, request.top_k, request.top_p, request.min_p), (0, 0, 1, 0))
        self.assertEqual(check(dict(temperature=1.5), temperature=0.6).temperature, 1.5)

    def test_a_bad_request_field_still_fails_with_a_default_set(self):
        for bad in (dict(temperature=3), dict(top_p=2), dict(top_k=-1), dict(min_p=None)):
            with self.assertRaises(ApiError, msg=str(bad)):
                check(bad, temperature=0.6, top_p=0.9)

    def test_top_k_default_is_capped_at_the_vocabulary(self):
        request = validate_request(CHAT, "chat", "crucible", vocab=1000, defaults=ServerDefaults(top_k=5000))
        self.assertEqual(request.top_k, 1000)

    def test_scoring_request_without_temperature_stays_greedy(self):
        body = dict(max_tokens=1, logprobs=True, top_logprobs=3)
        request = check(body, kind="completion", temperature=0.6, top_p=0.95)
        self.assertEqual((request.temperature, request.top_logprobs, request.top_p), (0, 3, 0.95))
        request = check(dict(max_tokens=1, top_logprobs=0), kind="completion", temperature=0.6)
        self.assertEqual((request.temperature, request.top_logprobs), (0, 1))

    def test_scoring_request_with_its_own_temperature_is_still_refused(self):
        with self.assertRaises(ApiError):
            check(dict(max_tokens=1, logprobs=True, temperature=0.5), kind="completion", temperature=0.6)

    def test_logprobs_off_uses_the_default(self):
        request = check(dict(max_tokens=1, logprobs=False), kind="completion", temperature=0.6)
        self.assertEqual(request.temperature, 0.6)

    def test_checked_defaults(self):
        self.assertEqual(ServerDefaults.checked(temperature=0.6, top_k=20, top_p=0.95, min_p=0.0, max_tokens="max"),
                         ServerDefaults(0.6, 20, 0.95, 0.0, "max"))
        for bad in (dict(temperature=2.5), dict(temperature=float("nan")), dict(top_p=1.5), dict(min_p=-0.1),
                    dict(top_k=-1), dict(top_k=VOCAB + 1), dict(max_tokens=0), dict(max_tokens=262145),
                    dict(max_tokens="all")):
            with self.assertRaises(ApiError, msg=str(bad)):
                ServerDefaults.checked(**bad)

    def test_non_default_description_names_only_the_changes(self):
        self.assertEqual(ServerDefaults().describe(), "")
        self.assertEqual(ServerDefaults(temperature=0.6, top_k=20, max_tokens="max").describe(),
                         "temperature 0.6, top_k 20, max_tokens max")
        self.assertEqual(ServerDefaults(max_tokens=512).describe(), "max_tokens 512")


class DefaultMaxTokensTest(unittest.TestCase):
    def test_default_is_128_and_a_number_replaces_it(self):
        self.assertEqual(check().max_tokens, 128)
        request = check(max_tokens=4096)
        self.assertEqual((request.max_tokens, request.max_tokens_all), (4096, False))

    def test_request_count_wins_and_keeps_its_rules(self):
        for field in ("max_tokens", "max_completion_tokens"):
            request = check({field: 64}, max_tokens="max")
            self.assertEqual((request.max_tokens, request.max_tokens_all), (64, False))
        with self.assertRaises(ApiError):
            check(dict(max_tokens=0), max_tokens="max")
        with self.assertRaises(ApiError):
            check(dict(max_tokens=1, max_completion_tokens=1), max_tokens="max")

    def test_max_marks_the_request(self):
        request = check(max_tokens="max")
        self.assertTrue(request.max_tokens_all)

    def test_logprobs_without_a_count_is_refused_as_before(self):
        with self.assertRaises(ApiError):
            check(dict(logprobs=True), kind="completion", max_tokens="max")


class FakeBackend:
    context = 100


class FitTest(unittest.TestCase):
    def service(self, **options):
        return Service(FakeBackend(), None, defaults=ServerDefaults(max_tokens="max"), **options)

    def request(self, **body):
        return validate_request(CHAT | body, "chat", "crucible", defaults=ServerDefaults(max_tokens="max"))

    def test_max_takes_the_room_after_the_prompt(self):
        fitted = self.service().fit(self.request(), 90)
        self.assertEqual(fitted.max_tokens, 11)  # the room that --clamp-max-tokens computes: 100 - 90 + 1
        self.assertFalse(fitted.max_tokens_all)

    def test_clamp_flag_does_not_change_it(self):
        self.assertEqual(self.service(clamp_max_tokens=True).fit(self.request(), 90).max_tokens, 11)

    def test_a_prompt_that_fills_the_context_still_fails_as_before(self):
        for prompt_count in (101, 500):
            fitted = self.service().fit(self.request(), prompt_count)
            self.assertEqual(fitted.max_tokens, 1)
            backend = EngineBackend.__new__(EngineBackend)
            backend.context = 100
            with self.assertRaises(ApiError) as caught:
                backend.submit(list(range(prompt_count)), fitted)
            self.assertIn("exceeds the context window", str(caught.exception))

    def test_a_request_with_its_own_count_is_not_filled(self):
        service = self.service()
        self.assertEqual(service.fit(self.request(max_tokens=64), 90).max_tokens, 64)

    def test_the_same_service_keeps_other_requests_unchanged(self):
        service = self.service()
        request = self.request()
        service.fit(request, 90)
        self.assertTrue(request.max_tokens_all)  # fit returns a copy: the request object is never changed
        self.assertEqual(service.fit(request, 10).max_tokens, 91)


class ThinkingTest(unittest.TestCase):
    def test_the_think_cap_sees_the_final_count(self):
        from server.tests.test_think_budget import ANSWER, IM_END, PROMPT, REASON, SequenceBackend, chat, run
        backend = SequenceBackend(([REASON] * 3000, "length"), ([ANSWER, IM_END], "stop"))
        backend.context = len(PROMPT) + 2047  # "max" gives 2048 tokens; the answer room is 1024 of them
        _, body = run(backend, chat(), defaults=ServerDefaults(max_tokens="max"))
        self.assertEqual(backend.submissions[0][1].max_tokens, 2048)
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_closed_at"], 1024)

    def test_a_small_context_leaves_the_answer_room_rule_unchanged(self):
        from server.tests.test_think_budget import ANSWER, IM_END, REASON, PROMPT, SequenceBackend, chat, run
        backend = SequenceBackend(([REASON] * 40, "length"), ([ANSWER, IM_END], "stop"))
        backend.context = len(PROMPT) + 39  # 40 tokens: below the 1024-token answer room, so the block closes at token 1
        run(backend, chat(), defaults=ServerDefaults(max_tokens="max"))
        self.assertEqual(backend.submissions[0][1].max_tokens, 40)
        self.assertEqual(backend.jobs[0].think_cap, 1)


class Tokenizer:
    def encode(self, request):
        return [5, 6, 7]

    def stream(self):
        return None

    def step(self, state, token):
        return "r "

    def decode(self, ids):
        return "r " * len(ids)


class EndToEndTest(unittest.TestCase):
    """Real serial backend on the fake engine: usage and finish_reason at the end of the context."""
    CONTEXT = 64

    @classmethod
    def setUpClass(cls):
        popen = subprocess.Popen
        env = dict(os.environ, FAKE_CONTEXT=str(cls.CONTEXT), FAKE_DELAY="0")
        with patch("server.backend.subprocess.Popen", side_effect=lambda *a, **k: popen(
                [sys.executable, str(TESTS / "fake_engine.py")], **{**k, "env": env})):
            cls.backend = EngineBackend("unused", "unused", context=cls.CONTEXT)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def ask(self, defaults=ServerDefaults(max_tokens="max"), **fields):
        body = dict(messages=[dict(role="user", content="hi")], return_token_ids=True) | fields
        response = invoke(self.backend, Tokenizer(), body, service_options=dict(defaults=defaults))
        return response.statuses[0], json.loads(response.body)

    def test_max_fills_the_context_and_ends_with_length(self):
        status, reply = self.ask()
        self.assertEqual(status, 200)
        self.assertEqual(reply["usage"]["prompt_tokens"], 3)
        self.assertEqual(reply["usage"]["completion_tokens"], self.CONTEXT - 3 + 1)
        self.assertEqual(reply["choices"][0]["finish_reason"], "length")
        self.assertEqual(len(reply["choices"][0]["token_ids"]), self.CONTEXT - 3 + 1)

    def test_a_request_count_keeps_today_rules(self):
        status, reply = self.ask(max_tokens=10)
        self.assertEqual((status, reply["usage"]["completion_tokens"]), (200, 10))
        status, reply = self.ask(max_tokens=self.CONTEXT)
        self.assertEqual(status, 400)
        self.assertIn("exceeds the context window", reply["error"]["message"])

    def test_no_flag_still_gives_128_and_fails_in_a_small_context(self):
        status, reply = self.ask(defaults=ServerDefaults())
        self.assertEqual(status, 400)  # 3 + 128 - 1 > 64
        self.assertIn("max_tokens (128)", reply["error"]["message"])


def parse(argv):
    """The Service options that main() builds for these flags, or the SystemExit code."""
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    err = io.StringIO()
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer") as load, patch("server.__main__.EngineBackend"), \
            patch("server.batch_backend.BatchedEngineBackend"), \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
            patch.object(startup, "Service") as service, contextlib.redirect_stderr(err):
        try:
            code = startup.main()
        except SystemExit as exit_:
            load.assert_not_called()  # a bad value stops before the engine or the tokenizer starts
            return exit_.code, err.getvalue()
    assert code == 0
    return service.call_args.kwargs, err.getvalue()


class FlagTest(unittest.TestCase):
    def test_no_flag_gives_the_built_in_defaults_and_no_extra_start_line(self):
        options, err = parse([])
        self.assertEqual(options["defaults"], ServerDefaults())
        self.assertNotIn("defaults:", err)

    def test_flags_reach_the_service(self):
        options, _ = parse(["--temperature", "0.6", "--top-k", "20", "--top-p", "0.95", "--min-p", "0",
                            "--default-max-tokens", "max"])
        self.assertEqual(options["defaults"], ServerDefaults(0.6, 20, 0.95, 0.0, "max"))
        options, _ = parse(["--default-max-tokens", "4096"])
        self.assertEqual(options["defaults"], ServerDefaults(max_tokens=4096))

    def test_start_line_names_the_non_default_values(self):
        _, err = parse(["--temperature", "0.6", "--top-p", "1", "--default-max-tokens", "max"])
        lines = [line for line in err.splitlines() if line.startswith("Request defaults:")]
        self.assertEqual(lines, ["Request defaults: temperature 0.6, max_tokens max"])

    def test_bad_values_are_refused_before_loading(self):
        for argv in (["--temperature", "2.1"], ["--temperature", "-1"], ["--temperature", "nan"],
                     ["--top-p", "1.1"], ["--min-p", "-0.5"], ["--top-k", "-1"], ["--top-k", "999999"],
                     ["--default-max-tokens", "0"], ["--default-max-tokens", "262145"],
                     ["--default-max-tokens", "all"], ["--default-max-tokens", "1.5"], ["--top-k", "x"]):
            with self.subTest(argv=argv):
                code, _ = parse(argv)
                self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
