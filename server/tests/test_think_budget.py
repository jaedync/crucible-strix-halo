"""The answer room and the thinking budget: the server closes a long think block at a fixed token, as Halogen does."""
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from server import metrics

from server.api import ApiError, validate_request
from server.tests.serve_fixtures import chunks, invoke
from server.think_budget import ThinkCap, answer_room, think_cap

REASON, ANSWER, END, SENTENCE, BREAK, IM_END = 30, 40, 50, 60, 61, 248046
CLOSE = (SENTENCE, END, BREAK)
PROMPT = [10, 11, 12]


class BudgetTokenizer:
    pieces = {REASON: "r ", ANSWER: "ok", END: "</think>", SENTENCE: "\n\nTime is up.\n", BREAK: "\n\n", IM_END: ""}

    def encode(self, request):
        return list(PROMPT)

    def stream(self):
        return None

    def step(self, stream, token):
        return self.pieces[token]

    def decode(self, ids):
        return "".join(self.pieces[token] for token in ids)

    def think_close_ids(self):
        return CLOSE


class SequenceBackend:
    """Each submit takes the next script: generated IDs and a finish reason. It records every submission."""
    context = 1 << 20

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.submissions, self.jobs = [], []

    def submit(self, ids, request, **options):
        tokens, reason, *cached = self.scripts.pop(0)
        opened = cached[1] if len(cached) > 1 else None
        self.submissions.append((list(ids), request, options))
        events = queue.Queue()
        for token in tokens:
            events.put(dict(event="token", id=token))
        done = dict(event="done", prompt_tokens=len(ids), completion_tokens=len(tokens), finish_reason=reason,
                    cached_tokens=cached[0] if cached else len(ids) - 1)
        events.put(done)
        turn = metrics.new_turn(request_id=len(self.jobs) + 1)
        turn["t_first_token"] = turn["t_done"] = turn["t_submit"]
        if opened is not None:
            turn["opened_cached"] = opened  # the batch backend's record of the engine's opened event
        if options.get("log_turn", True):
            # The real backends write this line when the engine reports done; log_turn=False leaves it to the server.
            metrics.emit(metrics.turn_line(metrics.finalize(turn, done), done))
        job = SimpleNamespace(events=events, cancelled=threading.Event(), turn=turn, think_cap=options.get("think_cap"))
        self.jobs.append(job)
        return job


class LinkedBackend(SequenceBackend):
    """A backend with continue_job (the batch rule): it records which job each continuation follows."""

    def continue_job(self, job, ids, request, **options):
        follow = self.submit(ids, request, **options)
        follow.after = job
        job.cancelled.set()
        return follow


def chat(**fields):
    return dict(model="crucible", messages=[dict(role="user", content="Hi")], enable_thinking=True) | fields


def run(backend, body, **service):
    response = invoke(backend, BudgetTokenizer(), body, service_options=service or None)
    if body.get("stream"):
        return response, chunks(response)
    return response, json.loads(response.body)


def logged(backend, body, **service):
    """The reply and the crucible_turn lines that the request wrote."""
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        _, reply = run(backend, body, **service)
    turns = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_turn ")]
    return reply, turns


class TurnLogTest(unittest.TestCase):
    """A log collector joins a crucible_turn line to its own request record by prompt and completion, so each request writes one line
    with the counts that the client sees."""

    def test_closed_turn_writes_one_line_with_client_totals(self):
        backend = SequenceBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", 9))
        reply, turns = logged(backend, chat(max_tokens=2048))
        self.assertEqual(len(turns), 1)
        (turn,) = turns
        usage = reply["usage"]
        self.assertEqual((turn["prompt_tokens"], turn["completion_tokens"], turn["finish"]),
                         (usage["prompt_tokens"], usage["completion_tokens"], "stop"))
        self.assertIsNone(turn["cached_tokens"])
        self.assertEqual((turn["think_closed_at"], turn["think_closed_by"]), (1024, "answer_room"))
        self.assertEqual((turn["cont_prompt_tokens"], turn["cont_cached_tokens"], turn["cont_completion_tokens"]),
                         (len(PROMPT) + 1024 + len(CLOSE), 9, 2))
        self.assertEqual(turn["request_id"], 1)
        self.assertEqual(turn["cont_request_id"], 2)

    def test_closed_turn_continues_through_the_backend_link(self):
        # The continuation names its first job, so the batch worker keeps that job's slot for it.
        backend = LinkedBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", 9))
        run(backend, chat(max_tokens=2048))
        first, follow = backend.jobs
        self.assertIs(follow.after, first)
        self.assertTrue(first.cancelled.is_set())
        self.assertEqual(len(backend.submissions[1][0]), len(PROMPT) + 1024 + len(CLOSE))

    def test_closed_turn_reports_the_first_jobs_restore(self):
        # Job 1 restored 2 of the 3 prompt tokens when it opened; the continuation's own hit (9) is not the client's.
        backend = SequenceBackend(([REASON] * 1500, "length", 0, 2), ([ANSWER, IM_END], "stop", 9))
        reply, turns = logged(backend, chat(max_tokens=2048))
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=2))
        self.assertEqual(turns[0]["cached_tokens"], 2)
        self.assertEqual(turns[0]["cont_cached_tokens"], 9)
        self.assertNotIn("replay_hit", turns[0])  # the opened event names no cache class: unknown, not false

    def test_capped_turn_that_closes_itself_writes_the_usual_line(self):
        backend = SequenceBackend(([REASON] * 5 + [END, BREAK, ANSWER, IM_END], "stop", 2))
        reply, turns = logged(backend, chat(max_tokens=2048))
        self.assertEqual(len(turns), 1)
        self.assertEqual((turns[0]["prompt_tokens"], turns[0]["completion_tokens"], turns[0]["cached_tokens"]),
                         (len(PROMPT), reply["usage"]["completion_tokens"], 2))
        self.assertNotIn("think_closed_at", turns[0])

    def test_truncated_close_writes_one_line(self):
        backend = SequenceBackend(([REASON] * 3, "length"))
        reply, turns = logged(backend, chat(max_tokens=3))
        self.assertEqual([(t["completion_tokens"], t["finish"]) for t in turns], [(3, "length")])

    def test_capped_turn_with_no_room_reports_the_restore(self):
        # max_tokens 1 with thinking. The cap closes the block at token 1 with no
        # room left, so the server cancels the job before its done event. The line and the reply still name the
        # restore of the opened event (2 of 3 prompt tokens), not cached null (a cold turn in the gateway).
        backend = SequenceBackend(([REASON], "length", 0, 2))
        reply, turns = logged(backend, chat(max_tokens=1))
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=2))
        self.assertEqual([(t["prompt_tokens"], t["cached_tokens"], t["completion_tokens"], t["finish"]) for t in turns],
                         [(len(PROMPT), 2, 1, "length")])
        self.assertNotIn("replay_hit", turns[0])  # the opened event names no cache class: unknown, not false

    def test_capped_turn_with_no_room_counts_its_prefill_rows(self):
        class Chunked(SequenceBackend):
            def submit(self, ids, request, **options):
                job = super().submit(ids, request, **options)
                metrics.add_chunk(job.turn, 0.0, 0.25, 1)  # the batch worker's record of the one prefill chunk
                return job
        _, turns = logged(Chunked(([REASON], "length", 0, 2)), chat(max_tokens=1))
        self.assertEqual((turns[0]["cached_tokens"], turns[0]["prefill_tokens"], turns[0]["prefill_s"]), (2, 1, 0.25))

    def test_capped_turn_without_an_opened_count_stays_unknown(self):
        # The serial backend has no opened event: the count stays null, never a guessed 0.
        reply, turns = logged(SequenceBackend(([REASON], "length", 0)), chat(max_tokens=1))
        self.assertNotIn("prompt_tokens_details", reply["usage"])
        self.assertIsNone(turns[0]["cached_tokens"])

    def test_stop_string_in_a_capped_turn_reports_the_restore(self):
        backend = SequenceBackend(([REASON, END, BREAK, ANSWER, ANSWER, IM_END], "stop", 0, 2))
        reply, turns = logged(backend, chat(max_tokens=2048, stop=["ok"]))
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=2))
        self.assertEqual([(t["cached_tokens"], t["finish"]) for t in turns], [(2, "stop")])

    def test_stop_string_reply_without_a_cap_reports_the_restore(self):
        # The stop path's reply change covers every stop-string reply, not only a capped one.
        backend = SequenceBackend(([ANSWER, ANSWER, IM_END], "stop", 0, 2))
        reply, _ = logged(backend, chat(max_tokens=2048, enable_thinking=False, stop=["ok"]))
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=2))

    def test_stop_string_reply_reports_a_whole_prompt_restore(self):
        # Re-send fix A: an identical re-send restores every prompt token (an End resume).
        backend = SequenceBackend(([ANSWER, ANSWER, IM_END], "stop", 0, 3))
        reply, _ = logged(backend, chat(max_tokens=2048, enable_thinking=False, stop=["ok"]))
        self.assertEqual(reply["usage"]["prompt_tokens"], len(backend.submissions[0][0]))
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=len(backend.submissions[0][0])))

    def test_stop_string_after_a_continuation_reports_the_first_jobs_restore(self):
        # The continuation restored 9 (the first job's snapshot); the client's restore is the first job's 2.
        backend = SequenceBackend(([REASON] * 1500, "length", 0, 2), ([ANSWER, ANSWER, IM_END], "stop", 9, 9))
        reply, turns = logged(backend, chat(max_tokens=2048, stop=["ok"]))
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
        self.assertEqual(reply["usage"]["prompt_tokens_details"], dict(cached_tokens=2))
        self.assertEqual([(t["cached_tokens"], t["finish"]) for t in turns], [(2, "stop")])
        self.assertNotIn("replay_hit", turns[0])

    def test_uncapped_turn_keeps_the_backend_line(self):
        backend = SequenceBackend(([REASON] * 3, "length"))
        _, turns = logged(backend, chat(max_tokens=3, enable_thinking=False))
        self.assertEqual(len(turns), 1)
        self.assertNotIn("log_turn", backend.submissions[0][2])

class PolicyTest(unittest.TestCase):
    def test_room_is_the_reference_rule(self):
        self.assertEqual(answer_room(32768), 4915)
        self.assertEqual(answer_room(2048), 1024)
        self.assertEqual(answer_room(2048, 100), 100)
        self.assertEqual(answer_room(2048, 0), 0)

    def test_cap(self):
        self.assertEqual(think_cap(32768), ThinkCap(27853, "answer_room"))
        self.assertEqual(think_cap(2048), ThinkCap(1024, "answer_room"))
        self.assertEqual(think_cap(256), ThinkCap(1, "answer_room"))
        self.assertEqual(think_cap(32768, budget=500), ThinkCap(500, "max_thinking_tokens"))
        self.assertEqual(think_cap(32768, budget=30000), ThinkCap(27853, "answer_room"))
        self.assertEqual(think_cap(32768, room_setting=0), None)
        self.assertEqual(think_cap(32768, budget=700, room_setting=0), ThinkCap(700, "max_thinking_tokens"))


class FieldsTest(unittest.TestCase):
    def budget(self, **fields):
        return validate_request(chat(**fields), "chat", "crucible").thinking_budget

    def test_every_alias(self):
        self.assertIsNone(self.budget())
        for name in ("max_thinking_tokens", "thinking_budget_tokens", "thinking_budget", "thinking_token_budget"):
            with self.subTest(name=name):
                self.assertEqual(self.budget(**{name: 700}), 700)
        self.assertEqual(self.budget(max_thinking_tokens=700, thinking_budget=700), 700)
        self.assertIsNone(self.budget(thinking_budget=None))

    def test_invalid_values(self):
        for fields in (dict(max_thinking_tokens=0), dict(thinking_budget="9"), dict(thinking_budget=True),
                       dict(max_thinking_tokens=5, thinking_budget_tokens=6)):
            with self.subTest(fields=fields), self.assertRaises(ApiError) as error:
                self.budget(**fields)
            self.assertIn(error.exception.param, ("max_thinking_tokens", "thinking_budget", "thinking_budget_tokens"))


class CloseTest(unittest.TestCase):
    def test_answer_room_closes_and_continues(self):
        # max_tokens 2048 leaves a 1024-token room, so the block closes after 1024 generated tokens.
        backend = SequenceBackend(([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))
        response, body = run(backend, chat(max_tokens=2048))
        self.assertEqual(response.statuses, [200])
        self.assertTrue(backend.jobs[0].cancelled.is_set())
        ids, request, options = backend.submissions[1]
        self.assertEqual(ids, PROMPT + [REASON] * 1024 + list(CLOSE))
        self.assertEqual(request.max_tokens, 2048 - 1024 - len(CLOSE))
        # Only the first job names its cap: the batch worker holds new chunks for its continuation from that token.
        self.assertEqual(backend.submissions[0][2], dict(options, think_cap=1024))
        self.assertNotIn("think_cap", options)
        message, usage = body["choices"][0]["message"], body["usage"]
        self.assertEqual(message["content"], "ok")
        self.assertEqual(message["reasoning_content"], ("r " * 1024 + "\n\nTime is up.").strip())
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(usage["prompt_tokens"], len(PROMPT))
        self.assertEqual(usage["completion_tokens"], 1024 + len(CLOSE) + 2)
        self.assertEqual(usage["completion_tokens_details"],
                         dict(reasoning_tokens=1024 + 2, reasoning_closed_at=1024, reasoning_closed_by="answer_room"))

    def test_continuation_prefills_the_close_as_the_next_turn_renders_it(self):
        # an earlier soak run: the model wrote a token split that the tokenizer does not make, so the next turn's prompt left the
        # replay tail inside the reasoning. The continuation prefills the reasoning again anyway, so it uses the
        # tokenizer's IDs of the text that the client sends back. The client still sees the generated tokens.
        class Canonical(BudgetTokenizer):
            def __init__(self, ids):
                self.ids, self.texts = ids, []

            def closed_think_ids(self, reasoning, ids=()):
                self.texts.append(reasoning)
                return self.ids
        for ids, expected in (((70, 71), PROMPT + [70, 71]), (tuple(range(1100)), PROMPT + [REASON] * 10 + list(CLOSE))):
            with self.subTest(length=len(ids)):
                backend = SequenceBackend(([REASON] * 40, "length"), ([ANSWER, IM_END], "stop"))
                tokenizer = Canonical(ids)
                response = invoke(backend, tokenizer, chat(max_tokens=4096, max_thinking_tokens=10))
                body = json.loads(response.body)
                # A longer canonical form could pass the context that the clamp left, so it keeps the generated IDs.
                self.assertEqual(backend.submissions[1][0], expected)
                self.assertEqual(tokenizer.texts, ["r " * 10 + "\n\nTime is up.\n"])
                self.assertEqual(backend.submissions[1][1].max_tokens, 4096 - 10 - len(CLOSE))
                self.assertEqual(body["usage"]["completion_tokens"], 10 + len(CLOSE) + 2)
                self.assertEqual(body["choices"][0]["message"]["content"], "ok")

    def test_closed_turn_does_not_report_the_continuations_cache_hit(self):
        # Job 1 is cold. The continuation restores the snapshot that job 1 wrote, which says nothing about the
        # client's prompt, and job 1 is cancelled before its done event, so the reply omits the cache count.
        backend = SequenceBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", len(PROMPT) + 1024 + 3))
        _, body = run(backend, chat(max_tokens=2048))
        self.assertNotIn("prompt_tokens_details", body["usage"])

    def test_stream_sends_nothing_past_the_close(self):
        backend = SequenceBackend(([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))
        _, events = run(backend, chat(max_tokens=2048, stream=True, stream_options=dict(include_usage=True)))
        deltas = [choice["delta"] for event in events for choice in event["choices"]]
        reasoning = "".join(delta.get("reasoning_content") or "" for delta in deltas)
        self.assertEqual(reasoning.count("r "), 1024)
        self.assertIn("Time is up.", reasoning)
        self.assertEqual("".join(delta.get("content") or "" for delta in deltas), "ok")
        self.assertEqual(events[-1]["usage"]["completion_tokens_details"]["reasoning_closed_at"], 1024)

    def test_request_budget_wins_when_smaller(self):
        backend = SequenceBackend(([REASON] * 40, "length"), ([ANSWER, IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=4096, max_thinking_tokens=10))
        self.assertEqual(backend.submissions[1][0], PROMPT + [REASON] * 10 + list(CLOSE))
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_closed_by"], "max_thinking_tokens")

    def test_model_that_closes_itself_is_untouched(self):
        backend = SequenceBackend(([REASON] * 5 + [END, BREAK, ANSWER, IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=2048))
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["usage"]["completion_tokens_details"], dict(reasoning_tokens=6))
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")

    def test_a_think_block_that_closes_itself_disarms_the_cap(self):
        # The batch worker holds new chunks at the cap token for a continuation: a block that
        # the model closed itself gets no continuation, so the server clears the job's cap at the close.
        backend = SequenceBackend(([REASON] * 5 + [END, BREAK, ANSWER, IM_END], "stop"))
        run(backend, chat(max_tokens=2048))
        self.assertEqual(backend.submissions[0][2]["think_cap"], 1024)
        self.assertIsNone(backend.jobs[0].think_cap)

    def test_a_capped_block_keeps_the_cap_armed(self):
        backend = SequenceBackend(([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))
        run(backend, chat(max_tokens=2048))
        self.assertEqual(backend.jobs[0].think_cap, 1024)

    def test_thinking_off_and_room_off_never_close(self):
        backend = SequenceBackend(([REASON] * 30, "length"))
        run(backend, chat(max_tokens=30, enable_thinking=False))
        backend = SequenceBackend(([REASON] * 30, "length"))
        _, body = run(backend, chat(max_tokens=30), thinking_answer_room=0)
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_tiny_budget_truncates_the_close(self):
        # max_tokens 3: one reasoning token, then two of the three close tokens; no room is left to continue.
        backend = SequenceBackend(([REASON] * 3, "length"))
        _, body = run(backend, chat(max_tokens=3))
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        # Only part of the close text fit, so the reply does not claim a close.
        self.assertEqual(body["usage"]["completion_tokens_details"], dict(reasoning_tokens=3))

    def test_budget_equal_to_max_tokens_writes_no_close(self):
        backend = SequenceBackend(([REASON] * 40, "length"))
        _, body = run(backend, chat(max_tokens=40, max_thinking_tokens=40), thinking_answer_room=0)
        self.assertEqual(len(backend.submissions), 1)
        self.assertNotIn("reasoning_closed_at", body["usage"]["completion_tokens_details"])

    def test_close_position_depends_only_on_the_request(self):
        submitted = []
        for spec in (0, 7):
            backend = SequenceBackend(([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))
            run(backend, chat(max_tokens=2048, spec=spec, seed=5))
            submitted.append(backend.submissions[1][0])
        self.assertEqual(submitted[0], submitted[1])

    def test_cap_uses_max_tokens_after_the_clamp(self):
        backend = SequenceBackend(([REASON] * 3000, "length"), ([ANSWER, IM_END], "stop"))
        backend.context = len(PROMPT) + 2047  # the clamp lowers max_tokens 32768 to 2048
        _, body = run(backend, chat(max_tokens=32768), clamp_max_tokens=True)
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_closed_at"], 1024)


if __name__ == "__main__":
    unittest.main()


class StreamErrorLogTest(unittest.TestCase):
    """Soak run Oct 5: a streamed thinking turn ended with an error chunk after HTTP 200, and no log line said why."""

    def test_error_after_the_stream_started_is_logged(self):
        # The engine reports a cache count above the prompt: the reply fails in finish(), after the first chunk.
        backend = SequenceBackend(([ANSWER, IM_END], "stop", 99))
        lines = []
        with mock.patch.object(metrics, "emit", lines.append):
            _, parts = run(backend, chat(max_tokens=64, stream=True, enable_thinking=False))
        self.assertIn("error", parts[-1])
        errors = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_error ")]
        self.assertEqual(errors, [dict(request_id=1, status=500, streamed=True,
                                       error="The engine returned an invalid cache count.")])

    def test_client_error_before_the_stream_logs_only_its_reason(self):
        # main (train-13 h2h): every 4xx gets a line with a fixed reason code, never its message.
        lines = []
        with mock.patch.object(metrics, "emit", lines.append):
            run(SequenceBackend(), dict(chat(max_tokens=64), messages=[]))
        errors = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_error ")]
        self.assertEqual(errors, [dict(request_id=None, status=400, streamed=False, reason="bad_request")])
