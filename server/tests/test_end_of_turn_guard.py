"""The end-of-turn guard: an end-of-turn token inside the reasoning or an open tool call is kept as text, as in Halogen."""
import json
import unittest
from unittest import mock

from server import metrics
from server.http import END_OF_TURN_KEPT
from server.tests.serve_fixtures import chunks, invoke
from server.tests.test_think_budget import (ANSWER, BREAK, END, IM_END, PROMPT, REASON, BudgetTokenizer,
                                            SequenceBackend, chat)
from server.tests.test_tools import TOOLS

ENDOFTEXT = 248044
TEXT = (70, 71, 72)  # "<|im_end|>" as plain characters
CALL, VALUE, CLOSE_CALL = 80, 81, 82


class GuardTokenizer(BudgetTokenizer):
    pieces = BudgetTokenizer.pieces | {ENDOFTEXT: "", 70: "<|", 71: "im_end", 72: "|>", 73: "<|endoftext|>",
                                       CALL: "<tool_call>\n<function=read>\n<parameter=path>\n", VALUE: "a",
                                       CLOSE_CALL: "\n</parameter>\n</function>\n</tool_call>"}

    def end_of_turn_text_ids(self, token):
        return TEXT if token == IM_END else (73,)


class QuotingTokenizer(GuardTokenizer):
    """A call whose argument value holds the literal text of the start tag."""
    pieces = GuardTokenizer.pieces | {VALUE: "<tool_call>"}


def run(backend, body, path="/v1/chat/completions", tokenizer=None, **service):
    response = invoke(backend, tokenizer or GuardTokenizer(), body, path=path, service_options=service or None)
    if body.get("stream"):
        return response, chunks(response)
    return response, json.loads(response.body)


class ReasoningTest(unittest.TestCase):
    def test_end_of_turn_in_the_reasoning_is_kept_as_text(self):
        backend = SequenceBackend(([REASON] * 3 + [IM_END], "stop"), ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        response, body = run(backend, chat(max_tokens=2048))
        self.assertEqual(response.statuses, [200])
        self.assertEqual(len(backend.submissions), 2)
        ids, request, options = backend.submissions[1]
        # The continuation prefills the output with the token's text in place of the token.
        self.assertEqual(ids, PROMPT + [REASON] * 3 + list(TEXT))
        self.assertEqual(request.max_tokens, 2048 - 3 - len(TEXT))
        # The answer room closes the block at output token 1024; the continuation counts its own tokens.
        self.assertEqual(backend.submissions[0][2]["think_cap"], 1024)
        self.assertEqual(options["think_cap"], 1024 - 3 - len(TEXT))
        message, usage = body["choices"][0]["message"], body["usage"]
        self.assertEqual(message["reasoning_content"], "r r r <|im_end|>r")
        self.assertEqual(message["content"], "ok")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(usage["completion_tokens"], 3 + len(TEXT) + 5)
        self.assertEqual(usage["completion_tokens_details"]["end_of_turn_kept"], 1)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 3 + len(TEXT) + 2)

    def test_endoftext_is_kept_with_its_own_text(self):
        backend = SequenceBackend(([REASON, ENDOFTEXT], "stop"), ([END, BREAK, ANSWER, IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=2048))
        self.assertEqual(backend.submissions[1][0], PROMPT + [REASON, 73])
        self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "r <|endoftext|>")

    def test_the_seventeenth_end_of_turn_ends_the_reply(self):
        limit = END_OF_TURN_KEPT["reasoning"]
        backend = SequenceBackend(*[([REASON, IM_END], "stop")] * (limit + 1))
        _, body = run(backend, chat(max_tokens=4096))
        self.assertEqual(len(backend.submissions), limit + 1)
        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["reasoning_content"].count("<|im_end|>"), limit)
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], limit)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_end_of_turn_in_the_answer_ends_the_reply(self):
        backend = SequenceBackend(([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=2048))
        self.assertEqual(len(backend.submissions), 1)
        self.assertNotIn("end_of_turn_kept", body["usage"]["completion_tokens_details"])

    def test_guard_off_keeps_the_old_reply(self):
        backend = SequenceBackend(([REASON] * 3 + [IM_END], "stop"))
        _, body = run(backend, chat(max_tokens=2048), end_of_turn_guard=False)
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "r r r")
        self.assertEqual(body["choices"][0]["message"]["content"], "")

    def test_text_that_would_pass_the_cap_ends_the_reply(self):
        # The cap closes the block at one exact output count; the text would jump over it.
        backend = SequenceBackend(([REASON] * 1021 + [IM_END], "stop"))
        run(backend, chat(max_tokens=2048))
        self.assertEqual(len(backend.submissions), 1)

    def test_text_that_fits_below_the_cap_is_kept(self):
        backend = SequenceBackend(([REASON] * 1020 + [IM_END], "stop"), ([END, BREAK, ANSWER, IM_END], "stop"))
        run(backend, chat(max_tokens=2048))
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(backend.submissions[1][2]["think_cap"], 1)

    def test_no_room_for_the_text_ends_the_reply(self):
        backend = SequenceBackend(([REASON] * 3 + [IM_END], "stop"))
        # max_tokens 6 leaves room for the text but not for one more token after it.
        run(backend, chat(max_tokens=6), thinking_answer_room=0)
        self.assertEqual(len(backend.submissions), 1)

    def test_completions_are_never_guarded(self):
        backend = SequenceBackend(([REASON, IM_END], "stop"))
        _, body = run(backend, dict(model="crucible", prompt="Hi", max_tokens=16), path="/v1/completions")
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_stream_shows_the_text_and_the_count(self):
        backend = SequenceBackend(([REASON] * 3 + [IM_END], "stop"), ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        _, events = run(backend, chat(max_tokens=2048, stream=True, stream_options=dict(include_usage=True)))
        deltas = [choice["delta"] for event in events for choice in event["choices"]]
        # The stream is not trimmed: the reply's reasoning_content is.
        self.assertEqual("".join(delta.get("reasoning_content") or "" for delta in deltas), "r r r <|im_end|>r ")
        self.assertEqual("".join(delta.get("content") or "" for delta in deltas), "ok")
        self.assertEqual(events[-1]["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)

    def test_each_kept_token_writes_a_log_line(self):
        backend = SequenceBackend(([REASON] * 3 + [IM_END], "stop"), ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        lines = []
        with mock.patch.object(metrics, "emit", lines.append):
            run(backend, chat(max_tokens=2048))
        kept = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_end_of_turn_kept ")]
        self.assertEqual(kept, [dict(request_id=1, where="reasoning", completion_tokens=3 + len(TEXT), kept=1)])
        # A capped request writes one crucible_turn line with the counts that the client sees.
        turns = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_turn ")]
        self.assertEqual([(turn["prompt_tokens"], turn["completion_tokens"]) for turn in turns],
                         [(len(PROMPT), 3 + len(TEXT) + 5)])


class ToolCallTest(unittest.TestCase):
    def call_body(self, **fields):
        return dict(model="crucible", messages=[dict(role="user", content="Read")], tools=TOOLS,
                    enable_thinking=False, max_tokens=256) | fields

    def test_end_of_turn_inside_an_open_call_is_kept(self):
        backend = SequenceBackend(([CALL, VALUE, IM_END], "stop"), ([VALUE, CLOSE_CALL, IM_END], "stop"))
        _, body = run(backend, self.call_body())
        self.assertEqual(backend.submissions[1][0], PROMPT + [CALL, VALUE] + list(TEXT))
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        arguments = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, dict(path="a<|im_end|>a"))
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)
        self.assertNotIn("think_cap", backend.submissions[1][2])

    def test_the_fifth_end_of_turn_in_a_call_ends_the_reply(self):
        limit = END_OF_TURN_KEPT["tool_call"]
        backend = SequenceBackend(([CALL, IM_END], "stop"), *[([VALUE, IM_END], "stop")] * limit)
        _, body = run(backend, self.call_body())
        self.assertEqual(len(backend.submissions), limit + 1)
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], limit)

    def test_end_of_turn_after_a_closed_call_ends_the_reply(self):
        backend = SequenceBackend(([CALL, VALUE, CLOSE_CALL, IM_END], "stop"))
        _, body = run(backend, self.call_body())
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_a_call_tag_in_an_argument_does_not_keep_the_call_open(self):
        # The block ends at the first </tool_call>: a start tag inside the value is argument text, not a block.
        backend = SequenceBackend(([CALL, VALUE, CLOSE_CALL, IM_END], "stop"))
        response, body = run(backend, self.call_body(), tokenizer=QuotingTokenizer())
        self.assertEqual(response.statuses, [200])
        self.assertEqual(len(backend.submissions), 1)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
                         dict(path="<tool_call>"))
        self.assertNotIn("tool_call_fallback", body["usage"])
        self.assertNotIn("end_of_turn_kept", body["usage"].get("completion_tokens_details", {}))

    def test_a_call_tag_in_an_argument_does_not_keep_the_call_open_in_the_stream(self):
        backend = SequenceBackend(([CALL, VALUE, CLOSE_CALL, IM_END], "stop"))
        _, events = run(backend, self.call_body(stream=True, stream_options=dict(include_usage=True)),
                        tokenizer=QuotingTokenizer())
        self.assertEqual(len(backend.submissions), 1)
        reasons = [event["choices"][0]["finish_reason"] for event in events
                   if event.get("choices") and event["choices"][0].get("finish_reason")]
        self.assertEqual(reasons, ["tool_calls"])
        sent = "".join(delta["function"]["arguments"] for event in events for choice in event.get("choices") or []
                       for delta in choice["delta"].get("tool_calls") or [] if "arguments" in delta["function"])
        self.assertEqual(json.loads(sent), dict(path="<tool_call>"))
        self.assertNotIn("tool_call_fallback", events[-1]["usage"])

    def test_the_guard_acts_in_an_open_call_that_holds_a_call_tag(self):
        # No </tool_call> yet, so the call is open: the start tag in the value does not change that.
        backend = SequenceBackend(([CALL, VALUE, IM_END], "stop"), ([VALUE, CLOSE_CALL, IM_END], "stop"))
        _, body = run(backend, self.call_body(), tokenizer=QuotingTokenizer())
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(backend.submissions[1][0], PROMPT + [CALL, VALUE] + list(TEXT))
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
                         dict(path="<tool_call><|im_end|><tool_call>"))
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)

    def test_a_client_stop_string_in_the_kept_text_ends_the_reply(self):
        # The kept text matches the client's stop string: the reply ends here, and no second job runs.
        backend = SequenceBackend(([CALL, VALUE, IM_END], "stop"))
        _, body = run(backend, self.call_body(stop=["<|im_end|>"]))
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["completion_tokens"], 2 + len(TEXT))
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)



if __name__ == "__main__":
    unittest.main()
