"""A model that ends its turn inside its reasoning, after complete tool calls, gets those calls made.

Halogen's rule tool_call_from_reasoning (the halogen-flash-server token budget guide, "The end-of-turn guard"; 0.13.4,
its an earlier change): "a model that writes a complete tool call inside its thinking block and then ends its turn on it gets
that call made, with finish_reason tool_calls; its text also stays in reasoning_content, where it was streamed".
Before this rule the end-of-turn guard kept the model's <|im_end|> 16 times and the reply ended with finish "stop" and
no tool call. The server reads the reasoning only when the turn ends there, or right after </think> with an answer of
whitespace only. A call that the model quotes in its reasoning before </think> and an answer with text does
not change the reply. The shapes here are synthetic: they repeat the token order of the fault, not its text.
"""
import unittest

from server.http import END_OF_TURN_KEPT
from server.tests.serve_fixtures import joined_calls
from server.tests.test_end_of_turn_guard import IM_END, TEXT, GuardTokenizer, run
from server.tests.test_think_budget import ANSWER, BREAK, END, PROMPT, REASON, SequenceBackend, chat
from server.tests.test_tools import TOOLS

TOOL_START, TOOL_END = 248058, 248059  # the checkpoint's <tool_call> and </tool_call> (server/tokenization.py)
THINK_END = 248069  # the checkpoint's </think> (server/metrics.py THINK_END_ID)
OPEN, VALUE, CLOSE, BREAK_LINE, PLAIN_START = 90, 91, 92, 93, 94
BLANK, EXAMPLE, SPACE = 95, 96, 97
CALL = [TOOL_START, OPEN, VALUE, CLOSE, TOOL_END]
EXAMPLE_CALL = [TOOL_START, OPEN, EXAMPLE, CLOSE, TOOL_END]
# The fault's order: reasoning, two calls with no </think> before them, then <|im_end|>.
H03_SHAPE = [REASON, BLANK] + CALL + [BREAK_LINE] + CALL + [IM_END]
# The review probe (toolcall-think-review F2): a quoted example call at a line start in the reasoning, more reasoning,
# </think>, then the real call. The base server gives the real call; the eager rule gave the example.
EXAMPLE_THEN_REAL = ([REASON, BREAK_LINE] + EXAMPLE_CALL + [BREAK_LINE, REASON, BREAK_LINE, END, BREAK] + CALL +
                     [IM_END])
# Prose between two calls, then <|im_end|>, all in the reasoning (toolcall-think-delta L1 item 2).
PROSE_BETWEEN_CALLS = [REASON, BLANK] + CALL + [BREAK_LINE, REASON, BREAK_LINE] + CALL + [IM_END]
# A call in the reasoning, </think>, an answer of whitespace only, then <|im_end|>.
CALL_THEN_EMPTY_ANSWER = [REASON, BLANK] + CALL + [BREAK_LINE, END, BREAK, IM_END]
CALL_THEN_SPACE_ANSWER = [REASON, BLANK] + CALL + [BREAK_LINE, END, BREAK, SPACE, BLANK, IM_END]


class CallTokenizer(GuardTokenizer):
    pieces = GuardTokenizer.pieces | {TOOL_START: "<tool_call>", TOOL_END: "</tool_call>",
                                      OPEN: "\n<function=read>\n<parameter=path>\n", VALUE: "a.py",
                                      CLOSE: "\n</parameter>\n</function>\n", BREAK_LINE: "\n",
                                      PLAIN_START: "<tool_call>", BLANK: "\n\n", THINK_END: "</think>",
                                      EXAMPLE: "example.py", SPACE: " "}


def call_chat(**fields):
    return chat(tools=TOOLS, max_tokens=4096) | fields


def reply(backend, body, **service):
    return run(backend, body, tokenizer=CallTokenizer(), **service)


def stream_parts(events):
    """(reasoning, content, [(name, arguments)], last finish_reason, usage) of a stream."""
    deltas = [event["choices"][0]["delta"] for event in events if event.get("choices")]
    calls = joined_calls(call for delta in deltas for call in delta.get("tool_calls") or ())
    return ("".join(delta.get("reasoning_content") or "" for delta in deltas),
            "".join(delta.get("content") or "" for delta in deltas),
            [(call["function"]["name"], call["function"]["arguments"]) for call in calls],
            [event["choices"][0]["finish_reason"] for event in events if event.get("choices")][-1],
            next((event["usage"] for event in events if event.get("usage")), {}))


class CallFromReasoningTest(unittest.TestCase):
    def test_calls_before_the_end_of_turn_in_the_reasoning_are_made(self):
        backend = SequenceBackend((H03_SHAPE, "stop"))
        _, body = reply(backend, call_chat())
        self.assertEqual(len(backend.submissions), 1)  # no end-of-turn guard continuation
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"]["tool_calls"]
        self.assertEqual([call["function"]["name"] for call in calls], ["read", "read"])
        self.assertEqual([call["function"]["arguments"] for call in calls], ['{"path":"a.py"}'] * 2)
        self.assertIsNone(choice["message"]["content"])
        # The call text stays in the reasoning, where the stream sent it (Halogen).
        self.assertTrue(choice["message"]["reasoning_content"].startswith("r \n\n<tool_call>\n<function=read>"))
        details = body["usage"]["completion_tokens_details"]
        self.assertNotIn("end_of_turn_kept", details)
        self.assertEqual(details["tool_call_from_reasoning"], 2)
        self.assertEqual(details["reasoning_tokens"], len(H03_SHAPE))
        self.assertEqual(body["usage"]["completion_tokens"], len(H03_SHAPE))

    def test_the_stream_sends_the_reasoning_then_the_calls(self):
        backend = SequenceBackend((H03_SHAPE, "stop"))
        _, events = reply(backend, call_chat(stream=True, stream_options=dict(include_usage=True)))
        reasoning, content, calls, finish, usage = stream_parts(events)
        self.assertIn("<tool_call>\n<function=read>", reasoning)
        self.assertEqual(content, "")
        self.assertEqual(calls, [("read", '{"path":"a.py"}')] * 2)
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(usage["completion_tokens_details"]["tool_call_from_reasoning"], 2)

    def test_a_quoted_call_then_the_end_tag_gives_the_real_call(self):
        backend = SequenceBackend((EXAMPLE_THEN_REAL, "stop"))
        _, body = reply(backend, call_chat())
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual([call["function"]["arguments"] for call in choice["message"]["tool_calls"]],
                         ['{"path":"a.py"}'])
        self.assertIn("example.py", choice["message"]["reasoning_content"])
        self.assertNotIn("tool_call_from_reasoning", body["usage"]["completion_tokens_details"])
        self.assertNotIn("tool_call_fallback", body["usage"])

    def test_a_quoted_call_then_the_end_tag_gives_the_real_call_in_the_stream(self):
        backend = SequenceBackend((EXAMPLE_THEN_REAL, "stop"))
        _, events = reply(backend, call_chat(stream=True, stream_options=dict(include_usage=True)))
        reasoning, _, calls, finish, usage = stream_parts(events)
        self.assertEqual(calls, [("read", '{"path":"a.py"}')])
        self.assertEqual(finish, "tool_calls")
        self.assertIn("example.py", reasoning)
        self.assertNotIn("tool_call_fallback", usage)

    def test_more_reasoning_after_the_call_keeps_the_guard(self):
        # The turn does not end on the call: the reasoning text after it is not a call, so the guard acts as before.
        backend = SequenceBackend(([REASON, BLANK] + CALL + [BREAK_LINE, REASON, IM_END], "stop"),
                                  ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        _, body = reply(backend, call_chat())
        self.assertEqual(len(backend.submissions), 2)
        message = body["choices"][0]["message"]
        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["content"], "ok")
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)

    def test_prose_between_calls_in_the_reasoning_makes_no_call(self):
        # Pins today's rule (the decision of Nov 18): from its first call the reasoning must be calls only, so
        # call, prose, call, end of turn makes no call and the guard keeps the token. Halogen (serve_api.py
        # call_in_block) makes only the last call here. The guard's second job ends on the kept text (end_of_turn_at).
        for stream in (False, True):
            with self.subTest(stream=stream):
                backend = SequenceBackend((PROSE_BETWEEN_CALLS, "stop"), ([IM_END], "stop"))
                body = call_chat(stream=True, stream_options=dict(include_usage=True)) if stream else call_chat()
                _, result = reply(backend, body)
                if stream:
                    reasoning, content, calls, finish, usage = stream_parts(result)
                else:
                    message, finish = result["choices"][0]["message"], result["choices"][0]["finish_reason"]
                    reasoning, content, usage = message["reasoning_content"], message["content"], result["usage"]
                    calls = message.get("tool_calls", [])
                self.assertEqual(len(backend.submissions), 2)
                self.assertEqual((finish, calls, content), ("stop", [], ""))
                self.assertTrue(reasoning.endswith("</tool_call><|im_end|>"))  # the kept token, as text
                details = usage["completion_tokens_details"]
                self.assertEqual(details["end_of_turn_kept"], 1)
                self.assertNotIn("tool_call_from_reasoning", details)

    def test_an_open_call_in_the_reasoning_keeps_the_guard(self):
        # An end-of-turn token inside an unfinished call (a quoted template in an argument) is kept as text; the call
        # that the model then completes is made with that text in its argument.
        backend = SequenceBackend(([REASON, BLANK, TOOL_START, OPEN, VALUE, IM_END], "stop"),
                                  ([CLOSE, TOOL_END, IM_END], "stop"))
        _, body = reply(backend, call_chat())
        self.assertEqual(len(backend.submissions), 2)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual([call["function"]["arguments"] for call in choice["message"]["tool_calls"]],
                         ['{"path":"a.py<|im_end|>"}'])

    def test_the_rule_is_part_of_the_guard(self):
        # --end-of-turn-guard off: the token ends the reply as before (Halogen: HALOGEN_EOS_GUARD=0).
        backend = SequenceBackend((H03_SHAPE, "stop"))
        _, body = reply(backend, call_chat(), end_of_turn_guard=False)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("tool_calls", body["choices"][0]["message"])

    def test_tool_choice_none_makes_no_call(self):
        backend = SequenceBackend((H03_SHAPE, "stop"), ([IM_END], "stop"))
        _, body = reply(backend, call_chat(tool_choice="none"))
        self.assertNotIn("tool_calls", body["choices"][0]["message"])
        self.assertNotIn("tool_call_from_reasoning", body["usage"]["completion_tokens_details"])

    def test_a_strict_tool_gets_its_call_after_the_schema_check(self):
        # The engine arms a grammar after </think> only (engine/runtime/constraint.h), so this call had no grammar. The
        # parser checks a strict tool's arguments strictly (tool_output.parse_function).
        strict = [dict(tool, function=tool["function"] | dict(strict=True)) for tool in TOOLS]
        backend = SequenceBackend((H03_SHAPE, "stop"))
        backend.constraint_ready = True  # the engine's ready event named the constraint key (Service.structured_options)
        _, body = reply(backend, call_chat(tools=strict))
        self.assertEqual(backend.submissions[0][1].constraint["kind"], "structural_tag")
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(len(body["choices"][0]["message"]["tool_calls"]), 2)

    def test_no_tools_keeps_the_guard(self):
        backend = SequenceBackend(([REASON, BLANK] + CALL + [IM_END], "stop"), ([IM_END], "stop"))
        _, body = reply(backend, chat(max_tokens=4096))
        message = body["choices"][0]["message"]
        self.assertIn("<tool_call>", message["reasoning_content"])
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_a_tool_call_token_inside_the_reasoning_does_not_end_it(self):
        # The eager rule (a line-start <tool_call> token ends the reasoning) is gone: the token is reasoning text.
        backend = SequenceBackend(([REASON, BLANK] + CALL + [BREAK_LINE, END, BREAK, ANSWER, IM_END], "stop"))
        _, body = reply(backend, call_chat())
        message = body["choices"][0]["message"]
        self.assertIn("<tool_call>", message["reasoning_content"])
        self.assertEqual(message["content"], "ok")
        self.assertNotIn("tool_calls", message)

    def test_a_call_then_the_end_tag_and_an_empty_answer_is_made(self):
        # The turn ends right after </think>, and the answer is whitespace only, so the reasoning holds the only
        # calls of the reply. The base server ended the reply with finish "stop" and no call, in plain and stream mode.
        for shape in (CALL_THEN_EMPTY_ANSWER, CALL_THEN_SPACE_ANSWER):
            for stream in (False, True):
                with self.subTest(space=shape is CALL_THEN_SPACE_ANSWER, stream=stream):
                    backend = SequenceBackend((shape, "stop"))
                    body = call_chat(stream=True, stream_options=dict(include_usage=True)) if stream else call_chat()
                    _, result = reply(backend, body)
                    self.assertEqual(len(backend.submissions), 1)  # no end-of-turn guard continuation
                    if stream:
                        reasoning, content, calls, finish, usage = stream_parts(result)
                    else:
                        choice = result["choices"][0]
                        message, finish, usage = choice["message"], choice["finish_reason"], result["usage"]
                        reasoning, content = message["reasoning_content"], message["content"]
                        calls = [(call["function"]["name"], call["function"]["arguments"])
                                 for call in message.get("tool_calls") or ()]
                    self.assertEqual(finish, "tool_calls")
                    self.assertEqual(calls, [("read", '{"path":"a.py"}')])
                    self.assertIn("<tool_call>\n<function=read>", reasoning)
                    self.assertFalse((content or "").strip())
                    self.assertEqual(usage["completion_tokens_details"]["tool_call_from_reasoning"], 1)

    def test_the_empty_answer_rule_is_part_of_the_guard(self):
        backend = SequenceBackend((CALL_THEN_EMPTY_ANSWER, "stop"))
        _, body = reply(backend, call_chat(), end_of_turn_guard=False)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("tool_calls", body["choices"][0]["message"])

    def test_the_tag_written_as_plain_characters_stays_reasoning(self):
        backend = SequenceBackend(([REASON, PLAIN_START, REASON, IM_END], "stop"),
                                  ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        _, body = reply(backend, call_chat())
        message = body["choices"][0]["message"]
        self.assertEqual(len(backend.submissions), 2)  # the guard kept the end-of-turn token in the reasoning
        self.assertTrue(message["reasoning_content"].startswith("r <tool_call>r <|im_end|>"))
        self.assertEqual(message["content"], "ok")


class RepeatedEndOfTurnTest(unittest.TestCase):
    def test_an_end_of_turn_right_after_the_kept_text_ends_the_reply(self):
        # The model writes the end-of-turn token again as the first token of the guard continuation: it means to end
        # the turn. Before this rule the guard kept it 16 times (end_of_turn_kept 16) and the reply ended the same way.
        backend = SequenceBackend(([REASON, IM_END], "stop"), ([IM_END], "stop"), ([IM_END], "stop"))
        _, body = reply(backend, call_chat())
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(backend.submissions[1][0], PROMPT + [REASON] + list(TEXT))
        message = body["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "r <|im_end|>")
        self.assertNotIn("tool_calls", message)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["completion_tokens_details"]["end_of_turn_kept"], 1)

    def test_a_kept_end_of_turn_with_text_after_it_still_goes_on(self):
        limit = END_OF_TURN_KEPT["reasoning"]
        self.assertEqual(limit, 16)
        backend = SequenceBackend(([REASON, IM_END], "stop"), ([REASON, IM_END], "stop"),
                                  ([REASON, END, BREAK, ANSWER, IM_END], "stop"))
        _, body = reply(backend, call_chat())
        self.assertEqual(len(backend.submissions), 3)
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
