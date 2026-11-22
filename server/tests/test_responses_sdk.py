"""Run the official OpenAI SDK Responses client against the real HTTP layer (POST /v1/responses) with CPU scripts."""
import json
import unittest
try:
    import openai
except ImportError:
    openai = None
from server.tests.sdk_fixture import HttpTarget, Script, fake_only, tool_block

ADD = dict(type="function", name="add", description="Add two integers.", parameters=dict(
    type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"]))
# A history input as a benchmark client's Responses test sends it (a system message, an assistant message item).
HISTORY_INPUT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "type": "message", "content": [{"text": "The LA Dodgers won in 2020.", "type": "output_text"}]},
    {"role": "user", "content": "What was the best play?"},
]


@unittest.skipUnless(openai, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class ResponsesSdkTests(HttpTarget, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, timeout = cls.start_target()
        cls.client = openai.OpenAI(api_key="crucible-local", base_url=f"{root}/v1", max_retries=0, timeout=timeout,
                                   http_client=openai.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.stop_target()

    def setUp(self):
        self.reset()

    @fake_only
    def test_history_non_streaming_request(self):
        response = self.client.responses.create(model="crucible", input=HISTORY_INPUT, stream=False, temperature=0.0,
                                                max_output_tokens=10, extra_body={"reasoning": {"effort": "none"}})
        self.assertEqual(response.output[0].content[0].text, "Hello snow 雪.")
        self.assertEqual(response.output_text, "Hello snow 雪.")
        self.assertEqual((response.usage.input_tokens, response.usage.output_tokens), (3, 2))
        request = self.backend.requests[0]
        self.assertEqual([m["role"] for m in request.messages], ["system", "user", "assistant", "user"])
        self.assertFalse(request.enable_thinking)

    @fake_only
    def test_history_streaming_request(self):
        self.reset(Script(("a", "b", "c")))
        stream = self.client.responses.create(model="crucible", input=HISTORY_INPUT, stream=True, temperature=0.0,
                                              max_output_tokens=2, extra_body={"reasoning": {"effort": "none"}})
        text, events = "", []
        for event in stream:
            events.append(event.type)
            if event.type == "response.output_text.delta":
                text += event.delta
            if event.type in ("response.completed", "response.incomplete"):
                self.assertEqual(event.response.output_text, text)
                last = event
        self.assertEqual(events[0], "response.created")
        self.assertEqual(events[-1], "response.incomplete")
        self.assertEqual(last.response.incomplete_details.reason, "max_output_tokens")
        self.assertEqual(text, "ab")

    @fake_only
    def test_stream_helper_final_response(self):
        with self.client.responses.stream(model="crucible", input="Hello") as stream:
            deltas = [event.delta for event in stream if event.type == "response.output_text.delta"]
            final = stream.get_final_response()
        self.assertEqual("".join(deltas), "Hello snow 雪.")
        self.assertEqual(final.output_text, "Hello snow 雪.")

    @fake_only
    def test_function_call_round_trip(self):
        self.reset(Script((tool_block(),)), Script(("The sum is 42.",)))
        first = self.client.responses.create(model="crucible", input="Add 17 and 25.", tools=[ADD])
        [call] = first.output
        self.assertEqual((call.type, call.name, json.loads(call.arguments)), ("function_call", "add", dict(a=17, b=25)))
        second = self.client.responses.create(model="crucible", tools=[ADD], input=[
            dict(role="user", content="Add 17 and 25."), *[item.model_dump(exclude_none=True) for item in first.output],
            dict(type="function_call_output", call_id=call.call_id, output="42")])
        self.assertEqual(second.output_text, "The sum is 42.")
        history = self.backend.requests[1].messages
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "tool"])

    @fake_only
    def test_stream_function_call(self):
        self.reset(Script((tool_block(),)))
        with self.client.responses.stream(model="crucible", input="Add.", tools=[ADD]) as stream:
            final = stream.get_final_response()
        self.assertEqual(json.loads(final.output[0].arguments), dict(a=17, b=25))

    @fake_only
    def test_stream_reasoning(self):
        self.reset(Script(("Think.", "</think>", "\n\nDone.")))
        with self.client.responses.stream(model="crucible", input="Go", reasoning=dict(effort="medium")) as stream:
            final = stream.get_final_response()
        self.assertEqual([item.type for item in final.output], ["reasoning", "message"])
        self.assertEqual(final.output[0].content[0].text, "Think.")
        self.assertEqual(final.output_text, "Done.")

    def test_state_fields_are_refused(self):
        for options in (dict(previous_response_id="resp_1"), dict(store=True)):
            with self.subTest(options=options), self.assertRaises(openai.BadRequestError) as caught:
                self.client.responses.create(model="crucible", input="x", **options)
            self.assertIn(next(iter(options)), str(caught.exception))
        self.assertNotSubmitted()
