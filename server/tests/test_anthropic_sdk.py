"""Run the official Anthropic SDK against the real HTTP layer (POST /v1/messages) with CPU token scripts."""
import unittest
try:
    import anthropic
except ImportError:
    anthropic = None
from server.tests.sdk_fixture import HttpTarget, Script, fake_only, tool_block

ADD = dict(name="add", description="Add two integers.", input_schema=dict(
    type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"]))


@unittest.skipUnless(anthropic, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class AnthropicSdkTests(HttpTarget, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root, timeout = cls.start_target()
        cls.client = anthropic.Anthropic(api_key="crucible-local", base_url=root, max_retries=0, timeout=timeout,
                                         http_client=anthropic.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.stop_target()

    def setUp(self):
        self.reset()

    def create(self, **options):
        options.setdefault("messages", [dict(role="user", content="Hello")])
        return self.client.messages.create(model="crucible", max_tokens=32, **options)

    @fake_only
    def test_text_and_usage(self):
        message = self.create(system="Be brief.", extra_body=dict(temperature=0))
        self.assertEqual(message.content[0].text, "Hello snow 雪.")
        self.assertEqual(message.stop_reason, "end_turn")
        self.assertEqual((message.usage.input_tokens, message.usage.output_tokens), (3, 2))
        self.assertEqual(self.backend.requests[0].messages[0]["role"], "system")

    @fake_only
    def test_stream_text(self):
        with self.client.messages.stream(model="crucible", max_tokens=32,
                                         messages=[dict(role="user", content="Hello")]) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        self.assertEqual(text, "Hello snow 雪.")
        self.assertEqual(final.content[0].text, text)
        self.assertEqual(final.stop_reason, "end_turn")
        self.assertEqual(final.usage.output_tokens, 2)

    @fake_only
    def test_max_tokens_and_stop_sequence(self):
        self.reset(Script(("a", "b", "c")))
        message = self.client.messages.create(model="crucible", max_tokens=2, messages=[dict(role="user", content="x")])
        self.assertEqual(message.stop_reason, "max_tokens")
        self.reset(Script(("one ", "END", " two")))
        message = self.create(stop_sequences=["END"])
        self.assertEqual((message.stop_reason, message.stop_sequence), ("stop_sequence", "END"))
        self.assertEqual(message.content[0].text, "one ")

    @fake_only
    def test_tool_use_round_trip(self):
        self.reset(Script((tool_block(),)), Script(("The sum is 42.",)))
        first = self.create(tools=[ADD])
        self.assertEqual(first.stop_reason, "tool_use")
        call = first.content[0]
        self.assertEqual((call.type, call.name, call.input), ("tool_use", "add", dict(a=17, b=25)))
        second = self.client.messages.create(model="crucible", max_tokens=32, tools=[ADD], messages=[
            dict(role="user", content="Hello"), dict(role="assistant", content=first.content),
            dict(role="user", content=[dict(type="tool_result", tool_use_id=call.id, content="42")])])
        self.assertEqual(second.content[0].text, "The sum is 42.")
        history = self.backend.requests[1].messages
        self.assertEqual([message["role"] for message in history], ["user", "assistant", "tool"])
        self.assertEqual(history[1]["tool_calls"][0]["id"], call.id)

    @fake_only
    def test_stream_tool_use(self):
        self.reset(Script((tool_block(),)))
        with self.client.messages.stream(model="crucible", max_tokens=64, tools=[ADD],
                                         messages=[dict(role="user", content="Add")]) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.stop_reason, "tool_use")
        self.assertEqual(final.content[-1].input, dict(a=17, b=25))

    @fake_only
    def test_stream_thinking(self):
        self.reset(Script(("Think.", "</think>", "\n\nDone.")))
        with self.client.messages.stream(model="crucible", max_tokens=64, thinking=dict(type="enabled", budget_tokens=1024),
                                         messages=[dict(role="user", content="Go")]) as stream:
            final = stream.get_final_message()
        self.assertEqual([block.type for block in final.content], ["thinking", "text"])
        self.assertEqual((final.content[0].thinking, final.content[1].text), ("Think.", "Done."))

    def test_errors_have_the_anthropic_types(self):
        with self.assertRaises(anthropic.NotFoundError):
            self.client.messages.create(model="other", max_tokens=8, messages=[dict(role="user", content="x")])
        with self.assertRaises(anthropic.BadRequestError) as caught:
            self.create(messages=[dict(role="user", content=[dict(type="document", source=dict(
                type="text", media_type="text/plain", data="x"))])])
        self.assertIn("messages[0].content[0].type", str(caught.exception))
        self.assertNotIn("document", str(caught.exception))  # the error never repeats the client's value
        self.assertNotSubmitted()

    @fake_only
    def test_count_tokens(self):
        self.assertEqual(self.client.messages.count_tokens(model="crucible", messages=[
            dict(role="user", content="Hello")]).input_tokens, 3)
