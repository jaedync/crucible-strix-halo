"""Run the official Anthropic SDK against the real HTTP layer (POST /v1/messages) with CPU token scripts."""
from http.server import ThreadingHTTPServer
import threading
import unittest
try:
    import anthropic
except ImportError:
    anthropic = None
from server.http import Service, handler_for
from server.tests.sdk_fixture import RecordedBackend, RecordedTokenizer, Script, tool_block

ADD = dict(name="add", description="Add two integers.", input_schema=dict(
    type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"]))


@unittest.skipUnless(anthropic, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class AnthropicSdkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RecordedTokenizer()
        cls.backend = RecordedBackend(cls.tokenizer)
        handler = handler_for(Service(cls.backend, cls.tokenizer))
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = anthropic.Anthropic(api_key="crucible-local", base_url=f"http://127.0.0.1:{cls.server.server_address[1]}",
                                         max_retries=0, timeout=5,
                                         http_client=anthropic.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.backend.reset()

    def create(self, **options):
        options.setdefault("messages", [dict(role="user", content="Hello")])
        return self.client.messages.create(model="crucible", max_tokens=32, **options)

    def test_text_and_usage(self):
        message = self.create(system="Be brief.", extra_body=dict(temperature=0))
        self.assertEqual(message.content[0].text, "Hello snow 雪.")
        self.assertEqual(message.stop_reason, "end_turn")
        self.assertEqual((message.usage.input_tokens, message.usage.output_tokens), (3, 2))
        self.assertEqual(self.backend.requests[0].messages[0]["role"], "system")

    def test_stream_text(self):
        with self.client.messages.stream(model="crucible", max_tokens=32,
                                         messages=[dict(role="user", content="Hello")]) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        self.assertEqual(text, "Hello snow 雪.")
        self.assertEqual(final.content[0].text, text)
        self.assertEqual(final.stop_reason, "end_turn")
        self.assertEqual(final.usage.output_tokens, 2)

    def test_max_tokens_and_stop_sequence(self):
        self.backend.reset(Script(("a", "b", "c")))
        message = self.client.messages.create(model="crucible", max_tokens=2, messages=[dict(role="user", content="x")])
        self.assertEqual(message.stop_reason, "max_tokens")
        self.backend.reset(Script(("one ", "END", " two")))
        message = self.create(stop_sequences=["END"])
        self.assertEqual((message.stop_reason, message.stop_sequence), ("stop_sequence", "END"))
        self.assertEqual(message.content[0].text, "one ")

    def test_tool_use_round_trip(self):
        self.backend.reset(Script((tool_block(),)), Script(("The sum is 42.",)))
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

    def test_stream_tool_use(self):
        self.backend.reset(Script((tool_block(),)))
        with self.client.messages.stream(model="crucible", max_tokens=64, tools=[ADD],
                                         messages=[dict(role="user", content="Add")]) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.stop_reason, "tool_use")
        self.assertEqual(final.content[-1].input, dict(a=17, b=25))

    def test_stream_thinking(self):
        self.backend.reset(Script(("Think.", "</think>", "\n\nDone.")))
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
        self.assertEqual(self.backend.requests, [])

    def test_count_tokens(self):
        self.assertEqual(self.client.messages.count_tokens(model="crucible", messages=[
            dict(role="user", content="Hello")]).input_tokens, 3)
