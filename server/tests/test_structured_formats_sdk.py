"""The official SDK parse helpers read the answer of a Responses text.format and an Anthropic output_config.format
from the real HTTP layer (CPU token scripts): the JSON is the output_text of the message item, and one text block."""
from http.server import ThreadingHTTPServer
import threading
import unittest
try:
    import anthropic
    import openai
    import pydantic
except ImportError:
    anthropic = openai = pydantic = None
from server.http import Service, handler_for
from server.tests.sdk_fixture import RecordedBackend, RecordedTokenizer, Script, tool_block

ANSWER = Script(('{"a":', '17,"b":25}'))


class StructuredBackend(RecordedBackend):
    constraint_ready = True  # the engine named the constraint key in its ready event


@unittest.skipUnless(openai and anthropic, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class FormatSdkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Sum(pydantic.BaseModel):
            a: int
            b: int
        cls.Sum = Sum
        cls.tokenizer = RecordedTokenizer()
        cls.backend = StructuredBackend(cls.tokenizer)
        handler = handler_for(Service(cls.backend, cls.tokenizer))
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        port = cls.server.server_address[1]
        cls.openai = openai.OpenAI(api_key="crucible-local", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0,
                                   timeout=5, http_client=openai.DefaultHttpxClient(trust_env=False))
        cls.anthropic = anthropic.Anthropic(api_key="crucible-local", base_url=f"http://127.0.0.1:{port}", max_retries=0,
                                            timeout=5, http_client=anthropic.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.openai.close()
        cls.anthropic.close()
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.backend.reset(ANSWER)

    def constraint(self):
        return self.backend.requests[0].constraint

    def test_responses_parse(self):
        response = self.openai.responses.parse(model="crucible", input="Add 17 and 25", max_output_tokens=16,
                                               text_format=self.Sum)
        self.assertEqual(response.output_parsed, self.Sum(a=17, b=25))
        self.assertEqual(self.constraint()["kind"], "json_schema")
        self.assertEqual(response.text.format.type, "json_schema")

    def test_responses_stream_with_a_text_format(self):
        with self.openai.responses.stream(model="crucible", input="Add 17 and 25", max_output_tokens=16,
                                          text_format=self.Sum) as stream:
            final = stream.get_final_response()
        self.assertEqual(final.output_parsed, self.Sum(a=17, b=25))

    def test_responses_json_object(self):
        response = self.openai.responses.create(model="crucible", input="Add", max_output_tokens=16,
                                                text={"format": {"type": "json_object"}})
        self.assertEqual(response.output_text, '{"a":17,"b":25}')
        self.assertEqual(self.constraint()["kind"], "json_object")

    def test_messages_parse(self):
        message = self.anthropic.messages.parse(model="crucible", max_tokens=16, output_format=self.Sum,
                                                messages=[dict(role="user", content="Add 17 and 25")])
        self.assertEqual(message.parsed_output, self.Sum(a=17, b=25))
        self.assertEqual([block.type for block in message.content], ["text"])
        self.assertEqual(self.constraint()["kind"], "json_schema")

    def test_messages_stream_with_an_output_format(self):
        with self.anthropic.messages.stream(model="crucible", max_tokens=16, output_format=self.Sum,
                                            messages=[dict(role="user", content="Add 17 and 25")]) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.parsed_output, self.Sum(a=17, b=25))

    def test_a_strict_anthropic_tool(self):
        self.backend.reset(Script((tool_block(),)))
        tool = dict(name="add", description="Add.", strict=True, input_schema=dict(
            type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"],
            additionalProperties=False))
        message = self.anthropic.messages.create(model="crucible", max_tokens=64, tools=[tool],
                                                 messages=[dict(role="user", content="Add 17 and 25")])
        self.assertEqual([(block.type, block.input) for block in message.content], [("tool_use", dict(a=17, b=25))])
        self.assertEqual(self.constraint()["kind"], "structural_tag")

    def test_a_refused_schema_is_an_sdk_error(self):
        schema = dict(allOf=[dict(type="object")])
        with self.assertRaises(openai.BadRequestError) as caught:
            self.openai.responses.create(model="crucible", input="x", max_output_tokens=4,
                                         text={"format": {"type": "json_schema", "name": "s", "schema": schema}})
        self.assertEqual(caught.exception.body["param"], "text.format")
        with self.assertRaises(anthropic.BadRequestError):
            self.anthropic.messages.create(model="crucible", max_tokens=4, messages=[dict(role="user", content="x")],
                                           output_config={"format": {"type": "json_schema", "schema": schema}})
        self.assertEqual(self.backend.requests, [])
