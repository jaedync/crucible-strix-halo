"""The official SDK parse helpers read the answer of a Responses text.format and an Anthropic output_config.format
from the real HTTP layer (CPU token scripts): the JSON is the output_text of the message item, and one text block."""
import unittest
try:
    import anthropic
    import openai
    import pydantic
except ImportError:
    anthropic = openai = pydantic = None
from server.tests.sdk_fixture import HttpTarget, RecordedBackend, Script, fake_only, tool_block

ANSWER = Script(('{"a":', '17,"b":25}'))


class StructuredBackend(RecordedBackend):
    constraint_ready = True  # the engine named the constraint key in its ready event


@unittest.skipUnless(openai and anthropic, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class FormatSdkTests(HttpTarget, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Sum(pydantic.BaseModel):
            a: int
            b: int
        cls.Sum = Sum
        root, timeout = cls.start_target(StructuredBackend)
        cls.openai = openai.OpenAI(api_key="crucible-local", base_url=f"{root}/v1", max_retries=0,
                                   timeout=timeout, http_client=openai.DefaultHttpxClient(trust_env=False))
        cls.anthropic = anthropic.Anthropic(api_key="crucible-local", base_url=root, max_retries=0,
                                            timeout=timeout, http_client=anthropic.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.openai.close()
        cls.anthropic.close()
        cls.stop_target()

    def setUp(self):
        self.reset(ANSWER)

    def constraint(self):
        return self.backend.requests[0].constraint

    @fake_only
    def test_responses_parse(self):
        response = self.openai.responses.parse(model="crucible", input="Add 17 and 25", max_output_tokens=16,
                                               text_format=self.Sum)
        self.assertEqual(response.output_parsed, self.Sum(a=17, b=25))
        self.assertEqual(self.constraint()["kind"], "json_schema")
        self.assertEqual(response.text.format.type, "json_schema")

    @fake_only
    def test_responses_stream_with_a_text_format(self):
        with self.openai.responses.stream(model="crucible", input="Add 17 and 25", max_output_tokens=16,
                                          text_format=self.Sum) as stream:
            final = stream.get_final_response()
        self.assertEqual(final.output_parsed, self.Sum(a=17, b=25))

    @fake_only
    def test_responses_json_object(self):
        response = self.openai.responses.create(model="crucible", input="Add", max_output_tokens=16,
                                                text={"format": {"type": "json_object"}})
        self.assertEqual(response.output_text, '{"a":17,"b":25}')
        self.assertEqual(self.constraint()["kind"], "json_object")

    @fake_only
    def test_messages_parse(self):
        message = self.anthropic.messages.parse(model="crucible", max_tokens=16, output_format=self.Sum,
                                                messages=[dict(role="user", content="Add 17 and 25")])
        self.assertEqual(message.parsed_output, self.Sum(a=17, b=25))
        self.assertEqual([block.type for block in message.content], ["text"])
        self.assertEqual(self.constraint()["kind"], "json_schema")

    @fake_only
    def test_messages_stream_with_an_output_format(self):
        with self.anthropic.messages.stream(model="crucible", max_tokens=16, output_format=self.Sum,
                                            messages=[dict(role="user", content="Add 17 and 25")]) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.parsed_output, self.Sum(a=17, b=25))

    @fake_only
    def test_a_strict_anthropic_tool(self):
        self.reset(Script((tool_block(),)))
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
        self.assertNotSubmitted()
