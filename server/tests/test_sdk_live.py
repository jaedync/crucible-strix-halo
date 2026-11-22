"""Live cases of the official SDKs (openai 3.24.0, anthropic 2.8.0) against a running engine.

They run only with CRUCIBLE_LIVE_BASE_URL set (server/tests/sdk_fixture.py); without it every class here skips. Synthetic
prompts only, temperature 0, at most 256 output tokens, thinking off (except the documented examples, which run as
docs/OPENAI_CLIENTS.md writes them). CRUCIBLE_LIVE_RECORD names a JSON lines file that gets the text of each case.

- a tool call round trip on Chat Completions, /v1/messages and /v1/responses, the first turn plain and streamed;
- response_format json_schema on chat and the text.format json_schema of /v1/responses (also output_config.format on
  /v1/messages): the result parses and matches the schema, plain and streamed;
- plain and streamed text on each API: the same text, the same token count;
- the three Python examples of docs/OPENAI_CLIENTS.md;
- exactness: one greedy request through chat and through each other API gives the same text (the adapters call the chat
  path; a difference is a bug: the failure message holds the request)."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import unittest
try:
    import anthropic
    import openai
    import pydantic
except ImportError:
    anthropic = openai = pydantic = None
from server.tests.sdk_fixture import LIVE, LIVE_ENV, LIVE_TIMEOUT_S
from server.tests.serve_fixtures import joined_calls

DOCS = Path(__file__).resolve().parents[2] / "docs/OPENAI_CLIENTS.md"
DOC_URL = "http://127.0.0.1:8790"
RECORD_ENV = "CRUCIBLE_LIVE_RECORD"
SKIP = f"set {LIVE_ENV} to the URL of a running server for the live SDK cases"
MODEL, MAX = "crucible", 256
ADD_PROMPT = "Use the add tool to add 17 and 25. Then tell me the result."
ADD_SCHEMA = dict(type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"],
                  additionalProperties=False)
CHAT_ADD = dict(type="function", function=dict(name="add", description="Add two integers.", parameters=ADD_SCHEMA))
MESSAGES_ADD = dict(name="add", description="Add two integers.", input_schema=ADD_SCHEMA)
RESPONSES_ADD = dict(type="function", name="add", description="Add two integers.", parameters=ADD_SCHEMA)
CITY_PROMPT = "Give the capital city of France as JSON: its name, its country and its population in millions."
CITY_SCHEMA = dict(type="object", properties=dict(name=dict(type="string"), country=dict(type="string"),
                                                  population_millions=dict(type="number")),
                   required=["name", "country", "population_millions"], additionalProperties=False)
TEXT_PROMPT = "Write four short sentences about the color of the sky at different times of the day."
# Each API's spelling of "thinking off, greedy". Chat has two spellings: the Anthropic adapter maps thinking disabled to
# enable_thinking false, the Responses adapter maps reasoning.effort to reasoning_effort.
CHAT_OFF = dict(temperature=0, extra_body=dict(enable_thinking=False))
CHAT_EFFORT_OFF = dict(temperature=0, reasoning_effort="none")
MESSAGES_OFF = dict(thinking=dict(type="disabled"), extra_body=dict(temperature=0))
RESPONSES_OFF = dict(temperature=0, reasoning=dict(effort="none"))


def record(case, **fields):
    path = os.environ.get(RECORD_ENV)
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(dict(case=case, **fields), ensure_ascii=False, sort_keys=True) + "\n")


def doc_examples(text):
    """The ```python blocks of docs/OPENAI_CLIENTS.md, in order."""
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)


if pydantic:
    class City(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid", strict=True)
        name: str
        country: str
        population_millions: float


@unittest.skipUnless(LIVE, SKIP)
@unittest.skipUnless(openai and anthropic, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class LiveSdkTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.openai = openai.OpenAI(api_key="crucible-local", base_url=f"{LIVE}/v1", max_retries=0, timeout=LIVE_TIMEOUT_S,
                                   http_client=openai.DefaultHttpxClient(trust_env=False))
        cls.anthropic = anthropic.Anthropic(api_key="crucible-local", base_url=LIVE, max_retries=0, timeout=LIVE_TIMEOUT_S,
                                            http_client=anthropic.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.openai.close()
        cls.anthropic.close()

    def assertAdd(self, name, arguments):
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        self.assertEqual(name, "add")
        self.assertEqual(set(args), {"a", "b"}, args)
        self.assertEqual(sorted(args.values()), [17, 25], args)

    def assertCity(self, text):
        city = City.model_validate_json(text)   # strict: the types and the keys of CITY_SCHEMA, no other key
        self.assertTrue(city.name.strip() and city.country.strip(), text)
        return city

    # ---- tool call round trips --------------------------------------------------------------------------------------

    def test_live_chat_tool_round_trip(self):
        user = dict(role="user", content=ADD_PROMPT)
        first = self.openai.chat.completions.create(model=MODEL, messages=[user], tools=[CHAT_ADD], tool_choice="required",
                                                    max_completion_tokens=MAX, **CHAT_OFF)
        self.assertEqual(first.choices[0].finish_reason, "tool_calls")
        calls = first.choices[0].message.tool_calls
        self.assertTrue(calls)
        for call in calls:
            self.assertTrue(call.id.startswith("call_"))
            self.assertAdd(call.function.name, call.function.arguments)
        chunks = list(self.openai.chat.completions.create(model=MODEL, messages=[user], tools=[CHAT_ADD],
                                                          tool_choice="required", max_completion_tokens=MAX, stream=True,
                                                          stream_options=dict(include_usage=True), **CHAT_OFF))
        deltas = [call.model_dump(exclude_none=True) for chunk in chunks for choice in chunk.choices
                  for call in (choice.delta.tool_calls or [])]
        streamed = joined_calls(deltas)
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in streamed],
                         [(c.function.name, json.loads(c.function.arguments)) for c in calls])
        self.assertEqual(chunks[-2].choices[0].finish_reason, "tool_calls")
        self.assertEqual(chunks[-1].usage.completion_tokens, first.usage.completion_tokens)
        history = [user, first.choices[0].message.model_dump(exclude_none=True),
                   *[dict(role="tool", tool_call_id=call.id, content="42") for call in calls]]
        second = self.openai.chat.completions.create(model=MODEL, messages=history, tools=[CHAT_ADD],
                                                     max_completion_tokens=MAX, **CHAT_OFF)
        self.assertEqual(second.choices[0].finish_reason, "stop")
        self.assertIn("42", second.choices[0].message.content)
        record("chat_tool", calls=[c.function.arguments for c in calls], answer=second.choices[0].message.content)

    def test_live_messages_tool_round_trip(self):
        user = dict(role="user", content=ADD_PROMPT)
        options = dict(model=MODEL, max_tokens=MAX, tools=[MESSAGES_ADD], **MESSAGES_OFF)
        first = self.anthropic.messages.create(messages=[user], tool_choice=dict(type="any"), **options)
        self.assertEqual(first.stop_reason, "tool_use")
        calls = [block for block in first.content if block.type == "tool_use"]
        self.assertTrue(calls)
        for call in calls:
            self.assertAdd(call.name, call.input)
        with self.anthropic.messages.stream(messages=[user], tool_choice=dict(type="any"), **options) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.stop_reason, "tool_use")
        self.assertEqual([(b.name, b.input) for b in final.content if b.type == "tool_use"], [(c.name, c.input) for c in calls])
        self.assertEqual(final.usage.output_tokens, first.usage.output_tokens)
        results = [dict(type="tool_result", tool_use_id=call.id, content="42") for call in calls]
        second = self.anthropic.messages.create(messages=[user, dict(role="assistant", content=first.content),
                                                          dict(role="user", content=results)], **options)
        self.assertEqual(second.stop_reason, "end_turn")
        text = "".join(block.text for block in second.content if block.type == "text")
        self.assertIn("42", text)
        record("messages_tool", calls=[c.input for c in calls], answer=text)

    def test_live_responses_tool_round_trip(self):
        options = dict(model=MODEL, tools=[RESPONSES_ADD], max_output_tokens=MAX, **RESPONSES_OFF)
        first = self.openai.responses.create(input=ADD_PROMPT, tool_choice="required", **options)
        calls = [item for item in first.output if item.type == "function_call"]
        self.assertTrue(calls, first.output)
        self.assertEqual(first.status, "completed")
        for call in calls:
            self.assertAdd(call.name, call.arguments)
        with self.openai.responses.stream(input=ADD_PROMPT, tool_choice="required", **options) as stream:
            final = stream.get_final_response()
        self.assertEqual([(i.name, json.loads(i.arguments)) for i in final.output if i.type == "function_call"],
                         [(c.name, json.loads(c.arguments)) for c in calls])
        self.assertEqual(final.usage.output_tokens, first.usage.output_tokens)
        second = self.openai.responses.create(input=[
            dict(role="user", content=ADD_PROMPT), *[item.model_dump(exclude_none=True) for item in first.output],
            *[dict(type="function_call_output", call_id=call.call_id, output="42") for call in calls]], **options)
        self.assertEqual(second.status, "completed")
        self.assertIn("42", second.output_text)
        record("responses_tool", calls=[c.arguments for c in calls], answer=second.output_text)

    # ---- structured output ------------------------------------------------------------------------------------------

    def test_live_chat_json_schema(self):
        options = dict(model=MODEL, messages=[dict(role="user", content=CITY_PROMPT)], max_completion_tokens=MAX,
                       response_format=dict(type="json_schema", json_schema=dict(name="city", schema=CITY_SCHEMA, strict=True)),
                       **CHAT_OFF)
        plain = self.openai.chat.completions.create(**options)
        self.assertEqual(plain.choices[0].finish_reason, "stop")
        city = self.assertCity(plain.choices[0].message.content)
        chunks = list(self.openai.chat.completions.create(stream=True, **options))
        self.assertEqual("".join(c.delta.content or "" for chunk in chunks for c in chunk.choices),
                         plain.choices[0].message.content)
        parsed = self.openai.chat.completions.parse(model=MODEL, messages=[dict(role="user", content=CITY_PROMPT)],
                                                    max_completion_tokens=MAX, response_format=City, **CHAT_OFF)
        self.assertIsInstance(parsed.choices[0].message.parsed, City)
        record("chat_json", text=plain.choices[0].message.content, city=city.model_dump())

    def test_live_responses_json_schema(self):
        options = dict(model=MODEL, input=CITY_PROMPT, max_output_tokens=MAX, **RESPONSES_OFF)
        fmt = dict(format=dict(type="json_schema", name="city", schema=CITY_SCHEMA, strict=True))
        plain = self.openai.responses.create(text=fmt, **options)
        self.assertEqual(plain.status, "completed")
        city = self.assertCity(plain.output_text)
        self.assertEqual(plain.text.format.type, "json_schema")
        parsed = self.openai.responses.parse(text_format=City, **options)
        self.assertIsInstance(parsed.output_parsed, City)
        with self.openai.responses.stream(text_format=City, **options) as stream:
            final = stream.get_final_response()
        self.assertEqual(final.output_parsed, parsed.output_parsed)
        record("responses_json", text=plain.output_text, city=city.model_dump())

    def test_live_messages_output_format(self):
        options = dict(model=MODEL, max_tokens=MAX, messages=[dict(role="user", content=CITY_PROMPT)], **MESSAGES_OFF)
        message = self.anthropic.messages.parse(output_format=City, **options)
        self.assertEqual([block.type for block in message.content], ["text"])
        self.assertIsInstance(message.parsed_output, City)
        self.assertCity(message.content[0].text)
        with self.anthropic.messages.stream(output_format=City, **options) as stream:
            final = stream.get_final_message()
        self.assertEqual(final.parsed_output, message.parsed_output)

    # ---- plain and streamed text ------------------------------------------------------------------------------------

    def test_live_chat_text_plain_and_stream(self):
        options = dict(model=MODEL, messages=[dict(role="user", content=TEXT_PROMPT)], max_completion_tokens=128, **CHAT_OFF)
        plain = self.openai.chat.completions.create(**options)
        chunks = list(self.openai.chat.completions.create(stream=True, stream_options=dict(include_usage=True), **options))
        self.assertTrue(plain.choices[0].message.content.strip())
        self.assertEqual("".join(c.delta.content or "" for chunk in chunks for c in chunk.choices),
                         plain.choices[0].message.content)
        self.assertEqual(chunks[-2].choices[0].finish_reason, plain.choices[0].finish_reason)
        self.assertEqual(chunks[-1].choices, [])
        self.assertEqual(chunks[-1].usage.completion_tokens, plain.usage.completion_tokens)
        self.assertLessEqual(plain.usage.completion_tokens, 128)

    def test_live_messages_text_plain_and_stream(self):
        options = dict(model=MODEL, max_tokens=128, messages=[dict(role="user", content=TEXT_PROMPT)], **MESSAGES_OFF)
        plain = self.anthropic.messages.create(**options)
        with self.anthropic.messages.stream(**options) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        self.assertEqual([block.type for block in plain.content], ["text"])
        self.assertTrue(plain.content[0].text.strip())
        self.assertEqual(text, plain.content[0].text)
        self.assertEqual((final.stop_reason, final.usage.output_tokens), (plain.stop_reason, plain.usage.output_tokens))

    def test_live_responses_text_plain_and_stream(self):
        options = dict(model=MODEL, input=TEXT_PROMPT, max_output_tokens=128, **RESPONSES_OFF)
        plain = self.openai.responses.create(**options)
        events, text = [], ""
        for event in self.openai.responses.create(stream=True, **options):
            events.append(event.type)
            if event.type == "response.output_text.delta":
                text += event.delta
            if event.type in ("response.completed", "response.incomplete"):
                last = event.response
        self.assertEqual(events[0], "response.created")
        self.assertIn(events[-1], ("response.completed", "response.incomplete"))
        self.assertTrue(plain.output_text.strip())
        self.assertEqual(text, plain.output_text)
        self.assertEqual((last.status, last.usage.output_tokens), (plain.status, plain.usage.output_tokens))

    # ---- the documented examples ------------------------------------------------------------------------------------

    def test_live_doc_examples(self):
        blocks = doc_examples(DOCS.read_text())
        self.assertEqual(len(blocks), 3, "docs/OPENAI_CLIENTS.md has three Python examples (chat, Responses, Anthropic)")
        for index, code in enumerate(blocks):
            with self.subTest(example=index):
                self.assertIn(DOC_URL, code)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(compile(code.replace(DOC_URL, LIVE), f"OPENAI_CLIENTS.md example {index + 1}", "exec"), {})
                printed = out.getvalue().strip()
                record("doc_example", index=index + 1, text=printed)
                self.assertTrue(printed and printed != "None", f"example {index + 1} printed {printed!r}")

    # ---- exactness across the APIs ----------------------------------------------------------------------------------

    def test_live_same_tokens_on_every_api(self):
        """One greedy request (thinking off, temperature 0, 128 tokens) on chat, /v1/messages and /v1/responses, plain and
        streamed. Every adapter maps to the chat path, so the text and the token count must be the same everywhere."""
        user = [dict(role="user", content=TEXT_PROMPT)]
        got = {}
        chat = self.openai.chat.completions.create(model=MODEL, messages=user, max_completion_tokens=128, **CHAT_OFF)
        got["chat"] = (chat.choices[0].message.content, chat.usage.completion_tokens)
        chunks = list(self.openai.chat.completions.create(model=MODEL, messages=user, max_completion_tokens=128, stream=True,
                                                          stream_options=dict(include_usage=True), **CHAT_OFF))
        got["chat stream"] = ("".join(c.delta.content or "" for chunk in chunks for c in chunk.choices),
                              chunks[-1].usage.completion_tokens)
        effort = self.openai.chat.completions.create(model=MODEL, messages=user, max_completion_tokens=128, **CHAT_EFFORT_OFF)
        got["chat effort none"] = (effort.choices[0].message.content, effort.usage.completion_tokens)
        message = self.anthropic.messages.create(model=MODEL, max_tokens=128, messages=user, **MESSAGES_OFF)
        got["messages"] = ("".join(b.text for b in message.content if b.type == "text"), message.usage.output_tokens)
        with self.anthropic.messages.stream(model=MODEL, max_tokens=128, messages=user, **MESSAGES_OFF) as stream:
            text = "".join(stream.text_stream)
            got["messages stream"] = (text, stream.get_final_message().usage.output_tokens)
        response = self.openai.responses.create(model=MODEL, input=TEXT_PROMPT, max_output_tokens=128, **RESPONSES_OFF)
        got["responses"] = (response.output_text, response.usage.output_tokens)
        with self.openai.responses.stream(model=MODEL, input=TEXT_PROMPT, max_output_tokens=128, **RESPONSES_OFF) as stream:
            final = stream.get_final_response()
            got["responses stream"] = (final.output_text, final.usage.output_tokens)
        record("exactness", texts={k: v[0] for k, v in got.items()}, tokens={k: v[1] for k, v in got.items()})
        request = dict(messages=user, temperature=0, max_tokens=128, thinking="off")
        differ = {name: value for name, value in got.items() if value != got["chat"]}
        self.assertTrue(got["chat"][0].strip())
        self.assertEqual(differ, {}, f"request {json.dumps(request)}: chat gave {got['chat']!r}")


if __name__ == "__main__":
    unittest.main()
