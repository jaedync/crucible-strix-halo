"""Thinking output: reasoning text goes to reasoning_content; with thinking off nothing changes."""
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
try:
    import openai
except ImportError:
    openai = None
from server.api import ApiError, validate_request
from server.http import Service, handler_for
from server.template import ChatTemplate
from server.thinking import ThinkSplitter
from server.tests.sdk_fixture import RecordedBackend, RecordedTokenizer, Script, TOOLS, tool_block

THOUGHT = ("\nThe user", " wants 17+25", " = 42.\n</th", "ink>\n\n", "The sum", " is 42.")


class SplitterTest(unittest.TestCase):
    def run_pieces(self, pieces, enabled=True):
        splitter = ThinkSplitter(enabled)
        reasoning, answer = "", ""
        for piece in pieces:
            r, a = splitter.feed(piece)
            reasoning, answer = reasoning + r, answer + a
        r, a = splitter.flush()
        return reasoning + r, answer + a

    def test_tag_split_across_pieces(self):
        self.assertEqual(self.run_pieces(THOUGHT), ("The user wants 17+25 = 42.\n", "The sum is 42."))

    def test_tag_one_character_at_a_time(self):
        self.assertEqual(self.run_pieces(list("abc</think>\n\nxyz")), ("abc", "xyz"))

    def test_answer_keeps_leading_indentation(self):
        self.assertEqual(self.run_pieces(("x</think>\n\n", "    code")), ("x", "    code"))

    def test_unclosed_thinking_is_all_reasoning(self):
        self.assertEqual(self.run_pieces(("still thinking </th",)), ("still thinking </th", ""))

    def test_disabled_passes_everything(self):
        text = ("\n\nHello </think> world",)
        self.assertEqual(self.run_pieces(text, enabled=False), ("", "\n\nHello </think> world"))


class ThinkingControlsTest(unittest.TestCase):
    def resolve(self, default=False, **fields):
        body = dict(model="crucible", messages=[dict(role="user", content="Hi")]) | fields
        request = validate_request(body, "chat", "crucible", default_thinking=default)
        return request.enable_thinking, request.reasoning_effort, request.preserve_thinking

    def test_server_default_applies_without_fields(self):
        self.assertEqual(self.resolve(), (False, "xhigh", None))
        self.assertEqual(self.resolve(default=True), (True, "xhigh", None))

    def test_every_client_shape(self):
        cases = [(dict(enable_thinking=True), (True, "xhigh", None)),
                 (dict(chat_template_kwargs=dict(enable_thinking=True)), (True, "xhigh", None)),
                 (dict(chat_template_kwargs=dict(enable_thinking=False)), (False, "xhigh", None)),
                 (dict(reasoning_effort="high"), (True, "xhigh", None)),
                 (dict(reasoning_effort="minimal"), (True, "low", None)),
                 (dict(reasoning_effort="medium"), (True, "medium", None)),
                 (dict(reasoning_effort=None), (False, "xhigh", None)),
                 (dict(enable_thinking=False, reasoning_effort="high"), (False, "xhigh", None)),
                 (dict(chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=False,
                                                 reasoning_effort="low")), (True, "low", False))]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                self.assertEqual(self.resolve(**fields), expected)
        self.assertEqual(self.resolve(default=True, reasoning_effort="none"), (False, "xhigh", None))
        self.assertEqual(self.resolve(default=True, enable_thinking=False), (False, "xhigh", None))

    def test_invalid_and_conflicting_fields(self):
        for fields, param in ((dict(reasoning_effort="extreme"), "reasoning_effort"),
                              (dict(chat_template_kwargs=dict(add_vision_id=True)), "chat_template_kwargs"),
                              (dict(chat_template_kwargs="on"), "chat_template_kwargs"),
                              (dict(chat_template_kwargs=dict(preserve_thinking="yes")), "preserve_thinking"),
                              (dict(enable_thinking=True, chat_template_kwargs=dict(enable_thinking=False)),
                               "enable_thinking"),
                              (dict(reasoning_effort="low", chat_template_kwargs=dict(reasoning_effort="high")),
                               "reasoning_effort")):
            with self.subTest(fields=fields), self.assertRaises(ApiError) as error:
                self.resolve(**fields)
            self.assertEqual(error.exception.param, param)


class ThinkingTemplateTest(unittest.TestCase):
    """The checkpoint template replays reasoning, so the next prompt extends the previous prompt plus its output."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        fixture = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
        (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": fixture.read_text()}))
        self.template = ChatTemplate(root)

    def tearDown(self):
        self.temp.cleanup()

    def turns(self, preserve=None):
        first = [dict(role="user", content="Add 17 and 25.")]
        prompt = self.template.render(first, enable_thinking=True, preserve_thinking=preserve)
        generated = "The user wants 17+25 = 42.\n</think>\n\nThe sum is 42.<|im_end|>\n"
        history = first + [dict(role="assistant", content="The sum is 42.", reasoning_content="The user wants 17+25 = 42."),
                           dict(role="user", content="Thanks.")]
        return prompt + generated, self.template.render(history, enable_thinking=True, preserve_thinking=preserve)

    def test_replayed_reasoning_extends_the_previous_turn(self):
        previous, following = self.turns()
        self.assertTrue(following.startswith(previous))

    def test_preserve_thinking_false_drops_earlier_reasoning(self):
        previous, following = self.turns(preserve=False)
        self.assertNotIn("17+25", following)
        self.assertFalse(following.startswith(previous))

    def test_reasoning_after_the_last_user_turn_stays_in_tool_chains(self):
        history = [dict(role="user", content="Read a."),
                   dict(role="assistant", content="", reasoning_content="I read a.",
                        tool_calls=[dict(id="c1", type="function", function=dict(name="read", arguments={"path": "a"}))]),
                   dict(role="tool", tool_call_id="c1", content="text")]
        rendered = self.template.render(history, enable_thinking=True, preserve_thinking=False)
        self.assertIn("<think>\nI read a.\n</think>\n\n<tool_call>", rendered)


@unittest.skipUnless(openai, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class ThinkingHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RecordedTokenizer()
        cls.backend = RecordedBackend(cls.tokenizer)
        cls.service = Service(cls.backend, cls.tokenizer)
        handler = handler_for(cls.service)
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # a free port: two sessions may run the tests at once
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = openai.OpenAI(api_key="crucible-local", base_url=f"http://127.0.0.1:{cls.server.server_address[1]}/v1", max_retries=0,
                                  timeout=5, http_client=openai.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def chat(self, script, thinking=True, **options):
        self.backend.reset(script)
        extra = dict(enable_thinking=True) if thinking else {}
        return self.client.chat.completions.create(model="crucible", messages=[dict(role="user", content="Hi")],
                                                   extra_body=extra, max_tokens=32, **options)

    def test_non_stream_message_fields(self):
        result = self.chat(Script(THOUGHT))
        message = result.choices[0].message
        self.assertEqual(message.content, "The sum is 42.")
        self.assertEqual(message.model_extra["reasoning_content"], "The user wants 17+25 = 42.")
        self.assertEqual(result.usage.completion_tokens_details.reasoning_tokens, 4)
        self.assertEqual(result.usage.completion_tokens, len(THOUGHT))

    def test_stream_deltas(self):
        chunks = list(self.chat(Script(THOUGHT), stream=True, stream_options=dict(include_usage=True)))
        deltas = [choice.delta for chunk in chunks for choice in chunk.choices]
        reasoning = "".join((delta.model_extra or {}).get("reasoning_content") or "" for delta in deltas)
        content = "".join(delta.content or "" for delta in deltas)
        self.assertEqual(reasoning, "The user wants 17+25 = 42.\n")
        self.assertEqual(content, "The sum is 42.")
        self.assertNotIn("</think>", reasoning + content)
        self.assertEqual(chunks[-1].usage.completion_tokens_details.reasoning_tokens, 4)

    def test_tool_call_after_thinking(self):
        result = self.chat(Script(("I should call add.", "</think>\n\n", tool_block())), tools=TOOLS)
        message = result.choices[0].message
        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        self.assertEqual(json.loads(message.tool_calls[0].function.arguments), dict(a=17, b=25))
        self.assertEqual(message.model_extra["reasoning_content"], "I should call add.")

    def test_tool_markup_inside_reasoning_is_not_a_call(self):
        result = self.chat(Script(("Maybe ", tool_block(), " later.", "</think>", "Done.")), tools=TOOLS)
        self.assertEqual(result.choices[0].finish_reason, "stop")
        self.assertEqual(result.choices[0].message.content, "Done.")

    def test_length_cut_inside_reasoning(self):
        script = Script(("Still", " thinking", " </th"), reason="length")
        result = self.chat(script)
        message = result.choices[0].message
        self.assertEqual(result.choices[0].finish_reason, "length")
        self.assertEqual(message.model_extra["reasoning_content"], "Still thinking </th")
        self.assertEqual(message.content, "")
        self.assertEqual(result.usage.completion_tokens_details.reasoning_tokens, 3)
        chunks = list(self.chat(script, stream=True))
        deltas = [choice.delta for chunk in chunks for choice in chunk.choices]
        self.assertEqual("".join((d.model_extra or {}).get("reasoning_content") or "" for d in deltas),
                         "Still thinking </th")
        self.assertEqual("".join(d.content or "" for d in deltas), "")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

    def test_chat_template_kwargs_and_server_default(self):
        self.backend.reset(Script(THOUGHT))
        result = self.client.chat.completions.create(model="crucible", messages=[dict(role="user", content="Hi")],
            extra_body=dict(chat_template_kwargs=dict(enable_thinking=True)), max_tokens=32)
        self.assertEqual(result.choices[0].message.content, "The sum is 42.")
        self.service.default_thinking = True
        try:
            result = self.chat(Script(THOUGHT), thinking=False)
            self.assertEqual(result.choices[0].message.model_extra["reasoning_content"], "The user wants 17+25 = 42.")
            result = self.chat(Script(("Plain.",)), thinking=False, reasoning_effort="none")
            self.assertEqual(result.choices[0].message.content, "Plain.")
            self.assertNotIn("reasoning_content", result.choices[0].message.model_extra or {})
        finally:
            self.service.default_thinking = False

    def test_thinking_off_keeps_text(self):
        result = self.chat(Script(("Plain", " </think> text")), thinking=False)
        self.assertEqual(result.choices[0].message.content, "Plain </think> text")
        self.assertNotIn("reasoning_content", result.choices[0].message.model_extra or {})
        self.assertIsNone(result.usage.completion_tokens_details)

    @unittest.skipUnless(os.environ.get("CRUCIBLE_PI_AI_DIR") and shutil.which("node"),
                         "set CRUCIBLE_PI_AI_DIR for the optional Pi transport check")
    def test_pi_stores_and_replays_thinking(self):
        self.backend.reset(Script(THOUGHT), Script(("Fine.",)))
        process = subprocess.run(["node", str(Path(__file__).with_name("pi_thinking_client.mjs"))],
                                 capture_output=True, text=True, timeout=15, check=True)
        content = json.loads(process.stdout)["content"]
        self.assertEqual([block["type"] for block in content], ["thinking", "text"])
        self.assertEqual(content[0]["thinking"].strip(), "The user wants 17+25 = 42.")
        self.assertEqual(content[1]["text"], "The sum is 42.")
        replay = self.backend.requests[-1]
        self.assertTrue(replay.enable_thinking)
        assistant = next(m for m in replay.messages if m["role"] == "assistant")
        self.assertEqual(assistant["reasoning_content"].strip(), "The user wants 17+25 = 42.")
        self.assertEqual(assistant["content"], "The sum is 42.")


if __name__ == "__main__":
    unittest.main()
