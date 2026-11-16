import json
from pathlib import Path
import tempfile
import unittest
from server.api import ApiError, validate_request
from server.template import ChatTemplate
from server.tool_stream import ToolCallStream

TOOLS = [dict(type="function", function=dict(name="read", description="Read a file.", parameters=dict(
    type="object", properties=dict(path=dict(type="string"), offset=dict(type="integer"),
                                   flags=dict(type="object")), required=["path"], additionalProperties=False)))]

class ToolRequestTests(unittest.TestCase):
    def request(self, **options):
        body = dict(messages=[dict(role="user", content="Read a.py")], tools=TOOLS) | options
        return validate_request(body, "chat", "crucible")

    def test_tools_and_choice(self):
        for choice in ("auto", "none", "required", dict(type="function", function=dict(name="read"))):
            request = self.request(tool_choice=choice)
            self.assertEqual(request.tools[0]["function"]["name"], "read")
            self.assertEqual(request.tool_choice, choice)

    def test_multi_turn_real_template(self):
        messages = [dict(role="user", content="Read a.py"),
                    dict(role="assistant", content=None, tool_calls=[dict(id="call_1", type="function", function=dict(
                        name="read", arguments=json.dumps(dict(path="a.py", offset=4, flags={"full": True}))))]),
                    dict(role="tool", tool_call_id="call_1", content="line four"),
                    dict(role="assistant", content="The file is ready."), dict(role="user", content="Read b.py")]
        request = self.request(messages=messages)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
            Path(directory, "tokenizer_config.json").write_text(json.dumps(dict(chat_template=fixture.read_text())))
            text = ChatTemplate(directory).render(request.messages, tools=request.tools)
        self.assertIn("<tools>", text)
        self.assertIn("<function=read>\n<parameter=path>\na.py\n</parameter>", text)
        self.assertIn('<parameter=flags>\n{"full": true}\n</parameter>', text)
        self.assertIn("<|im_start|>user\n<tool_response>\nline four\n</tool_response><|im_end|>", text)
        self.assertEqual(request.messages[1]["tool_calls"][0]["function"]["arguments"]["offset"], 4)
        self.assertIsNone(messages[1]["content"])
        self.assertIsInstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)

    def test_tool_prefix_uses_same_render_controls(self):
        request = self.request(tool_choice="required", enable_thinking=True)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
            Path(directory, "tokenizer_config.json").write_text(json.dumps(dict(chat_template=fixture.read_text())))
            template = ChatTemplate(directory)
            options = dict(enable_thinking=request.enable_thinking, reasoning_effort=request.reasoning_effort,
                           tools=request.tools, tool_choice=request.tool_choice)
            prompt = template.render(request.messages, **options)
            prefix = template.render(request.messages, **options, add_generation_prompt=False)
        self.assertTrue(prompt.startswith(prefix))
        self.assertIn("<tools>", prefix)
        self.assertIn("Call one of the available tools.", prefix)
        self.assertNotEqual(prefix, prompt)

    def test_reject_unknown_choice_or_tool(self):
        for options in (dict(tool_choice="bad"), dict(tool_choice=dict(type="function", function=dict(name="missing"))),
                        dict(tools=[TOOLS[0], TOOLS[0]]), dict(tools=[dict(type="function", function=dict(name="bad>name"))])):
            with self.assertRaises(ApiError): self.request(**options)

    def test_reject_bad_history(self):
        user = dict(role="user", content="Read")
        for messages in ([user, dict(role="tool", tool_call_id="missing", content="bad")],
                         [user, dict(role="assistant", content=None, tool_calls=[dict(id="x", type="function", function=dict(
                             name="read", arguments='{"path":"a","path":"b"}'))])],
                         [user, dict(role="assistant", content=None, tool_calls=[dict(id="x", type="function", function=dict(
                             name="read", arguments='{"path":NaN}'))])]):
            with self.assertRaises(ApiError): self.request(messages=messages)

class ToolOutputTests(unittest.TestCase):
    def parser(self, choice="auto"):
        from server.tool_output import ToolOutput
        request = validate_request(dict(messages=[dict(role="user", content="Read")], tools=TOOLS, tool_choice=choice),
                                   "chat", "crucible")
        return ToolOutput(request)

    def block(self, path="a.py", offset="4", flags='{"full":true}'):
        return ("<tool_call>\n<function=read>\n<parameter=path>\n" + path + "\n</parameter>\n"
                "<parameter=offset>\n" + offset + "\n</parameter>\n<parameter=flags>\n" + flags +
                "\n</parameter>\n</function>\n</tool_call>")

    def invoke(self, path="a.py", offset="4", flags='{"full":true}'):
        return ('<invoke name="read">\n<parameter name="path">' + path + '</parameter>\n'
                '<parameter name="offset">' + offset + '</parameter>\n'
                '<parameter name="flags">' + flags + '</parameter>\n</invoke>')

    def test_typed_values_and_openai_deltas(self):
        parser = self.parser()
        content = parser.feed("I will read it.\n" + self.block())
        message, deltas, finish = parser.finish("stop")
        self.assertEqual(content, "I will read it.\n")
        self.assertEqual(message["content"], content)
        call = message["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(json.loads(call["function"]["arguments"]), dict(path="a.py", offset=4, flags=dict(full=True)))
        self.assertEqual(deltas[0]["tool_calls"][0]["index"], 0)
        self.assertEqual(deltas[0]["tool_calls"][0]["id"], call["id"])
        self.assertEqual(finish, "tool_calls")

    def test_invoke_alias(self):
        parser = self.parser()
        content = parser.feed("I will read it.\n" + self.invoke())
        message, deltas, finish = parser.finish("stop")
        self.assertEqual(content, "I will read it.\n")
        self.assertEqual(message["content"], content)
        call = message["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         dict(path="a.py", offset=4, flags=dict(full=True)))
        self.assertEqual(finish, "tool_calls")
        self.assertFalse(parser.fallback)

    def test_prose_invoke_before_real_call(self):
        raw = 'See <invoke name="x"> text.\n' + self.block()
        for stream in (False, True):
            parser = self.parser()
            parser.stream = ToolCallStream(parser.request, parser.definitions) if stream else None
            parser.feed(raw)
            message, deltas, reason = parser.finish("stop")
            self.assertEqual(message["content"], 'See <invoke name="x"> text.\n')
            self.assertEqual(len(message["tool_calls"]), 1)
            self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")
            self.assertEqual(reason, "tool_calls")
            self.assertFalse(parser.fallback)
            self.assertEqual(len(deltas), 1)

    def test_incomplete_invoke_prefix_is_content(self):
        parser = self.parser()
        parser.feed("<invo")
        message, deltas, reason = parser.finish("stop")
        self.assertEqual(message["content"], "<invo")
        self.assertEqual(deltas, [])
        self.assertEqual(reason, "stop")
        self.assertFalse(parser.fallback)

    def test_plain_text_with_tag_like_prefix(self):
        for raw in ("Use a <", "The tag starts with <i",
                    'Claude writes <invoke name="read"> blocks; here is how they look. Done.'):
            parser = self.parser()
            parser.feed(raw)
            message, deltas, reason = parser.finish("stop")
            self.assertEqual(message["content"], raw)
            self.assertEqual(deltas, [])
            self.assertEqual(reason, "stop")
            self.assertFalse(parser.fallback)

    def test_all_stream_splits(self):
        raw = "Before.\n" + self.block(path="line one\n雪") + "\n" + self.block("b.py")
        for split in range(len(raw) + 1):
            parser = self.parser()
            text = parser.feed(raw[:split]) + parser.feed(raw[split:])
            message, deltas, reason = parser.finish("stop")
            self.assertEqual(text, "Before.\n")
            self.assertEqual(len(message["tool_calls"]), 2)
            self.assertEqual(len(deltas), 2)
            self.assertEqual(reason, "tool_calls")
            self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"])["path"], "line one\n雪")

    def test_normal_output_and_partial_less_than(self):
        parser = self.parser()
        text = parser.feed("a <") + parser.feed(" b")
        message, deltas, reason = parser.finish("length")
        self.assertEqual(message["content"], "a < b")
        self.assertEqual(text, message["content"])
        self.assertEqual(deltas, [])
        self.assertEqual(reason, "length")

    def test_malformed_falls_back_to_content(self):
        # As other engines: text that does not parse as a valid call is the answer, never a half-validated call or an error.
        # A complete, well-formed call that fails only the schema goes out as a call (design decision, Oct 8; test_tool_pass).
        for raw, value in ((self.block(offset="not an integer"), dict(offset="not an integer")),
                           (self.block(flags='{"x":1,"x":2}'), dict(flags='{"x":1,"x":2}')),
                           (self.block(flags='{"x":NaN}'), dict(flags='{"x":NaN}')),
                           (self.block(flags='{"x":1e999}'), dict(flags='{"x":1e999}')),
                           (self.block().replace("<function=read>", "<function=missing>"), dict(path="a.py"))):
            parser = self.parser()
            parser.feed(raw)
            message, _, reason = parser.finish("stop")
            self.assertEqual(reason, "tool_calls")
            arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
            self.assertEqual({key: arguments[key] for key in value}, value)
            self.assertFalse(parser.fallback)
            self.assertTrue(parser.unchecked)
        for raw in (self.block().replace("<parameter=offset>", "<parameter=path>"),
                    self.block()[:-5], self.block() + "suffix", "<tool_call>{\"name\":\"read\"}</tool_call>"):
            parser = self.parser()
            streamed = parser.feed("Before. ") + parser.feed(raw)
            message, deltas, reason = parser.finish("stop")
            self.assertEqual((message, deltas, reason), (dict(role="assistant", content="Before. " + raw), [], "stop"))
            self.assertEqual(streamed, "Before. ")
            self.assertTrue(parser.fallback)
        parser = self.parser()
        parser.feed(self.block()[:30])  # cut by max_tokens
        self.assertEqual(parser.finish("length")[2], "length")
        self.assertTrue(parser.fallback)
        parser = self.parser()
        parser.feed(self.block())
        parser.finish("stop")
        self.assertFalse(parser.fallback)

    def test_choice_enforcement(self):
        # A call that breaks tool_choice is not published; the text is the answer.
        parser = self.parser("required")
        parser.feed("No call.")
        self.assertEqual(parser.finish("stop")[0], dict(role="assistant", content="No call."))
        parser = self.parser("none")
        parser.feed(self.block())
        self.assertEqual(parser.finish("stop")[0], dict(role="assistant", content=self.block()))

if __name__ == "__main__": unittest.main()
