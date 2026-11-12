"""Issue 4: LiteLLM and Hermes send `reasoning` next to `reasoning_content` in assistant history."""
import json
from pathlib import Path
import tempfile
import unittest
from server.template import ChatTemplate
from server.api import ApiError, chat_messages, validate_request

FIXTURE = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
CALL = dict(id="call_1", type="function", function=dict(name="read", arguments='{"path": "a.py"}'))


def hermes(**assistant):
    return [dict(role="system", content="rules"), dict(role="user", content="read a.py"),
            dict(role="assistant", content="Reading.", tool_calls=[CALL]) | assistant,
            dict(role="tool", tool_call_id="call_1", content="print(1)")]


def refused(test, messages):
    with test.assertRaises(ApiError) as caught:
        chat_messages(messages)
    test.assertEqual(caught.exception.param, "messages")
    return str(caught.exception)


class ReasoningAliasTests(unittest.TestCase):
    def test_issue_curl_body(self):
        body = dict(model="crucible", max_tokens=32, messages=[
            dict(role="user", content="hi"),
            dict(role="assistant", content="hello", reasoning="greet back", reasoning_content="greet back"),
            dict(role="user", content="again")])
        request = validate_request(body, "chat", "crucible")
        self.assertEqual(request.messages[1]["reasoning_content"], "greet back")
        self.assertNotIn("reasoning", request.messages[1])

    def test_hermes_shape(self):
        messages = chat_messages(hermes(reasoning="plan", reasoning_content="plan"))
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["reasoning_content"], "plan")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_1")
        self.assertNotIn("reasoning", messages[2])

    def test_reasoning_only_is_reasoning_content(self):
        self.assertEqual(chat_messages(hermes(reasoning="plan"))[2], chat_messages(hermes(reasoning_content="plan"))[2])
        self.assertEqual(chat_messages(hermes(reasoning="plan"))[2]["reasoning_content"], "plan")

    def test_alias_on_every_role_that_takes_reasoning_content(self):
        messages = chat_messages([dict(role="system", content="s", reasoning="r0"), dict(role="user", content="u", reasoning="r1"),
                                  dict(role="assistant", content="", reasoning="r2", tool_calls=[CALL]),
                                  dict(role="tool", tool_call_id="call_1", content="t", reasoning="r3")])
        self.assertEqual([m["reasoning_content"] for m in messages], ["r0", "r1", "r2", "r3"])

    def test_different_values_name_field_and_index(self):
        text = refused(self, hermes(reasoning="alpha secret", reasoning_content="beta secret"))
        self.assertTrue(text.startswith("messages[2] has reasoning and reasoning_content with different text."), text)
        self.assertNotIn("secret", text)  # the error never repeats history text

    def test_empty_or_null_is_absent(self):
        for empty in ("", None):
            self.assertEqual(chat_messages(hermes(reasoning="one", reasoning_content=empty))[2]["reasoning_content"], "one")
            self.assertEqual(chat_messages(hermes(reasoning=empty, reasoning_content="two"))[2]["reasoning_content"], "two")
            self.assertEqual(chat_messages(hermes(reasoning_content=empty))[2]["reasoning_content"], "")
            self.assertEqual(chat_messages(hermes(reasoning=empty))[2]["reasoning_content"], "")
            for other in ("", None):
                self.assertEqual(chat_messages(hermes(reasoning=empty, reasoning_content=other))[2]["reasoning_content"], "")

    def test_non_string_reasoning_names_field_and_index(self):
        for value in (5, ["a"], dict(a=1), True, 0.0):
            text = refused(self, hermes(reasoning=value))
            self.assertIn("messages[2]", text)
            self.assertIn("reasoning must be a string", text)
            self.assertIn("messages[2] reasoning_content must be a string", refused(self, hermes(reasoning_content=value)))

    def test_unsupported_fields_name_fields_sorted_and_index(self):
        text = refused(self, [dict(role="user", content="a"), dict(role="assistant", content="b"),
                              dict(role="assistant", content="c", zeta=1, foo=2)])
        self.assertTrue(text.startswith("messages[2] (assistant) has unsupported fields: foo, zeta."), text)
        text = refused(self, [dict(role="user", content="a", tool_calls=[])])
        self.assertTrue(text.startswith("messages[0] (user) has unsupported fields: tool_calls."), text)

    def test_unsupported_field_with_unknown_role_omits_the_role(self):
        text = refused(self, [dict(role="developer", content="a", foo=1)])
        self.assertTrue(text.startswith("messages[0] has unsupported fields: foo."), text)


class ReasoningAliasPromptTests(unittest.TestCase):
    """Rule 1: equal values give the same prompt bytes as a request with reasoning_content only."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": FIXTURE.read_text()}))
        self.template = ChatTemplate(root)

    def tearDown(self):
        self.temp.cleanup()

    def render(self, messages, **options):
        return self.template.render(chat_messages(messages), **options)

    def test_prompt_bytes_equal(self):
        for options in (dict(), dict(enable_thinking=True), dict(preserve_thinking=True),
                        dict(enable_thinking=True, preserve_thinking=False)):
            plain = self.render(hermes(reasoning_content="plan"), **options)
            self.assertIn("plan", self.render(hermes(reasoning_content="plan"), preserve_thinking=True))
            for variant in (dict(reasoning="plan", reasoning_content="plan"), dict(reasoning="plan")):
                self.assertEqual(self.render(hermes(**variant), **options).encode(), plain.encode(), options)

    def test_null_and_empty_forms_render_as_the_field_left_out(self):
        for options in (dict(), dict(enable_thinking=True), dict(preserve_thinking=True)):
            plain = self.render(hermes(), **options).encode()
            for variant in (dict(reasoning=None), dict(reasoning=""), dict(reasoning_content=None), dict(reasoning_content=""),
                            dict(reasoning=None, reasoning_content=None), dict(reasoning="", reasoning_content=None)):
                self.assertEqual(self.render(hermes(**variant), **options).encode(), plain, variant)
            kept = self.render(hermes(reasoning_content="plan"), **options).encode()
            for variant in (dict(reasoning="plan", reasoning_content=""), dict(reasoning="plan", reasoning_content=None)):
                self.assertEqual(self.render(hermes(**variant), **options).encode(), kept, variant)

    def test_prompt_bytes_equal_with_content_and_no_tools(self):
        history = [dict(role="user", content="hi"), dict(role="assistant", content="hello"),
                   dict(role="user", content="again")]
        plain = history[:1] + [history[1] | dict(reasoning_content="greet")] + history[2:]
        both = history[:1] + [history[1] | dict(reasoning="greet", reasoning_content="greet")] + history[2:]
        for options in (dict(), dict(preserve_thinking=True), dict(enable_thinking=True)):
            self.assertEqual(self.render(both, **options).encode(), self.render(plain, **options).encode())


if __name__ == "__main__":
    unittest.main()
