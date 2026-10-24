import os
import unittest
from server.api import validate_request
from server.tokenization import ModelTokenizer

CHECKPOINT = os.environ.get("CRUCIBLE_TEST_CKPT")

@unittest.skipUnless(CHECKPOINT, "Set CRUCIBLE_TEST_CKPT to test the real checkpoint tokenizer.")
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = ModelTokenizer(CHECKPOINT)

    def test_unicode_stream_matches_batch(self):
        for text in ("hello\nworld", "🙂 café 日本語", "<think>\nreasoning\n</think>\n\nanswer"):
            ids = self.adapter.tokenizer.encode(text, add_special_tokens=False).ids
            stream = self.adapter.stream()
            decoded = "".join(self.adapter.step(stream, token) for token in ids)
            self.assertEqual(decoded, self.adapter.decode(ids))

    def test_real_chat_template(self):
        request = validate_request(dict(messages=[dict(role="user", content="hello")]), "chat", "crucible")
        ids = self.adapter.encode(request)
        text = self.adapter.tokenizer.decode(ids, skip_special_tokens=False)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_forced_think_close_text(self):
        ids = self.adapter.think_close_ids()
        self.assertEqual(self.adapter.decode(list(ids)), "\n\nConsidering the limited time by the user, I have to "
                         "give the solution based on the thinking directly now.\n</think>\n\n")
        self.assertIn(self.adapter.tokenizer.token_to_id("</think>"), ids)

    def test_thinking_turn_extends_previous_prompt_and_output(self):
        first = [dict(role="user", content="Add 17 and 25.")]
        prompt = self.adapter.encode(validate_request(dict(messages=first, enable_thinking=True), "chat", "crucible"))
        output = self.adapter.tokenizer.encode("The user wants 17+25 = 42.\n</think>\n\nThe sum is 42.<|im_end|>",
                                               add_special_tokens=False).ids
        history = first + [dict(role="assistant", content="The sum is 42.", reasoning_content="The user wants 17+25 = 42."),
                           dict(role="user", content="Thanks.")]
        following = self.adapter.encode(validate_request(dict(messages=history, enable_thinking=True), "chat", "crucible"))
        self.assertEqual(following[:len(prompt) + len(output)], prompt + output)

    def test_closed_think_turn_extends_to_the_next_prompt(self):
        # an earlier soak run: a 4.05 soak lost replay because the model wrote ":" then "<" where the tokenizer makes ":<". A
        # reasoning cut after a newline also merges with the close's "\n\n" when the client sends the text back.
        import json
        from server.tests.serve_fixtures import invoke
        from server.tests.test_think_budget import SequenceBackend
        encode = lambda text: self.adapter.tokenizer.encode(text, add_special_tokens=False).ids
        split = [self.adapter.tokenizer.token_to_id(":"), self.adapter.tokenizer.token_to_id("<")]
        reasonings = (encode("Check main.py") + split + encode("line> first"), encode("Read the file.\n"),
                      encode("Read the file. "))
        answer = encode("I will read it.") + [248046]
        first = [dict(role="user", content="Find the bug.")]
        for reasoning in reasonings:
            with self.subTest(reasoning=self.adapter.decode(reasoning)):
                body = dict(model="crucible", messages=first, enable_thinking=True, max_tokens=4096,
                            max_thinking_tokens=len(reasoning))
                backend = SequenceBackend((reasoning + [0] * 5, "length"), (answer, "stop"))
                message = json.loads(invoke(backend, self.adapter, body).body)["choices"][0]["message"]
                history = first + [dict(role="assistant", content=message["content"],
                                        reasoning_content=message["reasoning_content"]),
                                   dict(role="user", content="Thanks.")]
                following = self.adapter.encode(validate_request(dict(messages=history, enable_thinking=True),
                                                                 "chat", "crucible"))
                tail = backend.submissions[1][0] + answer[:-1]
                self.assertEqual(following[:len(tail)], tail)

    def test_tool_prefix_uses_identical_render_options(self):
        tools = [dict(type="function", function=dict(name="read_file", parameters=dict(
            type="object", properties=dict(path=dict(type="string")), required=["path"])))]
        messages = [dict(role="user", content="Read main.py."),
                    dict(role="assistant", content=None, tool_calls=[dict(id="call_1", type="function",
                        function=dict(name="read_file", arguments='{"path":"main.py"}'))]),
                    dict(role="tool", tool_call_id="call_1", content='print("café")'),
                    dict(role="user", content="Explain this file.")]
        for choice in ("auto", "none", "required", dict(type="function", function=dict(name="read_file"))):
            for thinking in (False, True):
                for effort in ("low", "medium", "xhigh"):
                    body = dict(messages=messages, tools=tools, tool_choice=choice,
                                enable_thinking=thinking, reasoning_effort=effort)
                    results = []
                    for control in ("auto", "bypass", "refresh"):
                        request = validate_request(body | dict(cache_control=control), "chat", "crucible")
                        ids = self.adapter.encode(request)
                        prefix = self.adapter.cache_prefix_tokens(request, ids)
                        self.assertGreater(prefix, 0)
                        expected = self.adapter.tokenizer.encode(
                            self.adapter.render_chat(request, add_generation_prompt=False), add_special_tokens=False).ids
                        self.assertEqual(ids[:prefix], expected)
                        results.append((ids, prefix))
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(results[0], results[2])

if __name__ == "__main__":
    unittest.main()
