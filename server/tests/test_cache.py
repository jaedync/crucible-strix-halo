import unittest
from server.api import ApiError, validate_request
from types import SimpleNamespace
from server.response import CompletionResponse
from server.tokenization import ModelTokenizer

class CacheRequestTests(unittest.TestCase):
    def test_controls(self):
        for control in ("auto", "bypass", "refresh"):
            request = validate_request(dict(prompt=[1, 2], cache_control=control), "completion", "crucible")
            self.assertEqual(request.cache_control, control)
        self.assertEqual(validate_request(dict(prompt=[1]), "completion", "crucible").cache_control, "auto")
        for value in (None, True, "purge", {}):
            with self.assertRaises(ApiError):
                validate_request(dict(prompt=[1], cache_control=value), "completion", "crucible")

class PrefixRenderTests(unittest.TestCase):
    def test_tools_and_bypass_keep_render_options(self):
        calls = []
        class Template:
            def render(self, messages, thinking, effort, tools, choice, add_generation_prompt=True):
                calls.append((messages, thinking, effort, tools, choice, add_generation_prompt))
                return repr((messages, thinking, effort, tools, choice)) + ("TAIL" if add_generation_prompt else "")
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return SimpleNamespace(ids=list(map(ord, text)))
        tokenizer = ModelTokenizer.__new__(ModelTokenizer)
        tokenizer.template, tokenizer.tokenizer = Template(), Tokenizer()
        for control in ("auto", "bypass", "refresh"):
            request = SimpleNamespace(kind="chat", messages=({"role": "user", "content": "x"},),
                enable_thinking=True, reasoning_effort="medium", tools=({"name": "test"},),
                tool_choice={"type": "function", "function": {"name": "test"}}, cache_control=control)
            ids = tokenizer.encode(request)
            prefix = tokenizer.cache_prefix_tokens(request, ids)
            self.assertEqual(prefix, len(ids) - 4)
            self.assertEqual(calls[-1][:-1], calls[-2][:-1])
            self.assertEqual(calls[-1][-1], False)
            self.assertEqual(tokenizer.cache_prefix_tokens(request, [0] + ids[1:]), 0)

    def test_shared_prefix_is_the_system_and_tools_block(self):
        class Template:
            user_separator = "\n"

            def render(self, messages, thinking, effort, tools, choice, add_generation_prompt=True):
                head = "<|im_start|>system\n" + "".join(m["content"] for m in messages if m["role"] == "system")
                head += "".join(t["name"] for t in tools) + "<|im_end|>\n" if head != "<|im_start|>system\n" or tools else ""
                if head == "<|im_start|>system\n": head = ""
                body = "".join(f"<|im_start|>{m['role']}{self.user_separator}{m['content']}<|im_end|>\n"
                               for m in messages if m["role"] != "system")
                return head + body + ("<|im_start|>assistant\n" if add_generation_prompt else "")
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return SimpleNamespace(ids=list(map(ord, text)))
        tokenizer = ModelTokenizer.__new__(ModelTokenizer)
        tokenizer.template, tokenizer.tokenizer = Template(), Tokenizer()
        def chat(messages, tools=()):
            return SimpleNamespace(kind="chat", messages=messages, enable_thinking=False, reasoning_effort="xhigh",
                                   tools=tools, tool_choice="auto", cache_control="auto")
        system = {"role": "system", "content": "rules"}
        turns = ({"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"})
        block = "<|im_start|>system\nrulestool<|im_end|>\n"
        for request in (chat((system,) + turns, ({"name": "tool"},)), chat((system,) + turns[:1], ({"name": "tool"},))):
            ids = tokenizer.encode(request)
            self.assertEqual(tokenizer.shared_prefix_tokens(request, ids), len(block))
            self.assertEqual(tokenizer.shared_prefix_tokens(request, [0] + ids[1:]), 0)
        tokenizer.template.user_separator = "\n\n"
        request = chat((system,) + turns, ({"name": "tool"},))
        self.assertEqual(tokenizer.shared_prefix_tokens(request, tokenizer.encode(request)), len(block))
        self.assertEqual(tokenizer.shared_prefix_tokens(chat(turns), tokenizer.encode(chat(turns))), 0)
        def refuse(*args, **kwargs): raise ApiError("No user query found in messages.", "messages")
        tokenizer.template = SimpleNamespace(render=refuse)
        self.assertEqual(tokenizer.shared_prefix_tokens(chat((system,) + turns), [1, 2]), 0)
        completion = SimpleNamespace(kind="completion")
        self.assertEqual(tokenizer.shared_prefix_tokens(completion, [1, 2]), 0)

    def test_shared_prefix_reaches_the_payload_only_when_known(self):
        import queue, threading
        from server.backend import EngineBackend
        backend = EngineBackend.__new__(EngineBackend)
        backend.context, backend.closed, backend.fatal, backend.jobs = 64, threading.Event(), None, queue.Queue()
        backend.ownership_lock, backend.owned_jobs = threading.Lock(), {}
        request = validate_request(dict(prompt=[1, 2, 3], max_tokens=1, spec=0), "completion", "crucible")
        self.assertNotIn("shared_prefix_tokens", backend.submit([1, 2, 3], request, 2).payload)
        self.assertEqual(backend.submit([1, 2, 3], request, 2, 1).payload["shared_prefix_tokens"], 1)

    def test_cached_usage_and_invalid_counts(self):
        request = validate_request(dict(prompt=[1, 2, 3], max_tokens=1), "completion", "crucible")
        sent = []
        handler = SimpleNamespace(send_json=lambda status, body: sent.append(body))
        tokenizer = SimpleNamespace(stream=lambda: None, step=lambda stream, token: "x", decode=lambda ids: "x")
        service = SimpleNamespace(tokenizer=tokenizer, model="crucible")
        for count in (0, 2, 3):
            response = CompletionResponse(handler, service, request, [1, 2, 3])
            response.token(4)
            response.finish(dict(finish_reason="length", cached_tokens=count))
            self.assertEqual(sent[-1]["usage"]["prompt_tokens_details"], dict(cached_tokens=count))
        for count in (-1, 4, True, "2"):
            response = CompletionResponse(handler, service, request, [1, 2, 3])
            response.token(4)
            with self.assertRaises(ApiError):
                response.finish(dict(finish_reason="length", cached_tokens=count))

if __name__ == "__main__":
    unittest.main()
