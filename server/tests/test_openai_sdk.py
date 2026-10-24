"""Run the official OpenAI SDK against the real HTTP layer with CPU scripts."""
import json
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import threading
import unittest
try:
    import openai
except ImportError:
    openai = None
from server.api import ApiError
from server.http import Service, handler_for
from server.tests.sdk_fixture import RecordedBackend, RecordedTokenizer, Script, TOOLS, tool_block
from server.tests.serve_fixtures import joined_calls

@unittest.skipUnless(openai, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class OpenAISdkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RecordedTokenizer()
        cls.backend = RecordedBackend(cls.tokenizer)
        handler = handler_for(Service(cls.backend, cls.tokenizer))
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # a free port: two sessions may run the tests at once
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = openai.OpenAI(api_key="crucible-local", base_url=f"http://127.0.0.1:{cls.server.server_address[1]}/v1",
                                  max_retries=0, timeout=5,
                                  http_client=openai.DefaultHttpxClient(trust_env=False))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.backend.reset()

    def chat(self, **options):
        return self.client.chat.completions.create(model="crucible", messages=[dict(role="user", content="Hello")],
                                                   **options)

    @unittest.skipUnless(os.environ.get("CRUCIBLE_PI_AI_DIR") and shutil.which("node"),
                         "set CRUCIBLE_PI_AI_DIR for the optional Pi transport check")
    def test_pi_transport_text_and_tools(self):
        for mode, script, reason in (("text", Script(), "stop"),
                                     ("tools", Script((tool_block(),)), "toolUse")):
            with self.subTest(mode=mode):
                self.backend.reset(script)
                process = subprocess.run(["node", str(Path(__file__).with_name("pi_client.mjs")), mode],
                                         capture_output=True, text=True, timeout=15, check=True)
                result = json.loads(process.stdout)
                self.assertEqual(result["stopReason"], reason)
                self.assertEqual(result["events"][0], "start")
                self.assertEqual(result["events"][-1], "done")
                self.assertEqual(result["usage"]["input"], 3)
                self.assertEqual(result["usage"]["output"], len(script.pieces))
                if mode == "tools":
                    self.assertEqual(result["content"][0]["arguments"], dict(a=17, b=25))

    @unittest.skipUnless(os.environ.get("CRUCIBLE_PI_AI_DIR") and shutil.which("node"),
                         "set CRUCIBLE_PI_AI_DIR for the optional Pi agent-loop check")
    def test_pi_agent_does_not_run_a_streamed_call_that_turns_invalid(self):
        # Item 4 (design decision, Oct 7): the id, name and a went out, then the call breaks. The fault must be structural (text
        # after </parameter>): since tool-pass, a call that fails only the schema goes out as a tool call. The server
        # ends with "length"; Pi's loop then sends an error tool result instead of running the call and asks again.
        broken = ("<tool_call>\n<function=add>\n<parameter=a>\n17\n</parameter>\n"
                  "<parameter=b>\n25\n</parameter>\njunk</function>\n</tool_call>")
        result = self.pi_agent(broken)
        self.assertEqual(result["executed"], [])
        self.assertEqual(result["turns"], [dict(stopReason="length", toolCalls=1), dict(stopReason="stop", toolCalls=0)])
        self.assertEqual(len(result["results"]), 1)
        self.assertTrue(result["results"][0]["isError"])
        self.assertIn("was not executed", result["results"][0]["text"])
        self.assertEqual(len(self.backend.requests), 2)

    @unittest.skipUnless(os.environ.get("CRUCIBLE_PI_AI_DIR") and shutil.which("node"),
                         "set CRUCIBLE_PI_AI_DIR for the optional Pi agent-loop check")
    def test_pi_agent_runs_a_valid_streamed_call(self):
        result = self.pi_agent(tool_block())
        self.assertEqual(result["executed"], [dict(a=17, b=25)])
        self.assertEqual(result["turns"], [dict(stopReason="toolUse", toolCalls=1), dict(stopReason="stop", toolCalls=0)])
        self.assertEqual(result["results"], [dict(isError=False, text="42")])
        self.assertEqual(len(self.backend.requests), 2)

    def pi_agent(self, first):
        self.backend.reset(Script((first,)), Script(("The sum is 42.",)))
        process = subprocess.run(["node", str(Path(__file__).with_name("pi_client.mjs")), "agent"],
                                 capture_output=True, text=True, timeout=30, check=True)
        return json.loads(process.stdout)

    def test_plain_and_usage(self):
        result = self.chat(max_tokens=8)
        self.assertEqual(result.object, "chat.completion")
        self.assertEqual(result.choices[0].message.role, "assistant")
        self.assertEqual(result.choices[0].message.content, "Hello snow 雪.")
        self.assertEqual(result.choices[0].finish_reason, "stop")
        self.assertEqual(result.usage.model_dump(exclude_none=True),
                         dict(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        # The handler cancels the job after it sends the response, so wait for it.
        self.assertTrue(self.backend.jobs[0].cancelled.wait(5))

    def test_stream_and_final_usage(self):
        chunks = list(self.chat(max_tokens=8, stream=True, stream_options=dict(include_usage=True)))
        self.assertTrue(all(chunk.object == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual(len({chunk.id for chunk in chunks}), 1)
        text = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        self.assertEqual(text, "Hello snow 雪.")
        self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[-1].choices, [])
        self.assertEqual(chunks[-1].usage.total_tokens, 5)
        self.assertTrue(all("usage" in chunk.model_fields_set and chunk.usage is None for chunk in chunks[:-1]))

    def test_stream_without_usage(self):
        chunks = list(self.chat(stream=True))
        self.assertTrue(all(chunk.usage is None for chunk in chunks))
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_both_token_limit_names(self):
        for field in ("max_tokens", "max_completion_tokens"):
            with self.subTest(field=field):
                result = self.chat(**{field: 1})
                self.assertEqual(result.choices[0].message.content, "Hello")
                self.assertEqual(result.choices[0].finish_reason, "length")
                self.assertEqual(result.usage.completion_tokens, 1)
                self.assertEqual(self.backend.requests[-1].max_tokens, 1)

    def test_token_limit_error_names_supplied_field(self):
        for field in ("max_tokens", "max_completion_tokens"):
            with self.subTest(field=field), self.assertRaises(openai.BadRequestError) as error:
                self.chat(**{field: 0})
            self.assertEqual(error.exception.body["param"], field)
        self.assertEqual(self.backend.requests, [])

    def test_conflicting_limits_reject_before_submission(self):
        with self.assertRaises(openai.BadRequestError) as error:
            self.chat(max_tokens=1, max_completion_tokens=1)
        self.assertEqual(error.exception.body["param"], "max_tokens")
        self.assertEqual(self.backend.requests, [])

    def test_plain_tool_choice_policies(self):
        for choice in ("auto", "required", dict(type="function", function=dict(name="add"))):
            with self.subTest(choice=choice):
                self.backend.reset(Script((tool_block(),)))
                result = self.chat(tools=TOOLS, tool_choice=choice)
                self.assertEqual(result.choices[0].finish_reason, "tool_calls")
                call = result.choices[0].message.tool_calls[0]
                self.assertEqual(call.function.name, "add")
                self.assertEqual(json.loads(call.function.arguments), dict(a=17, b=25))
                self.assertTrue(call.id.startswith("call_"))
        self.backend.reset(Script(("No function call.",)))
        result = self.chat(tools=TOOLS, tool_choice="none")
        self.assertIsNone(result.choices[0].message.tool_calls)
        self.assertEqual(result.choices[0].finish_reason, "stop")

    def test_multiple_tools_plain_and_stream(self):
        script = Script((tool_block(1, 2), "\n", tool_block(3, 4)))
        self.backend.reset(script, script)
        result = self.chat(tools=TOOLS, tool_choice="required")
        calls = result.choices[0].message.tool_calls
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0].id, calls[1].id)
        self.assertEqual([json.loads(call.function.arguments) for call in calls], [dict(a=1, b=2), dict(a=3, b=4)])
        chunks = list(self.chat(tools=TOOLS, tool_choice="required", stream=True,
                                stream_options=dict(include_usage=True)))
        deltas = [call.model_dump(exclude_none=True) for chunk in chunks for choice in chunk.choices
                  for call in (choice.delta.tool_calls or [])]
        # Streamed calls: the first delta of each index carries the id, type and name; arguments follow.
        heads = [call for call in deltas if "id" in call]
        self.assertEqual([call["index"] for call in heads], [0, 1])
        self.assertTrue(all(call["type"] == "function" and call["function"]["name"] == "add" for call in heads))
        calls = joined_calls(deltas)
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in calls], [dict(a=1, b=2), dict(a=3, b=4)])
        self.assertEqual(chunks[-2].choices[0].finish_reason, "tool_calls")
        self.assertEqual(chunks[-1].usage.completion_tokens, 3)

    def test_tool_result_next_turn(self):
        self.backend.reset(Script((tool_block(),)), Script(("The result is 42.",)))
        first = self.chat(tools=TOOLS, tool_choice="required")
        message = first.choices[0].message.model_dump(exclude_none=True)
        call = first.choices[0].message.tool_calls[0]
        result = self.client.chat.completions.create(model="crucible", tools=TOOLS, messages=[
            dict(role="user", content="Add 17 and 25"), message,
            dict(role="tool", tool_call_id=call.id, content="42")])
        self.assertEqual(result.choices[0].message.content, "The result is 42.")

    def test_warm_cache_text_stop_omits_unknown_usage_detail(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.backend.reset(Script(cached_tokens=0), Script(cached_tokens=3),
                                   Script(("abcENDhidden", "discarded"), cached_tokens=3))
                cold = self.chat(extra_body=dict(cache_control="auto"))
                warm = self.chat(extra_body=dict(cache_control="auto"))
                self.assertEqual(cold.usage.prompt_tokens_details.cached_tokens, 0)
                self.assertEqual(warm.usage.prompt_tokens_details.cached_tokens, 3)
                result = self.chat(extra_body=dict(cache_control="auto"), stop="END", stream=stream,
                                   stream_options=dict(include_usage=True))
                if stream:
                    chunks = list(result)
                    self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")
                    usage = chunks[-1].usage
                else:
                    self.assertEqual(result.choices[0].message.content, "abc")
                    usage = result.usage
                self.assertEqual(usage.completion_tokens, 1)
                self.assertNotIn("prompt_tokens_details", usage.model_fields_set)
                self.assertTrue(self.backend.jobs[-1].cancelled.wait(5))

    def test_stop_split_across_tokens_and_remove_suffix(self):
        self.backend.reset(Script(("abcE", "N", "Dhidden"), "length"),
                           Script(("abcE", "N", "Dhidden"), "length"))
        plain = self.chat(stop="END")
        chunks = list(self.chat(stop=["END", "OTHER"], stream=True, stream_options=dict(include_usage=True)))
        text = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        self.assertEqual(plain.choices[0].message.content, "abc")
        self.assertEqual(text, "abc")
        self.assertEqual(plain.choices[0].finish_reason, "stop")
        self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")
        self.assertEqual(plain.usage.completion_tokens, 3)
        self.assertEqual(chunks[-1].usage.completion_tokens, 3)

    def test_unmatched_stop_prefix_flushes_at_length(self):
        self.backend.reset(Script(("abcEN",), "length"))
        chunks = list(self.chat(stop="END", stream=True))
        self.assertEqual("".join(c.delta.content or "" for x in chunks for c in x.choices), "abcEN")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

    def test_invalid_stops_reject_before_submission(self):
        for stop in ("", [""], ["a"] * 5, [1], 3):
            with self.subTest(stop=stop), self.assertRaises(openai.BadRequestError) as error:
                self.chat(stop=stop)
            self.assertEqual(error.exception.body["param"], "stop")
        self.assertEqual(self.backend.requests, [])

    def test_null_stop_and_empty_stop_list(self):
        for stop in (None, []):
            with self.subTest(stop=stop):
                self.assertEqual(self.chat(stop=stop).choices[0].message.content, "Hello snow 雪.")

    def test_models_and_unknown_model(self):
        models = self.client.models.list()
        self.assertEqual(models.object, "list")
        self.assertEqual(models.data[0].id, "crucible")
        with self.assertRaises(openai.NotFoundError) as error:
            self.client.chat.completions.create(model="missing", messages=[dict(role="user", content="Hi")])
        self.assertEqual(error.exception.body["type"], "invalid_request_error")
        self.assertEqual(error.exception.body["param"], "model")
        self.assertEqual(self.backend.requests, [])

    def test_error_status_and_shape(self):
        for status in (400, 503):
            self.backend.reset(Script(error=ApiError("A bounded test error.", "messages", status)))
            with self.subTest(status=status), self.assertRaises(openai.APIStatusError) as error:
                self.chat()
            self.assertEqual(error.exception.status_code, status)
            self.assertEqual(set(error.exception.body), {"message", "type", "param", "code"})
            self.assertEqual(error.exception.body["type"], "invalid_request_error" if status == 400 else "server_error")

    def test_malformed_tool_output_returns_content(self):
        # A duplicate parameter stays content; a bad value of a known parameter now goes out as a call.
        text = tool_block().replace("<parameter=b>", "<parameter=a>")
        self.backend.reset(Script((text,)))
        response = self.chat(tools=TOOLS, tool_choice="required")
        self.assertEqual(response.choices[0].message.content, text)
        self.assertIsNone(response.choices[0].message.tool_calls)
        self.assertEqual(response.choices[0].finish_reason, "stop")

if __name__ == "__main__":
    unittest.main()
