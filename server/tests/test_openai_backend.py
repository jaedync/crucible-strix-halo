"""A text stop must not leave native protocol events for the next SDK request."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
try:
    import openai
except ImportError:
    openai = None
from server.backend import EngineBackend
from server.http import Service, handler_for

class Tokenizer:
    def encode(self, request):
        return [1 if request.messages[0]["content"] == "first" else 9]
    def stream(self):
        return None
    def step(self, stream, token):
        return {2: "abcENDhidden", 10: "fresh", 11: " ok"}.get(token, "hidden")
    def decode(self, ids):
        return "".join(self.step(None, token) for token in ids)

@unittest.skipUnless(openai, "install server/tests/requirements-sdk.txt for SDK compatibility checks")
class OpenAIBackendTests(unittest.TestCase):
    def test_stop_cancellation_drains_native_output_before_next_request(self):
        popen = subprocess.Popen
        fake = Path(__file__).with_name("fake_engine.py")
        with patch("server.backend.subprocess.Popen", side_effect=lambda *args, **kw: popen([sys.executable, str(fake)], **kw)):
            backend = EngineBackend("unused", "unused", context=32)
        server = None
        try:
            handler = handler_for(Service(backend, Tokenizer()))
            handler.log_message = lambda *args: None
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # a free port: two sessions may run the tests at once
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            with openai.OpenAI(api_key="crucible-local", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", max_retries=0,
                               timeout=5, http_client=openai.DefaultHttpxClient(trust_env=False)) as client:
                first = client.chat.completions.create(model="crucible", max_tokens=8, stop="END",
                    messages=[dict(role="user", content="first")])
                second = client.chat.completions.create(model="crucible", max_tokens=2,
                    messages=[dict(role="user", content="next")])
                self.assertEqual(first.choices[0].message.content, "abc")
                self.assertEqual(first.choices[0].finish_reason, "stop")
                self.assertEqual(first.usage.completion_tokens, 1)
                self.assertEqual(second.choices[0].message.content, "fresh ok")
                self.assertEqual(second.usage.completion_tokens, 2)
        finally:
            if server:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
            backend.close()
        self.assertIsNotNone(backend.process.poll())

if __name__ == "__main__":
    unittest.main()
