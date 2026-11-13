"""History rewrites through HTTP and the native CPU host-cache policy."""
import json
import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from server.api import validate_request
from server.batch_backend import BatchedEngineBackend
from server.tests.test_count_tokens import Server
from server.tokenization import ModelTokenizer

CHECKPOINT = os.environ.get("CRUCIBLE_TEST_CKPT")


@pytest.mark.skipif(not CHECKPOINT, reason="Set CRUCIBLE_TEST_CKPT to use the checkpoint tokenizer.")
def test_history_rewrite_retains_stable_point_on_failed_capture(tmp_path, capfd):
    executable = tmp_path / "cache_frontier_engine"
    source = Path(__file__).with_name("fixtures") / "cache_frontier_engine.cpp"
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++20",
                    str(source), "-o", str(executable)], check=True, timeout=60)
    tokenizer = ModelTokenizer(CHECKPOINT)

    def message(role, content):
        return dict(role=role, content=content)

    history = [message("system", " rule" * 11669), message("user", " item" * 45677),
               message("assistant", "Done."), message("user", "Review.")]
    prefix = " note" * 2024
    original = history + [message("user", prefix + " alpha" + " old" * 43931)]
    edited = history + [message("user", prefix + " beta" + " new" * 34246)]
    successor = edited + [message("user", " extra" * 4097)]
    rewrite_again = history + [message("user", prefix + " gamma" + " new" * 34246)]

    def tokens(messages):
        return tokenizer.encode(validate_request(dict(messages=messages), "chat", "crucible"))

    before, after = tokens(original), tokens(edited)
    assert next(i for i, (a, b) in enumerate(zip(before, after)) if a != b) == 59401

    def run(stable):
        def start(*unused):
            return BatchedEngineBackend(executable, "stable" if stable else "stock", context=131072, streams=2,
                                        output_timeout=5)

        with patch("server.tests.test_count_tokens.start_backend", start):
            server = Server(tokenizer, batch=True)
        try:
            def chat(messages, cache_control="auto"):
                status, body = server.post("/v1/chat/completions", dict(messages=messages, max_tokens=1, spec=0,
                                                                       cache_control=cache_control))
                assert status == 200, body
                response = json.loads(body)
                usage = response["usage"]
                count = usage["prompt_tokens"]
                cached = usage["prompt_tokens_details"]["cached_tokens"]
                return count, cached, count - cached, response["choices"][0]["message"]

            assert chat(original)[:3] == (103342, 0, 103342)
            rewrite = chat(edited)
            if stable:
                for _ in range(2):
                    assert chat(successor)[1] == 57344
                assert chat(rewrite_again) == rewrite
                cold = chat(rewrite_again, "bypass")
                assert cold[1] == 0 and cold[3] == rewrite[3]
            return rewrite[:3]
        finally:
            server.close()

    assert run(False) == (93657, 8192, 85465)
    assert run(True) == (93657, 57344, 36313)
    log = capfd.readouterr().err
    assert "stable capture refused: memory at 90112" in log
    assert "stable capture refused: host at 90112" in log
