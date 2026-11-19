"""The OpenAI-compatible `timings` object: names and units as OpenAI-compatible server writes them, from the server's turn clocks.

A benchmark client that reads OpenAI-compatible timings takes prompt_ms as TTFT, prompt_per_second as
prefill speed and predicted_per_second as decode speed.
"""
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from server import timings
from server.tests.serve_fixtures import PiecesTokenizer, chat, chunks, invoke, token_events

FIELDS = ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms",
          "predicted_per_token_ms", "predicted_per_second"]


def turn(submit=100.0, first=100.5, done=102.5):
    return dict(t_submit=submit, t_first_token=first, t_done=done)


def test_llama_cpp_names_and_units():
    jobs = [SimpleNamespace(turn=turn())]
    result = timings.timings(jobs, prompt_count=1000, completion_count=81, cached_tokens=200)
    assert list(result) == FIELDS
    assert result["cache_n"] == 200
    assert result["prompt_n"] == 800  # OpenAI-compatible counts the prompt tokens that it processed, not the cached ones
    assert result["prompt_ms"] == pytest.approx(500.0)
    assert result["prompt_per_token_ms"] == pytest.approx(500.0 / 800)
    assert result["prompt_per_second"] == pytest.approx(1600.0)
    assert result["predicted_n"] == 81
    assert result["predicted_ms"] == pytest.approx(2000.0)
    assert result["predicted_per_token_ms"] == pytest.approx(2000.0 / 81)
    assert result["predicted_per_second"] == pytest.approx(40.5)


def test_no_cache_count_means_the_whole_prompt():
    result = timings.timings([SimpleNamespace(turn=turn())], prompt_count=10, completion_count=2)
    assert result["cache_n"] == 0 and result["prompt_n"] == 10


def test_a_continuation_ends_at_the_last_job():
    jobs = [SimpleNamespace(turn=turn(done=101.0)), SimpleNamespace(turn=turn(submit=101.0, first=101.2, done=103.5))]
    result = timings.timings(jobs, prompt_count=4, completion_count=10)
    assert result["prompt_ms"] == pytest.approx(500.0)
    assert result["predicted_ms"] == pytest.approx(3000.0)


def test_a_job_without_a_done_time_ends_now():
    jobs = [SimpleNamespace(turn=turn(done=None))]
    result = timings.timings(jobs, prompt_count=4, completion_count=3, now=101.5)
    assert result["predicted_ms"] == pytest.approx(1000.0)


def test_zero_times_give_zero_rates_never_infinity():
    result = timings.timings([SimpleNamespace(turn=turn(first=100.0, done=100.0))], prompt_count=4, completion_count=1)
    assert result["prompt_per_second"] == 0 and result["predicted_per_second"] == 0
    assert result["prompt_per_token_ms"] == 0 and result["predicted_per_token_ms"] == 0
    json.dumps(result, allow_nan=False)


def test_no_tokens_and_no_prompt_tokens():
    result = timings.timings([SimpleNamespace(turn=turn(first=None))], prompt_count=4, completion_count=0,
                             cached_tokens=4)
    assert result["prompt_n"] == 0 and result["predicted_n"] == 0
    assert result["prompt_ms"] == pytest.approx(2500.0)  # no first token: the prompt time runs to the end
    assert result["predicted_ms"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("jobs", [[], [SimpleNamespace()], [SimpleNamespace(turn=None)],
                                  [SimpleNamespace(turn=dict(t_submit=None))]])
def test_no_clock_no_timings(jobs):
    assert timings.timings(jobs, prompt_count=4, completion_count=1) is None


class TurnBackend:
    """ScriptBackend with the turn clocks that the engine backend sets (backend._stamp)."""
    def __init__(self, events, clocks=None):
        self.source, self.jobs, self.clocks = events, [], clocks or turn()

    def submit(self, ids, request, **options):
        events = queue.Queue()
        for event in self.source:
            events.put(dict(event))
        job = SimpleNamespace(events=events, cancelled=threading.Event(), turn=dict(self.clocks))
        self.jobs.append(job)
        return job


def test_chat_response_has_timings():
    response = invoke(TurnBackend(token_events(["a", "b"], cached_tokens=0)), PiecesTokenizer(["a", "b"]), chat())
    body = json.loads(response.body)
    assert body["timings"]["predicted_n"] == body["usage"]["completion_tokens"] == 3
    assert body["timings"]["prompt_n"] == body["usage"]["prompt_tokens"] == 1
    assert body["timings"]["predicted_ms"] == pytest.approx(2000.0)


def test_completion_response_has_timings():
    payload = dict(prompt="x", max_tokens=16, seed=1)
    response = invoke(TurnBackend(token_events(["a"])), PiecesTokenizer(["a"]), payload, path="/v1/completions")
    assert json.loads(response.body)["timings"]["predicted_n"] == 2


def test_stream_has_timings_on_the_last_chunk_only():
    response = invoke(TurnBackend(token_events(["a", "b"])), PiecesTokenizer(["a", "b"]), chat(stream=True))
    parts = chunks(response)
    assert [("timings" in part) for part in parts] == [False] * (len(parts) - 1) + [True]
    assert parts[-1]["choices"][0]["finish_reason"] == "stop"
    assert parts[-1]["timings"]["predicted_n"] == 3


def test_stream_with_usage_has_timings_on_the_usage_chunk():
    response = invoke(TurnBackend(token_events(["a"])), PiecesTokenizer(["a"]),
                      chat(stream=True, stream_options=dict(include_usage=True)))
    parts = chunks(response)
    assert [("timings" in part) for part in parts] == [False] * (len(parts) - 1) + [True]
    assert parts[-1]["choices"] == [] and parts[-1]["usage"]["completion_tokens"] == parts[-1]["timings"]["predicted_n"]


def test_a_backend_without_clocks_sends_no_timings():
    from server.tests.serve_fixtures import ScriptBackend
    response = invoke(ScriptBackend(token_events(["a"])), PiecesTokenizer(["a"]), chat())
    assert "timings" not in json.loads(response.body)


@pytest.mark.parametrize("batch", [False, True])
def test_engine_backend_clocks_reach_the_response(tmp_path, batch):
    """The real backends (with the fake engine) stamp the turn clocks that timings reads."""
    from server.tests.test_count_tokens import Server, tokenizer
    server = Server(tokenizer("fixture", tmp_path), batch=batch)
    try:
        status, data = server.post("/v1/chat/completions", chat(max_tokens=3, spec=0))
        assert status == 200, data
        body = json.loads(data)
        result = body["timings"]
        assert result["predicted_n"] == body["usage"]["completion_tokens"]
        assert result["prompt_n"] + result["cache_n"] == body["usage"]["prompt_tokens"]
        assert result["prompt_ms"] > 0
        json.dumps(result, allow_nan=False)
    finally:
        server.close()
