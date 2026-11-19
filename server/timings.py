"""The OpenAI-compatible `timings` object of a finished reply, from the server's turn clocks (metrics.new_turn).

OpenAI-compatible server names and units: prompt_n is the prompt tokens that the engine processed (the cached ones are cache_n),
prompt_ms runs to the first token, predicted_n is every generated token, and predicted_ms runs from the first token to
the end. Crucible's prompt_ms is the server time from submit to the first token, so it includes a queue wait. A benchmark
client reads prompt_ms as TTFT and the two per_second values as prefill and decode speed.
"""
from . import metrics


def rate(count, ms):
    """(ms per token, tokens per second); 0 for both when a value is 0, so the JSON never holds infinity."""
    if count <= 0 or ms <= 0:
        return 0.0, 0.0
    return ms / count, count * 1000.0 / ms


def clock(jobs, index, name):
    turn = getattr(jobs[index], "turn", None) if jobs else None
    value = turn.get(name) if isinstance(turn, dict) else None
    return value if type(value) in (int, float) else None


def timings(jobs, prompt_count, completion_count, cached_tokens=None, now=None):
    """The timings of a reply that its engine jobs made (the first job holds the first token), or None without clocks.
    A last job with no done time (a stop string ended the reply first) ends now."""
    submit = clock(jobs, 0, "t_submit")
    if submit is None:
        return None
    done = clock(jobs, -1, "t_done")
    if done is None:
        done = metrics.now() if now is None else now
    first = clock(jobs, 0, "t_first_token")
    cache_n = cached_tokens if type(cached_tokens) is int else 0
    prompt_n = max(prompt_count - cache_n, 0)
    prompt_ms = max(((done if first is None else first) - submit) * 1000.0, 0.0)
    predicted_ms = max((done - first) * 1000.0, 0.0) if first is not None and completion_count else 0.0
    prompt_per_token, prompt_per_second = rate(prompt_n, prompt_ms)
    predicted_per_token, predicted_per_second = rate(completion_count, predicted_ms)
    return dict(cache_n=cache_n, prompt_n=prompt_n, prompt_ms=prompt_ms, prompt_per_token_ms=prompt_per_token,
                prompt_per_second=prompt_per_second, predicted_n=completion_count, predicted_ms=predicted_ms,
                predicted_per_token_ms=predicted_per_token, predicted_per_second=predicted_per_second)
