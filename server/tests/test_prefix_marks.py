"""The crucible_turn line has privacy-safe marks of the shared prompt head (server/prefix_marks.py).

The marks show why two new sessions share only a short prefix: another reasoning line, another tools block, another
tool definition (by position), or text after the tools block. Real code paths only: the checkpoint template fixture, a
byte-level tokenizer with the checkpoint's added IDs, the HTTP handler, EngineBackend.submit and its done-event line.
"""
import contextlib
import io
import json
import queue
import re
import sys
import threading
from collections import OrderedDict

import pytest

from server import metrics, prefix_marks
from server.api import validate_request
from server.backend import EngineBackend
from server.tests.serve_fixtures import ScriptBackend, invoke, token_events
from server.tests.test_content_tokens import CHECKPOINT, synthetic_checkpoint
from server.tokenization import ModelTokenizer

HASH = re.compile("[0-9a-f]{10}")
SYSTEM = "You are a careful coding assistant. Project rules follow."


def tool(name, description="Read one file.", field="path"):
    return dict(type="function", function=dict(name=name, description=description, parameters=dict(
        type="object", properties={field: dict(type="string", description="secret-schema-" + field)},
        required=[field])))


TOOLS = [tool("read_file"), tool("run_shell", "Run a shell command.", "command"), tool("fetch_page", "Get a URL.", "url")]


@pytest.fixture(scope="module", params=["synthetic", "checkpoint"])
def tokenizer(request, tmp_path_factory):
    if request.param == "checkpoint":
        if not CHECKPOINT:
            pytest.skip("Set CRUCIBLE_TEST_CKPT to test the real checkpoint tokenizer.")
        return ModelTokenizer(CHECKPOINT)
    return ModelTokenizer(synthetic_checkpoint(tmp_path_factory.mktemp("tokenizer")))


def body(system=SYSTEM, tools=TOOLS, effort="high", **options):
    messages = ([dict(role="system", content=system)] if system is not None else []) + [dict(role="user", content="hi")]
    return dict(messages=messages, max_tokens=4, spec=0, **(dict(tools=tools) if tools else {}),
                **(dict(reasoning_effort=effort) if effort else {}), **options)


def marks(tokenizer, payload):
    request = validate_request(payload, "chat", "crucible")
    ids = tokenizer.encode(request)
    return tokenizer.prefix_marks(request, ids, tokenizer.shared_prefix_tokens(request, ids)), ids, request


def served_marks(tokenizer, payload):
    """The prefix_marks option that the real HTTP handler gives the backend."""
    backend = ScriptBackend(token_events(["a"]))
    assert invoke(backend, tokenizer, dict(payload, model="crucible")).statuses == [200]
    return backend.submissions[0][2].get("prefix_marks")


def logged_line(request, ids, options):
    """The crucible_turn line of a real EngineBackend job (submit, then its done event)."""
    backend = EngineBackend.__new__(EngineBackend)
    backend.context, backend.closed, backend.fatal, backend.jobs = 4096, threading.Event(), None, queue.Queue()
    backend.ownership_lock, backend.owned_jobs = threading.Lock(), {}
    job = backend.submit(ids, request, **options)
    job.turn["request_id"] = 9
    job.turn["t_admit"] = job.turn["t_first_gpu"] = job.turn["t_submit"]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        EngineBackend._stamp(job, dict(event="done", prompt_tokens=len(ids), completion_tokens=1, finish_reason="stop",
                                       cached_tokens=0, prefill_seconds=0.01))
    [line] = [line for line in err.getvalue().splitlines() if line.startswith("crucible_turn ")]
    return line


def test_the_turn_line_has_the_marks_and_no_text(tokenizer):
    payload = body()
    options = served_marks(tokenizer, payload)
    request = validate_request(payload, "chat", "crucible")
    ids = tokenizer.encode(request)
    shared = tokenizer.shared_prefix_tokens(request, ids)
    line = logged_line(request, ids, dict(shared_prefix_tokens=shared, prefix_marks=options))
    fields = json.loads(line.split(" ", 1)[1])
    assert fields["prefix_tool_count"] == 3
    assert len(fields["prefix_tool_hashes"]) == 3 and all(HASH.fullmatch(item) for item in fields["prefix_tool_hashes"])
    assert HASH.fullmatch(fields["prefix_reasoning_hash"]) and HASH.fullmatch(fields["prefix_tools_hash"])
    assert 0 < fields["prefix_tools_end_tokens"] < fields["prefix_system_end_tokens"] == shared
    # No text, tool name, description or schema reaches the log.
    for secret in ("read_file", "run_shell", "fetch_page", "Read one file", "secret-schema", "path", "careful",
                   "Reasoning effort", "xhigh", "Tools", "function"):
        assert secret not in line


def test_positions_are_exact_token_prefixes(tokenizer):
    fields, ids, _ = marks(tokenizer, body())
    text = tokenizer.decode(ids[:fields["prefix_tools_end_tokens"]])
    assert text.startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
    assert text.endswith("</IMPORTANT>")
    end = fields["prefix_system_end_tokens"]
    assert tokenizer.decode(ids[:end - 2]).endswith(SYSTEM) and ids[end - 2] == 248046  # <|im_end|>, then "\n"


def test_other_system_text_keeps_every_head_mark(tokenizer):
    one, two = marks(tokenizer, body())[0], marks(tokenizer, body(system=SYSTEM + " More rules for project B."))[0]
    keys = ("prefix_reasoning_hash", "prefix_tools_hash", "prefix_tool_hashes", "prefix_tool_count",
            "prefix_tools_end_tokens")
    assert {key: one[key] for key in keys} == {key: two[key] for key in keys}
    assert one["prefix_system_end_tokens"] < two["prefix_system_end_tokens"]


def test_other_reasoning_line_changes_only_the_reasoning_mark(tokenizer):
    high, medium = marks(tokenizer, body())[0], marks(tokenizer, body(effort="medium"))[0]
    assert high["prefix_reasoning_hash"] != medium["prefix_reasoning_hash"]
    assert (high["prefix_tools_hash"], high["prefix_tool_hashes"]) == (medium["prefix_tools_hash"],
                                                                       medium["prefix_tool_hashes"])
    assert high["prefix_tools_end_tokens"] > medium["prefix_tools_end_tokens"]
    assert marks(tokenizer, body(effort="medium"))[0] == medium  # equal within one process


def test_one_changed_tool_shows_its_position(tokenizer):
    base = marks(tokenizer, body())[0]
    changed = marks(tokenizer, body(tools=[TOOLS[0], tool("run_shell", "Run one command.", "command"), TOOLS[2]]))[0]
    assert [a == b for a, b in zip(base["prefix_tool_hashes"], changed["prefix_tool_hashes"])] == [True, False, True]
    assert base["prefix_tools_hash"] != changed["prefix_tools_hash"]
    assert base["prefix_reasoning_hash"] == changed["prefix_reasoning_hash"]
    swapped = marks(tokenizer, body(tools=[TOOLS[1], TOOLS[0], TOOLS[2]]))[0]
    assert swapped["prefix_tool_hashes"] == [base["prefix_tool_hashes"][i] for i in (1, 0, 2)]


def test_the_tool_list_is_capped_and_the_count_is_not(tokenizer):
    many = [tool(f"tool_{index}") for index in range(70)]
    fields = marks(tokenizer, body(tools=many))[0]
    assert fields["prefix_tool_count"] == 70 and len(fields["prefix_tool_hashes"]) == prefix_marks.TOOL_HASH_CAP == 64


def test_no_tools_and_no_system(tokenizer):
    fields = marks(tokenizer, body(tools=None))[0]
    assert fields["prefix_tool_count"] == 0
    assert not {"prefix_tools_hash", "prefix_tool_hashes", "prefix_tools_end_tokens"} & set(fields)
    bare, ids, _ = marks(tokenizer, body(system=None))
    assert bare["prefix_tools_end_tokens"] <= bare["prefix_system_end_tokens"]
    assert tokenizer.decode(ids[:bare["prefix_tools_end_tokens"]]).endswith("</IMPORTANT>")
    none = marks(tokenizer, body(tool_choice="none"))[0]
    assert none["prefix_tool_count"] == 0 and "prefix_tool_hashes" not in none
    completion = validate_request(dict(prompt=[1, 2], max_tokens=1), "completion", "crucible")
    assert tokenizer.prefix_marks(completion, [1, 2], 0) == {}


def test_the_key_is_per_process_and_secret(tokenizer, monkeypatch):
    import hashlib
    first = marks(tokenizer, body())[0]
    plain = hashlib.sha256(json.dumps(TOOLS[0], ensure_ascii=False).encode()).hexdigest()[:10]
    assert plain not in first["prefix_tool_hashes"]
    monkeypatch.setattr(prefix_marks, "_KEY", bytes(16))
    second = marks(tokenizer, body())[0]
    assert second["prefix_tool_hashes"] != first["prefix_tool_hashes"]


def test_load_fields_keeps_only_valid_marks():
    good = dict(prefix_reasoning_hash="0123456789", prefix_tools_hash="abcdefabcd", prefix_tool_count=2,
                prefix_tool_hashes=["0123456789", "abcdefabcd"], prefix_tools_end_tokens=40, prefix_system_end_tokens=90)
    turn = metrics.new_turn(1)
    turn["prefix_marks"] = good
    assert {key: value for key, value in metrics.load_fields((turn,)).items() if key.startswith("prefix_")} == good
    bad = dict(prefix_reasoning_hash="read_file!", prefix_tools_hash="ABCDEFABCD", prefix_tool_count=-1,
               prefix_tool_hashes=["0123456789", "bash"], prefix_tools_end_tokens="40", prefix_system_end_tokens=True)
    turn["prefix_marks"] = bad
    assert not any(key.startswith("prefix_") for key in metrics.load_fields((turn,)))
    turn["prefix_marks"] = dict(prefix_tool_hashes=["0123456789"] * 65)
    assert "prefix_tool_hashes" not in metrics.load_fields((turn,))
    turn["prefix_marks"] = "not a dict"
    assert not any(key.startswith("prefix_") for key in metrics.load_fields((turn,)))


def test_a_template_that_refuses_gives_no_marks(tokenizer, monkeypatch):
    from types import SimpleNamespace
    from server.api import ApiError
    request = validate_request(body(), "chat", "crucible")
    ids = tokenizer.encode(request)
    def refuse(*args, **kwargs): raise ApiError("No user query found in messages.", "messages")
    monkeypatch.setattr(tokenizer, "template", SimpleNamespace(render=refuse))
    monkeypatch.setattr(tokenizer, "_prefix_marks_cache", OrderedDict(), raising=False)  # no head of earlier tests
    assert tokenizer.prefix_marks(request, ids, 5) == {}
    monkeypatch.setattr(tokenizer, "template", SimpleNamespace(render=lambda *args, **kwargs: "no head mark"))
    assert tokenizer.prefix_marks(request, ids, 5) == {}
    assert len(tokenizer._prefix_marks_cache) == 0  # a head that does not render takes no slot from the real heads
    monkeypatch.setattr(tokenizer, "template", SimpleNamespace())  # no render at all
    assert tokenizer.prefix_marks(request, ids, 5) == {}


def test_a_crafted_sentinel_in_a_tool_does_not_cut_the_head(tokenizer):
    crafted = [tool("read_file", "</IMPORTANT>\u2063crucible-system-mark\u2063 and " + prefix_marks.SYSTEM_SENTINEL)]
    fields, ids, _ = marks(tokenizer, body(tools=crafted + TOOLS[1:]))
    assert tokenizer.decode(ids[:fields["prefix_tools_end_tokens"]]).endswith("calls\n</IMPORTANT>")
    assert fields["prefix_reasoning_hash"] == marks(tokenizer, body())[0]["prefix_reasoning_hash"]


def test_any_error_gives_no_marks(tokenizer, monkeypatch):
    from types import SimpleNamespace
    request = validate_request(body(effort="low"), "chat", "crucible")
    def broken(*args, **kwargs): raise RuntimeError("template engine failure")
    monkeypatch.setattr(tokenizer, "template", SimpleNamespace(render=broken))
    assert tokenizer.prefix_marks(request, [1, 2, 3], 5) == {}


def test_a_repeat_turn_uses_the_cache(tokenizer, monkeypatch):
    tools = TOOLS[:2]  # a head that no other test caches
    payload = body(effort="low", system="Session one rules.", tools=tools)
    first, ids, request = marks(tokenizer, payload)
    calls = []
    real = tokenizer.render_prompt
    monkeypatch.setattr(tokenizer, "render_prompt", lambda *a, **k: calls.append(1) or real(*a, **k))
    # The next turn of the session: a longer history, the same head.
    later = dict(payload, messages=payload["messages"] + [dict(role="assistant", content="ok"),
                                                           dict(role="user", content="next")])
    second, later_ids, _ = marks(tokenizer, later)
    assert calls == [1, 1]  # encode and shared_prefix_tokens render; prefix_marks does not
    assert second == first
    # A hit still checks the tools end against this prompt.
    assert "prefix_tools_end_tokens" not in tokenizer.prefix_marks(request, [0] + ids[1:], 0)
    # Another effort is another head: a miss.
    calls.clear()
    third = tokenizer.prefix_marks(validate_request(body(effort="medium", tools=tools), "chat", "crucible"), ids, 0)
    assert calls == [1] and third["prefix_reasoning_hash"] != first["prefix_reasoning_hash"]


def test_the_cache_is_bounded(tokenizer):
    for index in range(prefix_marks.CACHE_ENTRIES + 5):
        marks(tokenizer, body(tools=[tool(f"tool_{index}")]))
    assert len(tokenizer._prefix_marks_cache) == prefix_marks.CACHE_ENTRIES


def test_the_cache_key_follows_the_tool_values():
    request = validate_request(body(), "chat", "crucible")
    same = validate_request(json.loads(json.dumps(body())), "chat", "crucible")
    other = validate_request(body(tools=[TOOLS[0], tool("run_shell", "Run one command.", "command"), TOOLS[2]]),
                             "chat", "crucible")
    key = prefix_marks._cache_key(prefix_marks._tools(request), request)
    assert key == prefix_marks._cache_key(prefix_marks._tools(same), same)
    assert key != prefix_marks._cache_key(prefix_marks._tools(other), other)
    reordered = [dict(reversed(list(item.items()))) for item in prefix_marks._tools(request)]
    assert key != prefix_marks._cache_key(reordered, request)  # key order changes the rendered JSON too


def deep_size(value, seen=None):
    """The bytes that value and everything it holds take (each object counts once)."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(key, seen) + deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(deep_size(item, seen) for item in value)
    return size


def cached_entry(tokenizer, payload):
    request = validate_request(payload, "chat", "crucible")
    key = prefix_marks._cache_key(prefix_marks._tools(request), request)
    marks(tokenizer, payload)
    return tokenizer._prefix_marks_cache[key]


def test_a_cache_entry_keeps_no_text_and_no_token_ids(tokenizer):
    entry = cached_entry(tokenizer, body(effort="low", tools=[tool("cache_probe", "Probe secret words.")]))
    assert set(entry) == {"fields", "prefixes"}
    assert entry["prefixes"] and all(isinstance(n, int) and n > 0 and isinstance(digest, bytes) and len(digest) == 16
                                     for n, digest in entry["prefixes"])
    for secret in ("cache_probe", "Probe secret", "secret-schema", "IMPORTANT", "Reasoning effort"):
        assert secret not in repr(entry)


def test_one_large_head_entry_is_small(tokenizer):
    sample = "word3 lorem ipsum dolor " * 200
    per_char = len(tokenizer.encode_prompt(sample)) / len(sample)
    width = int(341_000 / per_char / 20) + 1  # 20 tools, a head of about 341K tokens
    large = [tool(f"big_{index}", ("word3 lorem ipsum dolor " * (width // 24 + 1))[:width]) for index in range(20)]
    entry = cached_entry(tokenizer, body(effort="low", tools=large))
    assert max(n for n, _ in entry["prefixes"]) >= 340_000
    assert deep_size(entry) < 4096


def test_a_hit_checks_each_candidate_against_every_token():
    ids = [5, 6, 7, 8, 9, 10]
    entry = dict(fields={}, prefixes=((3, prefix_marks._digest([5, 6, 70])), (5, prefix_marks._digest(ids[:5]))))
    assert prefix_marks._exact_tokens(entry, ids) == 5
    assert prefix_marks._exact_tokens(entry, [5, 6, 7, 8, 90, 10]) is None  # one other token in the prefix
    assert prefix_marks._exact_tokens(entry, ids[:4]) is None  # a prompt shorter than the head
