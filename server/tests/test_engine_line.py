"""the engine's JSON-line limit follows its context (engine/serve/serve_line.h ServeLineBytes).

The y49 cell (Nov 11) found that every prompt above about 400,000 tokens got HTTP 413 from a 524,288-token YaRN server:
the server sends the prompt as token IDs in one line, and the engine and the server both stopped at 2 MiB. Now the
engine reports its limit in the ready event ("line_bytes"), and the server uses that value. An engine that does not
report it keeps the old 2 MiB limit."""
import io
import json
from email.message import Message
from types import SimpleNamespace

import pytest

from server.api import ApiError
from server.backend import ENGINE_LINE_BYTES, ProtocolError
from server.http import Service, handler_for
from server.tests.test_backend_paths import collect, request, start_backend  # noqa: F401 (pytest fixture)
from server.tests.test_yarn_host import service_kwargs

MIB = 1 << 20
# engine/serve/serve_line.h: max(2 MiB, 8 bytes a token + 1 MiB). tests/serve_line_test.cpp checks the same numbers.
LINE_BYTES = {32768: 2 * MIB, 262144: 3 * MIB, 524288: 5 * MIB}
# The widest prompt ID (6 digits): the longest line for a prompt of a given length.
WIDE_ID = 248000


def engine_env(context, streams, line_bytes, **extra):
    return {"FAKE_LINE_LIMIT": str(line_bytes), "FAKE_CONTEXT": str(context),
            "FAKE_SLOT_CONTEXTS": ",".join([str(context)] * streams), **extra}


def options(batch, streams):
    return dict(streams=streams) if batch else {}


@pytest.mark.parametrize("batch", [False, True])
def test_a_full_context_yarn_prompt_above_2_mib_reaches_the_engine(start_backend, batch):
    # 524,272 wide IDs are about 3.7 MB of JSON: above the old 2 MiB line, under the engine's 5 MiB for this context.
    context, streams = 524288, 2
    backend = start_backend(batch, env=engine_env(context, streams, LINE_BYTES[context]), context=context,
                            **options(batch, streams))
    ids = [WIDE_ID] * (context - 16)
    assert len(json.dumps(ids, separators=(",", ":"))) > 2 * MIB
    tokens, done = collect(backend.submit(ids, request(16)))
    assert len(tokens) == 16 and not backend.fatal
    assert backend.engine_line_bytes == LINE_BYTES[context]


@pytest.mark.parametrize("batch", [False, True])
def test_a_line_above_the_reported_limit_gets_413_and_the_engine_lives(start_backend, batch):
    # RA-2: the server refuses the line before the engine sees it, so the batch engine does not stop for every client.
    backend = start_backend(batch, env=engine_env(4096, 2, 4000), context=4096, **options(batch, 2))
    assert backend.engine_line_bytes == 4000
    with pytest.raises(ApiError) as error:
        backend.submit(list(range(1, 1200)), request())
    assert (error.value.status, error.value.code) == (413, "line_too_large") and not backend.fatal
    assert collect(backend.submit([9], request()))[0] == list(range(10, 18))


@pytest.mark.parametrize("batch", [False, True])
def test_an_engine_that_does_not_report_its_limit_keeps_2_mib(start_backend, batch):
    # An older engine reads 2 MiB lines only: a longer line would stop the batch engine (RA-2), so the server keeps 413.
    context = 400000
    env = engine_env(context, 1, ENGINE_LINE_BYTES, FAKE_NO_LINE_BYTES="1")
    backend = start_backend(batch, env=env, context=context, **options(batch, 1))
    assert backend.engine_line_bytes in (None, ENGINE_LINE_BYTES)
    with pytest.raises(ApiError) as error:
        backend.submit([WIDE_ID] * 300_000, request())  # about 2.1 MB of IDs
    assert error.value.status == 413 and not backend.fatal


@pytest.mark.parametrize("value", ["5242880", True, 0, 1023, 1 << 31, 1.5e6])
def test_a_bad_line_limit_in_the_ready_event_is_a_protocol_error(value):
    from server.backend import EngineBackend
    backend = EngineBackend.__new__(EngineBackend)
    backend.context = 4096
    ready = dict(event="ready", context=4096, line_bytes=value)
    with pytest.raises(ProtocolError):
        backend._validate_ready(ready)


class WordTokenizer:
    """One prompt token for each space-separated word, as wide as the widest real ID (the longest engine line)."""
    def encode(self, request):
        return [WIDE_ID] * sum(message["content"].count(" ") for message in request.messages)
    def cache_prefix_tokens(self, request, ids):
        return 0
    def stream(self):
        return None
    def step(self, stream, token):
        return ""
    def decode(self, ids):
        return ""


def post(service, payload):
    handler = handler_for(service)
    h = handler.__new__(handler)
    h.path = "/v1/chat/completions"
    body = json.dumps(payload).encode()
    h.headers = Message()
    for key, value in {"Host": "127.0.0.1:8790", "Content-Length": str(len(body)),
                       "Content-Type": "application/json"}.items():
        h.headers[key] = value
    h.server = SimpleNamespace(server_address=("127.0.0.1", 8790))
    h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
    status = []
    h.send_response = status.append
    h.send_header = lambda *args: None
    h.end_headers = lambda: None
    h.do_POST()
    return status, h.wfile.getvalue(), len(body)


@pytest.mark.parametrize("argv,context,streams", [
    (["--ctx", "524288", "--streams", "2"], 524288, 2),  # the YaRN start option, no vision
    (["--ctx", "262144", "--streams", "4"], 262144, 4),  # the default context, no YaRN, no vision
])
def test_http_gate_a_prompt_at_the_full_configured_context(start_backend, argv, context, streams):
    # Review 1111 item (a): prompt + reply = the whole configured context, through the HTTP handler with the start
    # options of server.__main__ (body limit, RoPE word and reply reserve). A word of 4 bytes is one token.
    kwargs = service_kwargs(argv, {})
    backend = start_backend(True, env=engine_env(context, streams, LINE_BYTES[context]), context=context,
                            streams=streams)
    reply = 16
    service = Service(backend, WordTokenizer(), kwargs.pop("model_id", "crucible"), **kwargs)
    text = "abc " * (context - reply)
    status, body, size = post(service, dict(messages=[dict(role="user", content=text)], max_tokens=reply, spec=0))
    assert size > MIB  # above the old 1 MiB body default
    assert status == [200], body[:300]
    assert json.loads(body)["usage"]["completion_tokens"] == reply
    assert json.loads(body)["usage"]["prompt_tokens"] == context - reply


def test_both_serve_loops_read_with_the_context_limit_and_report_it():
    # engine/serve/serve.h and batched_serve.h run on the GPU host only, so this reads their source.
    from pathlib import Path
    root = Path(__file__).parents[2] / "engine/serve"
    serial, batched = (root / "serve.h").read_text(), (root / "batched_serve.h").read_text()
    for source, name in ((serial, "ctx"), (batched, "context")):
        assert f"crucible::ServeLineReader reader(0, crucible::ServeLineBytes({name}));" in source
        assert '\\"line_bytes\\":%zu' in source
        assert "kMaxLine" not in source and "2 MiB" not in source


@pytest.mark.parametrize("argv,body_mib", [
    ([], 1),  # the default context (32768) keeps 1 MiB
    (["--ctx", "262144", "--streams", "4"], 4),
    (["--ctx", "524288", "--streams", "2"], 8),
    (["--ctx", "524288", "--streams", "2", "--max-body-mib", "2"], 2),  # an explicit value wins
    (["--ctx", "524288", "--streams", "2", "--vision"], 256),  # the vision default is larger
])
def test_the_default_body_limit_holds_a_prompt_that_fills_the_context(argv, body_mib):
    from unittest.mock import patch
    with patch("server.vision.tower_fault", return_value=None):
        assert service_kwargs(argv, {})["max_body_mib"] == body_mib
