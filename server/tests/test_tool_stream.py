"""Streamed tool calls (item 4): joined deltas equal the whole-call arguments byte for byte."""
import json
import pytest
from server.api import validate_request
from server.tests.serve_fixtures import ScriptBackend, PiecesTokenizer, chat, chunks, invoke, joined_calls, token_events
from server.tests.test_tools import TOOLS
from server.tool_output import ToolOutput

MORE_TOOLS = TOOLS + [
    dict(type="function", function=dict(name="ping", parameters=dict(type="object", properties={}))),
    dict(type="function", function=dict(name="write", parameters=dict(
        type="object", required=["path", "content"],
        properties=dict(path=dict(type="string"), content=dict(type="string"),
                        mode=dict(type="string", enum=["a", "w"]), note=dict(type=["string", "null"]),
                        short=dict(type="string", maxLength=3), ref=dict({"$ref": "#/$defs/text"}),
                        lines=dict(type="array", items=dict(type="integer"))),
        **{"$defs": dict(text=dict(type="string"))}))),
    dict(type="function", function=dict(name="free", parameters=dict(type="object"))),
]


def call(name, **parameters):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


VALID = [
    call("read", path="a.py", offset="4", flags='{"full":true}'),
    call("read", path="line one\n雪"),
    call("read", path='quote " back \\ tab \t bell \x07 end'),
    call("read", path=""),
    call("read", path="\n"),
    call("read", path="x</paramete y </tool z"),
    call("read", path="ends with newline\n"),
    call("ping"),
    call("write", path="f.txt", content="def f():\n    return '</p>'\n\n", mode="w", note="null", short="abc",
         ref="r", lines="[1, 2,3]"),
    call("write", content="first", path="second"),
    call("free", anything="1.5", other="text", obj='{"a": [1, {"b": null}]}'),
    call("read", path="a.py") + "\n" + call("ping") + "\n\n" + call("write", path="p", content="c"),
    "<tool_call><function=read><parameter=path>tight</parameter></function></tool_call>",
]


def request(stream=True, **options):
    body = dict(messages=[dict(role="user", content="Go")], tools=MORE_TOOLS, stream=stream) | options
    return validate_request(body, "chat", "crucible")


def run(text, pieces, **options):
    output = ToolOutput(request(**options), stream=True)
    content, deltas = "", []
    for piece in pieces:
        content += output.feed(piece)
        deltas += [call for delta in output.take_deltas() for call in delta["tool_calls"]]
    message, tail, reason = output.finish("stop")
    deltas += [call for delta in tail for call in delta["tool_calls"]]
    return output, content, message, deltas, reason


def splits(text):
    yield list(text)
    yield [text[index:index + 3] for index in range(0, len(text), 3)]
    for cut in range(len(text) + 1):
        yield [text[:cut], text[cut:]]


@pytest.mark.parametrize("raw", VALID)
def test_joined_deltas_equal_whole_call_arguments(raw):
    text = "Before.\n" + raw
    plain = ToolOutput(request(stream=False))
    plain.feed(text)
    expected, _, expected_reason = plain.finish("stop")
    assert expected_reason == "tool_calls", expected
    for pieces in splits(text):
        output, content, message, deltas, reason = run(text, pieces)
        assert (content, reason, output.fallback) == ("Before.\n", "tool_calls", False)
        joined = joined_calls(deltas)
        assert [c["function"]["name"] for c in joined] == [c["function"]["name"] for c in expected["tool_calls"]]
        assert [c["function"]["arguments"] for c in joined] == [c["function"]["arguments"]
                                                              for c in expected["tool_calls"]]
        assert [c["id"] for c in joined] == [c["id"] for c in message["tool_calls"]]
        assert [c["function"]["arguments"] for c in message["tool_calls"]] == [
            c["function"]["arguments"] for c in expected["tool_calls"]]


def test_header_goes_out_when_the_function_tag_closes():
    output = ToolOutput(request(), stream=True)
    output.feed("Sure.\n<tool_call>\n<function=read")
    assert output.take_deltas() == []
    output.feed(">")
    header = output.take_deltas()
    assert header == [dict(tool_calls=[dict(index=0, id=header[0]["tool_calls"][0]["id"], type="function",
                                            function=dict(name="read", arguments=""))])]


def test_string_value_streams_while_written_and_holds_tag_prefixes():
    output = ToolOutput(request(), stream=True)
    output.feed("<tool_call>\n<function=write>\n<parameter=content>\nline 1\nline 2</par")
    sent = "".join(d["tool_calls"][0]["function"]["arguments"] for d in output.take_deltas()[1:])
    # The trailing "</par" can still become the closing tag; the newline before it is held too.
    assert sent == '{"content":"line 1\\nline 2'
    output.feed("ade</parameter>")
    assert "".join(d["tool_calls"][0]["function"]["arguments"] for d in output.take_deltas()) == '</parade"'
    output.feed("\n<parameter=mode>\nw")
    assert output.take_deltas() == []  # an enum string is checked whole


@pytest.mark.parametrize("raw,code", [
    # Schema-only failures of a complete call now go out as calls (design decision, Oct 8): server/tests/test_tool_pass.py.
    (call("read", path="a.py") + " trailing", "unexpected_text"),
    ("<tool_call>\n<function=read>\n<parameter=path>\ncut", "incomplete_call"),
    (call("read", path="a</tool_call>b"), "incomplete_call"),
])
def test_invalid_after_deltas_ends_with_length(raw, code):
    text = "Before.\n" + raw
    for pieces in ([text], list(text)):
        output, content, message, deltas, reason = run(text, pieces)
        assert deltas and deltas[0]["function"]["name"] in ("read", "write")
        assert (message, reason) == (dict(role="assistant", content=text), "length")
        assert (output.fallback, output.invalid) == (True, code)


@pytest.mark.parametrize("raw,options,code", [
    (call("bad name", path="a"), {}, "unknown_function"),  # a valid unknown name goes out (test_tool_pass)
    (call("read", path="a"), dict(tool_choice="none"), "tool_choice"),
    (call("ping"), dict(tool_choice=dict(type="function", function=dict(name="read"))), "tool_choice"),
    ("<tool_call>{\"name\":\"read\"}</tool_call>", {}, "function_tag"),
])
def test_invalid_before_the_header_keeps_the_content_fallback(raw, options, code):
    output, content, message, deltas, reason = run(raw, list(raw), **options)
    assert deltas == [] and (message, reason) == (dict(role="assistant", content=raw), "stop")
    assert output.invalid == code


def test_stream_off_sends_calls_whole():
    text = call("read", path="a.py")
    backend = ScriptBackend(token_events(list(text)))
    response = invoke(backend, PiecesTokenizer(list(text)), chat(tools=TOOLS, stream=True, max_tokens=len(text) + 1),
                      service_options=dict(stream_tool_calls=False))
    deltas = [c for part in chunks(response) for choice in part["choices"] for c in choice["delta"].get("tool_calls", [])]
    assert len(deltas) == 1 and json.loads(deltas[0]["function"]["arguments"]) == dict(path="a.py")


DUPLICATE = "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n<parameter=path>\nb.py\n</parameter>\n</function>\n</tool_call>"
# Every reason that can end a stream after deltas went out (test_invalid_after_deltas_ends_with_length) and more.
AFTER_DELTAS = [
    DUPLICATE,
    call("read", path="a.py", offset="x"),
    call("read", offset="4"),
    call("read", path="a", path2="b"),
    call("write", path="a", content="b", mode="x"),
    call("read", path="a.py") + " trailing",
    "<tool_call>\n<function=read>\n<parameter=path>\ncut",
    "<tool_call>\n<function=read>\n",
    call("read", path="a</tool_call>b"),
    call("read", path="a.py") + "\n" + call("write", path="p", content="c", lines="[1,"),
]


def next_request(content, calls):
    """The client's next turn after a streamed reply, as an OpenAI client builds it from the joined deltas."""
    messages = [dict(role="user", content="Go"), dict(role="assistant", content=content or None, tool_calls=calls)]
    messages += [dict(role="tool", tool_call_id=c["id"], content="error: the call was cut") for c in calls]
    return validate_request(dict(messages=messages + [dict(role="user", content="Again")], tools=MORE_TOOLS),
                            "chat", "crucible")


def render(request):
    import tempfile
    from pathlib import Path
    from server.template import ChatTemplate
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(__file__).parent / "fixtures/qwen_chat_template.jinja"
        Path(directory, "tokenizer_config.json").write_text(json.dumps(dict(chat_template=fixture.read_text())))
        return ChatTemplate(directory).render(request.messages, tools=request.tools)


def test_a_streamed_duplicate_parameter_call_does_not_break_the_next_request():
    # An earlier soak run: the server streamed a call, judged it invalid (duplicate_parameter), and then refused the
    # client's next request, which carried that call back in its history (HTTP 400). The session stopped there.
    text = "Reading.\n" + DUPLICATE
    backend = ScriptBackend(token_events(list(text)))
    response = invoke(backend, PiecesTokenizer(list(text)), chat(tools=TOOLS, stream=True, max_tokens=len(text) + 1))
    parts = chunks(response)
    deltas = [c for part in parts for choice in part["choices"] for c in choice["delta"].get("tool_calls", [])]
    content = "".join(choice["delta"].get("content") or "" for part in parts for choice in part["choices"])
    assert deltas and parts[-1]["choices"][0]["finish_reason"] == "length"
    calls = joined_calls(deltas)
    assert calls[0]["function"]["arguments"] == '{"path":"a.py"'
    request = next_request(content, calls)
    assert request.messages[1]["tool_calls"][0]["function"]["arguments"] == dict(path="a.py")
    assert "<parameter=path>\na.py\n</parameter>" in render(request)


@pytest.mark.parametrize("raw", AFTER_DELTAS)
def test_no_streamed_prefix_can_make_the_next_request_fail(raw):
    text = "Before.\n" + raw
    for pieces in splits(text):
        output, content, message, deltas, reason = run(text, pieces)
        calls = joined_calls(deltas)
        if not calls:
            continue
        request = next_request(message.get("content"), calls)
        assert render(request)
        for sent, kept in zip(calls, request.messages[1]["tool_calls"]):
            assert isinstance(kept["function"]["arguments"], dict) and kept["function"]["name"] == sent["function"]["name"]


@pytest.mark.parametrize("arguments", ['{"path":"a\\', '{"path":"a" "b"}', '{"path":"a","path":"b"}',
                                       '[1', '{"path":', '{"path"', '{"path":"a"}}', ' ', '{"path":"a",'])
def test_history_still_refuses_arguments_that_no_stream_produces(arguments):
    from server.api import ApiError
    with pytest.raises(ApiError):
        next_request(None, [dict(id="c", type="function", function=dict(name="read", arguments=arguments))])


@pytest.mark.parametrize("linked", [False, True])
def test_a_call_streamed_by_the_think_cap_continuation_does_not_break_the_next_request(linked):
    # S5 request 16: answer_room closed the think block, the second (cont) job wrote the call with a duplicate
    # parameter, and the call had already streamed. linked=True gives the backend continue_job; without it the server cancels the first job and submits.
    from server.tests.test_think_budget import IM_END, REASON, BudgetTokenizer, SequenceBackend, run
    from server.tests.test_think_budget import chat as think_chat
    text = "Reading.\n" + DUPLICATE
    ids = list(range(1000, 1000 + len(text)))

    class CallTokenizer(BudgetTokenizer):
        pieces = BudgetTokenizer.pieces | dict(zip(ids, text))

    class Linked(SequenceBackend):
        def continue_job(self, job, ids, request, **options):
            follow = self.submit(ids, request, **options)
            job.cancelled.set()
            return follow

    backend = (Linked if linked else SequenceBackend)(([REASON] * 1500, "length"), (ids + [IM_END], "stop"))
    import server.tests.test_think_budget as think
    original = think.BudgetTokenizer
    think.BudgetTokenizer = CallTokenizer
    try:
        response, parts = run(backend, think_chat(max_tokens=2048, stream=True, tools=TOOLS))
    finally:
        think.BudgetTokenizer = original
    assert response.statuses == [200] and len(backend.submissions) == 2
    deltas = [c for part in parts for choice in part["choices"] for c in choice["delta"].get("tool_calls", [])]
    content = "".join(choice["delta"].get("content") or "" for part in parts for choice in part["choices"])
    assert deltas and parts[-1]["choices"][0]["finish_reason"] == "length"
    calls = joined_calls(deltas)
    assert calls[0]["function"]["arguments"] == '{"path":"a.py"'
    request = next_request(content, calls)
    assert request.messages[1]["tool_calls"][0]["function"]["arguments"] == dict(path="a.py")
    assert "<parameter=path>\na.py\n</parameter>" in render(request)
