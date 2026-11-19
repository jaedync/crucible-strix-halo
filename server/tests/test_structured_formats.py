"""The answer format of /v1/responses (text.format) and /v1/messages (output_config.format) on the chat path.

Each field maps to the chat field response_format (docs/OPENAI_CLIENTS.md "Structured output on the adapters"), so
the request gets the same grammar, the same schema checks and the same limits as the equal chat body. An error keeps
the adapter's own shape, names the adapter's field and never repeats the client's value.
"""
import json
from dataclasses import replace

import pytest

from server import structured
from server.tests.serve_fixtures import PiecesTokenizer
from server.tests.test_structured_adapters import ADD_A, ADD_C, ADD_R, SCHEMA, REFUSED_AT_OPEN, error, post, submitted

USER = [dict(role="user", content="Add 1 and 2")]
ANSWER = ('{"a":1,', '"b":2}')
REFUSED_SCHEMA = dict(allOf=[dict(type="object")])  # server/schema_exact.py REFUSED: a 400 constraint_keyword
SECRET = "client-value-7f3a"  # a value that no error may repeat


def responses(**fields):
    return dict(model="crucible", input="Add 1 and 2", max_output_tokens=9, seed=7) | fields


def messages(**fields):
    return dict(model="crucible", max_tokens=9, messages=USER, seed=7) | fields


def chat_body(**fields):
    return dict(model="crucible", messages=USER, max_tokens=9, seed=7) | fields


def same(request):
    return replace(request, stream=False, include_usage=False)


def chat_constraint(response_format, **fields):
    return submitted(post(chat_body(response_format=response_format, **fields)))


def r_format(**fmt):
    return dict(text=dict(format=fmt))


def a_format(**fmt):
    return dict(output_config=dict(format=fmt))


R_SCHEMA = dict(type="json_schema", name="sum", schema=SCHEMA)
C_SCHEMA = dict(type="json_schema", json_schema=dict(name="sum", schema=SCHEMA))


# /v1/responses text.format

@pytest.mark.parametrize("fmt,chat_format", [
    (R_SCHEMA, C_SCHEMA),
    (R_SCHEMA | dict(strict=True, description="The sum."),
     dict(type="json_schema", json_schema=dict(name="sum", schema=SCHEMA, strict=True, description="The sum."))),
    (dict(type="json_object"), dict(type="json_object")),
], ids=["json_schema", "json_schema strict description", "json_object"])
@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_responses_text_format_is_the_chat_response_format(fmt, chat_format, stream):
    via_r = submitted(post(responses(stream=stream, text=dict(format=fmt)), path="/v1/responses"))
    via_c = chat_constraint(chat_format, stream=stream)
    assert via_r[1].constraint is not None
    assert via_r[0] == via_c[0] and same(via_r[1]) == same(via_c[1])


def test_responses_text_format_text_is_the_request_without_it():
    via_r = submitted(post(responses(**r_format(type="text")), path="/v1/responses"))
    plain = submitted(post(responses(), path="/v1/responses"))
    assert via_r[:2] == plain[:2] and via_r[1].constraint is None


@pytest.mark.parametrize("strict", [None, False, True, "absent"])
def test_responses_strict_takes_the_chat_rule(strict):
    """The schema is enforced at every value of strict, as on the chat path."""
    fmt = R_SCHEMA if strict == "absent" else R_SCHEMA | dict(strict=strict)
    via_r = submitted(post(responses(text=dict(format=fmt)), path="/v1/responses"))
    via_c = chat_constraint(C_SCHEMA)
    assert via_r[1].constraint == via_c[1].constraint and via_r[1].constraint["kind"] == "json_schema"


def test_the_strict_default_is_one_named_constant():
    assert structured.STRICT_DEFAULT is True


@pytest.mark.parametrize("fmt,words", [
    (dict(type=SECRET), "text.format type must be text, json_schema, or json_object."),
    (dict(type=["json_schema"]), "text.format type must be text, json_schema, or json_object."),
    ("json_schema", "text.format type must be text, json_schema, or json_object."),
    (R_SCHEMA | {SECRET: 1}, "text.format json_schema supports name, schema, description, and strict only."),
    (dict(type="json_object", schema=SCHEMA), "text.format json_object takes no other field."),
    (dict(type="text", name=SECRET), "text.format text takes no other field."),
    (R_SCHEMA | dict(name=7), "name must be text"),
    (R_SCHEMA | dict(strict=SECRET), "strict must be true or false"),
    (R_SCHEMA | dict(schema=SECRET), "schema must be a JSON object"),
], ids=["type", "type list", "not an object", "extra key", "json_object key", "text key", "name", "strict", "schema"])
def test_a_bad_text_format_is_a_400_that_names_text_format(fmt, words):
    response = post(responses(text=dict(format=fmt)), path="/v1/responses")
    assert response.statuses == [400], response.body
    assert response.backend.submissions == []
    body = error(response)
    assert body["param"] == "text.format" and words in body["message"]
    assert SECRET not in response.body.decode()


def test_text_verbosity_keeps_its_own_check():
    response = post(responses(text=dict(format=R_SCHEMA, verbosity="high")), path="/v1/responses")
    assert response.statuses == [400] and error(response)["param"] == "text"


@pytest.mark.parametrize("schema", [REFUSED_SCHEMA, dict(enum=list(range(10_000)))], ids=["keyword", "limit"])
def test_a_refused_schema_has_the_chat_message_and_the_text_format_param(schema):
    via_r = post(responses(**r_format(type="json_schema", name="s", schema=schema)), path="/v1/responses")
    via_c = post(chat_body(response_format=dict(type="json_schema", json_schema=dict(name="s", schema=schema))))
    assert via_r.statuses == via_c.statuses == [400], via_r.body
    assert error(via_r)["message"] == error(via_c)["message"]
    assert (error(via_r)["param"], error(via_c)["param"]) == ("text.format", "response_format")
    assert via_r.backend.submissions == []


def test_text_format_without_structured_output_is_a_400_that_names_text_format():
    response = post(responses(**r_format(type="json_object")), path="/v1/responses", ready=False)
    assert response.statuses == [400]
    assert error(response)["param"] == "text.format" and "no structured output" in error(response)["message"]


def test_text_format_with_a_strict_tool_is_the_chat_constraint():
    strict_c = dict(ADD_C, function=ADD_C["function"] | dict(strict=True))
    via_r = submitted(post(responses(text=dict(format=R_SCHEMA), tools=[ADD_R | dict(strict=True)]),
                           path="/v1/responses"))
    via_c = chat_constraint(C_SCHEMA, tools=[strict_c])
    assert via_r[1].constraint is not None and via_r[1].constraint["kind"] == "structural_tag"
    assert via_r[0] == via_c[0] and same(via_r[1]) == same(via_c[1])


def test_response_format_on_responses_stays_refused():
    response = post(responses(response_format=dict(type="json_object")), path="/v1/responses")
    assert response.statuses == [400] and error(response)["param"] == "response_format"


def test_the_responses_reply_is_the_json_in_the_message_item_and_echoes_the_format():
    response = post(responses(text=dict(format=R_SCHEMA)), path="/v1/responses", pieces=ANSWER,
                    tokenizer=PiecesTokenizer(list(ANSWER)))
    assert response.statuses == [200], response.body
    reply = json.loads(response.body)
    [item] = reply["output"]
    assert item["type"] == "message" and json.loads(item["content"][0]["text"]) == dict(a=1, b=2)
    assert reply["text"]["format"] == R_SCHEMA | dict(strict=structured.STRICT_DEFAULT)


def test_the_responses_stream_sends_the_json_as_output_text_deltas():
    response = post(responses(text=dict(format=dict(type="json_object")), stream=True), path="/v1/responses",
                    pieces=ANSWER, tokenizer=PiecesTokenizer(list(ANSWER)))
    assert response.statuses == [200], response.body
    events = [json.loads(line[6:]) for line in response.body.decode().splitlines() if line.startswith("data: {")]
    deltas = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    assert json.loads(deltas) == dict(a=1, b=2)
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["text"]["format"] == dict(type="json_object")


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_a_text_format_refused_at_open_is_a_400(stream):
    response = post(responses(text=dict(format=R_SCHEMA), stream=stream), path="/v1/responses",
                    events=REFUSED_AT_OPEN)
    assert response.statuses == [400], response.body
    assert response.headers.get("Content-Type") == "application/json"
    assert "does not compile" in error(response)["message"]


# /v1/messages output_config.format

A_SCHEMA = dict(type="json_schema", schema=SCHEMA)
C_NAMELESS = dict(type="json_schema", json_schema=dict(schema=SCHEMA))


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("extra,chat_extra", [({}, {}), (dict(effort="low"), dict(reasoning_effort="low"))],
                         ids=["format", "format and effort"])
def test_messages_output_format_is_the_chat_response_format(stream, extra, chat_extra):
    via_a = submitted(post(messages(stream=stream, output_config=dict(format=A_SCHEMA) | extra), path="/v1/messages"))
    via_c = chat_constraint(C_NAMELESS, stream=stream, **chat_extra)
    assert via_a[1].constraint is not None and via_a[1].constraint["kind"] == "json_schema"
    assert via_a[0] == via_c[0] and same(via_a[1]) == same(via_c[1])


def test_a_null_output_format_is_the_request_without_it():
    via_a = submitted(post(messages(output_config=dict(format=None)), path="/v1/messages"))
    plain = submitted(post(messages(), path="/v1/messages"))
    assert via_a[:2] == plain[:2] and via_a[1].constraint is None


def test_messages_output_format_with_tools_is_the_chat_constraint():
    via_a = submitted(post(messages(output_config=dict(format=A_SCHEMA), tools=[ADD_A]), path="/v1/messages"))
    via_c = chat_constraint(C_NAMELESS, tools=[ADD_C])
    assert via_a[1].constraint is not None and via_a[1].constraint["kind"] == "structural_tag"
    assert via_a[0] == via_c[0] and same(via_a[1]) == same(via_c[1])


def anthropic_error(response):
    assert response.statuses == [400], response.body
    assert response.backend.submissions == []
    body = json.loads(response.body)
    assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"
    assert SECRET not in response.body.decode()
    return body["error"]["message"]


@pytest.mark.parametrize("fmt,words", [
    (dict(type="json_object"), "output_config.format type must be json_schema."),
    (dict(type=SECRET, schema=SCHEMA), "output_config.format type must be json_schema."),
    (SECRET, "output_config.format type must be json_schema."),
    (dict(type="json_schema"), "output_config.format json_schema needs a schema object."),
    (dict(type="json_schema", schema=SECRET), "output_config.format json_schema needs a schema object."),
    (A_SCHEMA | dict(name=SECRET), "output_config.format json_schema supports type and schema only."),
], ids=["json_object", "type", "not an object", "no schema", "schema text", "extra key"])
def test_a_bad_output_format_is_a_400_in_the_anthropic_shape(fmt, words):
    assert words in anthropic_error(post(messages(output_config=dict(format=fmt)), path="/v1/messages"))


def test_output_config_names_its_fields():
    message = anthropic_error(post(messages(output_config={SECRET: 1}), path="/v1/messages"))
    assert "output_config supports effort and format only." in message


@pytest.mark.parametrize("schema", [REFUSED_SCHEMA, dict(enum=list(range(10_000)))], ids=["keyword", "limit"])
def test_a_refused_output_schema_has_the_chat_message(schema):
    via_a = post(messages(output_config=dict(format=dict(type="json_schema", schema=schema))), path="/v1/messages")
    via_c = post(chat_body(response_format=dict(type="json_schema", json_schema=dict(schema=schema))))
    assert anthropic_error(via_a) == error(via_c)["message"]


def test_the_messages_reply_is_one_text_block_with_the_json():
    response = post(messages(output_config=dict(format=A_SCHEMA)), path="/v1/messages", pieces=ANSWER,
                    tokenizer=PiecesTokenizer(list(ANSWER)))
    assert response.statuses == [200], response.body
    reply = json.loads(response.body)
    [block] = reply["content"]
    assert block["type"] == "text" and json.loads(block["text"]) == dict(a=1, b=2)
    assert reply["stop_reason"] == "end_turn"


def test_the_messages_stream_sends_the_json_as_text_deltas():
    response = post(messages(output_config=dict(format=A_SCHEMA), stream=True), path="/v1/messages", pieces=ANSWER,
                    tokenizer=PiecesTokenizer(list(ANSWER)))
    assert response.statuses == [200], response.body
    events = [json.loads(line[6:]) for line in response.body.decode().splitlines() if line.startswith("data: {")]
    starts = [event for event in events if event["type"] == "content_block_start"]
    assert [event["content_block"]["type"] for event in starts] == ["text"]
    text = "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")
    assert json.loads(text) == dict(a=1, b=2)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_an_output_format_refused_at_open_is_a_400(stream):
    response = post(messages(output_config=dict(format=A_SCHEMA), stream=stream), path="/v1/messages",
                    events=REFUSED_AT_OPEN)
    assert response.statuses == [400], response.body
    assert response.headers.get("Content-Type") == "application/json"
    body = json.loads(response.body)
    assert body["error"]["type"] == "invalid_request_error" and "does not compile" in body["error"]["message"]


def test_output_format_without_structured_output_is_a_400():
    message = anthropic_error(post(messages(output_config=dict(format=A_SCHEMA)), path="/v1/messages", ready=False))
    assert "no structured output" in message


def test_count_tokens_takes_the_output_format():
    """The format adds no prompt text, so the count equals the count without it."""
    count = post(dict(messages=USER, output_config=dict(format=A_SCHEMA)), path="/v1/messages/count_tokens")
    plain = post(dict(messages=USER), path="/v1/messages/count_tokens")
    assert count.statuses == plain.statuses == [200], count.body
    assert json.loads(count.body) == json.loads(plain.body)


def test_the_client_guide_has_the_mapping_table():
    from pathlib import Path
    guide = (Path(__file__).parents[2] / "docs/OPENAI_CLIENTS.md").read_text()
    section = guide.split("## Structured output on the adapters", 1)[1].split("\n## ", 1)[0]
    for words in ("| Responses `text.format` |", "| Anthropic `output_config.format` |", "`STRICT_DEFAULT`"):
        assert words in section, words


@pytest.mark.parametrize("ready,schema,words", [(True, REFUSED_SCHEMA, "allOf"), (False, SCHEMA, "no structured output")],
                         ids=["refused schema", "no structured output"])
def test_a_count_tokens_format_error_names_output_config_format(ready, schema, words):
    response = post(dict(messages=USER, output_config=dict(format=dict(type="json_schema", schema=schema))),
                    path="/v1/messages/count_tokens", ready=ready)
    assert response.statuses == [400], response.body
    assert error(response)["param"] == "output_config.format" and words in error(response)["message"]
