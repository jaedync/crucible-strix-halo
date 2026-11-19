"""Structured output beside api-compat (lenient fields, /v1/responses, /v1/messages, timings).

The merge of the two branches keeps the rules of both sides:
- response_format is a known request field. --lenient-fields then does not touch it: the field has the same checks
  in both modes, and lenient.REFUSED names only fields that validate_request does not know.
- An adapter body that the chat path constrains (a strict function tool, a required or named tool choice) gets the
  same engine constraint, the same prompt IDs and the same rendered prompt as the equal chat body.
- An adapter field that the server does not map (response_format on /v1/responses) stays a 400: no request gets a
  reply with a constraint that the server did not apply. Responses text.format and Anthropic output_config.format map
  to response_format (server/tests/test_structured_formats.py); an Anthropic tool's strict maps to the chat function's
  strict (server/tests/test_structured_anthropic_strict.py).
- A constrained stream waits for the first engine event, so a constraint that the engine refuses at open is an HTTP
  400 in the adapter's own error shape, not an SSE stream.
"""
import json
from dataclasses import replace

import pytest

from server import api, lenient
from server.tests.sdk_fixture import tool_block
from server.tests.serve_fixtures import PiecesTokenizer, chat, invoke, token_events
from server.tests.test_count_tokens import tokenizer as fixture_tokenizer
from server.tests.test_messages_api import PromptBackend

SCHEMA = dict(type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"],
              additionalProperties=False)
ADD_R = dict(type="function", name="add", description="Add.", parameters=SCHEMA)
ADD_C = dict(type="function", function=dict(name="add", description="Add.", parameters=SCHEMA))
ADD_A = dict(name="add", description="Add.", input_schema=SCHEMA)
REFUSED_AT_OPEN = [dict(event="error", error="The schema does not compile to a grammar.", status=400)]


class StructuredBackend(PromptBackend):
    """A backend whose engine named the constraint key in its ready event (server/backend.py constraint_ready)."""
    constraint_ready = True


def post(payload, path="/v1/chat/completions", ready=True, options=None, events=None, tokenizer=None, pieces=("a",)):
    backend = (StructuredBackend if ready else PromptBackend)(events if events is not None else
                                                             token_events(list(pieces)))
    response = invoke(backend, tokenizer or PiecesTokenizer(list(pieces)), payload, path=path, service_options=options)
    response.backend = backend
    return response


def submitted(response):
    assert response.statuses == [200], response.body
    return response.backend.submissions[0]


def error(response):
    return json.loads(response.body)["error"]


# Lenient fields: response_format is known, so the mode does not change it.
STRICT = dict(lenient_fields=False)  # --lenient-fields off; the default (None) is on

def test_refused_names_only_fields_that_validate_request_does_not_know():
    assert not set(lenient.REFUSED) & api.REQUEST_FIELDS
    assert "response_format" in api.REQUEST_FIELDS


@pytest.mark.parametrize("mode", [STRICT, None, dict(lenient_fields=True)], ids=["strict", "default", "lenient"])
def test_text_response_format_is_the_request_without_it(mode):
    for ready in (False, True):
        ids, request, _ = submitted(post(chat(response_format=dict(type="text")), ready=ready, options=mode))
        plain_ids, plain, _ = submitted(post(chat(), ready=ready, options=mode))
        assert (ids, request) == (plain_ids, plain) and request.constraint is None


@pytest.mark.parametrize("fmt", [dict(type="json_object"),
                                 dict(type="json_schema", json_schema=dict(name="sum", schema=SCHEMA))],
                         ids=["json_object", "json_schema"])
def test_lenient_mode_gives_the_same_constraint(fmt):
    strict = submitted(post(chat(response_format=fmt), options=STRICT))
    loose = submitted(post(chat(response_format=fmt, user="u1"), options=dict(lenient_fields=True)))
    assert strict[1].constraint is not None
    assert strict[:2] == loose[:2]


@pytest.mark.parametrize("mode", [STRICT, None, dict(lenient_fields=True)], ids=["strict", "default", "lenient"])
def test_an_engine_without_structured_output_refuses_a_format_in_both_modes(mode):
    response = post(chat(response_format=dict(type="json_object")), ready=False, options=mode)
    assert response.statuses == [400]
    # The code constraint_unsupported goes to the server log only (api.ApiError); the client reads param and message.
    assert error(response)["param"] == "response_format" and "no structured output" in error(response)["message"]
    assert response.backend.submissions == []


# /v1/responses and /v1/messages: the same constraint as the equal chat body.

def responses_cases():
    user = [dict(role="user", content="Add 1 and 2")]
    strict_r, strict_c = ADD_R | dict(strict=True), dict(ADD_C, function=ADD_C["function"] | dict(strict=True))
    named_r, named_c = dict(type="function", name="add"), dict(type="function", function=dict(name="add"))
    return [
        ("strict tool", dict(tools=[strict_r]), dict(tools=[strict_c])),
        ("strict tool, no parallel", dict(tools=[strict_r], parallel_tool_calls=False),
         dict(tools=[strict_c], parallel_tool_calls=False)),
        # A Responses tool with no strict is strict (as OpenAI does, server/responses_strict.py): the chat body says it.
        ("no strict", dict(tools=[ADD_R]), dict(tools=[strict_c])),
        ("required choice", dict(tools=[ADD_R], tool_choice="required"), dict(tools=[strict_c], tool_choice="required")),
        ("named choice", dict(tools=[ADD_R], tool_choice=named_r), dict(tools=[strict_c], tool_choice=named_c)),
        ("strict false", dict(tools=[ADD_R | dict(strict=False)]),
         dict(tools=[dict(ADD_C, function=ADD_C["function"] | dict(strict=False))])),
    ], user


@pytest.mark.parametrize("kind", ["fixture", "checkpoint"])
@pytest.mark.parametrize("case", range(len(responses_cases()[0])), ids=[case[0] for case in responses_cases()[0]])
def test_responses_constraint_equals_the_chat_path(tmp_path, kind, case):
    cases, user = responses_cases()
    name, responses, chat_body = cases[case]
    template = fixture_tokenizer(kind, tmp_path)
    via_r = submitted(post(dict(model="crucible", input="Add 1 and 2", max_output_tokens=9, seed=7) | responses,
                           path="/v1/responses", tokenizer=template))
    via_c = submitted(post(dict(model="crucible", messages=user, max_tokens=9, seed=7) | chat_body,
                           tokenizer=template))
    assert via_r[0] == via_c[0]
    assert template.render_chat(via_r[1]).encode() == template.render_chat(via_c[1]).encode()
    assert replace(via_r[1], stream=False, include_usage=False) == replace(via_c[1], stream=False, include_usage=False)
    if name == "strict false":
        assert via_r[1].constraint is None
    else:
        assert via_r[1].constraint is not None and via_r[1].constraint["kind"] == "structural_tag"
    if name.startswith("strict tool"):
        # The template renders the tool without the API flag `strict`, on both paths.
        assert "strict" not in via_r[1].render_tools[0]["function"]


@pytest.mark.parametrize("kind", ["fixture", "checkpoint"])
@pytest.mark.parametrize("choice,chat_choice", [(dict(type="any"), "required"),
                                                (dict(type="tool", name="add"),
                                                 dict(type="function", function=dict(name="add")))],
                         ids=["any", "tool"])
def test_messages_constraint_equals_the_chat_path(tmp_path, kind, choice, chat_choice):
    template = fixture_tokenizer(kind, tmp_path)
    user = [dict(role="user", content="Add 1 and 2")]
    via_a = submitted(post(dict(model="crucible", max_tokens=9, messages=user, tools=[ADD_A], tool_choice=choice, seed=7),
                           path="/v1/messages", tokenizer=template))
    via_c = submitted(post(dict(model="crucible", max_tokens=9, messages=user, tools=[ADD_C], tool_choice=chat_choice, seed=7),
                           tokenizer=template))
    assert via_a[0] == via_c[0]
    assert via_a[1].constraint is not None and via_a[1].constraint == via_c[1].constraint
    assert template.render_chat(via_a[1]).encode() == template.render_chat(via_c[1]).encode()


@pytest.mark.parametrize("path,payload,param", [
    ("/v1/responses", dict(model="crucible", input="x", max_output_tokens=4, response_format=dict(type="json_object")),
     "response_format"),
], ids=["responses response_format"])
def test_an_adapter_format_that_the_server_does_not_map_stays_refused(path, payload, param):
    response = post(payload, path=path)
    assert response.statuses == [400], response.body
    assert response.backend.submissions == []
    body = json.loads(response.body)
    assert (body["error"].get("param") == param) if path == "/v1/responses" else body["type"] == "error"


# A constrained stream: a refusal at open is a plain 400 in the API's own shape.

def test_a_refusal_at_open_is_a_400_on_every_path():
    strict_c = dict(ADD_C, function=ADD_C["function"] | dict(strict=True))
    user = [dict(role="user", content="Add")]
    cases = [("/v1/chat/completions", dict(model="crucible", messages=user, max_tokens=8, tools=[strict_c])),
             ("/v1/responses", dict(model="crucible", input="Add", max_output_tokens=8, tools=[ADD_R | dict(strict=True)])),
             ("/v1/messages", dict(model="crucible", max_tokens=8, messages=user, tools=[ADD_A],
                                   tool_choice=dict(type="any")))]
    for path, payload in cases:
        for stream in (False, True):
            response = post(payload | dict(stream=stream), path=path, events=REFUSED_AT_OPEN)
            assert response.statuses == [400], (path, stream, response.body)
            assert response.headers.get("Content-Type") == "application/json", (path, stream)
            body = json.loads(response.body)
            if path == "/v1/messages":
                assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"
            else:
                assert "does not compile" in body["error"]["message"]


def test_an_unconstrained_stream_still_gets_its_headers_at_admission():
    response = post(dict(model="crucible", input="hi", max_output_tokens=8, stream=True), path="/v1/responses",
                    events=[dict(event="error", error="boom", status=400)])
    # No constraint: the headers go out first, so a later failure is an SSE error event, as before the merge.
    assert response.statuses == [200]
    assert "event: error" in response.body.decode()


# timings (api-compat) on a constrained reply.

def test_a_constrained_reply_has_timings():
    from server.tests.test_timings import TurnBackend

    class Backend(TurnBackend):
        constraint_ready = True
    pieces = [tool_block()]
    strict_c = dict(ADD_C, function=ADD_C["function"] | dict(strict=True))
    for stream in (False, True):
        backend = Backend(token_events(pieces, cached_tokens=0))
        response = invoke(backend, PiecesTokenizer(pieces), chat(tools=[strict_c], stream=stream))
        assert response.statuses == [200], response.body
        if stream:
            parts = [json.loads(part[6:]) for part in response.body.decode().split("\n\n") if part.startswith("data: {")]
            assert "timings" in parts[-1] and parts[-1]["timings"]["predicted_ms"] == pytest.approx(2000.0)
        else:
            reply = json.loads(response.body)
            assert reply["choices"][0]["finish_reason"] == "tool_calls"
            assert reply["timings"]["predicted_ms"] == pytest.approx(2000.0)
