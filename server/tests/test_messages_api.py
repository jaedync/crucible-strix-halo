"""POST /v1/messages (Anthropic Messages API) over the chat path: same prompt, same tokens, Anthropic shapes."""
import json
from dataclasses import replace

import pytest

from server import messages_api
from server.tests.sdk_fixture import tool_block
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, invoke, token_events
from server.tests.test_api_key import KEY, bearer, gate
from server.tests.test_count_tokens import TemplateTokenizer, corpus, tokenizer as fixture_tokenizer

PATH = "/v1/messages"
ADD_A = dict(name="add", input_schema=dict(type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")),
                                           required=["a", "b"]))


def body(**options):
    return dict(model="crucible", max_tokens=16, messages=[dict(role="user", content="hello")]) | options


class PromptBackend(ScriptBackend):
    """ScriptBackend whose done event counts the prompt that it got."""
    def submit(self, ids, request, **options):
        self.source = [dict(event, prompt_tokens=len(ids)) if event.get("event") == "done" else event
                       for event in self.source]
        return super().submit(ids, request, **options)


def post(pieces, payload, *, tokenizer=None, path=PATH, options=None, events=None, headers=None):
    backend = PromptBackend(events if events is not None else token_events(pieces))
    raw = json.dumps(payload).encode()
    if headers is not None:
        headers = [("Host", "127.0.0.1:8790"), ("Content-Length", str(len(raw))), ("Content-Type", "application/json"),
                   *headers]
    response = invoke(backend, tokenizer or PiecesTokenizer(pieces), raw=raw, path=path, service_options=options,
                      headers=headers)
    response.backend = backend
    return response


def events(response):
    """The (event name, data) pairs of an Anthropic stream."""
    result = []
    for part in response.body.decode().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in part.splitlines() if not line.startswith(":") and ": " in line)
        if "data" in lines:
            result.append((lines.get("event"), json.loads(lines["data"])))
    return result


def test_text_reply():
    response = post(["Hello", " there"], body())
    assert response.statuses == [200]
    reply = json.loads(response.body)
    assert reply["type"] == "message" and reply["role"] == "assistant" and reply["id"].startswith("msg_")
    assert reply["content"] == [dict(type="text", text="Hello there")]
    assert reply["stop_reason"] == "end_turn" and reply["stop_sequence"] is None
    assert reply["usage"]["input_tokens"] == 1 and reply["usage"]["output_tokens"] == 3


def test_fields_reach_the_request():
    response = post(["a"], body(temperature=0.5, top_p=0.9, top_k=20, stop_sequences=["xyz"], max_tokens=7))
    request = response.backend.submissions[0][1]
    assert (request.temperature, request.top_p, request.top_k, request.stop, request.max_tokens) == \
        (0.5, 0.9, 20, ("xyz",), 7)
    assert request.kind == "chat" and request.messages[0]["content"] == "hello"


def test_max_tokens_is_required():
    payload = body()
    del payload["max_tokens"]
    response = post(["a"], payload)
    assert response.statuses == [400]
    error = json.loads(response.body)
    assert error["type"] == "error" and error["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in error["error"]["message"]


def test_length_and_stop_sequence_reasons():
    reply = json.loads(post(["a", "b", "c"], body(max_tokens=2), events=token_events(["a", "b"], stop_token=False)).body)
    assert reply["stop_reason"] == "max_tokens"
    reply = json.loads(post(["one ", "STOP", " two"], body(stop_sequences=["STOP"])).body)
    assert reply["stop_reason"] == "stop_sequence" and reply["stop_sequence"] == "STOP"
    assert reply["content"] == [dict(type="text", text="one ")]


def test_thinking_block():
    pieces = ["I add.", "</think>", "\n\nIt is 4."]
    reply = json.loads(post(pieces, body(thinking=dict(type="enabled", budget_tokens=1024))).body)
    assert reply["content"][0] == dict(type="thinking", thinking="I add.", signature="")
    assert reply["content"][1] == dict(type="text", text="It is 4.")


def test_thinking_budget_maps_to_the_thinking_cap_field():
    response = post(["a"], body(thinking=dict(type="enabled", budget_tokens=2048)))
    request = response.backend.submissions[0][1]
    assert request.enable_thinking is True and request.thinking_budget == 2048
    response = post(["a"], body(thinking=dict(type="disabled")))
    assert response.backend.submissions[0][1].enable_thinking is False


def test_tool_use_reply():
    reply = json.loads(post([tool_block()], body(tools=[ADD_A])).body)
    assert reply["stop_reason"] == "tool_use"
    [block] = reply["content"]
    assert block["type"] == "tool_use" and block["name"] == "add" and block["input"] == dict(a=17, b=25)
    assert block["id"].startswith("call_")


def test_stream_text_event_sequence():
    response = post(["Hel", "lo"], body(stream=True))
    names = [name for name, _ in events(response)]
    assert names == ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"]
    data = [item for _, item in events(response)]
    assert all(item["type"] == name for name, item in events(response))
    assert data[0]["message"]["usage"]["input_tokens"] == 1 and data[0]["message"]["content"] == []
    assert data[2] == dict(type="content_block_start", index=0, content_block=dict(type="text", text=""))
    assert "".join(item["delta"]["text"] for item in data[3:5]) == "Hello"
    assert data[6]["delta"] == dict(stop_reason="end_turn", stop_sequence=None)
    assert data[6]["usage"]["output_tokens"] == 3
    assert "[DONE]" not in response.body.decode()


def test_stream_thinking_then_text_then_tool():
    pieces = ["Plan.", "</think>", "\n\nCalling.", tool_block()]
    response = post(pieces, body(stream=True, tools=[ADD_A], thinking=dict(type="enabled", budget_tokens=1024)))
    starts = [item["content_block"] for name, item in events(response) if name == "content_block_start"]
    assert [block["type"] for block in starts] == ["thinking", "text", "tool_use"]
    assert starts[2]["name"] == "add" and starts[2]["input"] == {}
    deltas = [item for name, item in events(response) if name == "content_block_delta"]
    thinking = "".join(d["delta"]["thinking"] for d in deltas if d["delta"]["type"] == "thinking_delta")
    text = "".join(d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta")
    arguments = "".join(d["delta"]["partial_json"] for d in deltas if d["delta"]["type"] == "input_json_delta")
    assert thinking == "Plan." and text.strip() == "Calling." and json.loads(arguments) == dict(a=17, b=25)
    stops = [item["index"] for name, item in events(response) if name == "content_block_stop"]
    assert stops == [0, 1, 2]
    assert events(response)[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_stream_error_after_start_is_an_anthropic_error_event():
    script = [dict(event="token", id=20), dict(event="error", error="The engine request failed.")]
    response = post(["a"], body(stream=True), events=script)
    name, data = events(response)[-1]
    assert name == "error" and data["type"] == "error" and data["error"]["type"] == "overloaded_error"


@pytest.mark.parametrize("block,word", [
    (dict(type="document", source=dict(type="text", media_type="text/plain", data="x")), "document"),
    (dict(type="search_result", source="s", title="t", content=[]), "search_result"),
    (dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA")), "image")])
def test_unsupported_blocks_are_refused_by_place(block, word):
    response = post(["a"], body(messages=[dict(role="user", content=[block])]))
    assert response.statuses == [400]
    error = json.loads(response.body)
    assert error["type"] == "error" and error["error"]["type"] == "invalid_request_error"
    if word == "image":  # an image goes to the chat path, which refuses it without --vision
        assert word in error["error"]["message"]
    else:  # the place and a fixed list of the supported blocks, never the client's value
        assert "messages[0].content[0].type" in error["error"]["message"] and word not in error["error"]["message"]
    assert response.backend.submissions == []


def anthropic_error(response):
    """The Anthropic error object of a 400 (a malformed value is a 400 with this shape, never a 500)."""
    assert response.statuses == [400], response.body
    result = json.loads(response.body)
    assert set(result) == {"type", "error"} and result["type"] == "error"
    assert set(result["error"]) == {"type", "message"} and result["error"]["type"] == "invalid_request_error"
    assert response.backend.submissions == []
    return result["error"]


@pytest.mark.parametrize("value", [[], {}, 1])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_a_malformed_block_type_is_a_400(role, value):
    """`type: []` gave a TypeError and HTTP 500."""
    history = [dict(role="user", content="Q")] if role == "assistant" else []
    payload = body(messages=history + [dict(role=role, content=[dict(type=value)])])
    message = anthropic_error(post(["a"], payload))["message"]
    assert f"messages[{len(history)}].content[0].type" in message


@pytest.mark.parametrize("value", ["canary_request_content", "CANARY-request-content"])
def test_a_block_error_never_echoes_the_value(value):
    response = post(["a"], body(messages=[dict(role="user", content=[dict(type=value)])]))
    anthropic_error(response)
    assert "canary" not in response.body.decode().lower()


@pytest.mark.parametrize("field", ["container", "mcp_servers", "context_management", "frequency"])
def test_unknown_fields_are_refused_by_name(field):
    response = post(["a"], body(**{field: {}}))
    assert response.statuses == [400]
    assert field in json.loads(response.body)["error"]["message"]


def test_diagnostics_is_refused():
    """Diagnostics asks for prompt-cache data that the server does not have."""
    response = post(["a"], body(diagnostics=dict(previous_message_id="msg_1")))
    assert response.statuses == [400] and response.backend.submissions == []
    assert "diagnostics" in json.loads(response.body)["error"]["message"]


def test_ignored_hint_fields():
    response = post(["a"], body(metadata=dict(user_id="u"), service_tier="auto", cache_control=dict(type="ephemeral"),
                                inference_geo="us", speed="standard", user_profile_id="p", workspace_id="w"))
    assert response.statuses == [200]


def test_unknown_model_is_not_found():
    response = post(["a"], body(model="claude-other"))
    assert response.statuses == [404]
    assert json.loads(response.body)["error"]["type"] == "not_found_error"


def test_beta_query_is_accepted():
    assert post(["a"], body(), path=PATH + "?beta=true").statuses == [200]


def test_missing_key_is_refused_when_a_key_is_set():
    response = post(["a"], body(), options=dict(api_key=gate()), headers=[])
    assert response.statuses == [401] and response.backend.submissions == []
    assert json.loads(response.body)["error"]["type"] == "authentication_error"
    response = post(["a"], body(), options=dict(api_key=gate()), headers=[("x-api-key", KEY)])
    assert response.statuses == [200]


def test_body_limit_applies():
    big = body(messages=[dict(role="user", content="x" * (2 << 20))])
    response = post(["a"], big, options=dict(max_body_mib=1))
    assert response.statuses == [413]
    assert json.loads(response.body)["error"]["type"] == "request_too_large"


def strip(request):
    return replace(request, stream=False, include_usage=False)


@pytest.mark.parametrize("kind", ["fixture", "checkpoint"])
@pytest.mark.parametrize("name,anthropic,openai", corpus(), ids=[case[0] for case in corpus()])
def test_the_prompt_and_request_equal_the_chat_path(tmp_path, kind, name, anthropic, openai):
    """Exactness: the same conversation, sampling and seed give the same prompt IDs and the same engine request.
    CRUCIBLE_TEST_CKPT adds the real checkpoint tokenizer and template."""
    template = fixture_tokenizer(kind, tmp_path)
    sampling = dict(temperature=0.6, top_p=0.95, top_k=20, seed=7)
    anthropic = {key: value for key, value in anthropic.items() if key not in ("max_tokens", "stream", "temperature",
                                                                              "top_p", "metadata")}
    budget = (anthropic.get("thinking") or {}).get("budget_tokens")
    if budget is not None:
        openai = openai | dict(max_thinking_tokens=budget)  # the Anthropic budget is the chat path's thinking cap
    via_messages = post(["a"], anthropic | sampling | dict(max_tokens=9), tokenizer=template)
    via_chat = post(["a"], openai | sampling | dict(max_tokens=9), tokenizer=template, path="/v1/chat/completions")
    assert via_messages.statuses == via_chat.statuses == [200], via_messages.body
    ids_m, request_m, _ = via_messages.backend.submissions[0]
    ids_c, request_c, _ = via_chat.backend.submissions[0]
    assert ids_m == ids_c
    assert template.render_chat(request_m).encode() == template.render_chat(request_c).encode()
    assert strip(request_m) == strip(request_c)


def test_the_servers_own_thinking_only_output_round_trips_like_the_chat_history(tmp_path):
    """The Messages twin of the Responses test: a one-token reply is a thinking block only. Fed back as the
    assistant turn, it gives the same prompt IDs as the chat history that holds the chat path's own reply."""
    template = fixture_tokenizer("fixture", tmp_path)
    events = token_events(["Think."], stop_token=False)
    user = [dict(role="user", content="Q")]
    own = json.loads(post(["Think."], body(messages=user, max_tokens=1, seed=7, thinking=dict(type="adaptive")),
                          events=events).body)
    assert own["stop_reason"] == "max_tokens" and [block["type"] for block in own["content"]] == ["thinking"]
    chat_first = post(["Think."], dict(messages=user, max_tokens=1, seed=7, enable_thinking=True), events=events,
                      path="/v1/chat/completions")
    message = json.loads(chat_first.body)["choices"][0]["message"]
    assert message["content"] == "" and message["reasoning_content"] == "Think."
    again = post(["a"], body(messages=user + [dict(role="assistant", content=own["content"]),
                                              dict(role="user", content="Continue")], max_tokens=9, seed=7),
                 tokenizer=template)
    history = user + [{key: message[key] for key in ("role", "content", "reasoning_content")},
                      dict(role="user", content="Continue")]
    via_chat = post(["a"], dict(messages=history, max_tokens=9, seed=7), tokenizer=template, path="/v1/chat/completions")
    assert again.statuses == via_chat.statuses == [200], again.body
    assert again.backend.submissions[0][0] == via_chat.backend.submissions[0][0]


def test_the_converter_names_the_path():
    assert messages_api.PATH == PATH
