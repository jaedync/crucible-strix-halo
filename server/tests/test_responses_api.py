"""POST /v1/responses (OpenAI Responses API) over the chat path: same prompt, same tokens, Responses shapes."""
import json
from dataclasses import replace

import pytest

from server import responses_api
from server.tests.sdk_fixture import tool_block
from server.tests.test_api_key import KEY, bearer, gate
from server.tests.test_count_tokens import tokenizer as fixture_tokenizer
from server.tests.test_messages_api import post as messages_post, events as named_events

PATH = "/v1/responses"
ADD = dict(type="function", name="add", description="Add.", parameters=dict(
    type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"]))
ADD_CHAT = dict(type="function", function=dict(name="add", description="Add.", parameters=ADD["parameters"]))


def body(**options):
    return dict(model="crucible", input="hello", max_output_tokens=16) | options


def post(pieces, payload, **options):
    return messages_post(pieces, payload, path=options.pop("path", PATH), **options)


def reply(response):
    assert response.statuses == [200], response.body
    return json.loads(response.body)


def error(response):
    return json.loads(response.body)["error"]


def test_text_reply():
    result = reply(post(["Hello", " there"], body()))
    assert result["object"] == "response" and result["id"].startswith("resp_") and result["status"] == "completed"
    [item] = result["output"]
    assert item["type"] == "message" and item["role"] == "assistant" and item["status"] == "completed"
    assert item["content"] == [dict(type="output_text", text="Hello there", annotations=[], logprobs=[])]
    assert result["usage"] == dict(input_tokens=1, input_tokens_details=dict(cached_tokens=0), output_tokens=3,
                                   output_tokens_details=dict(reasoning_tokens=0), total_tokens=4)
    assert result["incomplete_details"] is None and result["store"] is False


def test_fields_reach_the_request():
    response = post(["a"], body(instructions="Be brief.", temperature=0.5, top_p=0.9, max_output_tokens=7))
    request = response.backend.submissions[0][1]
    assert (request.temperature, request.top_p, request.max_tokens) == (0.5, 0.9, 7)
    assert [m["role"] for m in request.messages] == ["system", "user"]
    assert request.messages[0]["content"] == "Be brief."


def test_length_is_incomplete():
    from server.tests.serve_fixtures import token_events
    result = reply(post(["a", "b"], body(max_output_tokens=2), events=token_events(["a", "b"], stop_token=False)))
    assert result["status"] == "incomplete" and result["incomplete_details"] == dict(reason="max_output_tokens")


def test_reasoning_item_and_effort():
    pieces = ["Add.", "</think>", "\n\n4"]
    response = post(pieces, body(reasoning=dict(effort="high", summary="auto")))
    assert response.backend.submissions[0][1].enable_thinking is True
    result = reply(response)
    assert [item["type"] for item in result["output"]] == ["reasoning", "message"]
    assert result["output"][0]["content"] == [dict(type="reasoning_text", text="Add.")]
    assert result["output"][0]["summary"] == []
    assert result["usage"]["output_tokens_details"]["reasoning_tokens"] == 2
    response = post(["a"], body(reasoning=dict(effort="none")))
    assert response.backend.submissions[0][1].enable_thinking is False


def test_function_call_reply():
    result = reply(post([tool_block()], body(tools=[ADD])))
    [call] = result["output"]
    assert call["type"] == "function_call" and call["name"] == "add" and call["status"] == "completed"
    assert json.loads(call["arguments"]) == dict(a=17, b=25)
    assert call["call_id"].startswith("call_") and call["id"].startswith("fc_")


def test_stream_text_events():
    response = post(["Hel", "lo"], body(stream=True))
    pairs = named_events(response)
    names = [name for name, _ in pairs]
    assert names == ["response.created", "response.in_progress", "response.output_item.added",
                     "response.content_part.added", "response.output_text.delta", "response.output_text.delta",
                     "response.output_text.done", "response.content_part.done", "response.output_item.done",
                     "response.completed"]
    assert all(data["type"] == name for name, data in pairs)
    assert [data["sequence_number"] for _, data in pairs] == list(range(len(pairs)))
    assert pairs[0][1]["response"]["status"] == "in_progress" and pairs[0][1]["response"]["output"] == []
    assert pairs[6][1]["text"] == "Hello"
    final = pairs[-1][1]["response"]
    assert final["output"][0]["content"][0]["text"] == "Hello" and final["usage"]["output_tokens"] == 3
    assert "[DONE]" not in response.body.decode()


def test_stream_reasoning_then_call():
    pieces = ["Plan.", "</think>", "\n\n", tool_block()]
    response = post(pieces, body(stream=True, tools=[ADD], reasoning=dict(effort="medium")))
    pairs = named_events(response)
    added = [data["item"]["type"] for name, data in pairs if name == "response.output_item.added"]
    assert added == ["reasoning", "function_call"]
    reasoning = "".join(data["delta"] for name, data in pairs if name == "response.reasoning_text.delta")
    arguments = "".join(data["delta"] for name, data in pairs if name == "response.function_call_arguments.delta")
    assert reasoning == "Plan." and json.loads(arguments) == dict(a=17, b=25)
    done = [data for name, data in pairs if name == "response.function_call_arguments.done"]
    assert json.loads(done[0]["arguments"]) == dict(a=17, b=25)
    final = pairs[-1][1]["response"]
    assert [item["type"] for item in final["output"]] == ["reasoning", "function_call"]


def test_stream_incomplete():
    from server.tests.serve_fixtures import token_events
    response = post(["a", "b"], body(stream=True, max_output_tokens=2), events=token_events(["a", "b"], stop_token=False))
    name, data = named_events(response)[-1]
    assert name == "response.incomplete" and data["response"]["incomplete_details"] == dict(reason="max_output_tokens")


def test_stream_error_after_start():
    script = [dict(event="token", id=20), dict(event="error", error="The engine request failed.")]
    name, data = named_events(post(["a"], body(stream=True), events=script))[-1]
    assert name == "error" and data["type"] == "error" and data["message"] and "sequence_number" in data


@pytest.mark.parametrize("field,value", [("previous_response_id", "resp_1"), ("store", True),
                                         ("conversation", "conv_1"), ("background", True)])
def test_state_fields_are_refused_by_name(field, value):
    response = post(["a"], body(**{field: value}))
    assert response.statuses == [400]
    assert error(response)["param"] == field and field in error(response)["message"]
    assert response.backend.submissions == []


@pytest.mark.parametrize("field,value", [("store", False), ("previous_response_id", None), ("metadata", {"k": "v"}),
                                         ("user", "u"), ("safety_identifier", "s"), ("prompt_cache_key", "k"),
                                         ("service_tier", "auto"), ("truncation", "disabled"),
                                         ("include", ["reasoning.encrypted_content"]), ("text", {"format": {"type": "text"}}),
                                         ("stream_options", {"include_obfuscation": False}), ("parallel_tool_calls", True)])
def test_accepted_fields(field, value):
    assert post(["a"], body(**{field: value})).statuses == [200]


@pytest.mark.parametrize("field,value,param", [
    ("text", {"format": {"type": "grammar"}}, "text.format"), ("text", {"verbosity": "high"}, "text"),
    ("truncation", "auto", "truncation"),
    ("include", ["message.output_text.logprobs"], "include"), ("top_logprobs", 3, "top_logprobs"),
    ("tools", [dict(type="web_search")], "tools[0].type"), ("max_tool_calls", 2, "max_tool_calls"),
    ("prompt", dict(id="p"), "prompt"), ("frequency", 1, "frequency")])
def test_unsupported_fields_are_refused_by_name(field, value, param):
    response = post(["a"], body(**{field: value}))
    assert response.statuses == [400] and error(response)["param"] == param
    assert param in error(response)["message"]


@pytest.mark.parametrize("item,place", [(dict(type="item_reference", id="x"), "input[0].type"),
                                        (dict(type="web_search_call", id="x"), "input[0].type"),
                                        (dict(role="user", content=[dict(type="input_file", file_id="f")]),
                                         "input[0].content[0].type")])
def test_unsupported_items_are_refused_by_place(item, place):
    response = post(["a"], body(input=[item, dict(role="user", content="hi")]))
    assert response.statuses == [400] and place in error(response)["message"] and error(response)["param"] == place
    kind = item.get("type") if "content" not in item else item["content"][0]["type"]
    assert kind not in error(response)["message"]  # MEDIUM 4: a fixed list of supported values, never the client's


def openai_error(response):
    """The OpenAI error object of a 400 (MEDIUM 3: a malformed value is a 400 with this shape, never a 500)."""
    assert response.statuses == [400], response.body
    result = json.loads(response.body)
    assert set(result) == {"error"} and set(result["error"]) == {"message", "type", "param", "code"}
    assert result["error"]["type"] == "invalid_request_error"
    assert response.backend.submissions == []
    return result["error"]


@pytest.mark.parametrize("value", [[], {}, 1])
@pytest.mark.parametrize("make,param", [
    (lambda v: dict(input="Q", include=[v]), "include"),
    (lambda v: dict(input="Q", include=["reasoning.encrypted_content", v]), "include"),
    (lambda v: dict(input=[dict(type=v)]), "input[0].type"),
    (lambda v: dict(input=[dict(role="user", content=[dict(type=v, text="x")])]), "input[0].content[0].type"),
    (lambda v: dict(input=[dict(type="function_call_output", call_id="c", output=[dict(type=v)])]),
     "input[0].output[0].type"),
    (lambda v: dict(input="Q", tools=[dict(type=v)]), "tools[0].type")], ids=["include", "include-second", "item",
                                                                              "part", "output-part", "tool"])
def test_malformed_discriminators_are_a_400(make, param, value):
    assert openai_error(post(["a"], body(**make(value))))["param"] == param


CANARY = "canary_request_content"


@pytest.mark.parametrize("payload,place", [
    (dict(input=[dict(type=CANARY)]), "input[0].type"),
    (dict(input=[dict(type=CANARY.upper())]), "input[0].type"),
    (dict(input=[dict(role="user", content=[dict(type=CANARY, text="x")])]), "input[0].content[0].type"),
    (dict(input=[dict(type="function_call_output", call_id="c", output=[dict(type=CANARY)])]),
     "input[0].output[0].type"),
    (dict(input=[dict(role="user", content="Q"), dict(type="message", role="assistant", content=[dict(type=CANARY)])]),
     "input[1].content[0].type"),
    (dict(input="Q", tools=[dict(type=CANARY)]), "tools[0].type"),
    (dict(input="Q", include=[CANARY]), "include")])
def test_errors_name_the_place_and_never_echo_the_value(payload, place):
    """MEDIUM 4: the error names the place and a fixed list of supported values, never the client's value."""
    response = post(["a"], body(**payload))
    message = openai_error(response)["message"]
    assert place in message and CANARY not in response.body.decode().lower()


def test_missing_key_is_refused_when_a_key_is_set():
    response = post(["a"], body(), options=dict(api_key=gate()), headers=[])
    assert response.statuses == [401] and response.backend.submissions == []
    assert post(["a"], body(), options=dict(api_key=gate()), headers=[bearer(KEY)]).statuses == [200]


def test_body_limit_applies():
    response = post(["a"], body(input="x" * (2 << 20)), options=dict(max_body_mib=1))
    assert response.statuses == [413] and "error" in json.loads(response.body)


def strip(request):
    return replace(request, stream=False, include_usage=False)


def strict_default(request):
    """The chat request with the strict that /v1/responses gives a tool with no strict, as OpenAI does. The engine of
    these tests has no structured output: the tool falls back to false, and the prompt keeps the chat
    tools (render_tools)."""
    if not request.tools or all("strict" in tool["function"] for tool in request.tools):
        return request
    tools = tuple(dict(tool, function=dict(strict=False) | tool["function"]) for tool in request.tools)
    return replace(request, tools=tools, render_tools=request.tools)


CALL_ID = "call_1"


def corpus():
    """(name, Responses body, chat body)."""
    user = [dict(role="user", content="Hi")]
    calls = [dict(id=CALL_ID, type="function", function=dict(name="add", arguments='{"a":1,"b":2}'))]
    return [
        ("string input", dict(input="Hi"), dict(messages=user)),
        ("instructions", dict(instructions="Rules.", input="Hi"),
         dict(messages=[dict(role="system", content="Rules.")] + user)),
        ("developer and system lead", dict(instructions="A.", input=[
            dict(role="developer", content="B."), dict(role="system", content=[dict(type="input_text", text="C.")]),
            dict(role="user", content="Hi")]),
         dict(messages=[dict(role="system", content="A.\n\nB.\n\nC.")] + user)),
        ("typed message parts", dict(input=[dict(type="message", role="user", content=[
            dict(type="input_text", text="One. "), dict(type="input_text", text="Two.")])]),
         dict(messages=[dict(role="user", content=[dict(type="text", text="One. "), dict(type="text", text="Two.")])])),
        ("history", dict(input=[dict(role="user", content="Q"), dict(
            type="message", role="assistant", status="completed", id="msg_1",
            content=[dict(type="output_text", text="A", annotations=[])]), dict(role="user", content="Q2")]),
         dict(messages=[dict(role="user", content="Q"), dict(role="assistant", content="A"),
                        dict(role="user", content="Q2")])),
        ("tool round", dict(tools=[ADD], input=[
            dict(role="user", content="Add"),
            dict(type="reasoning", id="rs_1", summary=[], content=[dict(type="reasoning_text", text="Use add.")]),
            dict(type="message", role="assistant", content=[dict(type="output_text", text="Calling.")]),
            dict(type="function_call", id="fc_1", call_id=CALL_ID, name="add", arguments='{"a":1,"b":2}',
                 status="completed"),
            dict(type="function_call_output", call_id=CALL_ID, output="3")]),
         dict(tools=[ADD_CHAT], messages=[dict(role="user", content="Add"),
                                          dict(role="assistant", content="Calling.", reasoning_content="Use add.",
                                               tool_calls=calls),
                                          dict(role="tool", tool_call_id=CALL_ID, content="3")])),
        ("bare call", dict(tools=[ADD], input=[
            dict(role="user", content="Add"),
            dict(type="function_call", call_id=CALL_ID, name="add", arguments='{"a":1,"b":2}'),
            dict(type="function_call_output", call_id=CALL_ID, output=[dict(type="input_text", text="3")])]),
         dict(tools=[ADD_CHAT], messages=[dict(role="user", content="Add"),
                                          dict(role="assistant", content=None, tool_calls=calls),
                                          dict(role="tool", tool_call_id=CALL_ID, content=[dict(type="text", text="3")])])),
        ("tool choice", dict(tools=[ADD], tool_choice=dict(type="function", name="add"), parallel_tool_calls=False,
                             input="Hi"),
         dict(tools=[ADD_CHAT], tool_choice=dict(type="function", function=dict(name="add")),
              parallel_tool_calls=False, messages=user)),
        ("tool choice required", dict(tools=[ADD], tool_choice="required", input="Hi"),
         dict(tools=[ADD_CHAT], tool_choice="required", messages=user)),
        ("effort", dict(reasoning=dict(effort="low"), input="Hi"), dict(reasoning_effort="low", messages=user)),
        ("effort none", dict(reasoning=dict(effort="none"), input="Hi"), dict(reasoning_effort="none", messages=user)),
        # A reasoning-only assistant turn is the chat path's assistant message with
        # the reasoning and an empty content. Pending reasoning never crosses a user, system or tool boundary.
        ("reasoning-only turn", dict(input=[dict(role="user", content="Q"), think("Think."),
                                            dict(role="user", content="Continue")]),
         dict(messages=[dict(role="user", content="Q"), dict(role="assistant", content="", reasoning_content="Think."),
                        dict(role="user", content="Continue")])),
        ("reasoning stays in its turn", dict(input=[
            dict(role="user", content="Q"), think("R1"), dict(role="user", content="Q2"), think("R2"),
            dict(type="message", role="assistant", content=[dict(type="output_text", text="A")])]),
         dict(messages=[dict(role="user", content="Q"), dict(role="assistant", content="", reasoning_content="R1"),
                        dict(role="user", content="Q2"), dict(role="assistant", content="A", reasoning_content="R2")])),
        ("reasoning at the end", dict(input=[dict(role="user", content="Q"), think("Think.")]),
         dict(messages=[dict(role="user", content="Q"), dict(role="assistant", content="", reasoning_content="Think.")])),
        ("reasoning after a message starts a new turn", dict(tools=[ADD], input=[
            dict(role="user", content="Add"),
            dict(type="message", role="assistant", content=[dict(type="output_text", text="Calling.")]), think("Use add."),
            dict(type="function_call", call_id=CALL_ID, name="add", arguments='{"a":1,"b":2}'),
            dict(type="function_call_output", call_id=CALL_ID, output="3")]),
         dict(tools=[ADD_CHAT], messages=[dict(role="user", content="Add"), dict(role="assistant", content="Calling."),
                                          dict(role="assistant", content=None, reasoning_content="Use add.",
                                               tool_calls=calls),
                                          dict(role="tool", tool_call_id=CALL_ID, content="3")])),
    ]


def think(text):
    return dict(type="reasoning", summary=[], content=[dict(type="reasoning_text", text=text)])


@pytest.mark.parametrize("kind", ["fixture", "checkpoint"])
@pytest.mark.parametrize("name,responses,chat", corpus(), ids=[case[0] for case in corpus()])
def test_the_prompt_and_request_equal_the_chat_path(tmp_path, kind, name, responses, chat):
    """CRUCIBLE_TEST_CKPT adds the real checkpoint tokenizer and template."""
    template = fixture_tokenizer(kind, tmp_path)
    sampling = dict(temperature=0.6, top_p=0.95, top_k=20, seed=7)
    via_responses = post(["a"], responses | sampling | dict(max_output_tokens=9), tokenizer=template)
    via_chat = post(["a"], chat | sampling | dict(max_tokens=9), tokenizer=template, path="/v1/chat/completions")
    assert via_responses.statuses == via_chat.statuses == [200], via_responses.body
    ids_r, request_r, _ = via_responses.backend.submissions[0]
    ids_c, request_c, _ = via_chat.backend.submissions[0]
    assert ids_r == ids_c
    assert template.render_chat(request_r).encode() == template.render_chat(request_c).encode()
    assert strip(request_r) == strip(strict_default(request_c))


def test_the_servers_own_incomplete_output_round_trips_like_the_chat_history(tmp_path):
    """A one-token reply is a reasoning item only. Fed back as input, it gives the
    same prompt IDs as the chat history that holds the chat path's own reply to the same request."""
    from server.tests.serve_fixtures import token_events
    template = fixture_tokenizer("fixture", tmp_path)
    events = token_events(["Think."], stop_token=False)
    first = dict(seed=7, reasoning=dict(effort="high"))
    own = reply(post(["Think."], dict(input="Q", max_output_tokens=1) | first, events=events))
    assert own["status"] == "incomplete" and [item["type"] for item in own["output"]] == ["reasoning"]
    chat_first = post(["Think."], dict(messages=[dict(role="user", content="Q")], max_tokens=1, seed=7,
                                       reasoning_effort="high"), events=events, path="/v1/chat/completions")
    message = json.loads(chat_first.body)["choices"][0]["message"]
    assert message["content"] == "" and message["reasoning_content"] == "Think."
    sampling = dict(seed=7, max_output_tokens=9)
    again = post(["a"], dict(input=[dict(role="user", content="Q"), *own["output"],
                                    dict(role="user", content="Continue")]) | sampling, tokenizer=template)
    history = [dict(role="user", content="Q"), {key: message[key] for key in ("role", "content", "reasoning_content")},
               dict(role="user", content="Continue")]
    via_chat = post(["a"], dict(messages=history, seed=7, max_tokens=9), tokenizer=template, path="/v1/chat/completions")
    assert again.statuses == via_chat.statuses == [200], again.body
    assert again.backend.submissions[0][0] == via_chat.backend.submissions[0][0]
    assert any(m["role"] == "assistant" for m in again.backend.submissions[0][1].messages)


def test_reasoning_before_a_tool_result_is_refused():
    """Reasoning between a function_call and its output has no assistant turn (the chat path refuses that history)."""
    response = post(["a"], body(tools=[ADD], input=[
        dict(role="user", content="Add"),
        dict(type="function_call", call_id=CALL_ID, name="add", arguments='{"a":1,"b":2}'), think("Wait."),
        dict(type="function_call_output", call_id=CALL_ID, output="3")]))
    assert response.statuses == [400] and "input[3]" in error(response)["message"]
    assert response.backend.submissions == []


@pytest.mark.parametrize("verbosity,code", [(None, 200), ("medium", 200), ("low", 400), ("high", 400)])
def test_text_verbosity_is_medium_only(verbosity, code):
    """Low and high ask for another answer length, which the server cannot give."""
    response = post(["a"], body(text=dict(verbosity=verbosity)))
    assert response.statuses == [code]
    if code == 400:
        assert error(response)["param"] == "text" and response.backend.submissions == []


def test_a_late_system_message_is_refused():
    response = post(["a"], body(input=[dict(role="user", content="Hi"), dict(role="developer", content="X")]))
    assert response.statuses == [400] and "input[1]" in error(response)["message"]


def test_the_module_names_the_path():
    assert responses_api.PATH == PATH
