"""/v1/responses: a function tool with no strict (absent or null) is strict when it can be, as OpenAI does.

OpenAI, https://platform.openai.com/docs/guides/function-calling ("Strict mode"): "If you omit `strict`, the default
depends on the API: Responses requests will attempt to normalize your schema into strict mode when possible, and will
fall back to non-strict, best-effort function calling if the schema cannot be made compatible with strict mode. When
fallback happens, the response tool will show `strict: false`. Chat Completions requests remain non-strict by default."

So: no strict -> the strict call grammar when the chat path's strict check accepts the tool, else no grammar and
strict false in the reply's tools; strict true keeps its 400; strict false and chat completions do not change.
The replies and errors come from the real handler (serve_fixtures.invoke), not from hand-written shapes.
"""
import json

import pytest

from server.grammar_limits import MAX_LENGTH_BOUND, MAX_OPTIONAL_PROPERTIES_TOTAL
from server.tests.sdk_fixture import tool_block
from server.tests.test_count_tokens import tokenizer as fixture_tokenizer
from server.tests.test_structured_adapters import ADD_C, ADD_R, REFUSED_AT_OPEN, post, submitted

USER = [dict(role="user", content="Add 1 and 2")]
# The strict call grammar refuses a raw string parameter with a pattern (structured.RAW_STRING_REFUSED).
PATTERN = dict(type="object", properties=dict(code=dict(type="string", pattern="^[a-z]+$")), required=["code"])
# The grammar limits refuse a maxLength over MAX_LENGTH_BOUND (grammar_limits.GrammarBudget); call_format accepts it.
LONG = dict(type="object", properties=dict(names=dict(type="array", items=dict(type="string",
                                                                              maxLength=MAX_LENGTH_BOUND + 1))),
            required=["names"])


def responses(tools, **fields):
    return dict(model="crucible", input="Add 1 and 2", max_output_tokens=9, seed=7, tools=tools) | fields


def tool(parameters, name="add", **strict):
    return dict(type="function", name=name, description="Add.", parameters=parameters) | strict


def reply_tools(response):
    assert response.statuses == [200], response.body
    return json.loads(response.body)["tools"]


@pytest.mark.parametrize("missing", [{}, dict(strict=None)], ids=["absent", "null"])
def test_a_tool_without_strict_gets_the_strict_grammar(missing):
    plain = submitted(post(responses([ADD_R | missing]), path="/v1/responses"))
    strict = submitted(post(responses([ADD_R | dict(strict=True)]), path="/v1/responses"))
    assert plain[1].constraint is not None and plain[1].constraint["kind"] == "structural_tag"
    assert plain[:2] == strict[:2]


def test_strict_false_stays_without_a_grammar():
    ids, request, _ = submitted(post(responses([ADD_R | dict(strict=False)]), path="/v1/responses"))
    assert request.constraint is None
    assert request.tools[0]["function"]["strict"] is False


def test_chat_completions_keep_strict_false_when_absent():
    body = dict(model="crucible", messages=USER, max_tokens=9, seed=7, tools=[ADD_C])
    _, request, _ = submitted(post(body))
    assert request.constraint is None
    assert "strict" not in request.tools[0]["function"]


def test_the_reply_tools_show_the_strict_value_that_the_server_applied():
    tools = [ADD_R, tool(PATTERN, "code"), tool(PATTERN, "code2", strict=False), tool(ADD_R["parameters"], "sum",
                                                                                     strict=None)]
    echo = reply_tools(post(responses(tools), path="/v1/responses"))
    assert [item["strict"] for item in echo] == [True, False, False, True]
    assert [{key: value for key, value in item.items() if key != "strict"} for item in echo] == [
        {key: value for key, value in item.items() if key != "strict"} for item in tools]


@pytest.mark.parametrize("parameters", [PATTERN, LONG], ids=["call grammar", "grammar limit"])
def test_a_schema_that_strict_mode_refuses_falls_back_without_an_error(parameters):
    response = post(responses([tool(parameters)]), path="/v1/responses")
    assert response.statuses == [200], response.body
    request = response.backend.submissions[0][1]
    assert request.constraint is None and request.tools[0]["function"]["strict"] is False
    assert json.loads(response.body)["tools"][0]["strict"] is False


@pytest.mark.parametrize("parameters,code", [(PATTERN, "pattern"), (LONG, "MAX_LENGTH_BOUND")],
                         ids=["call grammar", "grammar limit"])
def test_explicit_strict_true_keeps_its_error(parameters, code):
    response = post(responses([tool(parameters, strict=True)]), path="/v1/responses")
    assert response.statuses == [400], response.body
    assert response.backend.submissions == []
    error = json.loads(response.body)["error"]
    assert error["param"] == "tools" and error["type"] == "invalid_request_error" and code in error["message"]
    # The chat path gives the same error for the same strict tool.
    chat_tool = dict(type="function", function=dict(name="add", description="Add.", parameters=parameters, strict=True))
    chat = post(dict(model="crucible", messages=USER, max_tokens=9, tools=[chat_tool]))
    assert chat.statuses == [400] and json.loads(chat.body)["error"]["message"] == error["message"]


def test_the_grammar_limits_of_the_whole_request_decide_each_tool_in_order():
    # Each tool has 100 optional parameters; the request may hold MAX_OPTIONAL_PROPERTIES_TOTAL of them.
    many = dict(type="object", properties={f"p{i}": dict(type="integer") for i in range(100)})
    count = MAX_OPTIONAL_PROPERTIES_TOTAL // 100 + 1
    tools = [tool(many, f"t{i}") for i in range(count)]
    response = post(responses(tools), path="/v1/responses")
    echo = reply_tools(response)
    assert [item["strict"] for item in echo] == [True] * (count - 1) + [False]
    assert response.backend.submissions[0][1].constraint is not None


def test_a_strict_tool_counts_before_a_tool_without_strict():
    many = dict(type="object", properties={f"p{i}": dict(type="integer") for i in range(100)})
    count = MAX_OPTIONAL_PROPERTIES_TOTAL // 100
    tools = [tool(many, f"t{i}", strict=True) for i in range(count)] + [tool(many, "last")]
    echo = reply_tools(post(responses(tools), path="/v1/responses"))
    assert [item["strict"] for item in echo] == [True] * count + [False]


def test_the_answer_format_counts_before_the_tools():
    many = dict(type="object", properties={f"p{i}": dict(type="integer") for i in range(100)})
    count = MAX_OPTIONAL_PROPERTIES_TOTAL // 100
    answer = dict(format=dict(type="json_schema", name="a", schema=dict(many, additionalProperties=False)))
    tools = [tool(many, f"t{i}") for i in range(count)]
    echo = reply_tools(post(responses(tools, text=answer), path="/v1/responses"))
    assert [item["strict"] for item in echo] == [True] * (count - 1) + [False]


def test_an_engine_refusal_at_open_stays_an_error():
    # The engine compiles the grammar after validation: the server cannot fall back there (docs/OPENAI_CLIENTS.md).
    # A constrained stream waits for the first engine event, so the refusal is a plain 400, not an SSE error.
    response = post(responses([ADD_R], stream=True), path="/v1/responses", events=REFUSED_AT_OPEN)
    assert response.statuses == [400], response.body
    assert response.headers.get("Content-Type") == "application/json"
    assert "does not compile" in json.loads(response.body)["error"]["message"]


def test_explicit_strict_tools_count_before_an_earlier_tool_without_strict():
    # A tool with no strict that comes first must not take the budget of the explicit strict tools.
    many = dict(type="object", properties={f"p{i}": dict(type="integer") for i in range(100)})
    count = MAX_OPTIONAL_PROPERTIES_TOTAL // 100
    tools = [tool(many, "a")] + [tool(many, f"s{i}", strict=True) for i in range(count)]
    response = post(responses(tools), path="/v1/responses")
    echo = reply_tools(response)
    assert [item["strict"] for item in echo] == [False] + [True] * count
    assert response.backend.submissions[0][1].constraint is not None


def test_an_engine_without_structured_output_runs_the_tool_without_a_grammar():
    # No grammar can hold the tool, so the server does not set strict: the tool falls back.
    response = post(responses([ADD_R]), path="/v1/responses", ready=False)
    request = response.backend.submissions[0][1]
    assert request.constraint is None and request.tools[0]["function"]["strict"] is False
    assert reply_tools(response)[0]["strict"] is False


def test_an_engine_without_structured_output_parses_a_call_softly():
    # The parse of a tool with no strict stays the soft parse of strict false (a schema fault is still a call).
    call = tool_block(a="foo")
    kinds = []
    for strict in ({}, dict(strict=False)):
        response = post(responses([ADD_R | strict]), path="/v1/responses", ready=False, pieces=(call,))
        assert response.statuses == [200], response.body
        kinds.append([item["type"] for item in json.loads(response.body)["output"]])
    assert kinds[0] == kinds[1] and "function_call" in kinds[0]


def test_an_engine_without_structured_output_keeps_the_prompt_of_the_client_flags(tmp_path):
    # That engine's template renders the strict flags (responses_api.RENDER_STRICT): only those that the client sent.
    sum_r, sum_c = tool(ADD_R["parameters"], "sum", strict=True), dict(type="function", function=dict(
        name="sum", description="Add.", parameters=ADD_R["parameters"], strict=True))
    template = fixture_tokenizer("fixture", tmp_path)
    via_r = submitted(post(responses([ADD_R, sum_r]), path="/v1/responses", ready=False, tokenizer=template))
    via_c = submitted(post(dict(model="crucible", messages=USER, max_tokens=9, seed=7, tools=[ADD_C, sum_c]),
                           ready=False, tokenizer=template))
    assert via_r[0] == via_c[0]
    assert via_r[1].render_tools == via_c[1].tools
    assert [item["function"]["strict"] for item in via_r[1].tools] == [False, True]
