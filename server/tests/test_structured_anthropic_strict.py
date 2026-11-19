"""An Anthropic custom tool with strict: true gets the strict call grammar of a chat function tool with strict: true.

The tool field strict maps to the chat function's strict (server/anthropic.py chat_tools). So the constraint, the
prompt IDs and the rendered prompt are those of the equal chat body, and the template never renders the flag.
"""
import json

import pytest

from server.tests.serve_fixtures import PiecesTokenizer
from server.tests.sdk_fixture import tool_block
from server.tests.test_count_tokens import tokenizer as fixture_tokenizer
from server.tests.test_structured_adapters import ADD_A, ADD_C, ADD_R, post, submitted
from server.tests.test_structured_formats import A_SCHEMA, C_NAMELESS, SECRET, USER, anthropic_error, chat_body, messages, same

STRICT_A = ADD_A | dict(strict=True)
STRICT_C = dict(ADD_C, function=ADD_C["function"] | dict(strict=True))


@pytest.mark.parametrize("kind", ["fixture", "checkpoint"])
@pytest.mark.parametrize("choice,chat_choice", [(None, None), (dict(type="any"), "required"),
                                                (dict(type="tool", name="add"),
                                                 dict(type="function", function=dict(name="add")))],
                         ids=["auto", "any", "tool"])
def test_a_strict_anthropic_tool_is_the_strict_chat_tool(tmp_path, kind, choice, chat_choice):
    template = fixture_tokenizer(kind, tmp_path)
    a_choice = {} if choice is None else dict(tool_choice=choice)
    c_choice = {} if chat_choice is None else dict(tool_choice=chat_choice)
    via_a = submitted(post(messages(tools=[STRICT_A], **a_choice), path="/v1/messages", tokenizer=template))
    via_c = submitted(post(chat_body(tools=[STRICT_C], **c_choice), tokenizer=template))
    assert via_a[1].constraint is not None and via_a[1].constraint["kind"] == "structural_tag"
    assert via_a[0] == via_c[0] and same(via_a[1]) == same(via_c[1])
    assert template.render_chat(via_a[1]).encode() == template.render_chat(via_c[1]).encode()
    assert "strict" not in via_a[1].render_tools[0]["function"]


@pytest.mark.parametrize("strict", [False, None])
def test_strict_false_or_null_is_the_tool_without_it(strict):
    via_a = submitted(post(messages(tools=[ADD_A | dict(strict=strict)]), path="/v1/messages"))
    plain = submitted(post(messages(tools=[ADD_A]), path="/v1/messages"))
    assert via_a[1].constraint is None and via_a[0] == plain[0]


@pytest.mark.parametrize("value", [SECRET, 1, [True]], ids=["text", "number", "list"])
def test_a_strict_value_that_is_not_a_boolean_is_a_400(value):
    message = anthropic_error(post(messages(tools=[ADD_A | dict(strict=value)]), path="/v1/messages"))
    assert "strict must be true or false" in message


def test_a_strict_tool_with_an_output_format_is_the_chat_constraint():
    via_a = submitted(post(messages(tools=[STRICT_A], output_config=dict(format=A_SCHEMA)), path="/v1/messages"))
    via_c = submitted(post(chat_body(tools=[STRICT_C], response_format=C_NAMELESS)))
    assert via_a[1].constraint is not None and via_a[1].constraint == via_c[1].constraint


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_a_strict_call_is_a_tool_use_block(stream):
    pieces = [tool_block()]
    response = post(messages(tools=[STRICT_A], tool_choice=dict(type="any"), stream=stream), path="/v1/messages",
                    pieces=pieces, tokenizer=PiecesTokenizer(pieces))
    assert response.statuses == [200], response.body
    if stream:
        events = [json.loads(line[6:]) for line in response.body.decode().splitlines() if line.startswith("data: {")]
        starts = [event["content_block"] for event in events if event["type"] == "content_block_start"]
        assert [block["type"] for block in starts] == ["tool_use"]
        partial = "".join(event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta")
        assert json.loads(partial) == dict(a=17, b=25)
    else:
        reply = json.loads(response.body)
        assert [block["type"] for block in reply["content"]] == ["tool_use"]
        assert reply["content"][0]["input"] == dict(a=17, b=25) and reply["stop_reason"] == "tool_use"


def test_count_tokens_takes_a_strict_tool():
    count = post(dict(messages=USER, tools=[STRICT_A]), path="/v1/messages/count_tokens")
    plain = post(dict(messages=USER, tools=[ADD_A]), path="/v1/messages/count_tokens")
    assert count.statuses == plain.statuses == [200], count.body
    assert json.loads(count.body) == json.loads(plain.body)


# An engine with no structured output (no constraint key in its ready event): the rule of the chat path. A strict tool
# runs with no grammar (200). Chat and Responses keep the prompt of the base, which renders the client's flag. An
# Anthropic tool never had the flag in its prompt, so the template does not render it there either.

def test_a_strict_tool_without_structured_output_runs_unconstrained_on_every_path(tmp_path):
    template = fixture_tokenizer("fixture", tmp_path)
    cases = [("/v1/chat/completions", chat_body(tools=[STRICT_C])),
             ("/v1/responses", dict(model="crucible", input="Add 1 and 2", max_output_tokens=9, seed=7,
                                    tools=[ADD_R | dict(strict=True)])),
             ("/v1/messages", messages(tools=[STRICT_A]))]
    for path, payload in cases:
        for stream in (False, True):
            ids, request, _ = submitted(post(payload | dict(stream=stream), path=path, ready=False, tokenizer=template))
            assert request.constraint is None, (path, stream)
            if path == "/v1/messages":
                assert "strict" not in template.render_chat(request), stream
            else:
                assert request.render_tools is None and '"strict": true' in template.render_chat(request), path


@pytest.mark.parametrize("strict", [True, False])
def test_an_anthropic_strict_flag_never_reaches_the_prompt_without_structured_output(tmp_path, strict):
    template = fixture_tokenizer("fixture", tmp_path)
    via_a = submitted(post(messages(tools=[ADD_A | dict(strict=strict)]), path="/v1/messages", ready=False,
                           tokenizer=template))
    plain = submitted(post(messages(tools=[ADD_A]), path="/v1/messages", ready=False, tokenizer=template))
    assert via_a[0] == plain[0]
    assert template.render_chat(via_a[1]).encode() == template.render_chat(plain[1]).encode()


def test_count_tokens_of_a_strict_tool_without_structured_output(tmp_path):
    template = fixture_tokenizer("fixture", tmp_path)
    count = post(dict(messages=USER, tools=[STRICT_A]), path="/v1/messages/count_tokens", ready=False,
                 tokenizer=template)
    plain = post(dict(messages=USER, tools=[ADD_A]), path="/v1/messages/count_tokens", ready=False, tokenizer=template)
    assert count.statuses == plain.statuses == [200], count.body
    assert json.loads(count.body) == json.loads(plain.body)
