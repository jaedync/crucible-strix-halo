"""Hostile structured output requests: the bounds that the server applies before the
engine compiles a client's schema with xgrammar 0.2.9.

Each refused shape crashed or stalled the engine compile in the grammar probe (tests/constraint_hostile_test.cpp
is the engine side). A refusal is a 400 with a fixed message: no schema text goes back to the client. The controls keep
ordinary schemas working, so that no test passes only because the server refuses everything.
Proposed bounds (a cheap first line; the engine compile also needs a deadline and a memory cap): a pattern nests groups
at most 64 levels, a quantifier bound {m,n} is at most 1000, the quantifier bounds of one schema sum to at most 4000,
and one schema holds at most 32 strings with pattern or format. Compile cost on the real vocabulary: 64 email strings
2.5 s, one (a|b|c){1,1000} 0.5 s, 64 of them 26 s (raw vocabulary): the cost adds up over strings.
"""
import pytest

from server.api import ApiError, validate_request

NEST = "(" * 40000 + "a" + ")" * 40000
DEEP_STAR = "(" * 20000 + "a" + ")*" * 20000


def chat(**fields):
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=8) | fields
    return validate_request(body, "chat", "crucible")


def answer(schema):
    return chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=schema)))


def strict_tool(parameters):
    tools = [dict(type="function", function=dict(name="f", strict=True, parameters=parameters))]
    return chat(tools=tools, tool_choice="required")


def refused(make, *args):
    with pytest.raises(ApiError) as error:
        make(*args)
    assert "(" * 8 not in str(error.value)          # the message never quotes the schema
    return error.value


@pytest.mark.parametrize("pattern", [NEST, DEEP_STAR, "(" * 65 + "a" + ")" * 65, "(?:" * 65 + "a" + ")" * 65],
                         ids=["nest-40000", "nested-star-20000", "nest-65", "group-65"])
def test_a_deeply_nested_pattern_is_refused(pattern):
    refused(answer, {"type": "string", "pattern": pattern})


def test_a_nested_pattern_in_an_object_property_is_refused():
    refused(answer, {"type": "object", "properties": {"q": {"type": "string", "pattern": NEST}}, "required": ["q"]})


def test_a_nested_pattern_in_a_strict_tool_parameter_is_refused():
    refused(strict_tool, {"type": "object", "properties": {"q": {"type": "object", "properties": {
        "r": {"type": "string", "pattern": NEST}}}}, "required": ["q"]})


def test_a_nested_pattern_in_a_definition_is_refused():
    refused(answer, {"$defs": {"s": {"type": "string", "pattern": NEST}}, "type": "object",
                     "properties": {"q": {"$ref": "#/$defs/s"}}})


@pytest.mark.parametrize("pattern", ["(a|b|c){1,5000}", "[ab]{1,50000}x", "a{1001}", "(ab){2,1001}"])
def test_a_large_quantifier_bound_is_refused(pattern):
    # 45 bytes: "(a|b|c){1,5000}" compiled in 11 s, "[ab]{1,50000}x" in more than 300 s (grammar probe).
    refused(answer, {"type": "string", "pattern": pattern})


@pytest.mark.parametrize("pattern", [".{1,129}", ".{999}", ".{129,}", "[^\"]{1,500}", "(.|x){1,300}", "a{0,129}"])
def test_a_repeat_over_128_is_refused(pattern):
    # xgrammar 0.2.9 unrolls a repeat of 128 or less (fsm_builder.cc kLargeRepeatThreshold); a larger one is a counted
    # repeat that the token mask cache does not cover. A fuzz test: one masked token of ".{1,129}" took 312 ms
    # (mean, 694 ms max), of ".{1,128}" 0.15 ms. The mask runs on the serve loop, so every other stream waits.
    refused(answer, {"type": "string", "pattern": pattern})


@pytest.mark.parametrize("pattern", [".{1,128}", "[^x]{1,128}", "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", "a{128,}"])
def test_a_repeat_of_128_is_accepted(pattern):
    assert answer({"type": "string", "pattern": pattern}).constraint["kind"] == "json_schema"


def test_quantifier_bounds_that_add_up_are_refused():
    # 64 strings of (a|b|c){1,1000} compiled in 26 s (grammar probe); at the bound of 128 they still add up.
    refused(answer, {"type": "object", "properties": {
        f"p{i}": {"type": "string", "pattern": "(a|b|c){1,128}"} for i in range(64)}})


def test_many_format_strings_are_refused():
    # 680 email strings (28 KB, under every size limit today) compiled in 19.8 s and kept 1.1 GB (grammar probe).
    refused(answer, {"type": "object", "properties": {f"p{i}": {"type": "string", "format": "email"} for i in range(680)}})


def test_many_pattern_strings_in_tool_parameters_are_refused():
    refused(strict_tool, {"type": "object", "properties": {"q": {"type": "object", "properties": {
        f"p{i}": {"type": "string", "pattern": "[a-z]{1,200}"} for i in range(600)}}}, "required": ["q"]})


def test_a_schema_deeper_than_64_levels_stays_refused():
    # The present bound is load bearing: 1000 nested arrays compiled in 114 s, 5000 used more than 5.7 GB.
    schema = {"type": "string"}
    for _ in range(70):
        schema = {"type": "array", "items": schema}
    refused(answer, schema)


@pytest.mark.parametrize("pattern", ["^[a-z]{1,16}$", "(a|b)(c|d)", "\\(" * 100 + "x", "[(]{1,128}", "a{0,128}"],
                         ids=["plain", "groups", "escaped-parens", "class-paren", "bound-128"])
def test_an_ordinary_pattern_is_accepted(pattern):
    request = answer({"type": "string", "pattern": pattern})
    assert request.constraint["kind"] == "json_schema"


def test_thirty_two_format_strings_are_accepted():
    request = answer({"type": "object", "properties": {f"p{i}": {"type": "string", "format": "email"} for i in range(32)}})
    assert request.constraint["kind"] == "json_schema"
