"""Structured output: response_format, strict tools and enforced tool_choice.

The server only checks the request and builds the engine's constraint object; the engine compiles and enforces the
grammar (xgrammar, engine/runtime/constraint_xgr.cc). No grammar library is imported here.

The tool-call grammar is written here, not taken from a library preset: its bytes must equal what
server/tool_output.py parses (`<tool_call>\\n<function=NAME>\\n`, then `<parameter=KEY>\\nVALUE\\n</parameter>\\n` for each
argument in schema order, then `</function>\\n</tool_call>`). A string parameter is raw text; every other type is JSON.
"""
import json
from .api import ApiError
from .format_grammar import grammar_schema
from .grammar_limits import GrammarBudget
from .schema_limits import WorkBudget
from .schema_exact import check_exact
from .tool_schema import check_schema, resolve, types

# The longest run of JSON whitespace inside a tool call value (the engine caps json_schema answers itself).
MAX_WHITESPACE = 16
# Raw string parameters of a tool call take only these string keywords (the rest needs a grammar of raw text), unless
# the parameter is a const or an enum (each value is checked against them: server/schema_exact.py).
RAW_STRING_REFUSED = ("pattern", "format", "minLength", "maxLength", "anyOf")
# Root keywords of a strict tool's parameters that the call grammar does not encode (review H2): a 400 that names the
# keyword. The call grammar encodes properties, required and the boolean schema false; it emits no other property,
# so additionalProperties holds; the keyword allow-list (server/schema_exact.py) checks the rest of the schema.
TOOL_ROOT_REFUSED = ("const", "enum", "anyOf")


def refuse(keyword, field):
    raise ApiError(f"Structured output does not support the JSON schema keyword {keyword}.", field,
                   code="constraint_keyword")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# The chat field of an answer format, and the error place of the adapters' fields that map to it
# (docs/OPENAI_CLIENTS.md "Structured output on the adapters"): an error names the field that the client sent.
FORMAT_FIELD = "response_format"
# json_schema strict (chat response_format.json_schema.strict, Responses text.format.strict): the OpenAI APIs default
# it to false, but the server enforces the schema at every value of strict. So a missing or null
# strict takes this value, and false gets the same grammar as true. A Responses reply echoes it in text.format.
STRICT_DEFAULT = True


def response_format(value, kind, field=FORMAT_FIELD):
    """The constraint of a response_format field: None for text, else (kind, schema or None). field: the place that
    the client's error names (the adapter's own field when an adapter mapped it)."""
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in ("text", "json_object", "json_schema"):
        raise ApiError(f"{field} type must be text, json_object, or json_schema.", field)
    if kind != "chat":
        raise ApiError(f"{field} is supported by chat completions only.", field)
    if value["type"] == "text":
        if set(value) != {"type"}:
            raise ApiError(f"{field} text takes no other field.", field)
        return None
    if value["type"] == "json_object":
        if set(value) != {"type"}:
            raise ApiError(f"{field} json_object takes no other field.", field)
        return ("json_object", None)
    if set(value) != {"type", "json_schema"} or not isinstance(value["json_schema"], dict):
        raise ApiError(f"{field} json_schema needs a json_schema object.", field)
    return ("json_schema", answer_schema(value["json_schema"], field))


def answer_schema(spec, field):
    """The schema of a json_schema answer (name, description, schema, strict), after the checks of every schema."""
    if set(spec) - {"name", "description", "schema", "strict"}:
        raise ApiError(f"{field} json_schema supports name, description, schema, and strict only.", field)
    if "name" in spec and not isinstance(spec["name"], str):
        raise ApiError(f"{field} json_schema name must be text.", field)
    if "description" in spec and not isinstance(spec["description"], str):
        raise ApiError(f"{field} json_schema description must be text.", field)
    if "strict" in spec and type(spec["strict"]) is not bool and spec["strict"] is not None:
        raise ApiError(f"{field} json_schema strict must be true or false.", field)
    schema = spec.get("schema", {})  # enforced at every value of strict (STRICT_DEFAULT)
    if not isinstance(schema, dict):
        raise ApiError(f"{field} json_schema schema must be a JSON object.", field)
    budget = WorkBudget()
    try:
        budget.inspect(schema)
        check_schema(schema, schema, budget=budget)
    except (ValueError, TypeError, RecursionError):
        raise ApiError("The JSON schema is invalid or exceeds its size, depth, reference, or work limit.",
                       field) from None
    check_exact(schema, field)
    return schema


def raw_string(schema, root, budget):
    """True when the parser keeps this parameter as raw text (tool_output.argument_value)."""
    return types(schema, root, budget=budget) == {"string"}


def const(value):
    return dict(type="const_string", value=value)


def parameter_value(schema, root, field, budget, end_tag="</parameter>"):
    """The grammar of one parameter value, between `<parameter=KEY>\\n` and `\\n</parameter>\\n`. end_tag: the tag
    that ends a raw string value in the call form (Hy4-flash: hy4_format.ARG_VALUE_END)."""
    if schema is True or schema == {}:
        schema = {}
    resolved = resolve(schema, root, budget=budget)
    if raw_string(schema, root, budget):
        choices = None
        if isinstance(resolved, dict) and "const" in resolved:
            choices = [resolved["const"]]
        elif isinstance(resolved, dict) and "enum" in resolved:
            choices = resolved["enum"]
        for keyword in RAW_STRING_REFUSED if choices is None else ():
            if isinstance(resolved, dict) and keyword in resolved:
                refuse(f"{keyword} (string tool parameter)", field)
        if choices is not None:
            if any(not isinstance(item, str) or end_tag in item for item in choices):
                # The call form cannot hold the value; the other values can still match (not constraint_impossible).
                raise ApiError("A string tool parameter enum must hold text without a parameter end tag.", field,
                               code="constraint_keyword")
            return dict(type="or", elements=[const(item) for item in dict.fromkeys(choices)]) if len(choices) > 1 \
                else const(choices[0])
        return None  # raw text: the tag's any_text content
    sub = dict(schema) if isinstance(schema, dict) else {}
    # References resolve against the tool's parameters object: keep its definitions beside the value's schema (the
    # allow-list refuses definitions below the root, so each reference binds as in the whole schema). Only a value
    # with a reference gets them: a key beside the plain {"type": "array"} closes it in the converter's strict mode.
    for key in ("$defs", "definitions"):
        if key in root and key not in sub and has_ref(sub):
            sub[key] = root[key]
    return dict(type="json_schema", json_schema=grammar_schema(sub), max_whitespace_cnt=MAX_WHITESPACE)


def has_ref(schema):
    if isinstance(schema, dict):
        return "$ref" in schema or any(has_ref(value) for value in schema.values())
    if isinstance(schema, list):
        return any(has_ref(value) for value in schema)
    return False


def parameter_tag(key, value):
    begin, end = f"<parameter={key}>\n", "\n</parameter>\n"
    if value is None:
        return dict(type="tag", begin=begin, content=dict(type="any_text", excludes=["</parameter>"]), end=end)
    return dict(type="sequence", elements=[const(begin), value, const(end)])


def tool_root(parameters, budget):
    """The object schema of a strict tool's arguments (its root reference resolved), after the keyword checks."""
    resolved = resolve(parameters, parameters, budget=budget)
    if resolved is False:
        raise ApiError("A strict tool's parameters schema is false: no call matches it.", "tools",
                       code="constraint_impossible")
    resolved = {} if resolved is True else resolved
    check_exact(parameters, "tools", tool=True)
    for schema in (parameters, resolved):
        for keyword in TOOL_ROOT_REFUSED:
            if keyword in schema:
                refuse(f"{keyword} (strict tool parameters root)", "tools")
    declared = types(resolved, parameters, budget=budget)
    if declared and "object" not in declared:
        raise ApiError("A strict tool's parameters schema must describe a JSON object.", "tools",
                       code="constraint_impossible")
    return resolved


def call_format(tool, strict):
    """One call of `tool` in the checkpoint's form. A tool that is not strict takes any parameter text."""
    function = tool["function"]
    begin, end = f"<tool_call>\n<function={function['name']}>\n", "</function>\n</tool_call>"
    if not strict:
        return dict(type="tag", begin=begin, content=dict(type="any_text", excludes=["</tool_call>"]), end=end)
    parameters = function.get("parameters", dict(type="object", properties={}))
    budget = WorkBudget()
    root = parameters
    resolved = tool_root(parameters, budget)
    properties = resolved.get("properties", {})
    required = set(resolved.get("required", []))
    if required - set(properties):
        # The call grammar writes only the defined parameters, but such a call can match (not constraint_impossible).
        raise ApiError("A strict tool requires a parameter that it does not define.", "tools",
                       code="constraint_keyword")
    elements = []
    for key, schema in properties.items():
        # The boolean schema false forbids the parameter: the call never holds it.
        if resolve(schema, root, budget=budget) is False:
            if key in required:
                raise ApiError("A strict tool requires a parameter whose schema is false.", "tools",
                               code="constraint_impossible")
            continue
        element = parameter_tag(key, parameter_value(schema, root, "tools", budget))
        elements.append(element if key in required else dict(type="optional", content=element))
    content = dict(type="sequence", elements=elements) if elements else const("")
    return dict(type="tag", begin=begin, content=content, end=end)


def enforced(request_tools):
    return any(tool["function"].get("strict") is True for tool in request_tools)


def violation(request):
    """True when a tool-call fallback of `request` is an engine fault (log code constraint_violation): the grammar
    is on and every tool that it lets the model call is strict, so each call parses and matches its schema."""
    tools, choice = getattr(request, "tools", ()), getattr(request, "tool_choice", "auto")
    if getattr(request, "constraint", None) is None or not tools or choice == "none":
        return False
    if isinstance(choice, dict):
        tools = [tool for tool in tools if tool["function"]["name"] == choice["function"]["name"]]
    return bool(tools) and all(tool["function"].get("strict") is True for tool in tools)


def tool_format(tools, choice, parallel):
    """The call grammar for tool_choice required or named, or the triggered calls for auto; None when not enforced."""
    if not tools or choice == "none":
        return None
    if isinstance(choice, dict):
        name = choice["function"]["name"]
        tool = next(tool for tool in tools if tool["function"]["name"] == name)
        return call_format(tool, tool["function"].get("strict") is True)
    calls = [call_format(tool, tool["function"].get("strict") is True) for tool in tools]
    one = calls[0] if len(calls) == 1 else dict(type="or", elements=calls)
    if choice == "required":
        if not parallel:
            return one
        return dict(type="sequence", elements=[one, dict(type="star", content=dict(
            type="sequence", elements=[const("\n"), one]))])
    return ("auto", calls)


def constraint(body, kind, tools, choice, parallel, field=FORMAT_FIELD, form=None):
    """The engine constraint of a request: None, or dict(kind=..., spec=JSON text). The engine adds the phase.
    field: the place that an error of body's response_format names. form "hy4": the Hy4-flash call tags
    (server/hy4_structured.py); None: the checkpoint's Qwen form."""
    answer = response_format(body.get("response_format"), kind, field)
    calls = None
    if kind == "chat" and tools and choice != "none" and (answer is not None or choice != "auto" or enforced(tools)):
        if form == "hy4":
            from .hy4_structured import tool_format as hy4_tool_format
            calls = ("hy4", hy4_tool_format(tools, choice, parallel))
        else:
            calls = tool_format(tools, choice, parallel)
    # The grammar limits of every schema that goes into this request's grammar (server/grammar_limits.py).
    budget = GrammarBudget()
    if answer is not None and answer[0] == "json_schema":
        budget.check(answer[1], field)
    if calls is not None:
        named = choice["function"]["name"] if isinstance(choice, dict) else None
        for tool in tools:
            if tool["function"].get("strict") is True and named in (None, tool["function"]["name"]):
                budget.check(tool["function"].get("parameters", {}), "tools")
    if calls is None:
        if answer is None:
            return None
        if answer[0] == "json_object":
            return dict(kind="json_object", spec="")
        return dict(kind="json_schema", spec=compact(grammar_schema(answer[1])))
    if isinstance(calls, tuple) and calls[0] == "hy4":
        fmt = calls[1]
        if answer is not None and fmt["type"] == "triggered_tags":
            # auto with an answer format: the JSON answer, or text and then at least one tool block.
            schema = {"type": "object"} if answer[0] == "json_object" else answer[1]
            fmt = dict(type="or", elements=[
                dict(type="json_schema", json_schema=grammar_schema(schema), max_whitespace_cnt=MAX_WHITESPACE),
                fmt | dict(at_least_one=True)])
    elif isinstance(calls, tuple):
        # auto: free text (or the JSON answer), then calls that open at `<tool_call>`. tool_output.py takes text
        # before the first call only, so the output ends with the calls: one call, or with parallel calls more calls
        # each after "\n", and no text after or between them.
        _, tags = calls
        one = tags[0] if len(tags) == 1 else dict(type="or", elements=tags)
        more = [dict(type="star", content=dict(type="sequence", elements=[const("\n"), one]))] if parallel else []

        def text_then_calls(at_least_one):
            first = dict(type="triggered_tags", triggers=["<tool_call>"], tags=tags, at_least_one=at_least_one,
                         stop_after_first=True)
            return dict(type="sequence", elements=[first, *more]) if more else first
        if answer is not None:
            schema = {"type": "object"} if answer[0] == "json_object" else answer[1]
            fmt = dict(type="or", elements=[
                dict(type="json_schema", json_schema=grammar_schema(schema), max_whitespace_cnt=MAX_WHITESPACE),
                text_then_calls(True)])
        else:
            fmt = text_then_calls(False)
    else:
        fmt = calls
    return dict(kind="structural_tag", spec=compact(dict(type="structural_tag", format=fmt)))


def engine_constraint(request):
    """The constraint object of the engine line, with the phase: a chat request with thinking on starts in reasoning."""
    if request.constraint is None:
        return None
    reasoning = 1 if request.kind == "chat" and request.enable_thinking else 0
    return dict(kind=request.constraint["kind"], spec=request.constraint["spec"], reasoning=reasoning)


# Fixed engine refusals of a constrained open (engine/runtime/constraint*.h) and the client's 400 message.
ENGINE_CODES = {
    "constraint_compile": "The response format or strict tool schema does not compile to a grammar.",
    "constraint_kind": "The structured output kind is not supported.",
    "constraint_spec": "The structured output spec is invalid.",
    "constraint_unsupported": "This engine has no structured output.",
    "constraint_stop": "A structured output request must stop on the end-of-turn token.",
    "constraint_replay": "The output so far does not match the structured output constraint.",
    "constraint_from": "The structured output start is outside the prompt.",
    "constraint_vocab": "The engine could not read the tokenizer for structured output.",
    "constraint_scoring": "logprobs is not supported with structured output.",
}


# The compile helper's refusals (engine/runtime/constraint_helper.h HelperMessage): the client learns which bound
# its schema hit. The real bounds are the helper's deadline and memory; the count limits of grammar_limits.py come first.
ENGINE_TEXTS = {
    "constraint_limit: the grammar compile exceeds kHelperDeadlineMs = 5000":
        "The response format or strict tool schema exceeds the grammar compile time limit (5 s).",
    "constraint_limit: the grammar compile exceeds kHelperMemoryBytes = 2 GiB":
        "The response format or strict tool schema exceeds the grammar compile memory limit (2 GiB).",
    "constraint_limit: the compiled grammar exceeds kMaxGrammarBytes = 128 MiB":
        "The response format or strict tool schema exceeds the compiled grammar size limit (128 MiB).",
    "constraint_compile: the grammar compiler stopped on this schema":
        "The grammar compiler stopped on this response format or strict tool schema.",
    "constraint_spec: the spec exceeds kMaxSpecBytes = 1 MiB":
        "The structured output spec exceeds its size limit (1 MiB).",
    "constraint_busy: the grammar compile queue is full (kCompileQueue = 8)":
        "The engine is compiling too many structured output grammars. Retry the request.",
}
# An open of a spec that the engine still compiles is refused with this code (engine/runtime/constraint_vocab.h
# ServeConstraintFactory): the worker opens it again every COMPILE_POLL_SECONDS. The engine bounds each compile
# (5 s), but the first one also starts the helper (up to 60 s) and the queue holds 8, so the wait stops after
# COMPILE_WAIT_SECONDS with COMPILE_TIMEOUT.
PENDING_CODE = "constraint_pending"
COMPILE_POLL_SECONDS = 0.02
COMPILE_WAIT_SECONDS = 120.0
COMPILE_TIMEOUT = "The engine did not compile the structured output grammar in time. Retry the request."


# A grammar with no legal token in the middle of an output (engine/runtime/constraint.h kConstraintDeadEndMessage):
# the engine ends only that request (one recoverable error event, the slot is free). The client gets a 503 with this
# message after the tokens before it; the peers of the command go on (batch_backend._command).
DEAD_END_TEXT = "constraint_dead_end: the grammar has no legal token"
DEAD_END = "The structured output grammar has no legal next token. Check the schema, or retry the request."


def engine_refusal(message):
    """The client's message for an engine refusal that names a constraint code, else None."""
    if not isinstance(message, str):
        return None
    if message in ENGINE_TEXTS:
        return ENGINE_TEXTS[message]
    code = message.split(":", 1)[0]
    return ENGINE_CODES.get(code)


# A full compile queue is load, not a client fault: 503, so the client retries. Every other refusal is a 400.
BUSY_CODE = "constraint_busy"


def refusal_status(message):
    """The HTTP status of an engine refusal of a constrained open (engine_refusal gives its message)."""
    return 503 if isinstance(message, str) and message.split(":", 1)[0] == BUSY_CODE else 400


def pending(message):
    """True for the engine's refusal of an open whose grammar still compiles."""
    return isinstance(message, str) and message.split(":", 1)[0] == PENDING_CODE
