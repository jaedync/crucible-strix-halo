"""The Hy4-flash call grammar: xgrammar structural tags in the Hy4 tag form (server/hy4_format.py).

The form follows the Hy4 structural tag of another engine: the calls sit
back-to-back with no separator inside <tool_calls>...</tool_calls>. As server/structured.py call_format does for the
Qwen form, a strict tool's value gets its schema's grammar: raw text for a string (the template writes it as is),
JSON for another type (the template's tojson). Another engine leaves each value free; this is the stricter choice.
"""
from . import hy4_format as hy4
from .api import ApiError
from .structured import const, parameter_value, tool_root
from .tool_schema import WorkBudget, resolve


def argument_tag(key, value):
    begin = hy4.ARG_KEY + key + hy4.ARG_KEY_END + hy4.ARG_VALUE
    if value is None:
        return dict(type="tag", begin=begin, content=dict(type="any_text", excludes=[hy4.ARG_VALUE_END]),
                    end=hy4.ARG_VALUE_END)
    return dict(type="sequence", elements=[const(begin), value, const(hy4.ARG_VALUE_END)])


def call_format(tool, strict):
    """One call of `tool`. A tool that is not strict takes any argument text."""
    function = tool["function"]
    begin = hy4.CALL + function["name"]
    if not strict:
        return dict(type="tag", begin=begin, content=dict(type="any_text", excludes=[hy4.CALL_END]), end=hy4.CALL_END)
    root = function.get("parameters", dict(type="object", properties={}))
    budget = WorkBudget()
    resolved = tool_root(root, budget)
    properties = resolved.get("properties", {})
    required = set(resolved.get("required", []))
    if required - set(properties):
        raise ApiError("A strict tool requires a parameter that it does not define.", "tools",
                       code="constraint_keyword")
    elements = []
    for key, schema in properties.items():
        if resolve(schema, root, budget=budget) is False:
            if key in required:
                raise ApiError("A strict tool requires a parameter whose schema is false.", "tools",
                               code="constraint_impossible")
            continue
        element = argument_tag(key, parameter_value(schema, root, "tools", budget, hy4.ARG_VALUE_END))
        elements.append(element if key in required else dict(type="optional", content=element))
    content = dict(type="sequence", elements=elements) if elements else const("")
    return dict(type="tag", begin=begin, content=content, end=hy4.CALL_END)


def tool_format(tools, choice, parallel):
    """The format for tool_choice auto, required or named; None for none or no tools."""
    if not tools or choice == "none":
        return None
    named = isinstance(choice, dict)
    if named:
        tools = [tool for tool in tools if tool["function"]["name"] == choice["function"]["name"]]
    tags = [call_format(tool, tool["function"].get("strict") is True) for tool in tools]
    calls = dict(type="tags_with_separator", tags=tags, separator="", at_least_one=True,
                 stop_after_first=named or not parallel)
    if choice == "auto":
        block = dict(type="tag", begin=hy4.CALLS, content=calls, end=hy4.CALLS_END)
        return dict(type="triggered_tags", triggers=[hy4.CALLS], tags=[block], at_least_one=False,
                    stop_after_first=True)
    return dict(type="sequence", elements=[const(hy4.CALLS), calls, const(hy4.CALLS_END)])
