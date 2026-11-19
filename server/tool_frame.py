"""Read a strict call by its structure (review round 3, finding 2).

The grammar of a strict tool (server/structured.py call_format) lets an argument hold `</tool_call>`: a raw string
excludes only `</parameter>`, and a JSON value can hold any tag inside a JSON string. So the end of a strict call is
not its first `</tool_call>`. Here a call in the grammar's form is read tag by tag: a raw string ends at its first
`</parameter>`, a JSON array, object or string at its closing bracket or quote. Any other text (a tool that is not
strict, a call off the grammar's form) keeps the first end tag, as before.
"""
import re
from .tool_schema import resolve, types

START, END = "<tool_call>", "</tool_call>"
FUNCTION = re.compile(r"<function=([^<>\r\n]+)>\s*")
PARAMETER = re.compile(r"<parameter=([^<>\r\n]+)>")
FUNCTION_CLOSE, PARAMETER_OPEN, PARAMETER_CLOSE = "</function>", "<parameter=", "</parameter>"
OPEN = -1  # the text ends inside the structure
WHITESPACE = " \t\n\r"
SCALAR = frozenset("0123456789+-.eEtruefalsn")


def skip(text, pos):
    while pos < len(text) and text[pos] in WHITESPACE:
        pos += 1
    return pos


def json_end(text, pos):
    """The index after the JSON array, object or string at text[pos], or OPEN. Brackets are counted, not matched:
    the parser's JSON check (tool_output.argument_value) reads the value itself."""
    depth, string, index = 0, False, pos
    while index < len(text):
        char = text[index]
        if string:
            if char == "\\":
                index += 1
            elif char == '"':
                string = False
                if depth == 0:
                    return index + 1
        elif char == '"':
            string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return OPEN


def raw_string(schema, root, budget):
    """True for a parameter that the parser keeps as raw text (tool_output.argument_value)."""
    return types(schema, root, budget=budget) == {"string"}


def value_close(text, pos, raw):
    """The index of the `</parameter>` that closes the value at text[pos:], OPEN, or None when the text is not in
    the grammar's form."""
    if raw:
        return text.find(PARAMETER_CLOSE, pos)
    start = skip(text, pos)
    if start == len(text):
        return OPEN
    if text[start] in '"[{':
        end = json_end(text, start)
        if end == OPEN:
            return OPEN
    else:  # a number, true, false or null: it holds no tag
        end = start
        while end < len(text) and text[end] in SCALAR:
            end += 1
    close = skip(text, end)
    if text.startswith(PARAMETER_CLOSE, close):
        return close
    return OPEN if PARAMETER_CLOSE.startswith(text[close:]) else None


def strict_parameters(name, definitions, budget):
    """The (properties, root) of a strict tool, or None for a tool that is not strict or not advertised."""
    definition = definitions.get(name)
    if definition is None or definition.get("strict") is not True:
        return None
    root = definition.get("parameters", {})
    resolved = resolve(root, root, budget=budget)
    return (resolved.get("properties", {}) if isinstance(resolved, dict) else {}), root


def strict_end(text, pos, definitions, budget):
    """The index of the `</tool_call>` that closes a strict call whose function tag is at text[pos:], OPEN, or None
    when the call is not a strict call in the grammar's form."""
    function = FUNCTION.match(text, skip(text, pos))
    strict = strict_parameters(function.group(1), definitions, budget) if function else None
    if strict is None:
        return None
    (properties, root), index = strict, function.end()
    while True:
        index = skip(text, index)
        rest = text[index:]
        if rest.startswith(FUNCTION_CLOSE):
            index = skip(text, index + len(FUNCTION_CLOSE))
            if text.startswith(END, index):
                return index
            return OPEN if END.startswith(text[index:]) else None
        parameter = PARAMETER.match(text, index)
        if parameter is None:
            name = rest[len(PARAMETER_OPEN):] if rest.startswith(PARAMETER_OPEN) else None
            partial = PARAMETER_OPEN.startswith(rest) or FUNCTION_CLOSE.startswith(rest) or (
                name is not None and not any(char in name for char in "<>\r\n"))
            return OPEN if partial else None
        if parameter.group(1) not in properties:
            return None
        close = value_close(text, parameter.end(), raw_string(properties[parameter.group(1)], root, budget))
        if close is None or close == OPEN:
            return close
        index = close + len(PARAMETER_CLOSE)


def call_end(text, start, definitions, budget):
    """The index of the `</tool_call>` that closes the call at text[start:] (it starts with `<tool_call>`), or OPEN."""
    end = strict_end(text, start + len(START), definitions, budget)
    return text.find(END, start + len(START)) if end is None else end
