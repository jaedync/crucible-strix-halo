"""Read an Anthropic-style tool call (`<invoke name="...">`) by its structure.

Models trained on that form, and agent clients such as the Hermes Agent CLI, write a call as:

    <invoke name="read">
    <parameter name="path">a.py</parameter>
    </invoke>

A block is the start tag, parameter tags with only whitespace between them, and the end tag. A value ends at its
first `</parameter>`; a value of a strict tool ends where tool_frame.value_close ends it, as in a `<tool_call>`.
tool_output.parse_function reads the block in the INVOKE form, so a block gets the same checks as a `<tool_call>`.
Text that leaves this form before its first parameter tag (prose that names the tag) is not a call. Text that leaves
it later is a malformed call: the output falls back to content, as for a malformed `<tool_call>`.
"""
import re
from collections import namedtuple
from .tool_frame import OPEN, PARAMETER_CLOSE, raw_string, strict_parameters, value_close

START, END = "<invoke", "</invoke>"
BROKEN = -2  # block_end: a block that has a parameter tag and then leaves the form (OPEN is -1)
NAME = r'"([^"<>\r\n]+)"'
OPEN_TAG = re.compile(r"<invoke\s+name=" + NAME + r"\s*>")
PARAMETER_TAG = re.compile(r"<parameter\s+name=" + NAME + r"\s*>")
SPACE = re.compile(r"\s*")  # the whitespace of str.lstrip, as parse_function skips it
# The tags of one call form: the function start tag (its name is group 1), the parameter tag, the function end tag.
Form = namedtuple("Form", "function parameter close")
INVOKE = Form(OPEN_TAG, PARAMETER_TAG, END)


def growing(rest, word):
    """True when rest (it holds no complete tag) can still grow into `<word name="...">`."""
    tag = "<" + word
    if tag.startswith(rest):
        return True
    return re.fullmatch(re.escape(tag) + r'\s+(?:n|na|nam|name|name=|name="[^"<>\r\n]*|name=' + NAME + r"\s*)?",
                        rest) is not None


def block_end(text, pos, definitions, budget, final=False):
    """The index of the `</invoke>` that closes the block at text[pos:] (it starts with `<invoke`), OPEN while the
    text can still grow into a block, BROKEN for a malformed call, or None for prose. final: the output has ended, so
    a block that is still open is prose before its first parameter tag and a malformed call after it."""
    match = OPEN_TAG.match(text, pos)
    if match is None:
        return OPEN if growing(text[pos:], "invoke") and not final else None
    strict = strict_parameters(match.group(1), definitions, budget)
    index, prose = match.end(), None
    while True:
        index = SPACE.match(text, index).end()
        if text.startswith(END, index):
            return index
        parameter = PARAMETER_TAG.match(text, index)
        if parameter is None:
            rest = text[index:]
            return OPEN if (END.startswith(rest) or growing(rest, "parameter")) and not final else prose
        prose, close = BROKEN, None
        if strict is not None and parameter.group(1) in strict[0]:
            properties, root = strict
            close = value_close(text, parameter.end(), raw_string(properties[parameter.group(1)], root, budget))
        if close is None:
            close = text.find(PARAMETER_CLOSE, parameter.end())
        if close == OPEN:
            return BROKEN if final else OPEN
        index = close + len(PARAMETER_CLOSE)
