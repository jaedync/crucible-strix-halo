"""Read a Hy4-flash tool block by its structure (the Hy4 form of server/tool_frame.py).

The call grammar of a strict tool (server/hy4_structured.py) gives a JSON value its schema's grammar, and a JSON
string can hold any tag: </arg_value>, </tool_call>, </tool_calls>. With the engine's Hy4 vocabulary each tag is
grammar text (engine/runtime/constraint_vocab.h Hy4ConstraintFamily), so the model can write one there. So the end
of a strict value is its closing bracket or quote, and the end of a strict call is not its first </tool_call>. A raw
string ends at its first </arg_value> (the grammar excludes that tag in it). A call that is not strict, or not in the
grammar's form, ends at its first </tool_call>, as other engines hy_v4_tool_parser reads it.
"""
from .hy4_format import ARG_KEY, ARG_KEY_END, ARG_VALUE, ARG_VALUE_END, CALL, CALL_END, CALLS, CALLS_END
from .tool_frame import OPEN, SCALAR, json_end, raw_string, skip, strict_parameters


def value_close(text, pos, raw):
    """The index of the </arg_value> that closes the value at text[pos:], OPEN, or None when the text is not in the
    grammar's form. raw: the value is raw text (tool_frame.raw_string)."""
    if raw:
        return text.find(ARG_VALUE_END, pos)
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
    if text.startswith(ARG_VALUE_END, close):
        return close
    return OPEN if ARG_VALUE_END.startswith(text[close:]) else None


def name_end(text, pos):
    """(index, tag) of the first <arg_key> or </tool_call> after the name at text[pos:], or (-1, None)."""
    found = [(index, tag) for tag in (ARG_KEY, CALL_END) if (index := text.find(tag, pos)) >= 0]
    return min(found, default=(-1, None))


def strict_end(text, pos, definitions, budget):
    """The index of the </tool_call> that closes a strict call whose name is at text[pos:], OPEN, or None when the
    call is not a strict call in the grammar's form."""
    index, tag = name_end(text, pos)
    strict = strict_parameters(text[pos:index].strip(), definitions, budget) if tag is not None else None
    if strict is None:
        return None
    (properties, root) = strict
    while True:
        index = skip(text, index)
        if text.startswith(CALL_END, index):
            return index
        if not text.startswith(ARG_KEY, index):
            rest = text[index:]
            return OPEN if ARG_KEY.startswith(rest) or CALL_END.startswith(rest) else None
        key_end = text.find(ARG_KEY_END, index + len(ARG_KEY))
        if key_end < 0:
            return OPEN if "<" not in text[index + len(ARG_KEY):] else None
        key = text[index + len(ARG_KEY):key_end]
        value = key_end + len(ARG_KEY_END)
        if not text.startswith(ARG_VALUE, value):
            return OPEN if ARG_VALUE.startswith(text[value:]) else None
        if key not in properties:
            return None
        close = value_close(text, value + len(ARG_VALUE), raw_string(properties[key], root, budget))
        if close is None or close == OPEN:
            return close
        index = close + len(ARG_VALUE_END)


def call_end(text, start, definitions, budget):
    """The index of the </tool_call> that closes the call at text[start:] (it starts with <tool_call>), or OPEN."""
    end = strict_end(text, start + len(CALL), definitions, budget)
    return text.find(CALL_END, start + len(CALL)) if end is None else end


def block_end(text, definitions, budget):
    """For a tool block (text starts with <tool_calls>): the index of its </tool_calls>, OPEN while the text can still
    become a block, or None for other text between the calls."""
    pos = skip(text, len(CALLS))
    while True:
        if text.startswith(CALLS_END, pos):
            return pos
        if not text.startswith(CALL, pos):
            rest = text[pos:]
            return OPEN if CALLS_END.startswith(rest) or CALL.startswith(rest) else None
        end = call_end(text, pos, definitions, budget)
        if end == OPEN:
            return OPEN
        pos = skip(text, end + len(CALL_END))
