"""Stream Hy4-flash tool calls while the model writes them (the Hy4 tag form of server/hy4_format.py).

The contract is the one of server/tool_stream.py: the id and name go out when the name ends (at the first <arg_key>
or at </tool_call>) and pass the request's tool policy. A string-only value streams as JSON-escaped text; another
value goes out when it closes and parses. The rest of the arguments goes out when </tool_call> arrives and the whole
call parses with hy4_format.parse_call, the final parser. For a valid block the joined deltas are byte-equal to the
arguments of ToolOutput.finish. At any text that the final parser can read in another way, the stream stops; finish
then decides the response from the complete text.

As other engines hy_v4_tool_parser, the stream reads the tags as text: each tag is one token, so a token never splits a tag.
A plain-text copy of a tag reads the same way in the final parser, so this stream also holds a partial tag.
"""
import uuid
from .hy4_format import (ARG_KEY, ARG_KEY_END, ARG_VALUE, ARG_VALUE_END, CALL, CALL_END, CALLS, CALLS_END, TAGS,
                         parse_call)
from .hy4_frame import OPEN, value_close
from .tool_frame import raw_string
from .tool_output import argument_value
from .tool_schema import resolve
from .tool_stream import Stopped, ToolCallStream, dumps, held

# A value that is not a strict one ends at its first </arg_value>; a </tool_call> before it ends the call there in the
# final parser (hy4_frame.call_end), so the stream stops at it.
LONGEST = max(len(CALL_END), len(ARG_VALUE_END))


def first_tag(text, tags):
    """(index, tag) of the first of tags in text, or (-1, None)."""
    found = [(index, tag) for index, tag in ((text.find(tag), tag) for tag in tags) if index >= 0]
    return min(found, default=(-1, None))


class Hy4CallStream(ToolCallStream):
    """advance(text) reads the calls text, which starts with <tool_calls>."""

    def __init__(self, request, definitions, whole=False):
        super().__init__(request, definitions, whole)
        self.state = "open"

    def _open(self, text, deltas):
        if not text.startswith(CALLS):
            raise Stopped
        self.pos = len(CALLS)
        self.state = "between"
        return True

    def tag_ahead(self, text, tags):
        """(skip, tag) when the text at pos starts, after whitespace, with one of tags; None while it can still grow
        into one. Other text stops the stream."""
        rest = text[self.pos:]
        stripped = rest.lstrip()
        for tag in tags:
            if stripped.startswith(tag):
                return len(rest) - len(stripped), tag
        if not stripped or any(tag.startswith(stripped) for tag in tags):
            return None
        raise Stopped

    def _between(self, text, deltas):
        ahead = self.tag_ahead(text, (CALL, CALLS_END))
        if ahead is None:
            return False
        skip, tag = ahead
        self.pos += skip + len(tag)
        self.call_start = self.pos
        self.state = "function" if tag == CALL else "done"
        return True

    def _done(self, text, deltas):
        if text[self.pos:].strip():
            raise Stopped
        return False

    def _function(self, text, deltas):
        rest = text[self.pos:]
        index, tag = first_tag(rest, TAGS)
        if tag is None:
            # No advertised name is this long: stop, so a long text is not read again for each piece.
            if len(rest.lstrip()) > max(map(len, self.definitions), default=0) + LONGEST:
                raise Stopped
            return False
        name = rest[:index].strip()
        if tag not in (ARG_KEY, CALL_END) or not name:
            raise Stopped
        self.check_policy(name)
        self.start_call(name, deltas)
        self.pos += index
        self.state = "parameters"
        return True

    def start_call(self, name, deltas):
        definition = self.definitions[name]
        self.strict = definition.get("strict") is True
        self.schema = definition.get("parameters", {})
        resolved = resolve(self.schema, self.schema, budget=self.budget)
        self.properties = resolved.get("properties", {}) if isinstance(resolved, dict) else {}
        self.keys = set()
        call = dict(id="call_" + uuid.uuid4().hex, name=name, sent="")
        self.calls.append(call)
        deltas.append(dict(tool_calls=[dict(index=len(self.calls) - 1, id=call["id"], type="function",
                                            function=dict(name=name, arguments=""))]))

    def _parameters(self, text, deltas):
        ahead = self.tag_ahead(text, (ARG_KEY, CALL_END))
        if ahead is None:
            return False
        skip, tag = ahead
        if tag == CALL_END:
            self.end_call(text, self.pos + skip, deltas)
            return True
        self.pos += skip + len(tag)
        self.state = "key"
        return True

    def _key(self, text, deltas):
        rest = text[self.pos:]
        index = rest.find("<")
        if index < 0:
            return False
        if not rest.startswith(ARG_KEY_END + ARG_VALUE, index):
            if (ARG_KEY_END + ARG_VALUE).startswith(rest[index:]):
                return False
            raise Stopped
        key = rest[:index]
        if not key or key in self.keys:
            raise Stopped
        self.keys.add(key)
        self.key = key
        self.pos += index + len(ARG_KEY_END + ARG_VALUE)
        self.value_start, self.value_sent, self.scan = self.pos, 0, 0
        # A strict value in the grammar's form: raw is its reader's mode (hy4_frame.value_close).
        self.raw = (raw_string(self.properties[key], self.schema, self.budget)
                    if self.strict and key in self.properties else None)
        self.state = "string" if self.streams_as_string(key) else "value"
        if self.state == "string":
            self.send(deltas, ("{" if len(self.keys) == 1 else ",") + dumps(key) + ':"')
        return True

    def value_end(self, raw):
        """Where the value ends, or None while open. A strict value ends where the final parser ends it, and it is
        read again only when a new </arg_value> arrives. Another value ends at its first </arg_value>."""
        if self.raw is not None:
            tag = raw.find(ARG_VALUE_END, self.scan)
            if tag < 0:
                return None
            self.scan = tag + len(ARG_VALUE_END)
            close = value_close(raw, 0, self.raw)
            if close is None:
                raise Stopped  # off the grammar's form: finish decides
            return None if close == OPEN else close
        start = max(0, self.scan - LONGEST)
        close = raw.find(ARG_VALUE_END, start)
        if raw.find(CALL_END, start, close if close >= 0 else len(raw)) >= 0:
            raise Stopped
        self.scan = len(raw)
        return close if close >= 0 else None

    def held_tags(self):
        return (ARG_VALUE_END,) if self.raw is not None else (ARG_VALUE_END, CALL_END)

    def _string(self, text, deltas):
        raw = text[self.value_start:]
        close = self.value_end(raw)
        value = raw[:len(raw) - held(raw, self.held_tags())] if close is None else raw[:close]
        if len(value) > self.value_sent:
            self.send(deltas, dumps(value[self.value_sent:])[1:-1])
            self.value_sent = len(value)
        if close is None:
            return False
        self.send(deltas, '"')
        self.pos = self.value_start + close + len(ARG_VALUE_END)
        self.state = "parameters"
        return True

    def _value(self, text, deltas):
        raw = text[self.value_start:]
        close = self.value_end(raw)
        if close is None:
            return False
        # As parse_call: argument_value trims one newline at each end, so the raw value gets one of each.
        parsed = argument_value("\n" + raw[:close] + "\n", self.properties.get(self.key, {}), self.schema,
                                self.budget, None if self.strict else [], self.whole)
        self.send(deltas, ("{" if len(self.keys) == 1 else ",") + dumps(self.key) + ":" + dumps(parsed))
        self.pos = self.value_start + close + len(ARG_VALUE_END)
        self.state = "parameters"
        return True

    def end_call(self, text, end, deltas):
        call = parse_call(text[self.call_start:end], self.definitions, self.budget, [], whole=self.whole)
        arguments, sent = call["function"]["arguments"], self.calls[-1]["sent"]
        if not arguments.startswith(sent):
            raise Stopped
        self.send(deltas, arguments[len(sent):])
        self.pos = end + len(CALL_END)
        self.state = "between"
