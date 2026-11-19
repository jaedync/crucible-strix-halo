"""Stream tool calls while the model writes them (item 4).

The id and name go out when the function tag closes and the name passes the request's tool policy. A string-only
parameter streams as JSON-escaped text; any other parameter goes out when it closes and validates. The closing
brace waits for the whole call to validate. For a valid call, the joined argument deltas are byte-equal to the
arguments that ToolOutput.finish returns. Once this stream finds a problem it sends nothing more; ToolOutput.finish
then decides the response from the complete text. A value of a strict tool ends where the parser ends it
(tool_frame.value_close): a JSON string can hold `</parameter>`, and any strict value can hold `</tool_call>`.
"""
import json
import uuid
from .api import ApiError
from .schema_limits import WorkBudget
from .tool_frame import OPEN, raw_string, value_close
from .tool_schema import resolve, types

START, END = "<tool_call>", "</tool_call>"
FUNCTION_OPEN, FUNCTION_CLOSE = "<function=", "</function>"
PARAMETER_OPEN, PARAMETER_CLOSE = "<parameter=", "</parameter>"
# A string with any of these rules is checked whole before it goes out.
STRING_RULES = ("enum", "const", "minLength", "maxLength", "anyOf", "allOf", "oneOf", "not", "$ref")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def held(text, tags):
    """Length of the longest suffix of text that can still grow into one of tags."""
    for length in range(min(len(text), max(len(tag) for tag in tags) - 1), 0, -1):
        if any(tag.startswith(text[-length:]) for tag in tags):
            return length
    return 0


class Stopped(Exception):
    pass


class ToolCallStream:
    def __init__(self, request, definitions, whole=False):
        self.request, self.definitions = request, definitions
        self.whole = whole  # tool_output.whole_numbers on the values (--extend-tool-tails)
        self.calls = []  # dict(id, name, sent): sent is the argument text already sent
        self.pos = 0
        self.state = "between"
        self.stopped = False
        self.budget = WorkBudget()

    def advance(self, text):
        """Deltas for the new part of text (everything from the first <tool_call>)."""
        deltas = []
        if self.stopped:
            return deltas
        try:
            while getattr(self, "_" + self.state)(text, deltas):
                pass
        except (Stopped, ApiError, ValueError, TypeError, RecursionError):
            self.stopped = True
        return deltas

    # Each state returns True when it moved on and the loop should run the next state.
    def _between(self, text, deltas):
        rest = text[self.pos:]
        stripped = rest.lstrip()
        if not stripped:
            return False
        if stripped.startswith(START):
            self.pos += len(rest) - len(stripped) + len(START)
            self.call_start = self.pos
            self.state = "function"
            return True
        if START.startswith(stripped):
            return False
        raise Stopped

    def _function(self, text, deltas):
        rest = text[self.pos:]
        stripped = rest.lstrip()
        if not stripped:
            return False
        if not stripped.startswith(FUNCTION_OPEN):
            if FUNCTION_OPEN.startswith(stripped):
                return False
            raise Stopped
        close = stripped.find(">")
        if close < 0:
            if any(char in stripped[len(FUNCTION_OPEN):] for char in "<\r\n"):
                raise Stopped
            return False
        name = stripped[len(FUNCTION_OPEN):close]
        if not name or any(char in name for char in "<\r\n"):
            raise Stopped
        self.check_policy(name)
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
        self.pos += len(rest) - len(stripped) + close + 1
        self.state = "parameters"
        return True

    def check_policy(self, name):
        """The checks that can fail before the first delta: the name and the request's tool policy."""
        choice = self.request.tool_choice
        if (name not in self.definitions or choice == "none" or len(self.calls) >= 128 or
                (isinstance(choice, dict) and choice["function"]["name"] != name) or
                (not self.request.parallel_tool_calls and self.calls)):
            raise Stopped

    def _parameters(self, text, deltas):
        rest = text[self.pos:]
        stripped = rest.lstrip()
        if not stripped:
            return False
        skip = len(rest) - len(stripped)
        if stripped.startswith(FUNCTION_CLOSE):
            self.pos += skip + len(FUNCTION_CLOSE)
            self.state = "call_end"
            return True
        if not stripped.startswith(PARAMETER_OPEN):
            if PARAMETER_OPEN.startswith(stripped) or FUNCTION_CLOSE.startswith(stripped):
                return False
            raise Stopped
        close = stripped.find(">")
        if close < 0:
            if any(char in stripped[len(PARAMETER_OPEN):] for char in "<\r\n"):
                raise Stopped
            return False
        key = stripped[len(PARAMETER_OPEN):close]
        if not key or any(char in key for char in "<\r\n") or key in self.keys:
            raise Stopped
        self.keys.add(key)
        self.key = key
        self.pos += skip + close + 1
        self.value_start = self.pos
        self.value_sent = 0
        # A strict value in the grammar's form: raw is its reader's mode; scan is where the next end tag search starts.
        self.raw = (raw_string(self.properties[key], self.schema, self.budget)
                    if self.strict and key in self.properties else None)
        self.scan = 0
        self.state = "string" if self.streams_as_string(key) else "value"
        if self.state == "string":
            self.send(deltas, ("{" if len(self.keys) == 1 else ",") + dumps(key) + ':"')
        return True

    def streams_as_string(self, key):
        schema = self.properties.get(key)
        if not isinstance(schema, dict) or any(rule in schema for rule in STRING_RULES):
            return False
        return types(schema, self.schema, budget=self.budget) == {"string"}

    def value_end(self, raw):
        """Where the value ends, or None while open. A strict value in the grammar's form ends as the final parser
        reads it (tool_frame.value_close). Any other value ends at its first </parameter>, and the final parser cuts
        its call at the first </tool_call>."""
        if self.raw is not None:
            # The value can close only at a </parameter>: read it again only when a new one arrives, so a long value
            # costs one read per tag, not one per piece. Off-form text found later only waits; finish decides it.
            tag = raw.find(PARAMETER_CLOSE, self.scan)
            if tag < 0:
                return None
            self.scan = tag + len(PARAMETER_CLOSE)
            close = value_close(raw, 0, self.raw)
            if close is not None:
                return None if close == OPEN else close
        close, end = raw.find(PARAMETER_CLOSE), raw.find(END)
        if end >= 0 and (close < 0 or end < close):
            raise Stopped
        return close if close >= 0 else None

    def _string(self, text, deltas):
        raw = text[self.value_start:]
        close = self.value_end(raw)
        if close is None:
            # Hold text that can still become a closing tag, and a newline that the closing tag would strip.
            ready = raw[:len(raw) - held(raw, (PARAMETER_CLOSE, END))]
            if ready.endswith("\n"):
                ready = ready[:-1]
            if not ready:
                return False
            value = ready[1:] if ready.startswith("\n") else ready
        else:
            value = raw[:close]
            value = value[1:] if value.startswith("\n") else value
            value = value[:-1] if value.endswith("\n") else value
        if len(value) > self.value_sent:
            self.send(deltas, dumps(value[self.value_sent:])[1:-1])
            self.value_sent = len(value)
        if close is None:
            return False
        self.send(deltas, '"')
        self.pos = self.value_start + close + len(PARAMETER_CLOSE)
        self.state = "parameters"
        return True

    def _value(self, text, deltas):
        from .tool_output import argument_value
        raw = text[self.value_start:]
        close = self.value_end(raw)
        if close is None:
            return False
        # As the final parser: a schema-only failure keeps the value (tool_output.UNCHECKED), so the stream goes on.
        parsed = argument_value(raw[:close], self.properties.get(self.key, {}), self.schema, self.budget,
                                None if self.strict else [], self.whole)
        self.send(deltas, ("{" if len(self.keys) == 1 else ",") + dumps(self.key) + ":" + dumps(parsed))
        self.pos = self.value_start + close + len(PARAMETER_CLOSE)
        self.state = "parameters"
        return True

    def _call_end(self, text, deltas):
        from .tool_output import parse_function
        rest = text[self.pos:]
        end = rest.find(END)
        if end < 0:
            if rest.strip() and not END.startswith(rest.strip()):
                raise Stopped
            return False
        if rest[:end].strip():
            raise Stopped
        call = parse_function(text[self.call_start:self.pos + end], self.definitions, self.budget, [],
                              whole=self.whole)
        arguments = call["function"]["arguments"]
        sent = self.calls[-1]["sent"]
        if not arguments.startswith(sent):
            raise Stopped
        self.send(deltas, arguments[len(sent):])
        self.pos += end + len(END)
        self.state = "between"
        return True

    def send(self, deltas, piece):
        if not piece:
            return
        call = self.calls[-1]
        call["sent"] += piece
        deltas.append(dict(tool_calls=[dict(index=len(self.calls) - 1, function=dict(arguments=piece))]))
