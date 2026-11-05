"""Parse the checkpoint's typed XML-like calls into strict OpenAI tool calls."""
import json
import re
import uuid
from .api import ApiError
from .json_values import json_object, json_value
from .tool_schema import matches, types, resolve, WorkBudget
from .tool_stream import ToolCallStream

START = "<tool_call>"
END = "</tool_call>"
FUNCTION = re.compile(r"<function=([^<>\r\n]+)>\s*")
PARAMETER = re.compile(r"<parameter=([^<>\r\n]+)>")

def malformed(message, code):
    """code is a fixed reason (usage tool_call_invalid); the message stays server text, never model text."""
    error = ApiError("Malformed model tool call: " + message, status=502)
    error.code = code
    return error

# Design decision (Oct 8): a complete, well-formed call that fails only these checks goes out as a tool call. The
# client checks it against its own schema and tells the model the exact error. Structural and policy failures keep
# the content fallback. The record keeps the reason code only.
UNCHECKED = ("schema_mismatch", "parameter_json", "unknown_function")
# A passed-through name must be one that the next request can send back in its history (tool_requests.NAME).
PASS_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def unchecked(soft, code, message):
    """Record a schema-only failure when the call may pass through (soft is a list), else raise it."""
    if soft is None:
        raise malformed(message, code)
    soft.append(code)


def argument_value(raw, schema, root, budget, soft=None):
    """The parsed parameter. With soft (a list), a value that fails its schema keeps its parsed form, and a value
    that is not JSON for a non-string type stays the raw string; the codes go to soft."""
    value = raw[1:] if raw.startswith("\n") else raw
    value = value[:-1] if value.endswith("\n") else value
    declared = types(schema, root, budget=budget)
    if declared == {"string"}:
        parsed = value
    else:
        try:
            parsed = json.loads(value, object_pairs_hook=json_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (ValueError, RecursionError):
            if "string" not in declared and declared:
                unchecked(soft, "parameter_json", "a parameter is not valid JSON for its declared type.")
                return json_value(value, "tool_calls", 1)
            parsed = value
    try:
        # Depth 1: the value sits inside the arguments object, as the next request's history check counts it.
        parsed = json_value(parsed, "tool_calls", 1)
    except ApiError:
        if soft is None:
            raise
        soft.append("parameter_json")  # for example 1e999: JSON, but not a finite value; the raw text goes out
        return value
    if not matches(parsed, schema, root, budget=budget):
        unchecked(soft, "schema_mismatch", "a parameter does not match its advertised schema.")
    return parsed

def parse_function(block, definitions, budget, soft=None):
    """One call. soft=None checks the schema strictly; a list lets schema-only failures pass (see UNCHECKED). A tool
    with strict: true always checks strictly: its client expects only valid arguments."""
    match = FUNCTION.match(block.lstrip())
    if not match:
        raise malformed("expected a function tag.", "function_tag")
    body = block.lstrip()
    name = match.group(1)
    if name not in definitions:
        if not PASS_NAME.match(name):
            soft = None
        unchecked(soft, "unknown_function", "the function is not advertised.")
    elif definitions[name].get("strict") is True:
        soft = None
    schema = definitions.get(name, {}).get("parameters", {})
    resolved = resolve(schema, schema, budget=budget)
    properties = resolved.get("properties", {}) if isinstance(resolved, dict) else {}
    arguments, position, first = {}, match.end(), len(soft) if soft is not None else 0
    while not body[position:].lstrip().startswith("</function>"):
        position += len(body[position:]) - len(body[position:].lstrip())
        parameter = PARAMETER.match(body, position)
        if not parameter:
            raise malformed("expected a parameter tag or function end.", "parameter_tag")
        key = parameter.group(1)
        if key in arguments:
            raise malformed("parameter names must be unique.", "duplicate_parameter")
        close = body.find("</parameter>", parameter.end())
        if close < 0:
            raise malformed("a parameter end tag is missing.", "incomplete_call")
        arguments[key] = argument_value(body[parameter.end():close], properties.get(key, {}), schema, budget, soft)
        position = close + len("</parameter>")
    rest = body[position:].lstrip()[len("</function>"):]
    if rest.strip():
        raise malformed("unexpected text after the function end.", "unexpected_text")
    # A parameter that already failed also fails the object; one code per cause.
    if not matches(arguments, schema, schema, budget=budget) and (soft is None or len(soft) == first):
        unchecked(soft, "schema_mismatch", "arguments do not match the advertised object schema.")
    return dict(id="call_" + uuid.uuid4().hex, type="function", function=dict(
        name=name, arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))

class ToolOutput:
    def __init__(self, request, stream=False):
        self.request = request
        self.definitions = {tool["function"]["name"]: tool["function"] for tool in request.tools}
        self.pending = ""
        self.content = ""
        self.calls_text = None
        self.fallback = False  # the output did not parse as valid calls and is returned as content
        self.invalid = None  # the fixed reason code of a fallback
        self.unchecked = []  # reason codes of calls that went out with a schema-only failure (UNCHECKED)
        # Opt-out with --stream-tool-calls off: calls then go out whole when the output ends.
        self.stream = ToolCallStream(request, self.definitions) if stream else None
        self.deltas = []

    def open_call(self):
        """True inside a <tool_call> block that the model has not closed yet."""
        return self.calls_text is not None and self.calls_text.count(START) > self.calls_text.count(END)

    def take_deltas(self):
        """Tool-call deltas that are ready to send now (stream mode)."""
        deltas, self.deltas = self.deltas, []
        return deltas

    def advance(self):
        if self.stream is not None:
            self.deltas += self.stream.advance(self.calls_text)

    def feed(self, text):
        if self.calls_text is not None:
            self.calls_text += text
            self.advance()
            return ""
        combined = self.pending + text
        marker = combined.find(START)
        if marker >= 0:
            delta = combined[:marker]
            self.calls_text = combined[marker:]
            self.pending = ""
            self.advance()
        else:
            keep = next((length for length in range(min(len(combined), len(START) - 1), 0, -1)
                         if START.startswith(combined[-length:])), 0)
            delta = combined[:-keep] if keep else combined
            self.pending = combined[-keep:] if keep else ""
        self.content += delta
        return delta

    def finish(self, reason):
        """The message, its tool-call deltas and the finish reason. As in other engines, output that does not parse as
        valid calls under the request's tool policy is the answer text (design decision, Oct 6): an agent client
        then keeps its turn instead of an error after HTTP 200. The reason stays "stop" or "length"."""
        text = self.content + (self.calls_text or "") + self.pending
        try:
            calls = self.checked_calls()
            calls, deltas = self.streamed(calls)
        except ApiError as error:
            if error.status != 502:
                raise
            self.fallback, self.invalid = True, getattr(error, "code", "malformed")
            # Deltas of a call already went out. With finish "stop" a client such as Pi repairs and runs the partial
            # call, so the reason is "length": clients do not run a call that a length stop cut (design decision, Oct 7).
            started = self.stream is not None and bool(self.stream.calls)
            return dict(role="assistant", content=text), [], "length" if started else reason
        message = dict(role="assistant", content=self.content or (None if calls else ""))
        if calls: message["tool_calls"] = calls
        return message, deltas, "tool_calls" if calls else reason

    def streamed(self, calls):
        """The calls with their streamed ids, and the deltas that complete them."""
        sent = self.stream.calls if self.stream is not None else []
        deltas = self.take_deltas()
        if len(sent) > len(calls):
            raise malformed("the streamed calls differ from the output.", "stream_mismatch")
        result = []
        for index, call in enumerate(calls):
            if index >= len(sent):
                result.append(call)
                deltas.append(dict(tool_calls=[dict(index=index, **call)]))
                continue
            arguments = call["function"]["arguments"]
            if sent[index]["name"] != call["function"]["name"] or not arguments.startswith(sent[index]["sent"]):
                raise malformed("the streamed calls differ from the output.", "stream_mismatch")
            result.append(dict(call, id=sent[index]["id"]))
            if len(arguments) > len(sent[index]["sent"]):
                deltas.append(dict(tool_calls=[dict(index=index, function=dict(
                    arguments=arguments[len(sent[index]["sent"]):]))]))
        return result, deltas

    def checked_calls(self):
        try:
            calls = self.parse_calls()
        except (ValueError, TypeError, RecursionError) as error:
            if isinstance(error, ApiError) and error.status == 502:
                raise
            raise malformed("invalid JSON value or parameter schema.", "parameter_json") from error
        choice = self.request.tool_choice
        if calls and choice == "none": raise malformed("tool_choice none forbids calls.", "tool_choice")
        if not calls and (choice == "required" or isinstance(choice, dict)):
            raise malformed("the required tool call is missing.", "tool_choice")
        if isinstance(choice, dict) and any(call["function"]["name"] != choice["function"]["name"] for call in calls):
            raise malformed("the model did not use the selected function.", "tool_choice")
        if not self.request.parallel_tool_calls and len(calls) > 1:
            raise malformed("parallel_tool_calls false permits one call only.", "parallel_calls")
        return calls

    def parse_calls(self):
        if self.calls_text is None:
            if self.pending and START.startswith(self.pending) and self.pending.startswith("<tool"):
                raise malformed("the tool-call start tag is incomplete.", "incomplete_call")
            self.content += self.pending
            self.pending = ""
            return []
        calls, remaining = [], self.calls_text
        budget = WorkBudget()
        while remaining.strip():
            remaining = remaining.lstrip()
            if not remaining.startswith(START): raise malformed("unexpected suffix after a tool call.", "unexpected_text")
            end = remaining.find(END, len(START))
            if end < 0: raise malformed("the tool-call end tag is missing.", "incomplete_call")
            calls.append(parse_function(remaining[len(START):end], self.definitions, budget, self.unchecked))
            if len(calls) > 128: raise malformed("at most 128 calls are supported.", "too_many_calls")
            remaining = remaining[end + len(END):]
        return calls
