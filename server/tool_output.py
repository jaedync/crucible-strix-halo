"""Parse the checkpoint's typed XML-like calls into strict OpenAI tool calls."""
import json
import re
import uuid
from .api import ApiError
from .json_values import json_object, json_value
from .tool_frame import END, FUNCTION, OPEN, PARAMETER, START, call_end, raw_string, value_close
from .tool_invoke import BROKEN, END as INVOKE_END, INVOKE, START as INVOKE_START, Form, block_end
from .tool_schema import matches, types, resolve, WorkBudget
from .tool_stream import ToolCallStream

CALL_STARTS = (START, INVOKE_START)
CHECKPOINT = Form(FUNCTION, PARAMETER, "</function>")
# Prose that starts with `<invoke` is checked again while it is at most this long, then each time it doubles.
RECHECK = 256


def first_start(text, start=0):
    """The index of the first call start tag in text[start:], or -1."""
    return min((index for index in (text.find(tag, start) for tag in CALL_STARTS) if index >= 0), default=-1)


def held(text):
    """Length of the longest suffix of text that can still grow into a call start tag."""
    return next((length for length in range(min(len(text), len(START) - 1), 0, -1)
                 if any(tag.startswith(text[-length:]) for tag in CALL_STARTS)), 0)


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


# JavaScript writes an integral number below 1e21 with no fraction. The rule stops at 2**53: above it a JavaScript
# integer loses digits, and a client could put a large integer in a fixed-width integer type.
WHOLE_LIMIT = 2 ** 53


def whole_numbers(value):
    """The value with each integral float of at most 2**53 in magnitude as an integer of the same value (1.0 -> 1,
    -0.0 -> 0). A JavaScript client (Pi) sends the arguments back after JSON.parse and JSON.stringify; with this
    rule that round trip gives the same values, so the next request renders the call as tool_tail extended it.
    Only with --extend-tool-tails (ToolOutput whole): with the switch off the reply is as on trunk."""
    if isinstance(value, float):
        return int(value) if value.is_integer() and abs(value) <= WHOLE_LIMIT else value
    if isinstance(value, dict):
        return type(value)((key, whole_numbers(item)) for key, item in value.items())
    if isinstance(value, list):
        return [whole_numbers(item) for item in value]
    return value


def argument_value(raw, schema, root, budget, soft=None, whole=False):
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
        parsed = whole_numbers(parsed) if whole else parsed
    except ApiError:
        if soft is None:
            raise
        soft.append("parameter_json")  # for example 1e999: JSON, but not a finite value; the raw text goes out
        return value
    if not matches(parsed, schema, root, budget=budget):
        unchecked(soft, "schema_mismatch", "a parameter does not match its advertised schema.")
    return parsed

def parse_function(block, definitions, budget, soft=None, form=CHECKPOINT, whole=False):
    """One call. soft=None checks the schema strictly; a list lets schema-only failures pass (see UNCHECKED). A tool
    with strict: true always checks strictly: its client expects only valid arguments. form gives the tags: the
    checkpoint's (the block is the text between `<tool_call>` and `</tool_call>`) or tool_invoke.INVOKE (the block is
    the whole `<invoke>` block)."""
    match = form.function.match(block.lstrip())
    if not match:
        raise malformed("expected a function tag.", "function_tag")
    body = block.lstrip()
    name = match.group(1)
    strict = name in definitions and definitions[name].get("strict") is True
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
    while not body[position:].lstrip().startswith(form.close):
        position += len(body[position:]) - len(body[position:].lstrip())
        parameter = form.parameter.match(body, position)
        if not parameter:
            raise malformed("expected a parameter tag or function end.", "parameter_tag")
        key = parameter.group(1)
        if key in arguments:
            raise malformed("parameter names must be unique.", "duplicate_parameter")
        close = None
        if strict and key in properties:
            # A strict value can hold the tag inside a JSON string (tool_frame.py).
            close = value_close(body, parameter.end(), raw_string(properties[key], schema, budget))
        close = body.find("</parameter>", parameter.end()) if close is None else close
        if close == OPEN:
            raise malformed("a parameter end tag is missing.", "incomplete_call")
        arguments[key] = argument_value(body[parameter.end():close], properties.get(key, {}), schema, budget, soft,
                                        whole)
        position = close + len("</parameter>")
    rest = body[position:].lstrip()[len(form.close):]
    if rest.strip():
        raise malformed("unexpected text after the function end.", "unexpected_text")
    # A parameter that already failed also fails the object; one code per cause.
    if not matches(arguments, schema, schema, budget=budget) and (soft is None or len(soft) == first):
        unchecked(soft, "schema_mismatch", "arguments do not match the advertised object schema.")
    return dict(id="call_" + uuid.uuid4().hex, type="function", function=dict(
        name=name, arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))

class ToolOutput:
    def __init__(self, request, stream=False, whole=False):
        self.request = request
        self.whole = whole  # whole_numbers on the arguments (--extend-tool-tails)
        self.definitions = {tool["function"]["name"]: tool["function"] for tool in request.tools}
        self.pending = ""
        self.content = ""
        self.calls_text = None
        self.fallback = False  # the output did not parse as valid calls and is returned as content
        self.invalid = None  # the fixed reason code of a fallback
        self.unchecked = []  # reason codes of calls that went out with a schema-only failure (UNCHECKED)
        # Opt-out with --stream-tool-calls off: calls then go out whole when the output ends.
        self.stream = ToolCallStream(request, self.definitions, whole) if stream else None
        self.deltas = []
        self.checked = 0  # the length of calls_text at the last prose check (release)
        # True while the parser reads the free text of the call grammar (an <invoke> block before the first <tool_call>;
        # structured.py), and after the parse when a call came from that text: a fault there is not a grammar fault.
        self.free_text = False
        self.free_calls = 0

    def open_call(self):
        """True inside a <tool_call> or <invoke> block that the model has not closed yet. This walks the structure of
        parse_calls (tool_frame.call_end): the start tag inside an argument value (a path, a quoted template) does not
        open a block, and the end tag inside a strict argument does not close one. After a <tool_call>, only a
        <tool_call> start counts, as before the <invoke> form (the call grammar takes no other text there)."""
        text = self.calls_text
        if text is None:
            return False
        pos, free = 0, True
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos == len(text):
                return False
            if text.startswith(START, pos):
                end = call_end(text, pos, self.definitions, WorkBudget())
                if end == OPEN:
                    return True
                pos, free = end + len(END), False
                continue
            if free and text.startswith(INVOKE_START, pos):
                end = block_end(text, pos, self.definitions, WorkBudget())
                if end in (None, OPEN, BROKEN):
                    return end == OPEN  # None: prose that names the tag; BROKEN: a malformed call
                pos = end + len(INVOKE_END)
                continue
            # Between calls: text that is not a start tag may still be a partial start tag.
            return START.startswith(text[pos:]) or (free and INVOKE_START.startswith(text[pos:]))

    def take_deltas(self):
        """Tool-call deltas that are ready to send now (stream mode)."""
        deltas, self.deltas = self.deltas, []
        return deltas

    def advance(self):
        # The stream reads the checkpoint form only. Calls in the <invoke> form go out whole when the output ends.
        if self.stream is not None and not self.calls_text.startswith(INVOKE_START):
            self.deltas += self.stream.advance(self.calls_text)

    def release(self, final=False):
        """Content: prose at the start of calls_text that names `<invoke` but is not a block (tool_invoke.block_end).
        The search for the first call goes on after it. A check reads the whole text, so a long open block is checked
        again only when its length doubles. final: the output has ended (check now)."""
        released = ""
        while self.calls_text is not None and self.calls_text.startswith(INVOKE_START):
            text = self.calls_text
            if not final and RECHECK < len(text) < 2 * self.checked:
                break
            self.checked = len(text)
            if block_end(text, 0, self.definitions, WorkBudget(), final) is not None:
                break
            marker = first_start(text, 1)
            if marker < 0:
                keep = held(text)
                marker, self.calls_text, self.pending = len(text) - keep, None, text[len(text) - keep:]
            else:
                self.calls_text = text[marker:]
            released += text[:marker]
            self.checked = 0
        return released

    def feed(self, text):
        if self.calls_text is not None:
            self.calls_text += text
            delta = ""
        else:
            combined = self.pending + text
            marker = first_start(combined)
            if marker < 0:
                keep = held(combined)
                self.content += combined[:len(combined) - keep]
                self.pending = combined[len(combined) - keep:]
                return combined[:len(combined) - keep]
            delta, self.calls_text, self.pending = combined[:marker], combined[marker:], ""
        delta += self.release()
        if self.calls_text is not None:
            self.advance()
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
        self.free_text = self.free_calls > 0  # a policy fault below counts the calls from the free text
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
        self.content += self.release(final=True)
        if self.calls_text is None:
            if self.pending and self.pending.startswith("<tool"):
                raise malformed("the tool-call start tag is incomplete.", "incomplete_call")
            self.content += self.pending
            self.pending = ""
            return []
        calls, remaining = [], self.calls_text
        # Only the auto grammar has free text before its calls; required and named start with <tool_call>.
        self.free_text = remaining.startswith(INVOKE_START) and self.request.tool_choice == "auto"
        budget = WorkBudget()
        while remaining.strip():
            remaining = remaining.lstrip()
            if remaining.startswith(START):
                self.free_text = False
                end = call_end(remaining, 0, self.definitions, budget)
                if end == OPEN: raise malformed("the tool-call end tag is missing.", "incomplete_call")
                calls.append(parse_function(remaining[len(START):end], self.definitions, budget, self.unchecked,
                                            whole=self.whole))
                remaining = remaining[end + len(END):]
            elif remaining.startswith(INVOKE_START):
                end = block_end(remaining, 0, self.definitions, budget, final=True)
                if end is None: raise malformed("unexpected suffix after a tool call.", "unexpected_text")
                if end == BROKEN: raise malformed("the invoke block is malformed or cut.", "parameter_tag")
                end += len(INVOKE_END)
                calls.append(parse_function(remaining[:end], self.definitions, budget, self.unchecked, INVOKE,
                                            self.whole))
                self.free_calls += self.free_text
                remaining = remaining[end:]
            else:
                raise malformed("unexpected suffix after a tool call.", "unexpected_text")
            if len(calls) > 128: raise malformed("at most 128 calls are supported.", "too_many_calls")
        return calls
