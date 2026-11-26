"""The Hy4-flash (model_type hy_v4_flash) chat format: tags, token IDs, effort words and the tool-call parser.

Sources: tencent/Hy4-flash tokenizer.json and chat_template.jinja, and the Hy4 tool parser of another engine.
Each structural tag is one added token with the checkpoint suffix ":opensource".
A tool block is <tool_calls>, then for each call <tool_call>NAME, then pairs <arg_key>K</arg_key><arg_value>V</arg_value>,
then </tool_call>, and at the end </tool_calls>. A string value is raw text; another value is JSON (the template's tojson).
"""
import json
import uuid
from pathlib import Path
from .api import ApiError
from .tool_frame import raw_string
from .tool_output import PASS_NAME, ToolOutput, argument_value, malformed, unchecked as unchecked_code
from .tool_schema import WorkBudget, matches, resolve

SUFFIX = ":opensource"
VOCAB_SIZE = 120832
TURN_START, TURN_MIDDLE, TURN_END = ("<｜hy_" + name + SUFFIX + "｜>" for name in ("start", "middle", "end"))
THINK, THINK_END = "<think" + SUFFIX + ">", "</think" + SUFFIX + ">"
CALLS, CALLS_END = "<tool_calls" + SUFFIX + ">", "</tool_calls" + SUFFIX + ">"
CALL, CALL_END = "<tool_call" + SUFFIX + ">", "</tool_call" + SUFFIX + ">"
ARG_KEY, ARG_KEY_END = "<arg_key" + SUFFIX + ">", "</arg_key" + SUFFIX + ">"
ARG_VALUE, ARG_VALUE_END = "<arg_value" + SUFFIX + ">", "</arg_value" + SUFFIX + ">"
ASSISTANT_PROMPT = TURN_START + "assistant" + TURN_MIDDLE
# IDs from tokenizer.json; the tokenizer check compares them at load time.
SPECIAL_IDS = {TURN_START: 120000, TURN_MIDDLE: 120001, TURN_END: 120025, THINK: 120029, THINK_END: 120030,
               CALL: 120033, CALL_END: 120034, CALLS: 120035, CALLS_END: 120036,
               ARG_KEY: 120040, ARG_KEY_END: 120041, ARG_VALUE: 120042, ARG_VALUE_END: 120043}
STOP_IDS = (SPECIAL_IDS[TURN_END],)
# The token classes of crucible_turn (metrics.count_token): the think end, then a tool block from start to end.
MARKER_IDS = (SPECIAL_IDS[THINK_END], SPECIAL_IDS[CALLS], SPECIAL_IDS[CALLS_END])
TAGS = (CALLS, CALLS_END, CALL, CALL_END, ARG_KEY, ARG_KEY_END, ARG_VALUE, ARG_VALUE_END)
MODEL_TYPE = "hy_v4_flash"
# The ROCm runtime words of the engine child for a Hy4-flash checkpoint. ROCr's fragment allocator puts each hipMalloc
# of 2 MiB or less in its own 2 MiB block: the full pack would commit 127.3 GiB, over the 121.6 GiB KFD limit of a
# 128 GB machine; with the switch, 74.0 GiB (the test machine probe of 2026-11-19). ROCr reads it
# at its first init, so it must be in the child's environment or set by the engine main before any HIP call (both
# happen: engine/cli/gen_cli.h).
ALLOCATOR_ENV = {"HSA_DISABLE_FRAGMENT_ALLOCATOR": "1"}


def is_hy4(checkpoint):
    """True when the checkpoint's config.json names model_type hy_v4_flash. No config, or a config that does not
    parse, means another family: that family's own loader reports the file."""
    try:
        config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and config.get("model_type") == MODEL_TYPE


def template_effort(thinking, word):
    """The template's reasoning_effort word for the thinking switch and the client's effort word
    (api.Request.effort_word, None without one). Design decision (keep the client word): max and xhigh give max;
    high, the lower words and no word give high, the card's default. Thinking off gives no_think."""
    if not thinking:
        return "no_think"
    return "max" if word in ("xhigh", "max") else "high"


def gap(text, pos):
    """The index after the whitespace at text[pos:] (str.isspace, as the stream's lstrip)."""
    return len(text) - len(text[pos:].lstrip())


def read_pairs(rest, properties, root, strict, budget):
    """[(key, raw value)] of the argument pairs in rest (the text after the name). A strict tool's value ends where its
    grammar ends it (hy4_frame.value_close); another value ends at its first </arg_value>."""
    from .hy4_frame import value_close
    pairs, pos = [], gap(rest, 0)
    while pos < len(rest):
        if not rest.startswith(ARG_KEY, pos):
            raise malformed("unexpected text between the argument tags.", "parameter_tag")
        key_end = rest.find(ARG_KEY_END, pos + len(ARG_KEY))
        value = key_end + len(ARG_KEY_END)
        if key_end < 0 or not rest.startswith(ARG_VALUE, value):
            raise malformed("expected an argument key and value tag.", "parameter_tag")
        key, value = rest[pos + len(ARG_KEY):key_end], value + len(ARG_VALUE)
        close = (value_close(rest, value, raw_string(properties[key], root, budget))
                 if strict and key in properties else None)
        close = rest.find(ARG_VALUE_END, value) if close is None else close
        if close < 0:
            raise malformed("an argument end tag is missing.", "incomplete_call")
        pairs.append((key, rest[value:close]))
        pos = gap(rest, close + len(ARG_VALUE_END))
    return pairs


def parse_call(body, definitions, budget, unchecked=None, whole=False):
    """One call from the text between <tool_call> and </tool_call>, in tool_output.parse_function's shape.
    unchecked (a list) collects the schema-only codes (tool_output.UNCHECKED); a strict tool checks strictly."""
    start = body.find(ARG_KEY)
    name, rest = (body, "") if start < 0 else (body[:start], body[start:])
    name = name.strip()
    if not name or any(tag in name for tag in TAGS):
        raise malformed("expected a function name.", "function_tag")
    definition = definitions.get(name)
    strict = definition is not None and definition.get("strict") is True
    soft = None if strict else unchecked
    if definition is None:
        unchecked_code(None if not PASS_NAME.match(name) else soft, "unknown_function", "the function is not advertised.")
    schema = (definition or {}).get("parameters", {})
    resolved = resolve(schema, schema, budget=budget)
    properties = resolved.get("properties", {}) if isinstance(resolved, dict) else {}
    arguments, first = {}, len(soft) if soft is not None else 0
    for key, raw in read_pairs(rest, properties, schema, strict, budget):
        if key in arguments:
            raise malformed("parameter names must be unique.", "duplicate_parameter")
        # argument_value trims one newline at each end (the Qwen form); the Hy4 value has none, so add them.
        arguments[key] = argument_value("\n" + raw + "\n", properties.get(key, {}), schema, budget, soft, whole)
    if not matches(arguments, schema, schema, budget=budget) and (soft is None or len(soft) == first):
        unchecked_code(soft, "schema_mismatch", "arguments do not match the advertised object schema.")
    return dict(id="call_" + uuid.uuid4().hex, type="function", function=dict(
        name=name, arguments=json.dumps(arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":"))))


def parse_block(text, definitions, budget, unchecked=None, whole=False):
    """The calls of a tool block (text starts with <tool_calls>), read by its structure (hy4_frame.call_end). A
    malformed block raises the 502 ApiError of tool_output.malformed, as other engines's strict mode refuses it."""
    from .hy4_frame import call_end
    calls, pos = [], gap(text, len(CALLS))
    while not text.startswith(CALLS_END, pos):
        if not text.startswith(CALL, pos):
            if CALL.startswith(text[pos:]) or CALLS_END.startswith(text[pos:]):
                raise malformed("the tool block is not closed.", "incomplete_call")
            raise malformed("expected a tool call tag.", "function_tag")
        end = call_end(text, pos, definitions, budget)
        if end < 0:
            raise malformed("a tool call end tag is missing.", "incomplete_call")
        calls.append(parse_call(text[pos + len(CALL):end], definitions, budget, unchecked, whole))
        pos = gap(text, end + len(CALL_END))
    if text[pos + len(CALLS_END):].strip():
        raise malformed("unexpected text after the tool block.", "unexpected_text")
    if not calls:
        raise malformed("the tool block holds no call.", "function_tag")
    return calls


def parse_tool_calls(text, definitions, budget):
    """(content, calls) for the full generated answer. A malformed block keeps the whole text as content."""
    start = text.find(CALLS)
    if start < 0:
        return text, []
    try:
        return text[:start], parse_block(text[start:], definitions, budget, [])
    except ApiError:
        return text, []


def held(text):
    """The length of the suffix of text that can start the tool block marker."""
    return next((length for length in range(min(len(text), len(CALLS) - 1), 0, -1)
                 if CALLS.startswith(text[-length:])), 0)


class Hy4ToolOutput(ToolOutput):
    """ToolOutput for the Hy4 tag form. Prose goes out until <tool_calls>. With stream, the calls go out while the
    model writes them (server/hy4_stream.py); else whole when the output ends (serve --stream-tool-calls off).
    finish() applies the same tool policy and content fallback."""

    def __init__(self, request, stream=False, whole=False):
        super().__init__(request, False, whole)
        if stream:
            from .hy4_stream import Hy4CallStream  # hy4_stream imports this module
            self.stream = Hy4CallStream(request, self.definitions, whole)

    def open_call(self):
        """True inside a tool block that the model has not closed yet, read by its structure (hy4_frame)."""
        from .hy4_frame import OPEN, block_end
        return self.calls_text is not None and block_end(self.calls_text, self.definitions, WorkBudget()) == OPEN

    def feed(self, text):
        if self.calls_text is not None:
            self.calls_text += text
            self.advance()
            return ""
        combined = self.pending + text
        marker = combined.find(CALLS)
        if marker < 0:
            keep = held(combined)
            delta, self.pending = combined[:len(combined) - keep], combined[len(combined) - keep:]
        else:
            delta, self.calls_text, self.pending = combined[:marker], combined[marker:], ""
            self.advance()
        self.content += delta
        return delta

    def advance(self):
        if self.stream is not None:
            self.deltas += self.stream.advance(self.calls_text)

    def parse_calls(self):
        if self.calls_text is None:
            self.content, self.pending = self.content + self.pending, ""
            return []
        return parse_block(self.calls_text, self.definitions, WorkBudget(), self.unchecked, self.whole)
