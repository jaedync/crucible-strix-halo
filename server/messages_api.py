"""POST /v1/messages (Anthropic Messages API) on the chat path.

The request maps to a chat request body (server/anthropic.py for messages, system, tools, tool_choice and thinking;
this module for the generation fields), so validate_request, the prompt and the tokens are those of
/v1/chat/completions. The reply maps back: reasoning_content -> a thinking block, content -> a text block, tool calls ->
tool_use blocks, finish reasons -> stop reasons, usage -> Anthropic usage. A stream is the Anthropic event sequence:
message_start, ping, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop.
"""
import uuid

from . import anthropic
from .api import ApiError
from .output_sink import Sink, call_input, stop_sequence
from .response import CompletionResponse

PATH = "/v1/messages"
FORMAT_FIELD = anthropic.FORMAT_FIELD  # the field that maps to response_format (server/anthropic.py output_format)
RENDER_STRICT = anthropic.RENDER_STRICT
# Generation fields that this module maps. seed is a Crucible extension (Anthropic has none) for repeatable sampling.
GENERATION = {"max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "stream", "seed"}
# Client hints with no effect on the reply: account, routing and speed hints of the hosted API, and prompt caching
# (Crucible caches by itself).
HINTS = {"metadata", "service_tier", "cache_control", "inference_geo", "speed", "user_profile_id", "workspace_id"}
CONVERTED = {"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config"}
FIELDS = GENERATION | HINTS | CONVERTED
BLOCKS = {"text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"}
BLOCK_LIST = "text, image, tool_use, tool_result, thinking, and redacted_thinking"  # errors never repeat the client's value
MAX_STOP_SEQUENCES = 4
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
ERROR_TYPES = {401: "authentication_error", 403: "permission_error", 404: "not_found_error", 413: "request_too_large",
               429: "rate_limit_error", 503: "overloaded_error"}


def refuse_blocks(messages):
    """A content block type that the chat path cannot take is refused here with its place in the request and the
    supported types, never the client's value."""
    if not isinstance(messages, list):
        return
    for i, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        for j, block in enumerate(content if isinstance(content, list) else ()):
            kind = block.get("type") if isinstance(block, dict) else None
            if not isinstance(kind, str) or kind not in BLOCKS:  # a list or an object as the type is a 400, not a 500
                raise ApiError(f"messages[{i}].content[{j}].type must be one of {BLOCK_LIST}.", "messages")


def chat_body(body):
    """The chat request body for one Anthropic Messages generation body. validate_request checks the result."""
    if not isinstance(body, dict):
        raise ApiError("The request must be a JSON object.")
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        raise ApiError(f"{unknown[0]} is not supported by /v1/messages.", unknown[0])
    if "max_tokens" not in body:
        raise ApiError("max_tokens is required.", "max_tokens")
    refuse_blocks(body.get("messages"))
    result = anthropic.chat_body({key: value for key, value in body.items() if key in anthropic.FIELDS})
    result["max_tokens"] = body["max_tokens"]
    for name in ("temperature", "top_p", "top_k", "seed"):
        if body.get(name) is not None:
            result[name] = body[name]
    stops = body.get("stop_sequences")
    if stops is not None:
        if isinstance(stops, list) and len(stops) > MAX_STOP_SEQUENCES:
            raise ApiError("stop_sequences takes at most four strings.", "stop_sequences")
        result["stop"] = stops
    if body.get("stream") is not None:
        result["stream"] = body["stream"]
        if body["stream"] is True:
            result["stream_options"] = dict(include_usage=True)  # the counts of message_delta
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled" and "budget_tokens" in thinking:
        result["max_thinking_tokens"] = thinking["budget_tokens"]  # the chat path closes the think block there
    return result


def error_body(error):
    kind = getattr(error, "kind", None) or ERROR_TYPES.get(error.status) or (
        "invalid_request_error" if error.status < 500 else "api_error")
    return dict(type="error", error=dict(type=kind, message=str(error)))


def stream_error(handler, error):
    Sink(handler).event("error", error_body(error))


def usage(chat_usage):
    """Anthropic usage: input_tokens leaves out the cached prompt tokens (cache_read_input_tokens)."""
    prompt = chat_usage.get("prompt_tokens", 0)
    cached = (chat_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return dict(input_tokens=prompt - cached, output_tokens=chat_usage.get("completion_tokens", 0),
                cache_read_input_tokens=cached, cache_creation_input_tokens=0)


def stop_fields(reason, output):
    matched = stop_sequence(output)
    if reason == "stop" and matched is not None:
        return "stop_sequence", matched
    return STOP_REASONS.get(reason, "end_turn"), None


class MessagesSink(Sink):
    def __init__(self, handler, model):
        super().__init__(handler)
        self.model, self.id = model, "msg_" + uuid.uuid4().hex
        self.count, self.open, self.calls = 0, None, {}  # blocks so far, (kind, index) of the open one, call -> block
        self.started, self.reason, self.counts = False, None, {}

    def start(self):
        if self.started:
            return
        self.started = True
        prompt = self.output.prompt_count if self.output is not None else 0
        self.event("message_start", dict(type="message_start", message=dict(
            id=self.id, type="message", role="assistant", model=self.model, content=[], stop_reason=None,
            stop_sequence=None, usage=dict(input_tokens=prompt, output_tokens=0))))
        self.event("ping", dict(type="ping"))

    def open_block(self, kind, block):
        self.close_block()
        self.open, self.count = (kind, self.count), self.count + 1
        self.event("content_block_start", dict(type="content_block_start", index=self.open[1], content_block=block))
        return self.open[1]

    def close_block(self):
        if self.open is not None:
            self.event("content_block_stop", dict(type="content_block_stop", index=self.open[1]))
            self.open = None

    def delta(self, index, delta):
        self.event("content_block_delta", dict(type="content_block_delta", index=index, delta=delta))

    def block(self, kind, block):
        return self.open[1] if self.open is not None and self.open[0] == kind else self.open_block(kind, block)

    def chunk(self, body):
        self.start()
        if not body.get("choices"):
            self.counts = body.get("usage") or self.counts
            return
        choice = body["choices"][0]
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            index = self.block("thinking", dict(type="thinking", thinking="", signature=""))
            self.delta(index, dict(type="thinking_delta", thinking=delta["reasoning_content"]))
        if delta.get("content"):
            index = self.block("text", dict(type="text", text=""))
            self.delta(index, dict(type="text_delta", text=delta["content"]))
        for call in delta.get("tool_calls") or ():
            function = call.get("function") or {}
            if "id" in call:
                self.calls[call["index"]] = self.open_block("tool_use", dict(type="tool_use", id=call["id"],
                                                                             name=function.get("name"), input={}))
            if function.get("arguments") and call["index"] in self.calls:
                self.delta(self.calls[call["index"]], dict(type="input_json_delta", partial_json=function["arguments"]))
        if choice.get("finish_reason"):
            self.reason = choice["finish_reason"]

    def done(self):
        self.start()
        self.close_block()
        reason, matched = stop_fields(self.reason, self.output)
        self.event("message_delta", dict(type="message_delta", delta=dict(stop_reason=reason, stop_sequence=matched),
                                         usage=usage(self.counts)))
        self.event("message_stop", dict(type="message_stop"))

    def reply(self, body):
        choice = body["choices"][0]
        message = choice["message"]
        content = []
        if message.get("reasoning_content"):
            content.append(dict(type="thinking", thinking=message["reasoning_content"], signature=""))
        if message.get("content"):
            content.append(dict(type="text", text=message["content"]))
        for call in message.get("tool_calls") or ():
            content.append(dict(type="tool_use", id=call["id"], name=call["function"]["name"],
                                input=call_input(call["function"]["arguments"])))
        reason, matched = stop_fields(choice["finish_reason"], self.output)
        return dict(id=self.id, type="message", role="assistant", model=self.model, content=content,
                    stop_reason=reason, stop_sequence=matched, usage=usage(body["usage"]))


def output(handler, service, request, prompt_ids):
    """The CompletionResponse of a /v1/messages request, with a sink that writes the Anthropic shapes."""
    sink = MessagesSink(handler, request.model)
    sink.output = CompletionResponse(sink, service, request, prompt_ids)
    return sink.output
