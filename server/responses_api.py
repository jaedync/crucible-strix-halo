"""POST /v1/responses (OpenAI Responses API) on the chat path, with no server-side state.

The request maps to a chat request body, so validate_request, the prompt and the tokens are those of
/v1/chat/completions: instructions and leading system or developer items -> one system message, message items -> chat
messages, reasoning items -> the next assistant message's reasoning_content (or a reasoning-only assistant turn),
function_call items -> its tool_calls, function_call_output items -> tool messages. The reply maps back to reasoning,
message and function_call output items. Crucible keeps no conversation store, so previous_response_id, conversation,
store true and background are refused.
"""
import time
import uuid

from .api import ApiError
from .output_sink import Sink
from .response import CompletionResponse
# http.py calls strict_default of the adapter before validate_request and prompt_request after it.
from .responses_strict import prompt_request, strict_default
from .structured import STRICT_DEFAULT

PATH = "/v1/responses"
# text.format maps to the chat field response_format (server/structured.py); its errors name this place.
FORMAT_FIELD = "text.format"
# On an engine with no structured output, the template renders a function tool's strict flag, as chat does (the
# prompt before structured output).
RENDER_STRICT = True
FORMAT_TYPES = "text, json_schema, or json_object"
SCHEMA_KEYS = ("name", "schema", "description", "strict")
# Hints with no effect on the reply (account, cache and routing hints of the hosted API).
HINTS = {"metadata", "user", "safety_identifier", "prompt_cache_key", "prompt_cache_retention", "service_tier",
         "stream_options"}
FIELDS = HINTS | {"model", "input", "instructions", "max_output_tokens", "temperature", "top_p", "stream", "tools",
                  "tool_choice", "parallel_tool_calls", "reasoning", "text", "store", "previous_response_id",
                  "conversation", "background", "include", "truncation", "top_logprobs", "max_tool_calls", "prompt",
                  "top_k", "min_p", "seed"}
# Fields that need server-side state: they must be absent, null or (store, background) false.
STATE = ("previous_response_id", "conversation", "prompt")
INCLUDE = {"reasoning.encrypted_content"}  # Crucible returns no encrypted reasoning; asking for it changes nothing
SYSTEM_ROLES = ("system", "developer")


def fail(message, param):
    raise ApiError(message, param)


def check_fields(body):
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        fail(f"{unknown[0]} is not supported by /v1/responses.", unknown[0])
    for field in STATE:
        if body.get(field) is not None:
            fail(f"{field} is not supported: Crucible keeps no conversation state. Send the whole input.", field)
    for field in ("store", "background"):
        if body.get(field) not in (None, False):
            fail(f"{field} must be false: Crucible keeps no conversation state.", field)
    include = body.get("include")
    if include is not None and (not isinstance(include, list) or
                                any(not isinstance(value, str) or value not in INCLUDE for value in include)):
        fail("include supports reasoning.encrypted_content only.", "include")
    if body.get("truncation") not in (None, "disabled"):
        fail("truncation must be disabled.", "truncation")
    if body.get("top_logprobs") not in (None, 0):
        fail("top_logprobs is not supported by /v1/responses.", "top_logprobs")
    if body.get("max_tool_calls") is not None:
        fail("max_tool_calls is not supported.", "max_tool_calls")
    text = body.get("text")
    if text is not None:
        if (not isinstance(text, dict) or set(text) - {"format", "verbosity"} or
                text.get("verbosity") not in (None, "medium")):
            fail("text supports format and verbosity medium only.", "text")


def text_format(value):
    """text.format -> the chat response_format, or None for text. The chat path checks the values and the schema
    (server/structured.py response_format), so the grammar, the refused keywords and the limits are the same."""
    kind = value.get("type") if isinstance(value, dict) else None
    if value is None or (kind == "text" and set(value) == {"type"}):
        return None
    if kind == "json_object" and set(value) == {"type"}:
        return dict(type="json_object")
    if not isinstance(kind, str) or kind not in ("text", "json_schema", "json_object"):
        fail(f"{FORMAT_FIELD} type must be {FORMAT_TYPES}.", FORMAT_FIELD)
    if kind != "json_schema":
        fail(f"{FORMAT_FIELD} {kind} takes no other field.", FORMAT_FIELD)
    if set(value) - {"type", *SCHEMA_KEYS}:
        fail(f"{FORMAT_FIELD} json_schema supports name, schema, description, and strict only.", FORMAT_FIELD)
    # A null description or strict counts as absent, as for a function tool (chat_tools).
    spec = {key: value[key] for key in SCHEMA_KEYS if key in value and not (key != "name" and value[key] is None)}
    return dict(type="json_schema", json_schema=spec)


def echo_format(body):
    """The reply's text.format: the request's format, with strict at the value that the server applied."""
    value = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
    if not isinstance(value, dict) or value.get("type") not in ("json_schema", "json_object"):
        return dict(type="text")
    if value["type"] == "json_object":
        return dict(type="json_object")
    echo = {key: value[key] for key in ("type", *SCHEMA_KEYS) if value.get(key) is not None}
    return echo | dict(strict=value["strict"] if isinstance(value.get("strict"), bool) else STRICT_DEFAULT)


def echo_tools(body, request):
    """The reply's tools: the request's tools, with strict at the value that the server applied (OpenAI shows
    strict false when a tool with no strict falls back to non-strict)."""
    applied = getattr(request, "tools", ())
    tools = body.get("tools") or []
    if len(applied) != len(tools):
        return tools
    return [tool | dict(strict=chat["function"].get("strict") is True) for tool, chat in zip(tools, applied)]


def text_parts(content, where, kinds):
    """A string, or a part list of the given text kinds (and input_image) -> chat content. where: the place of the
    content (input[0].content). An error names the place and the supported types, never the client's value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        fail(f"{where} must be text or a list of parts.", where)
    parts = []
    for j, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if not isinstance(kind, str) or kind not in kinds:
            place = f"{where}[{j}].type"
            fail(f"{place} must be {' or '.join(kinds)}.", place)
        if kind == "input_image":
            if not isinstance(part.get("image_url"), str):
                fail(f"{where}[{j}]: an input_image part needs an image_url string.", f"{where}[{j}].image_url")
            parts.append(dict(type="image_url", image_url=dict(url=part["image_url"])))
        else:
            if not isinstance(part.get("text"), str):
                fail(f"{where}[{j}]: a {kind} part needs a text string.", f"{where}[{j}].text")
            parts.append(dict(type="text", text=part["text"]))
    return parts


def joined(content):
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


def reasoning_text(item, where):
    content = item.get("content") or []
    if not isinstance(content, list) or any(not isinstance(part, dict) or part.get("type") != "reasoning_text" or
                                            not isinstance(part.get("text"), str) for part in content):
        fail(f"{where}: reasoning content must be reasoning_text parts.", "input")
    return "".join(part["text"] for part in content)  # summary and encrypted_content do not reach the prompt


class History:
    """Responses input items -> chat messages, in order. A reasoning item waits for the assistant message or the
    function_call after it. A user, system or developer item, or the end of the input, first ends the waiting
    reasoning as its own assistant turn with an empty content: the message of a reasoning-only chat reply."""
    def __init__(self):
        self.messages, self.reasoning = [], ""

    def assistant(self):
        """The assistant message that the next function_call joins: the last message, if it is an assistant one and no
        reasoning waits. Reasoning after a message starts a new assistant turn, so it stays in its place."""
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "assistant" and not self.reasoning:
            return last
        message = dict(role="assistant", content=None)
        self.take_reasoning(message)
        self.messages.append(message)
        return message

    def take_reasoning(self, message):
        if self.reasoning:
            message["reasoning_content"], self.reasoning = self.reasoning, ""

    def end_reasoning(self):
        """Waiting reasoning with no assistant message or function_call after it -> a reasoning-only assistant turn."""
        if self.reasoning:
            self.messages.append(dict(role="assistant", content="", reasoning_content=self.reasoning))
            self.reasoning = ""

    def add(self, item, i, system):
        where = f"input[{i}]"
        if not isinstance(item, dict):
            fail(f"{where} must be an object.", "input")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role in SYSTEM_ROLES or role == "user":
                self.end_reasoning()
            if role in SYSTEM_ROLES:
                if self.messages:
                    fail(f"{where}: a {role} message must come before the conversation.", "input")
                system.append(joined(text_parts(item.get("content"), f"{where}.content", ("input_text",))))
            elif role == "user":
                self.messages.append(dict(role="user", content=text_parts(item.get("content"), f"{where}.content",
                                                                          ("input_text", "input_image"))))
            elif role == "assistant":
                content = text_parts(item.get("content"), f"{where}.content", ("output_text",))
                message = dict(role="assistant", content=joined(content))
                self.take_reasoning(message)
                self.messages.append(message)
            else:
                fail(f"{where}: role must be user, assistant, system, or developer.", "input")
        elif kind == "reasoning":
            self.reasoning += reasoning_text(item, where)
        elif kind == "function_call":
            if not all(isinstance(item.get(key), str) for key in ("call_id", "name", "arguments")):
                fail(f"{where}: a function_call needs call_id, name, and arguments.", "input")
            message = self.assistant()
            message.setdefault("tool_calls", []).append(dict(id=item["call_id"], type="function", function=dict(
                name=item["name"], arguments=item["arguments"])))
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                fail(f"{where}: a function_call_output needs a call_id.", "input")
            if self.reasoning:
                fail(f"{where}: a function_call_output cannot follow a reasoning item. Put the reasoning item before "
                     "its function_call.", "input")
            output = text_parts(item.get("output"), f"{where}.output", ("input_text", "input_image"))
            self.messages.append(dict(role="tool", tool_call_id=item["call_id"], content=output))
        else:
            fail(f"{where}.type must be message, reasoning, function_call, or function_call_output.", f"{where}.type")


def chat_messages(body):
    system = [body["instructions"]] if body.get("instructions") is not None else []
    if system and not isinstance(system[0], str):
        fail("instructions must be text.", "instructions")
    value = body.get("input")
    items = [dict(role="user", content=value)] if isinstance(value, str) else value
    if not isinstance(items, list) or not items:
        fail("input must be text or a nonempty list of items.", "input")
    history = History()
    for i, item in enumerate(items):
        history.add(item, i, system)
    history.end_reasoning()
    lead = [dict(role="system", content="\n\n".join(system))] if system else []
    return lead + history.messages


def chat_tools(value):
    """The chat tools of a Responses tools list. A tool with no strict (absent or null) has no strict key here:
    strict_default sets it (server/responses_strict.py)."""
    if not isinstance(value, list):
        fail("tools must be a list.", "tools")
    tools = []
    for i, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            fail(f"tools[{i}].type must be function: Crucible supports function tools only.", f"tools[{i}].type")
        function = {key: tool[key] for key in ("name", "description", "parameters", "strict")
                    if key in tool and not (key in ("strict", "description") and tool[key] is None)}
        if set(tool) - {"type", "name", "description", "parameters", "strict"}:
            fail(f"tools[{i}]: a function tool supports name, description, parameters, and strict only.", "tools")
        tools.append(dict(type="function", function=function))
    return tools


def chat_choice(value):
    if value in ("auto", "none", "required"):
        return value
    if isinstance(value, dict) and set(value) == {"type", "name"} and value["type"] == "function":
        return dict(type="function", function=dict(name=value["name"]))
    fail("tool_choice must be auto, none, required, or a function by name.", "tool_choice")


def chat_body(body):
    """The chat request body for one Responses body. validate_request checks the result."""
    if not isinstance(body, dict):
        raise ApiError("The request must be a JSON object.")
    check_fields(body)
    result = dict(messages=chat_messages(body))
    for name in ("model", "temperature", "top_p", "top_k", "min_p", "seed", "parallel_tool_calls"):
        if body.get(name) is not None:
            result[name] = body[name]
    if body.get("max_output_tokens") is not None:
        result["max_tokens"] = body["max_output_tokens"]
    if body.get("tools") is not None:
        result["tools"] = chat_tools(body["tools"])
    if body.get("tool_choice") is not None:
        result["tool_choice"] = chat_choice(body["tool_choice"])
    answer = text_format((body.get("text") or {}).get("format"))
    if answer is not None:
        result["response_format"] = answer
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary", "generate_summary"}:
            fail("reasoning supports effort and summary only.", "reasoning")
        if reasoning.get("effort") is not None:
            result["reasoning_effort"] = reasoning["effort"]  # validate_request maps it (server/api.py EFFORTS)
    if body.get("stream") is not None:
        result["stream"] = body["stream"]
        if body["stream"] is True:
            result["stream_options"] = dict(include_usage=True)  # the counts of the final event
    return result


def error_body(error):
    kind = getattr(error, "kind", None) or ("invalid_request_error" if error.status < 500 else "server_error")
    return dict(error=dict(message=str(error), type=kind, param=error.param, code=None))


def usage(chat_usage):
    prompt, completion = chat_usage.get("prompt_tokens", 0), chat_usage.get("completion_tokens", 0)
    cached = (chat_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = (chat_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return dict(input_tokens=prompt, input_tokens_details=dict(cached_tokens=cached), output_tokens=completion,
                output_tokens_details=dict(reasoning_tokens=reasoning), total_tokens=prompt + completion)


def new_id(prefix):
    return prefix + uuid.uuid4().hex


def reasoning_item(text, status="completed", item_id=None):
    return dict(type="reasoning", id=item_id or new_id("rs_"), summary=[],
                content=[dict(type="reasoning_text", text=text)] if text is not None else [], status=status)


def message_item(text, status="completed", item_id=None):
    content = [dict(type="output_text", text=text, annotations=[], logprobs=[])] if text is not None else []
    return dict(type="message", id=item_id or new_id("msg_"), status=status, role="assistant", content=content)


def call_item(call_id, name, arguments, status="completed", item_id=None):
    return dict(type="function_call", id=item_id or new_id("fc_"), call_id=call_id, name=name, arguments=arguments,
                status=status)


class ResponsesSink(Sink):
    def __init__(self, handler, request, body):
        super().__init__(handler)
        self.request, self.body = request, body if isinstance(body, dict) else {}
        self.id, self.created = new_id("resp_"), int(time.time())
        self.sequence, self.items, self.open = 0, [], None  # open: dict(kind, index, item, text)
        self.calls, self.reason, self.counts, self.started = {}, None, {}, False

    def response(self, status, output, counts=None, reason=None):
        request, body = self.request, self.body
        incomplete = dict(reason="max_output_tokens") if reason == "length" else None
        return dict(id=self.id, object="response", created_at=self.created, status=status, model=request.model,
                    output=output, error=None, incomplete_details=incomplete, instructions=body.get("instructions"),
                    max_output_tokens=body.get("max_output_tokens"), parallel_tool_calls=request.parallel_tool_calls,
                    temperature=request.temperature, top_p=request.top_p, tool_choice=body.get("tool_choice", "auto"),
                    tools=echo_tools(body, request), reasoning=dict(effort=(body.get("reasoning") or {}).get("effort"),
                                                                  summary=None),
                    text=dict(format=echo_format(body)), store=False, previous_response_id=None, truncation="disabled",
                    metadata=body.get("metadata") or {}, usage=usage(counts) if counts is not None else None)

    def emit(self, event_type, /, **fields):
        self.event(event_type, dict(type=event_type, sequence_number=self.sequence) | fields)
        self.sequence += 1

    def start(self):
        if self.started:
            return
        self.started = True
        self.emit("response.created", response=self.response("in_progress", []))
        self.emit("response.in_progress", response=self.response("in_progress", []))

    def open_item(self, kind, item):
        self.close_item()
        self.open = dict(kind=kind, index=len(self.items), item=item, text="")
        self.items.append(item)
        self.emit("response.output_item.added", output_index=self.open["index"], item=item)
        part = dict(reasoning=dict(type="reasoning_text", text=""),
                    message=dict(type="output_text", text="", annotations=[])).get(kind)
        if part is not None:
            self.emit("response.content_part.added", item_id=item["id"], output_index=self.open["index"],
                      content_index=0, part=part)
        return self.open

    def close_item(self):
        state, self.open = self.open, None
        if state is None:
            return
        item, index, text = state["item"], state["index"], state["text"]
        if state["kind"] == "reasoning":
            done = reasoning_item(text, item_id=item["id"])
            self.emit("response.reasoning_text.done", item_id=item["id"], output_index=index, content_index=0, text=text)
            self.emit("response.content_part.done", item_id=item["id"], output_index=index, content_index=0,
                      part=done["content"][0])
        elif state["kind"] == "message":
            done = message_item(text, item_id=item["id"])
            self.emit("response.output_text.done", item_id=item["id"], output_index=index, content_index=0, text=text,
                      logprobs=[])
            self.emit("response.content_part.done", item_id=item["id"], output_index=index, content_index=0,
                      part=done["content"][0])
        else:
            done = call_item(item["call_id"], item["name"], text, item_id=item["id"])
            self.emit("response.function_call_arguments.done", item_id=item["id"], output_index=index, name=item["name"],
                      arguments=text)
        self.items[index] = done
        self.emit("response.output_item.done", output_index=index, item=done)

    def current(self, kind, make):
        return self.open if self.open is not None and self.open["kind"] == kind else self.open_item(kind, make())

    def chunk(self, body):
        self.start()
        if not body.get("choices"):
            self.counts = body.get("usage") or self.counts
            return
        choice = body["choices"][0]
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            state = self.current("reasoning", lambda: reasoning_item(None, "in_progress"))
            state["text"] += delta["reasoning_content"]
            self.emit("response.reasoning_text.delta", item_id=state["item"]["id"], output_index=state["index"],
                      content_index=0, delta=delta["reasoning_content"])
        if delta.get("content"):
            state = self.current("message", lambda: message_item(None, "in_progress"))
            state["text"] += delta["content"]
            self.emit("response.output_text.delta", item_id=state["item"]["id"], output_index=state["index"],
                      content_index=0, delta=delta["content"], logprobs=[])
        for call in delta.get("tool_calls") or ():
            function = call.get("function") or {}
            if "id" in call:
                self.calls[call["index"]] = self.open_item("function_call", call_item(
                    call["id"], function.get("name"), "", "in_progress"))
            state = self.calls.get(call["index"])
            if function.get("arguments") and state is not None:
                state["text"] += function["arguments"]
                self.emit("response.function_call_arguments.delta", item_id=state["item"]["id"],
                          output_index=state["index"], delta=function["arguments"])
        if choice.get("finish_reason"):
            self.reason = choice["finish_reason"]

    def done(self):
        self.start()
        self.close_item()
        status = "incomplete" if self.reason == "length" else "completed"
        self.emit("response.incomplete" if status == "incomplete" else "response.completed",
                  response=self.response(status, self.items, self.counts, self.reason))

    def reply(self, body):
        choice = body["choices"][0]
        message, output = choice["message"], []
        if message.get("reasoning_content"):
            output.append(reasoning_item(message["reasoning_content"]))
        if message.get("content"):
            output.append(message_item(message["content"]))
        for call in message.get("tool_calls") or ():
            output.append(call_item(call["id"], call["function"]["name"], call["function"]["arguments"]))
        reason = choice["finish_reason"]
        return self.response("incomplete" if reason == "length" else "completed", output, body["usage"], reason)


def stream_error(handler, error):
    sink = getattr(getattr(handler, "output", None), "handler", None)
    target = sink if isinstance(sink, ResponsesSink) else None
    sequence = target.sequence if target is not None else 0
    Sink(handler).event("error", dict(type="error", code="server_error" if error.status >= 500 else "invalid_request",
                                      message=str(error), param=error.param, sequence_number=sequence))


def output(handler, service, request, prompt_ids):
    """The CompletionResponse of a /v1/responses request, with a sink that writes the Responses shapes."""
    sink = ResponsesSink(handler, request, getattr(handler, "api_body", None))
    sink.output = CompletionResponse(sink, service, request, prompt_ids)
    return sink.output

