"""Anthropic Messages request -> the server's chat request body, for POST /v1/messages/count_tokens.

The result goes through validate_request and the chat path's prompt-ID function, so the count is the number of IDs
that /v1/chat/completions prefills for the same conversation with the same server flags.
"""
import json
from .api import ApiError

# Known Anthropic fields that do not change this server's prompt. They are accepted so that a client can send its
# generation body as it is. Any other field is refused, because it could change the prompt.
IGNORED = {"max_tokens", "stream", "temperature", "top_p", "top_k", "stop_sequences", "metadata", "service_tier",
           "context_management", "cache_control", "container"}
FIELDS = IGNORED | {"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config"}
# cache_control and citations are client hints with no prompt text.
BLOCK_HINTS = {"cache_control", "citations"}
# output_config.format maps to the chat field response_format (server/structured.py); its errors name this place.
FORMAT_FIELD = "output_config.format"
# The template never renders a tool's strict flag (api.validate_request render_strict), also on an engine with no
# structured output: an Anthropic tool never had the flag in its prompt.
RENDER_STRICT = False


def fail(message, param="messages"):
    raise ApiError(message, param)


def block_fields(block, allowed):
    if not isinstance(block, dict) or set(block) - allowed - BLOCK_HINTS - {"type"}:
        fail("A content block has unsupported fields.")  # a fixed text: the block type is client data


def text_of(block):
    if not isinstance(block, dict) or block.get("type") != "text":
        fail("This content supports text blocks only.")
    block_fields(block, {"text"})
    if not isinstance(block.get("text"), str):
        fail("A text block needs text.")
    return block["text"]


def image_part(block):
    """An image block -> an OpenAI image_url part. The chat validation decodes it, or refuses it without --vision."""
    block_fields(block, {"source"})
    source = block.get("source")
    if not isinstance(source, dict):
        fail("An image block needs a source.")
    if source.get("type") == "base64" and set(source) <= {"type", "media_type", "data"}:
        if not isinstance(source.get("media_type"), str) or not isinstance(source.get("data"), str):
            fail("A base64 image source needs media_type and data.")
        return dict(type="image_url", image_url=dict(url=f"data:{source['media_type']};base64,{source['data']}"))
    if source.get("type") == "url" and set(source) == {"type", "url"} and isinstance(source["url"], str):
        return dict(type="image_url", image_url=dict(url=source["url"]))
    fail("An image source must be base64 or url.")


def blocks(content):
    """String content -> one text block, so every role reads blocks."""
    if isinstance(content, str):
        return [dict(type="text", text=content)]
    if not isinstance(content, list):
        fail("Message content must be text or a list of content blocks.")
    return content


def tool_result(block):
    block_fields(block, {"tool_use_id", "content", "is_error"})
    if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
        fail("A tool_result block needs a tool_use_id.")
    if "is_error" in block and type(block["is_error"]) is not bool:
        fail("tool_result is_error must be true or false.")
    parts = []
    for item in blocks(block.get("content", "")):
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            parts.append(dict(type="text", text=text_of(item)))
        elif kind == "image":
            parts.append(image_part(item))
        else:
            fail("tool_result content supports text and image blocks.")
    if block.get("is_error"):
        # The chat template has no error field, so the flag is kept as text.
        parts.insert(0, dict(type="text", text="Error: "))
    images = any(part["type"] == "image_url" for part in parts)
    content = parts if images else "".join(part["text"] for part in parts)
    return dict(role="tool", tool_call_id=block["tool_use_id"], content=content)


def user_messages(content):
    """Tool results become tool messages, in order, before the user's own text and images."""
    results, parts = [], []
    for block in blocks(content):
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "tool_result":
            results.append(tool_result(block))
        elif kind == "text":
            parts.append(dict(type="text", text=text_of(block)))
        elif kind == "image":
            parts.append(image_part(block))
        else:
            fail("User content supports text, image, and tool_result blocks.")
    return results + ([dict(role="user", content=parts)] if parts else [])


def assistant_message(content):
    text, reasoning, calls = [], [], []
    for block in blocks(content):
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            text.append(text_of(block))
        elif kind == "thinking":
            block_fields(block, {"thinking", "signature"})
            if not isinstance(block.get("thinking"), str):
                fail("A thinking block needs thinking text.")
            reasoning.append(block["thinking"])
        elif kind == "redacted_thinking":
            block_fields(block, {"data"})  # encrypted text that this model cannot read
        elif kind == "tool_use":
            block_fields(block, {"id", "name", "input"})
            if not isinstance(block.get("id"), str) or not isinstance(block.get("input"), dict):
                fail("A tool_use block needs an id, a name, and an input object.")
            arguments = json.dumps(block["input"], ensure_ascii=False, allow_nan=False)
            calls.append(dict(id=block["id"], type="function", function=dict(name=block.get("name"), arguments=arguments)))
        else:
            fail("Assistant content supports text, thinking, and tool_use blocks.")
    message = dict(role="assistant", content="".join(text) if text or not calls else None)
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    return message


def chat_messages(body):
    messages = []
    if "system" in body:
        system = body["system"]
        if not isinstance(system, str):
            system = [dict(type="text", text=text_of(block)) for block in blocks(system)]
        messages.append(dict(role="system", content=system))
    value = body.get("messages")
    if not isinstance(value, list) or not value:
        fail("messages must be a nonempty list.")
    for message in value:
        if not isinstance(message, dict) or set(message) - {"role", "content"} or "content" not in message:
            fail("Each message needs role and content only.")
        role = message.get("role")
        if role == "system":
            content = message["content"]
            messages.append(dict(role="system", content=content if isinstance(content, str) else
                                 [dict(type="text", text=text_of(block)) for block in blocks(content)]))
        elif role == "user":
            messages.extend(user_messages(message["content"]))
        elif role == "assistant":
            messages.append(assistant_message(message["content"]))
        else:
            fail("Message roles must be system, user, or assistant.")
    return messages


def chat_tools(value):
    if not isinstance(value, list):
        fail("tools must be a list.", "tools")
    tools = []
    for tool in value:
        if (not isinstance(tool, dict) or set(tool) - {"name", "description", "input_schema", "type", "cache_control",
                                                       "strict"}
                or tool.get("type", "custom") != "custom"):
            fail("Each tool needs name, description, and input_schema. Server tools are not supported.", "tools")
        # The template renders each tool as JSON, so the keys keep the usual OpenAI order: name, description, parameters.
        function = dict(name=tool.get("name"))
        if "description" in tool:
            function["description"] = tool["description"]
        function["parameters"] = tool.get("input_schema", dict(type="object", properties={}))
        # strict: true is the strict call grammar of a chat function tool (server/structured.py); null is absent.
        # The template never renders the flag (RENDER_STRICT, Request.render_tools).
        if tool.get("strict") is not None:
            if type(tool["strict"]) is not bool:
                fail("A tool's strict must be true or false.", "tools")
            function["strict"] = tool["strict"]
        tools.append(dict(type="function", function=function))
    return tools


def chat_choice(value, body):
    if not isinstance(value, dict) or set(value) - {"type", "name", "disable_parallel_tool_use"}:
        fail("tool_choice must be an object with a type.", "tool_choice")
    if "disable_parallel_tool_use" in value:
        if type(value["disable_parallel_tool_use"]) is not bool:
            fail("disable_parallel_tool_use must be true or false.", "tool_choice")
        body["parallel_tool_calls"] = not value["disable_parallel_tool_use"]
    kind = value.get("type")
    if kind in ("auto", "none") and "name" not in value:
        return kind
    if kind == "any" and "name" not in value:
        return "required"
    if kind == "tool":
        return dict(type="function", function=dict(name=value.get("name")))
    fail("tool_choice type must be auto, none, any, or tool.", "tool_choice")


def thinking_switch(value, body):
    if not isinstance(value, dict) or value.get("type") not in ("enabled", "disabled", "adaptive"):
        fail("thinking type must be enabled, disabled, or adaptive.", "thinking")
    if set(value) - {"type", "budget_tokens"} or ("budget_tokens" in value and value["type"] != "enabled"):
        fail("thinking supports type and, when enabled, budget_tokens.", "thinking")
    body["enable_thinking"] = value["type"] != "disabled"  # the budget limits output, not the prompt


def chat_body(body):
    """The server's chat request body for one Anthropic Messages body. validate_request checks the result."""
    if not isinstance(body, dict):
        fail("The request must be a JSON object.", None)
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        fail(f"{unknown[0]} is not supported by count_tokens.", unknown[0])
    result = dict(messages=chat_messages(body))
    if "model" in body:
        result["model"] = body["model"]
    if body.get("tools") is not None:
        result["tools"] = chat_tools(body["tools"])
    if body.get("tool_choice") is not None:
        result["tool_choice"] = chat_choice(body["tool_choice"], result)
    if body.get("thinking") is not None:
        thinking_switch(body["thinking"], result)
    config = body.get("output_config")
    if config is not None:
        if not isinstance(config, dict) or set(config) - {"effort", "format"}:
            fail("output_config supports effort and format only.", "output_config")
        if "effort" in config:
            result["reasoning_effort"] = config["effort"]  # validate_request maps the level (server/api.py EFFORTS)
        if config.get("format") is not None:
            result["response_format"] = output_format(config["format"])
    return result


def output_format(value):
    """output_config.format -> the chat response_format. The chat path checks the schema (server/structured.py
    response_format), so the grammar, the refused keywords and the limits are the same."""
    if not isinstance(value, dict) or value.get("type") != "json_schema":
        fail(f"{FORMAT_FIELD} type must be json_schema.", FORMAT_FIELD)
    if not isinstance(value.get("schema"), dict):
        fail(f"{FORMAT_FIELD} json_schema needs a schema object.", FORMAT_FIELD)
    if set(value) != {"type", "schema"}:
        fail(f"{FORMAT_FIELD} json_schema supports type and schema only.", FORMAT_FIELD)
    return dict(type="json_schema", json_schema=dict(schema=value["schema"]))
