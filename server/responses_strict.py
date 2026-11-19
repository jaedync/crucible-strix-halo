"""The strict flag of a /v1/responses function tool that has none, as OpenAI's Responses API does.

https://platform.openai.com/docs/guides/function-calling ("Strict mode"): "If you omit `strict`, the default depends on
the API: Responses requests will attempt to normalize your schema into strict mode when possible, and will fall back to
non-strict, best-effort function calling if the schema cannot be made compatible with strict mode. When fallback
happens, the response tool will show `strict: false`. Chat Completions requests remain non-strict by default."

A tool with no strict (absent or null) gets strict true when the chat path's strict checks accept it: the tool checks
(tool_requests.validate_tools), the strict call grammar (structured.call_format) and the grammar limits of the whole
request (grammar_limits.GrammarBudget: the answer format and the explicit strict tools first, then the tools with no
strict in list order). Else it gets strict false and no grammar, with no error. On an engine with no structured
output no grammar can hold a tool, so each one falls back. An explicit strict keeps its value and errors.
The engine compiles the grammar after validation, so a compile refusal at open stays a 400 (no fallback there).
On an engine with no structured output the template renders only the strict flags that the client sent
(prompt_request), so the prompt there stays the prompt before this default.
"""
import copy
from dataclasses import replace

from .api import ApiError
from .grammar_limits import GrammarBudget
from .structured import call_format
from .tool_requests import validate_tools


def answer_schema(answer):
    """The schema of a chat response_format that the grammar limits count, or None."""
    if not isinstance(answer, dict) or answer.get("type") != "json_schema":
        return None
    spec = answer.get("json_schema")
    return spec.get("schema", {}) if isinstance(spec, dict) else None


def counted(function, choice):
    """True when structured.constraint counts this strict tool in the grammar limits: tool_choice none builds no
    grammar, and a named choice counts its tool only."""
    if choice == "none":
        return False
    named = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
    return named in (None, function.get("name"))


def fits(function, budget, choice):
    """The budget after one tool with strict true, or None when the strict checks refuse the tool."""
    try:
        tool = validate_tools([dict(type="function", function=function | dict(strict=True))])[0]
        call_format(tool, True)
        trial = copy.copy(budget)
        if counted(function, choice):
            trial.check(tool["function"].get("parameters", {}), "tools")
        return trial
    except ApiError:
        return None


def strict_budget(tools, unset, answer, choice):
    """The grammar budget after the answer format and every explicit strict true tool, or None when they alone go
    over a limit (validate_request then gives its own 400). The total limits are sums, so these count first: a tool
    with no strict never takes the room of a tool that the client marked strict."""
    budget = GrammarBudget()
    try:
        schema = answer_schema(answer)
        if schema is not None:
            budget.check(schema, "text.format")
        for i, tool in enumerate(tools):
            function = tool["function"]
            if i not in unset and function.get("strict") is True and counted(function, choice):
                budget.check(function.get("parameters", {}), "tools")
    except ApiError:
        return None
    return budget


def apply_default(tools, unset, answer=None, choice=None, structured=True):
    """Set strict on each chat function tool whose index is in `unset`, in place, in list order with the budget that
    the explicit strict tools leave. answer: the chat response_format; choice: the chat tool_choice. structured:
    False for an engine with no structured output; no grammar can hold a tool there, so each one falls back."""
    budget = strict_budget(tools, unset, answer, choice) if structured else None
    for i in sorted(unset):
        function = tools[i]["function"]
        after = fits(function, budget, choice) if budget is not None else None
        function["strict"] = after is not None
        budget = after or budget


def strict_default(chat, structured=True):
    """http.py calls this with the chat body of a Responses request before validate_request: it sets strict on each
    function tool that has none (chat_tools leaves the key out)."""
    tools = chat.get("tools") if isinstance(chat, dict) else None
    if not tools:
        return
    unset = {i for i, tool in enumerate(tools) if "strict" not in tool["function"]}
    apply_default(tools, unset, chat.get("response_format"), chat.get("tool_choice"), structured)


def prompt_request(request, body):
    """The request with the template's tools when the template renders strict flags (an engine with no structured
    output: Request.render_tools None, RENDER_STRICT): a strict that the server set is left out of the prompt."""
    tools = body.get("tools") if isinstance(body, dict) else None
    if request.render_tools is not None or not request.tools or not isinstance(tools, list):
        return request
    sent = [isinstance(tool, dict) and tool.get("strict") is not None for tool in tools]
    if all(sent) or len(sent) != len(request.tools):
        return request
    rendered = tuple(tool if keep else dict(tool, function={key: value for key, value in tool["function"].items()
                                                            if key != "strict"})
                     for tool, keep in zip(request.tools, sent))
    return replace(request, render_tools=rendered)
