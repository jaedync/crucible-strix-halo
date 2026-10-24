"""Build text and tool responses without changing accepted engine tokens."""
import json
import math
import time
import uuid
from . import metrics
from .api import ApiError
from .tool_output import ToolOutput
from .stop_sequences import StopSequences
from .thinking import ThinkSplitter

def scoring_field(request, token, scores):
    """The `logprobs` field of a scoring completion. Check the engine's list before it reaches a client."""
    if (type(scores) is not list or not 0 < len(scores) <= 20 or
            any(type(pair) is not list or len(pair) != 2 or type(pair[0]) is not int or
                type(pair[1]) not in (int, float) or not math.isfinite(pair[1]) for pair in scores)):
        raise ApiError("The engine returned an invalid logprobs list.", status=500)
    top = [dict(token_id=pair[0], logprob=pair[1]) for pair in scores]
    chosen = next((entry["logprob"] for entry in top if entry["token_id"] == token), None)
    return dict(content=[dict(token_id=token, logprob=chosen, top_logprobs=top[:max(request.top_logprobs, 1)])])

class CompletionResponse:
    def __init__(self, handler, service, request, prompt_ids):
        self.handler, self.service, self.request = handler, service, request
        self.prompt_count = len(prompt_ids)
        self.base = dict(id=("chatcmpl-" if request.kind == "chat" else "cmpl-") + uuid.uuid4().hex,
                         object="chat.completion.chunk" if request.kind == "chat" else "text_completion",
                         created=int(time.time()), model=service.model, seed=request.seed)
        self.stream = service.tokenizer.stream()
        self.ids, self.text, self.raw_text, self.displayed = [], "", "", 0
        self.scores = None  # the engine's [[id, logprob], ...] for a scoring request
        self.stop_filter = StopSequences(request.stop)
        stream_calls = request.stream and getattr(service, "stream_tool_calls", True)
        self.tools = ToolOutput(request, stream_calls) if request.kind == "chat" and request.tools else None
        self.think = ThinkSplitter(request.kind == "chat" and request.enable_thinking)
        self.reasoning_tokens = 0
        self.closed = None  # the ThinkCap when the server closed the think block
        self.token_time = None
        self.headers_sent = False

    def headers(self):
        """Send the stream's HTTP 200 headers. The server calls this at admission, before the prompt prefills."""
        if not self.request.stream or self.headers_sent:
            return
        self.handler.send_response(200)
        self.handler.send_header("Content-Type", "text/event-stream")
        self.handler.send_header("Cache-Control", "no-cache")
        self.handler.send_header("Connection", "close")
        self.handler.end_headers()
        self.handler.wfile.flush()
        self.headers_sent = True

    def start(self):
        if not self.request.stream:
            return
        self.headers()
        if self.request.kind == "chat":
            self.chunk(dict(role="assistant", content=""))

    def chunk(self, delta, reason=None, token=None):
        choice = dict(index=0, finish_reason=reason)
        choice["delta" if self.request.kind == "chat" else "text"] = delta
        if token is not None and self.request.return_token_ids:
            choice["token_id"] = token
        usage = dict(usage=None) if self.request.include_usage else {}
        stamp = dict(crucible_t=self.token_time) if self.request.crucible_metrics and token is not None else {}
        self.handler.sse(self.base | dict(choices=[choice]) | usage | stamp)

    def content(self, text, token=None):
        if self.request.stream and (text or (token is not None and self.request.return_token_ids)):
            self.chunk(dict(content=text) if self.request.kind == "chat" else text, token=token)
            self.displayed += len(text)

    def reasoning(self, text, token=None):
        if self.request.stream and (text or (token is not None and self.request.return_token_ids)):
            self.chunk(dict(reasoning_content=text), token=token)

    def token(self, token, server_time=None):
        self.ids.append(token)
        self.token_time = server_time
        raw_delta = self.service.tokenizer.step(self.stream, token)
        self.raw_text += raw_delta
        thinking = self.think.thinking
        reasoning, answer = self.think.feed(raw_delta)
        if thinking:
            # The token belongs to the reasoning; any answer text after the end tag carries no second token ID.
            self.reasoning_tokens += 1
            self.reasoning(reasoning, token)
            token = None
        delta = self.stop_filter.feed(answer)
        self.text += delta
        self.content(self.tools.feed(delta) if self.tools else delta, token)
        if self.tools:
            for tool_delta in self.tools.take_deltas():
                self.chunk(tool_delta)

    def finish(self, event):
        decoded = self.service.tokenizer.decode(self.ids)
        if not decoded.startswith(self.raw_text):
            raise ApiError("The incremental tokenizer output is inconsistent.", status=500)
        reasoning, answer = self.think.feed(decoded[len(self.raw_text):])
        held, _ = self.think.flush()
        reasoning_tail = reasoning + held
        tail = self.stop_filter.feed(answer) + self.stop_filter.flush()
        full_text = self.text + tail
        reason = "stop" if self.stop_filter.stopped else event["finish_reason"]
        deltas = []
        if self.tools:
            self.tools.feed(tail)
            message, deltas, reason = self.tools.finish(reason)
        else:
            message = dict(role="assistant", content=full_text)
        if self.think.enabled:
            message["reasoning_content"] = self.think.reasoning.strip()
        remember = getattr(self.service.tokenizer, "remember_output", None)
        if remember is not None and self.request.kind == "chat":
            remember(self.ids, message)  # replay: the next turn keeps the added tokens that the model wrote here
        usage = dict(prompt_tokens=self.prompt_count, completion_tokens=len(self.ids),
                     total_tokens=self.prompt_count + len(self.ids))
        if self.think.enabled:
            usage["completion_tokens_details"] = dict(reasoning_tokens=self.reasoning_tokens)
            if self.closed:
                usage["completion_tokens_details"] |= dict(reasoning_closed_at=self.closed.tokens,
                                                           reasoning_closed_by=self.closed.by)
        elif self.request.thinking_skipped_by:
            usage["completion_tokens_details"] = dict(reasoning_skipped_by=self.request.thinking_skipped_by)
        if "cached_tokens" in event:
            cached = event["cached_tokens"]
            if type(cached) is not int or not 0 <= cached <= self.prompt_count:
                raise ApiError("The engine returned an invalid cache count.", status=500)
            usage["prompt_tokens_details"] = dict(cached_tokens=cached)
        if self.request.crucible_metrics:
            usage["crucible_metrics"] = event.get("crucible_metrics")
        jobs = getattr(self.handler, "jobs", None) or [None]
        turn = getattr(jobs[0], "turn", None)
        request_id = turn.get("request_id") if isinstance(turn, dict) else None
        if self.tools and self.tools.fallback:
            usage["tool_call_fallback"] = True
            usage["tool_call_invalid"] = self.tools.invalid
            metrics.emit(metrics.fallback_line(request_id, len(self.ids), self.request.stream, self.tools.invalid))
        elif self.tools and self.tools.unchecked and message.get("tool_calls"):
            metrics.emit(metrics.tool_pass_line(request_id, self.request.stream, len(message["tool_calls"]),
                                                self.tools.unchecked))
        if self.request.stream:
            self.reasoning(reasoning_tail)
            self.finish_stream(message, deltas, reason, usage)
        else:
            choice = dict(index=0, finish_reason=reason)
            choice["message" if self.request.kind == "chat" else "text"] = message if self.request.kind == "chat" else full_text
            if self.request.return_token_ids:
                choice["token_ids"] = self.ids
            if getattr(self.request, "top_logprobs", 0):
                choice["logprobs"] = scoring_field(self.request, self.ids[0], self.scores)
            self.handler.send_json(200, self.base | dict(
                object="chat.completion" if self.request.kind == "chat" else "text_completion", choices=[choice], usage=usage))

    def finish_stream(self, message, deltas, reason, usage):
        self.content((message["content"] or "")[self.displayed:])
        for delta in deltas:
            self.chunk(delta)
        self.chunk({} if self.request.kind == "chat" else "", reason)
        if self.request.include_usage:
            self.handler.sse(self.base | dict(choices=[], usage=usage))
        self.handler.sse("[DONE]")
        self.handler.close_connection = True
