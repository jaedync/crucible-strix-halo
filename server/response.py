"""Build text and tool responses without changing accepted engine tokens."""
import dataclasses
import json
import math
import time
import uuid
from . import hy4_format, metrics
from .api import ApiError
from .timings import timings
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
        # Client stop strings do not cut a constrained answer: it would no longer parse.
        self.stop_filter = StopSequences(() if getattr(request, "constraint", None) is not None else request.stop)
        stream_calls = request.stream and getattr(service, "stream_tool_calls", True)
        # The Hy4-flash family has its own tool tags and think close tag (server/hy4_format.py).
        hy4 = getattr(service.tokenizer, "hy4", False) is True
        output = hy4_format.Hy4ToolOutput if hy4 else ToolOutput
        self.tools = (output(request, stream_calls, getattr(service, "extend_tool_tails", False))
                      if request.kind == "chat" and request.tools else None)
        end_id = getattr(service.tokenizer, "think_end_id", None)  # None for a stub tokenizer: the text ends it
        self.think = ThinkSplitter(request.kind == "chat" and request.enable_thinking,
                                   end_id=end_id if type(end_id) is int else None,
                                   **(dict(end=hy4_format.THINK_END) if hy4 else {}))
        self.reasoning_tokens = 0
        self.closed = None  # the ThinkCap when the server closed the think block
        self.end_of_turn_kept = dict(reasoning=0, tool_call=0)  # http.keep_end_of_turn
        self.end_of_turn_at = None  # len(ids) after the last kept text (http.keep_end_of_turn)
        self.reasoning_calls = None  # the reasoning text whose calls the reply makes (calls_from_reasoning)
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

    def chunk(self, delta, reason=None, token=None, extra=None, ended=False):
        choice = dict(index=0, finish_reason=reason)
        if ended:
            choice["stop_reason"] = self.stop_reason()  # only on the chunk that ends the choice, as its finish_reason
        choice["delta" if self.request.kind == "chat" else "text"] = delta
        if token is not None and self.request.return_token_ids:
            choice["token_id"] = token
        usage = dict(usage=None) if self.request.include_usage else {}
        stamp = dict(crucible_t=self.token_time) if self.request.crucible_metrics and token is not None else {}
        self.handler.sse(self.base | dict(choices=[choice]) | usage | stamp | (extra or {}))

    def content(self, text, token=None):
        if self.request.stream and (text or (token is not None and self.request.return_token_ids)):
            self.chunk(dict(content=text) if self.request.kind == "chat" else text, token=token)
            self.displayed += len(text)

    def reasoning(self, text, token=None):
        if self.request.stream and (text or (token is not None and self.request.return_token_ids)):
            self.chunk(dict(reasoning_content=text), token=token)

    def stop_reason(self):
        """other engines's stop_reason: the client stop string that ended the reply, else None (an end-of-turn token or
        max_tokens). token_ids keep the IDs of that string, as other engines keeps them: a client that rebuilds the text from
        the IDs cuts it at this string, and with None it removes only a last end-of-turn ID."""
        return self.stop_filter.matched if self.stop_filter.stopped else None

    def token(self, token, server_time=None):
        self.ids.append(token)
        self.token_time = server_time
        raw_delta = self.service.tokenizer.step(self.stream, token)
        self.raw_text += raw_delta
        thinking = self.think.thinking
        reasoning, answer = self.think.feed(raw_delta, token)
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

    def calls_from_reasoning(self):
        """Halogen's tool_call_from_reasoning (the end-of-turn guard). The
        model ends its turn inside its reasoning: if that reasoning parses as text and then complete calls under the
        request's tool policy (the class of self.tools, the parser of the answer), the reply makes those calls. The call
        text stays in reasoning_content, where the stream sent it. Returns True when finish() makes the calls.
        The rule also reads the reasoning when the turn ends right after the end tag with an answer of whitespace only:
        those calls are then the only calls of the reply. An answer with text keeps them as reasoning text."""
        if self.tools is None:
            return False
        if self.think.thinking:
            text = self.think.reasoning + self.think.pending
        elif self.think.enabled and not self.think.answer.strip():
            # The splitter's answer: with an end token ID, a quoted end tag before it is reasoning.
            text = self.think.reasoning
        else:
            return False
        try:
            calls = self.reasoning_parser(text).checked_calls()
        except ApiError as error:
            if error.status != 502:
                raise
            return False
        if calls:
            self.reasoning_calls = text
        return bool(calls)

    def reasoning_parser(self, text):
        # The class of the answer parser: ToolOutput, or Hy4ToolOutput for a Hy4 tokenizer (__init__). The grammar starts
        # after the end tag and never constrains the reasoning, so the reasoning parser reads it with no grammar: prose
        # that names <tool_call> there is prose (ToolOutput.prose_start).
        request = self.request
        if getattr(request, "constraint", None) is not None:
            request = dataclasses.replace(request, constraint=None)
        parser = type(self.tools)(request, False, self.tools.whole)
        parser.feed(text)
        return parser

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
        if self.reasoning_calls is not None:
            # The calls of the reasoning (calls_from_reasoning); its text is not the answer.
            self.tools = self.reasoning_parser(self.reasoning_calls)
            message, deltas, reason = self.tools.finish(reason)
            message["content"] = None
        elif self.tools:
            self.tools.feed(tail)
            message, deltas, reason = self.tools.finish(reason)
        else:
            message = dict(role="assistant", content=full_text)
        if self.think.enabled:
            message["reasoning_content"] = self.think.reasoning.strip()
        self.message = message  # the returned reply (tool_tail.extend renders it as the next request does)
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
        if self.reasoning_calls is not None:
            usage["completion_tokens_details"] |= dict(tool_call_from_reasoning=len(message.get("tool_calls") or ()))
        kept = sum(self.end_of_turn_kept.values())
        if kept:
            # Halogen's field: end-of-turn tokens that the server wrote as text instead of ending the reply.
            usage["completion_tokens_details"] = usage.get("completion_tokens_details", {}) | dict(end_of_turn_kept=kept)
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
            from .structured import violation
            metrics.emit(metrics.fallback_line(request_id, len(self.ids), self.request.stream, self.tools.invalid,
                                                getattr(self.handler, "client_request_id", None),
                                                # A fault in the grammar's free text is not an engine fault.
                                                violation=violation(self.request) and not self.tools.free_text))
        elif self.tools and self.tools.unchecked and message.get("tool_calls"):
            metrics.emit(metrics.tool_pass_line(request_id, self.request.stream, len(message["tool_calls"]),
                                                self.tools.unchecked))
        # OpenAI-compatible `timings` (server/timings.py): on the response, or on the last chunk of a stream.
        timing = timings(jobs if jobs != [None] else [], self.prompt_count, len(self.ids), event.get("cached_tokens"))
        timing = dict(timings=timing) if timing else {}
        if self.request.stream:
            self.reasoning(reasoning_tail)
            self.finish_stream(message, deltas, reason, usage, timing)
        else:
            choice = dict(index=0, finish_reason=reason, stop_reason=self.stop_reason())
            choice["message" if self.request.kind == "chat" else "text"] = message if self.request.kind == "chat" else full_text
            if self.request.return_token_ids:
                choice["token_ids"] = self.ids
            if getattr(self.request, "top_logprobs", 0):
                choice["logprobs"] = scoring_field(self.request, self.ids[0], self.scores)
            self.handler.send_json(200, self.base | dict(
                object="chat.completion" if self.request.kind == "chat" else "text_completion", choices=[choice], usage=usage) |
                timing)

    def finish_stream(self, message, deltas, reason, usage, timing=None):
        self.content((message["content"] or "")[self.displayed:])
        for delta in deltas:
            self.chunk(delta)
        last = timing if not self.request.include_usage else None
        self.chunk({} if self.request.kind == "chat" else "", reason, extra=last, ended=True)
        if self.request.include_usage:
            self.handler.sse(self.base | dict(choices=[], usage=usage) | (timing or {}))
        self.handler.sse("[DONE]")
        self.handler.close_connection = True
