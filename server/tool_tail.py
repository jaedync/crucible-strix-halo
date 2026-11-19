"""Session replay after a tool-call turn: extend the session tail with the next request's render of the reply.

The engine resumes a turn from the state after prompt + generated IDs only when the next prompt repeats those IDs.
The chat template renders a returned tool call from its parsed arguments, so the next prompt can leave that tail
inside the call: the model wrote compact or indented JSON or \\u escapes, or a token split that the tokenizer does not
make. The engine then prefilled the rest of the earlier output again, on the next turn's first-token path.

Rule E1: the prompt stays a function of the request only. When a turn ends with tool
calls, the server renders the returned message as the next request renders it (the request's messages, then this
reply) and encodes it. If those IDs leave prompt + generated IDs inside the reply, the server sends one more job with
those IDs, after the response, while the client runs its tools. The engine resumes it from its best saved point.

The job shape follows the engine's tail rule: a job that generates fewer than 2 tokens keeps no tail
(engine/runtime/conversation_resume.h:85; batched_serve.h:1034-1038 logs `session_replay skip`), and a longer one keeps
ids + generated[:-1] (batched_serve.h:1044-1046). So the job is greedy with max_tokens 2. After a closed assistant
message the model writes the template's start of the next message (<|im_start|>), which the next request renders
there too: the next request then resumes at the end of the reply + 1 row. If the model writes another token, the
next request diverges at that last row and resumes from the job's slot points (the end-64 Pre point, else a host
frontier on the 4,096-row grid), as a max_tokens 1 job would. Nothing is remembered: a restart or another session
changes no prompt. The think close does the same on the generation side (http.close_thinking, closed_think_ids).
"""
from dataclasses import replace
import json
import queue
import threading
import time

from . import metrics
from .api import ApiError
from .tool_requests import normalize_message

# The extension job's own wait for its events (it holds no client): a queue wait plus the prefill of a long reply.
EXTEND_SECONDS = 900.0
# A next request whose prompt extends a running extension waits for it at most this long, then goes on as on trunk.
WAIT_SECONDS = 120.0
# The wait checks the client (idle) this often, as the handler's event wait does (http.POLL_SECONDS).
POLL_SECONDS = 0.25
# The tokens that the job generates: 2, the fewest that the engine keeps as a tail (ConversationTailState).
JOB_TOKENS = 2


class Pending:
    """The extension jobs that run now. A next request that extends one waits for it: else it would restore the
    older tail and prefill the reply again while the extension job prefills the same tokens."""
    def __init__(self):
        self.items, self.lock = [], threading.Lock()

    def add(self, ids):
        entry = (tuple(ids), threading.Event())
        with self.lock:
            self.items.append(entry)
        return entry

    def finish(self, entry):
        entry[1].set()
        with self.lock:
            if entry in self.items:
                self.items.remove(entry)

    def running(self, prompt_ids):
        """The running extensions whose IDs start prompt_ids."""
        with self.lock:
            return [entry for entry in self.items if len(entry[0]) <= len(prompt_ids) and
                    tuple(prompt_ids[:len(entry[0])]) == entry[0]]

    def wait(self, prompt_ids, timeout=None, idle=None, poll=None):
        """Wait for the running extensions whose IDs start prompt_ids. idle() runs between polls (the handler sends
        its keepalives and checks the client there); an exception from it ends the wait. Returns (seconds, done) or
        None when no extension runs."""
        found = self.running(prompt_ids)
        if not found:
            return None
        start = time.monotonic()
        deadline = start + (WAIT_SECONDS if timeout is None else timeout)
        step = POLL_SECONDS if poll is None else poll
        for _, event in found:
            while not event.wait(max(0.0, min(step, deadline - time.monotonic()))):
                if time.monotonic() >= deadline:
                    break
                if idle is not None:
                    idle()
        done = all(event.is_set() for _, event in found)
        seconds = round(time.monotonic() - start, 3)
        metrics.emit("crucible_tail_wait " + json.dumps(dict(seconds=seconds, outcome="done" if done else "expired")))
        return seconds, done


def next_prompt_ids(tokenizer, request, message):
    """The IDs that the next request renders up to the end of this reply, when it sends the reply back unchanged."""
    reply = dict(role="assistant", content=message.get("content"), tool_calls=message["tool_calls"])
    if message.get("reasoning_content"):
        reply["reasoning_content"] = message["reasoning_content"]
    following = replace(request, messages=tuple(request.messages) + (normalize_message(reply, len(request.messages)),))
    return tokenizer.encode_prompt(*tokenizer.render_prompt(following, add_generation_prompt=False))


def shared(left, right):
    return next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def extension(tokenizer, request, prompt_ids, output_ids, message):
    """(IDs of the extension job, the tokens they share with prompt + output), or None when none is needed or useful."""
    if (getattr(request, "kind", None) != "chat" or not message.get("tool_calls") or getattr(request, "images", ())
            or getattr(request, "constraint", None) is not None or request.cache_control != "auto"
            or not hasattr(tokenizer, "render_prompt")):
        return None
    ids = next_prompt_ids(tokenizer, request, message)
    tail = list(prompt_ids) + list(output_ids)
    common = shared(tail, ids)
    # The next prompt extends the tail already; or it leaves the prompt itself, where the reply cannot help.
    return None if common == len(tail) or common < len(prompt_ids) else (ids, common)


def extend(service, request, prompt_ids, output_ids, message):
    """Send the extension job of a finished tool-call turn, and drain it on a thread of its own. Returns the job.
    It runs after the response, so any fault leaves the turn as on trunk: one line of counts, no traceback."""
    fields = dict(previous_tokens=len(prompt_ids) + len(output_ids))
    try:
        found = extension(service.tokenizer, request, prompt_ids, output_ids, message)
    except Exception as error:
        metrics.emit("crucible_tail_extend " + json.dumps(fields | dict(outcome="fault", error=type(error).__name__)))
        return None
    if found is None:
        return None
    ids, common = found
    # Greedy and plain: the job's output is not sent; the engine keeps ids + its first token as the tail. It keeps the
    # request's RoPE mode (the tail's engine policy) but not its clamp, so it writes no rope clamp or cut line.
    job_request = replace(request, max_tokens=JOB_TOKENS, stream=False, top_logprobs=0, temperature=0, stop=(),
                          rope_clamped_from=0)
    fields |= dict(prompt_tokens=len(ids), shared_tokens=common)
    pending = getattr(service, "tail_extensions", None)
    entry = pending.add(ids) if pending is not None else None
    try:
        job = service.backend.submit(ids, job_request, log_turn=False)
    except Exception as error:
        if entry is not None:
            pending.finish(entry)
        refused = isinstance(error, ApiError)
        metrics.emit("crucible_tail_extend " + json.dumps(fields | (dict(outcome="refused", status=error.status)
                                                                    if refused else
                                                                    dict(outcome="fault", error=type(error).__name__))))
        return None
    threading.Thread(target=drain, args=(job, fields, pending, entry), daemon=True, name="tail-extend").start()
    return job


def drain(job, fields, pending=None, entry=None):
    """Read the job's events to its end and write one line of counts (no text)."""
    start, outcome, cached = time.monotonic(), "expired", None
    deadline = start + EXTEND_SECONDS
    try:
        while (left := deadline - time.monotonic()) > 0:
            try:
                event = job.events.get(timeout=left)
            except queue.Empty:
                break
            if event.get("event") == "error":
                outcome = "error"
                break
            if event.get("event") == "done":
                outcome, cached = "done", event.get("cached_tokens")
                break
    finally:
        job.cancelled.set()
        if entry is not None:
            pending.finish(entry)
    seconds = round(time.monotonic() - start, 3)
    metrics.emit("crucible_tail_extend " + json.dumps(fields | dict(outcome=outcome, cached_tokens=cached,
                                                                     seconds=seconds)))
