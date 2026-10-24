"""Opt-in per-turn timing for the multi-session benchmark (gap G16).

All times use one server clock (time.monotonic). Unknown values stay None, never estimates.
"""
from collections import deque
import hashlib
import json
from os.path import basename
import re
import sys
import threading
import time
from . import engine_messages
from traceback import extract_tb

# --log-engine-errors: also log the raw engine error text. Off by default, because a future engine text could quote
# request content.
LOG_ENGINE_TEXT = False

TRACE_LIMIT = 10000
# The 8-row path can run 16K chunks for one 128K prompt; keep the response bounded.
CHUNK_LIMIT = 1024
now = time.monotonic


def new_turn(request_id=None):
    return dict(enabled=False, request_id=request_id, queued=None, t_submit=now(), t_admit=None, t_first_gpu=None,
                t_prefill_end=None, t_first_token=None, t_done=None, prefill_chunks=[], chunk_count=0, prefill_sum=0.0)


def add_chunk(turn, start, end, rows):
    turn["chunk_count"] += 1
    turn["prefill_sum"] += end - start
    if len(turn["prefill_chunks"]) < CHUNK_LIMIT: turn["prefill_chunks"].append((start, end, rows))


def _span(end, start):
    return None if end is None or start is None else end - start


def finalize(turn, event):
    """Build usage.crucible_metrics. A metrics fault must not stop the worker, so it yields None."""
    try:
        return _finalize(turn, event)
    except Exception:
        return None


def _finalize(turn, event):
    cached = event.get("cached_tokens")
    label = None
    if type(cached) is int:
        label = "miss" if cached == 0 else "hit_" + str(event.get("cache_class", "session"))
        # G15: the restored state is the previous turn's decode-time tail, not a bulk prompt state.
        if cached and event.get("session_replay") is True: label = "hit_replay"
    restore = event.get("restore_seconds")
    prefill = turn["prefill_sum"] if turn["chunk_count"] else None
    if prefill is None and type(event.get("prefill_seconds")) in (int, float):
        prefill = float(event["prefill_seconds"])
    return dict(
        request_id=turn["request_id"], session_label=label, queued=turn["queued"],
        t_submit=turn["t_submit"], t_admit=turn["t_admit"], t_first_gpu=turn["t_first_gpu"],
        t_prefill_end=turn["t_prefill_end"], t_first_token=turn["t_first_token"], t_done=turn["t_done"],
        queue_s=_span(turn["t_first_gpu"], turn["t_submit"]),
        restore_s=restore if type(restore) in (int, float) else None,
        prefill_s=prefill,
        post_prefill_s=_span(turn["t_first_token"], turn["t_prefill_end"]),
        # Everything from t_first_gpu to t_prefill_end that is not a prefill chunk: the open command
        # and other sessions' chunks and steps. TTFT = queue + prefill + between_chunks + post_prefill.
        between_chunks_s=None if prefill is None or _span(turn["t_prefill_end"], turn["t_first_gpu"]) is None
        else max(0.0, turn["t_prefill_end"] - turn["t_first_gpu"] - prefill),
        restored_tokens=cached if type(cached) is int else None,
        prefill_tokens=event.get("prefill_tokens") if type(event.get("prefill_tokens")) is int else None,
        prompt_tokens=event.get("prompt_tokens"), completion_tokens=event.get("completion_tokens"),
        prefill_chunks=[list(chunk) for chunk in turn["prefill_chunks"]], prefill_chunk_count=turn["chunk_count"],
        memory=event.get("memory") if isinstance(event.get("memory"), dict) else None)


class Trace:
    """Bounded scheduler log: one entry per GPU command."""
    def __init__(self, limit=TRACE_LIMIT):
        self.entries = deque(maxlen=limit)
        self.lock = threading.Lock()

    def add(self, t_start, t_end, op, request_ids, rows=None, engine_s=None):
        with self.lock:
            self.entries.append(dict(t_start=t_start, t_end=t_end, op=op, request_ids=list(request_ids), rows=rows,
                                     engine_s=engine_s))

    def since(self, t):
        with self.lock:
            return [dict(entry) for entry in self.entries if entry["t_end"] >= t]


def _round(value):
    return round(value, 4) if isinstance(value, float) else value


# Cache visibility from the engine's done event (batch serve). An older engine omits them; a wrong type is dropped.
CACHE_INT_FIELDS = ("cache_saved_tokens", "cache_tail_tokens", "cache_entries", "cache_bytes", "cache_evictions",
                    "host_free_mib", "slot_context", "cache_pinned_bytes",
                    # An earlier change: the state nearest to a miss and how much of it the prompt repeats.
                    "closest_shared_tokens",
                    # An earlier change: per-tier lookups, hits and reused tokens (slot, host, disk), cumulative.
                    "cache_slot_lookups", "cache_slot_hits", "cache_slot_tokens",
                    "cache_host_lookups", "cache_host_hits", "cache_host_tokens",
                    "cache_disk_lookups", "cache_disk_hits", "cache_disk_tokens")
# The engine's fixed miss causes (engine/serve/cache_status.h) plus ssd_read_failed, which the server adds for
# a failed disk read. A log reader can map a word it does not know to "other", so a new reason never costs the turn's metrics.
MISS_REASONS = frozenset(("none", "bypass", "too_short", "new", "diverged", "evicted", "not_saved",
                          "above_limit", "above_prompt", "off_grid", "schedule", "rows", "tail", "empty",
                          "unknown", "ssd_read_failed"))
CLOSEST_OUTCOMES = frozenset(("all", "prompt_differs", "reply_differs"))


def _cache_fields(event):
    fields = {key: event[key] for key in CACHE_INT_FIELDS if type(event.get(key)) is int}
    if type(event.get("cache_save_refused")) is str:
        fields["cache_save_refused"] = event["cache_save_refused"]
    # The miss diagnosis: a closed set of words, so no request text can reach the line.
    if event.get("miss_reason") in MISS_REASONS:
        fields["miss_reason"] = event["miss_reason"]
    if event.get("closest_outcome") in CLOSEST_OUTCOMES:
        fields["closest_outcome"] = event["closest_outcome"]
    age = event.get("closest_age_s")
    if type(age) in (int, float) and not isinstance(age, bool) and age >= 0:
        fields["closest_age_s"] = float(age)
    # A count from the opened event comes with no cache class (request_line): replay_hit is then unknown, not false.
    if type(event.get("cached_tokens")) is int and not event.get("cache_class_unknown"):
        fields["replay_hit"] = bool(event["cached_tokens"]) and event.get("session_replay") is True
    if event.get("slot_resume") is True:
        fields["slot_resume"] = True  # the replay state was still on the slot: no host restore
    if event.get("checkpoint_resume") is True:
        fields["checkpoint_resume"] = True  # an in-slot checkpoint: the prompt left the previous output
    if event.get("frontier_copy") is True:
        fields["frontier_copy"] = True  # a copy of another slot's Frontier point
    if event.get("end_resume") is True:
        fields["end_resume"] = True  # the whole prompt from an End point with outputs (re-send fix A)
    return fields


ENGINE_DRAFT_FIELDS = ("draft_proposed", "draft_accepted", "spec_steps")
# Draft positions K and the list fields of the done event: counts by position 1..K and
# by draft length 0..K. draft_passes and draft_ms follow. All are counters of the engine's accept step.
DRAFT_POSITIONS = 7
DRAFT_LIST_FIELDS = (("draft_pos_proposed", DRAFT_POSITIONS), ("draft_pos_tested", DRAFT_POSITIONS),
                     ("draft_pos_accepted", DRAFT_POSITIONS), ("draft_len_hist", DRAFT_POSITIONS + 1))


def _count_list(value, size):
    """True for a list of `size` non-negative ints (a bool is not a count)."""
    return type(value) is list and len(value) == size and all(type(n) is int and n >= 0 for n in value)


def draft_position_fields(events):
    """The position counters of the engine's done events, summed; {} unless every event has all six, well formed."""
    try:
        if not events or not all(
                all(_count_list(event.get(name), size) for name, size in DRAFT_LIST_FIELDS) and
                type(event.get("draft_passes")) is int and event["draft_passes"] >= 0 and
                type(event.get("draft_ms")) in (int, float) and 0 <= event["draft_ms"] < 1e12
                for event in events):
            return {}
        fields = {name: [sum(column) for column in zip(*(event[name] for event in events))]
                  for name, _ in DRAFT_LIST_FIELDS}
        return fields | dict(draft_passes=sum(event["draft_passes"] for event in events),
                             draft_ms=float(sum(event["draft_ms"] for event in events)))
    except (AttributeError, TypeError, ValueError):  # A counter must never fail a request.
        return {}


def draft_fields(*parts):
    """Draft statistics of a turn from its jobs, (turn, done event or None) each, or None.

    The engine's done event names a speculative job's counters (spec_steps, draft_proposed, draft_accepted). Without
    them the batch worker's counts stand in: spec_rounds is the decode steps or passes that served the job, and
    draft_accepted their tokens less one correction per round (low by a cut round's rest, never high). The engine
    values count only when every job that decoded reported them. A closed think turn's first job is cancelled
    before its done event: its last step counters (note_token, draft_live) stand in, and without them the turn
    uses the worker's counts for both jobs.
    """
    engine = [all(type((event or {}).get(key)) is int for key in ENGINE_DRAFT_FIELDS) for _, event in parts]
    counted = [turn for turn, _ in parts if turn and type(turn.get("spec_rounds")) is int]
    if any(engine) and all(ok or not (turn or {}).get("spec_rounds") for ok, (turn, _) in zip(engine, parts)):
        reported = [event for ok, (_, event) in zip(engine, parts) if ok]
        return dict(spec_rounds=sum(event["spec_steps"] for event in reported),
                    draft_accepted=sum(event["draft_accepted"] for event in reported),
                    draft_proposed=sum(event["draft_proposed"] for event in reported),
                    **draft_position_fields(reported))
    if not counted: return None
    rounds = sum(turn["spec_rounds"] for turn in counted)
    return dict(spec_rounds=rounds, draft_accepted=sum(turn.get("round_tokens", 0) for turn in counted) - rounds)


# crucible_decode (live drafts for a log reader): at most one line a second for each decoding
# speculative turn, and one final line with its crucible_turn values. Counts only, never token IDs or text.
DECODE_LINE_SECONDS = 1.0
DECODE_DRAFT_KEYS = ("spec_rounds", "draft_proposed", "draft_accepted")


def note_token(turn, event):
    """Count one published token, and keep the cumulative draft counters that the first token event of a speculative
    step carries (engine/serve/batch_session.h BatchStepDraftFields). Counters only: nothing reads them to decide."""
    try:
        turn["decode_tokens"] = turn.get("decode_tokens", 0) + 1
        if all(type(event.get(key)) is int and event[key] >= 0 for key in ENGINE_DRAFT_FIELDS):
            turn["draft_live"] = {key: event[key] for key in ENGINE_DRAFT_FIELDS}
    except (AttributeError, TypeError):  # A counter must never fail a request.
        return


def _decode_text(fields):
    try:
        return "crucible_decode " + json.dumps({key: _round(value) for key, value in fields.items()}, allow_nan=False)
    except (TypeError, ValueError):  # A log line must never stop the worker.
        return None


def decode_line(job, t):
    """The live crucible_decode line of a speculative turn after one of its decode steps, or None (no speculation, no
    round yet, or less than DECODE_LINE_SECONDS since the turn's last line or its first token).

    A think-cap continuation (job.after) writes for its first job: the same request_id and rate limit, the client's
    completion count (the first job's tokens up to the close and the close text, closed_count, then its own tokens),
    and draft counts over both jobs, as request_line sums them.
    """
    try:
        first = getattr(job, "after", None)
        owner = (first or job).turn
        start = owner.get("t_first_token")
        if (not owner.get("speculative") or owner.get("decode_final") or start is None or
                t - owner.get("decode_line_t", start) < DECODE_LINE_SECONDS):
            return None  # decode_final: the final line is out (a stop string ends the client before its job)
        turn = job.turn
        if first is None:
            draft, done = draft_fields((turn, turn.get("draft_live"))), turn.get("decode_tokens", 0)
        else:
            draft = draft_fields((owner, owner.get("draft_live")), (turn, turn.get("draft_live")))
            done = owner.get("closed_count", owner.get("decode_tokens", 0)) + turn.get("decode_tokens", 0)
        if getattr(job, "think_cap", None) is not None:
            done = min(done, job.think_cap)  # tokens past an armed cap are discarded, so the count stays monotonic
        # a log parser needs draft_proposed: without the engine's step counters (an older engine, or packed_op step
        # before the first speculative step) the turn writes no line.
        if not draft or not draft.get("spec_rounds") or "draft_proposed" not in draft or type(done) is not int:
            return None
        owner["decode_line_t"] = t
        return _decode_text(dict(request_id=owner.get("request_id"), completion_tokens=done,
                                 **{key: draft[key] for key in DECODE_DRAFT_KEYS if key in draft}, elapsed_s=t - start))
    except (AttributeError, TypeError, ValueError):  # A counter must never fail a request.
        return None


def final_decode_line(turn_text, owner):
    """The last crucible_decode line of a speculative turn: the values of its crucible_turn line (decode_s is the
    elapsed time), or None for a turn without speculation, without a decode round or without draft_proposed."""
    try:
        if not owner.get("speculative") or not turn_text.startswith("crucible_turn "): return None
        fields = json.loads(turn_text.split(" ", 1)[1])
        if type(fields.get("spec_rounds")) is not int or not fields["spec_rounds"] or "draft_proposed" not in fields:
            return None
        return _decode_text(dict(request_id=fields.get("request_id"), completion_tokens=fields.get("completion_tokens"),
                                 **{key: fields[key] for key in DECODE_DRAFT_KEYS if key in fields},
                                 elapsed_s=fields.get("decode_s")))
    except (AttributeError, TypeError, ValueError):
        return None


def emit_turn(turn_text, owner):
    """Write a turn's crucible_turn line, after its final crucible_decode line. owner: the turn's first job's turn."""
    owner = owner or {}
    decode = final_decode_line(turn_text, owner)
    if decode is not None:
        owner["decode_final"] = True  # no live line after the final one
        emit(decode)
    emit(turn_text)


# Token IDs of the checkpoint's markers (server/tokenization.py checks <tool_call> and </tool_call>; </think> is the
# end tag of the think block, as think_budget.THINK_END names it). Classes count IDs, not text, so the engine thread
# can count them when it stamps a token, before the HTTP thread has read it.
THINK_END_ID, TOOL_START_ID, TOOL_END_ID = 248069, 248058, 248059
ENGINE_ID_DIGITS = 12
TURN_COUNTS = ("passes", "pass_decoders", "pass_rows")


def file_id(path):
    """The first 12 hex digits of the sha256 of a file (the engine executable), or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""): digest.update(block)
        return digest.hexdigest()[:ENGINE_ID_DIGITS]
    except OSError:
        return None


def count_token(turn, token_id):
    """Count one generated token as thinking, tool call or answer. Counters only: nothing reads them to decide.

    turn["think_open"] is True from the start of a job whose output begins inside a think block. The token that
    holds the end tag still counts as thinking; a tool call runs from its start marker to its end marker.
    """
    try:
        if turn.get("think_open"):
            key = "think_tokens"
            if token_id == THINK_END_ID: turn["think_open"] = False
        elif turn.get("in_tool") or token_id == TOOL_START_ID:
            key = "tool_tokens"
            turn["in_tool"] = token_id != TOOL_END_ID
        else:
            key = "answer_tokens"
        turn[key] = turn.get(key, 0) + 1
    except (AttributeError, TypeError):  # A counter must never fail a request.
        return


def count_pass(turn, decoders, rows):
    """One decode pass that served the turn: how many decoders it held and how many rows it asked for."""
    try:
        counts = [turn.get(key, 0) + more for key, more in zip(TURN_COUNTS, (1, decoders, rows))]
    except (AttributeError, TypeError):  # A counter must never fail a request.
        return
    turn.update(zip(TURN_COUNTS, counts))


STAGE_STATES = ("none", "reading", "ready", "failed")  # server/batch_backend.py STAGE_STATES


def load_fields(turns):
    """Numbers about the engine and the load that a turn ran under, from its jobs' counters (see
    server/tests/test_turn_fields.py). A field with no counter is left out. A bad counter leaves all of them out."""
    try:
        return _load_fields(turns)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):  # A counter must never fail a request.
        return {}


def _load_fields(turns):
    fields = {}
    engine = next((turn["engine_id"] for turn in turns if isinstance(turn.get("engine_id"), str)), None)
    if engine is not None and re.fullmatch("[0-9a-f]{%d}" % ENGINE_ID_DIGITS, engine):
        fields["engine_id"] = engine
    total = {key: sum(turn.get(key, 0) for turn in turns) for key in TURN_COUNTS}
    if total["passes"]:
        fields.update(passes=total["passes"], decoders_per_pass=total["pass_decoders"] / total["passes"],
                      rows_per_pass=total["pass_rows"] / total["passes"])
    if any(type(turn.get("stall_s")) is float for turn in turns):
        fields["stall_s"] = sum(turn.get("stall_s", 0.0) for turn in turns)
    # The last stage reply of the turn's first job (server/batch_backend.py _staged): links a turn to its disk read
    # (an earlier change). Fixed words and numbers only.
    staged = next((turn for turn in turns if turn.get("state_disk") in STAGE_STATES), None)
    if staged is not None:
        fields["state_disk"] = staged["state_disk"]
        if type(staged.get("state_disk_wait_s")) is float: fields["state_disk_wait_s"] = staged["state_disk_wait_s"]
        reason = staged.get("state_disk_reason")
        if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,24}|-", reason): fields["state_disk_reason"] = reason
        for key in ("state_disk_polls", "state_disk_tokens", "state_disk_tag", "state_disk_job"):
            if type(staged.get(key)) is int: fields[key] = staged[key]
    keys = ("think_tokens", "tool_tokens", "answer_tokens")
    if any(key in turn for turn in turns for key in keys):
        fields.update({key: sum(turn.get(key, 0) for turn in turns) for key in keys})
    return fields


CLASS_KEYS = ("think_tokens", "tool_tokens", "answer_tokens")


def _miss_reason(fields):
    """The turn's cache miss cause: the engine's fixed word, or ssd_read_failed for a failed read.

    The engine names why its cache served nothing. A failed disk read is the server's own evidence (the stage reply of
    this turn), so it wins for a turn that cached nothing: the state existed, and its read failed. A turn that restored
    state keeps the engine's word: the failed read was for another state, or for a turn ahead of this one.
    """
    if "miss_reason" in fields and fields.get("cached_tokens") == 0 and fields.get("state_disk") == "failed":
        fields["miss_reason"] = "ssd_read_failed"


def _drop_unequal_classes(fields):
    """The engine thread counts every token it decodes, but a stop string or a cancel ends the turn before the engine's
    done event, and the client got fewer tokens. Then the classes do not describe the completion: leave them out."""
    total = fields.get("completion_tokens")
    if type(total) is not int or sum(fields.get(key, 0) for key in CLASS_KEYS) != total:
        for key in CLASS_KEYS: fields.pop(key, None)


def turn_line(summary, event, draft=None, load=None):
    """One stderr line per finished turn for log collectors. Always on; never part of a response.

    draft: draft_fields() of the turn's jobs. load: load_fields() of the turn's jobs.
    """
    summary = summary or {}
    first, submit, done = summary.get("t_first_token"), summary.get("t_submit"), summary.get("t_done")
    fields = dict(request_id=summary.get("request_id"), prompt_tokens=event.get("prompt_tokens"),
                  cached_tokens=event.get("cached_tokens"), prefill_tokens=summary.get("prefill_tokens"),
                  completion_tokens=event.get("completion_tokens"), queue_s=summary.get("queue_s"),
                  prefill_s=summary.get("prefill_s"), decode_s=_span(done, first), ttft_s=_span(first, submit),
                  finish=event.get("finish_reason"), session_label=summary.get("session_label"),
                  **_cache_fields(event))
    if draft is not None: fields.update(draft)
    if load: fields.update(load)
    _miss_reason(fields)
    _drop_unequal_classes(fields)
    try:
        return "crucible_turn " + json.dumps({key: _round(value) for key, value in fields.items()}, allow_nan=False)
    except (TypeError, ValueError):  # A log line must never stop the worker.
        return "crucible_turn " + json.dumps(dict(request_id=fields["request_id"], error="unloggable metrics"))


def error_line(request_id, status, message, streamed):
    """A failed request, for log collectors. The messages are the server's own fixed texts, never content."""
    try:
        return "crucible_error " + json.dumps(dict(request_id=request_id, status=status, streamed=streamed,
                                                    error=str(message)[:300]), allow_nan=False)
    except (TypeError, ValueError):
        return "crucible_error " + json.dumps(dict(request_id=None, status=status, streamed=streamed))


# The reasons of a 4xx in the server log: a closed set, so no request text can reach the log.
CLIENT_ERROR_REASONS = frozenset(("bad_request", "context_length", "invalid_tool_call_history", "line_too_large",
                                  "body_too_large", "length_required", "unsupported_media_type", "forbidden_host",
                                  "not_found", "too_many_requests"))
_STATUS_REASONS = {403: "forbidden_host", 404: "not_found", 411: "length_required", 413: "body_too_large",
                   415: "unsupported_media_type", 429: "too_many_requests"}
_COUNT_KEYS = ("prompt_tokens", "max_tokens", "context")


def client_error_line(request_id, status, error, streamed=False):
    """A refused request (4xx): a fixed reason code and counts, never the message (it can name request values)."""
    code = getattr(error, "code", None)
    reason = code if code in CLIENT_ERROR_REASONS else _STATUS_REASONS.get(status, "bad_request")
    counts = getattr(error, "counts", None) or {}
    fields = dict(request_id=request_id if type(request_id) is int else None, status=status, streamed=streamed,
                  reason=reason)
    fields |= {key: counts[key] for key in _COUNT_KEYS if type(counts.get(key)) is int}
    return "crucible_error " + json.dumps(fields)


def engine_error_line(request_id, op, error, recoverable):
    """The engine's error for an engine command (server log only; the client gets a fixed message).
    Only the engine's own fixed messages are logged (RA-14). Any other text is "other" with its length and a sha256
    prefix, so repeats match and a developer can check a candidate text; LOG_ENGINE_TEXT adds the text itself."""
    fields = dict(request_id=request_id, op=op, error=engine_messages.loggable(error), recoverable=recoverable)
    if fields["error"] == "other" and isinstance(error, str):
        raw = error.encode("utf-8", "replace")
        fields |= dict(error_bytes=len(raw), error_sha256=hashlib.sha256(raw).hexdigest()[:16])
        if LOG_ENGINE_TEXT:
            fields["text"] = error[:2000]
    return "crucible_engine_error " + json.dumps(fields)


def code_frames(error):
    """The last three code locations of an exception (file basename, line, function): no text, no values."""
    try:
        return " > ".join(f"{basename(frame.filename)}:{frame.lineno}:{frame.name}"
                          for frame in extract_tb(error.__traceback__)[-3:])
    except Exception:
        return ""


def worker_error_line(error):
    """Why a worker stopped. A server-written ProtocolError text is fixed; any other exception text could quote a
    payload value, so only its type is logged (RA-14)."""
    if isinstance(error, engine_messages.ProtocolError):
        return "crucible_worker_error " + json.dumps(dict(type=type(error).__name__, error=str(error)[:500]))
    return "crucible_worker_error " + json.dumps(dict(type=type(error).__name__, error="other",
                                                       at=code_frames(error)))


def cancel_line(request_id, completion_tokens, streamed):
    """The client went away; the server cancelled its job (between prefill chunks in the batch worker)."""
    return "crucible_cancel " + json.dumps(dict(request_id=request_id, streamed=streamed,
                                                 completion_tokens=completion_tokens))


def fallback_line(request_id, completion_tokens, streamed, invalid=None):
    """Output that did not parse as valid tool calls went back as content (counts and a fixed reason code only)."""
    return "crucible_error " + json.dumps(dict(request_id=request_id, status=200, streamed=streamed,
                                                error="tool_call_fallback", completion_tokens=completion_tokens) |
                                           (dict(invalid=invalid) if invalid else {}))


def tool_pass_line(request_id, streamed, calls, reasons):
    """Tool calls that went out with a schema-only failure (tool_output.UNCHECKED): fixed reason codes and counts,
    never a parameter name or value. Not an error: the client checks the call and the model gets the error."""
    return "crucible_tool_pass " + json.dumps(dict(request_id=request_id, streamed=streamed, calls=calls,
                                                    reasons=sorted(set(reasons))))


def request_line(first_turn, event, closed=None, cont_turn=None, cont_event=None):
    """The crucible_turn line of a request whose think block the server closed, or that the server logs itself.

    Counts and finish are what the client sees, so a log collector can join the line to its own request record. The first job's times
    give queue, prefill and TTFT. The continuation's own figures go in cont_* fields.
    """
    summary = finalize(first_turn, event if cont_turn is None else {})
    if cont_turn is None:
        if event.get("event") != "done":
            # the server cancelled the job before its done event (no room after the think cap, or a stop
            # string). Its opened event and its chunks still name the restore and the rows it prefilled.
            summary = dict(summary or {}, prefill_tokens=_chunk_rows(first_turn))
            event = dict(event, cached_tokens=_opened(first_turn), cache_class_unknown=True)
            # Its last step counters (note_token) stand in for the done event's.
            return turn_line(summary, event, draft_fields((first_turn, first_turn.get("draft_live"))),
                             load_fields((first_turn,)))
        return turn_line(summary, event, draft_fields((first_turn, event)), load_fields((first_turn,)))
    fields = dict(summary or {}, prefill_tokens=_chunk_rows(first_turn), t_done=cont_turn.get("t_done"))
    # The first job is cancelled before its done event: the cache after the turn is the continuation's report.
    after = {key: cont_event[key] for key in CACHE_INT_FIELDS + ("cache_save_refused",) if key in (cont_event or {})}
    line = turn_line(fields, dict(event, **after, cached_tokens=_opened(first_turn), cache_class_unknown=True),
                     draft_fields((first_turn, first_turn.get("draft_live")), (cont_turn, cont_event)),
                     load_fields((first_turn, cont_turn)))
    cont = finalize(cont_turn, cont_event) or {}
    extra = dict(think_closed_at=closed.tokens, think_closed_by=closed.by, cont_request_id=cont_turn.get("request_id"),
                 cont_prompt_tokens=cont_event.get("prompt_tokens"), cont_cached_tokens=cont_event.get("cached_tokens"),
                 cont_prefill_tokens=cont.get("prefill_tokens"),
                 cont_completion_tokens=cont_event.get("completion_tokens"),
                 cont_ms=None if _span(cont_turn.get("t_first_token"), cont_turn.get("t_submit")) is None
                 else 1000 * _span(cont_turn["t_first_token"], cont_turn["t_submit"]))
    # completion_tokens is what the client got: the first job's tokens, the close text that the server wrote at once,
    # and the continuation's tokens. decode_s spans the whole turn, so the engine's rate is generated_tokens / decode_s.
    total, after_close = event.get("completion_tokens"), cont_event.get("completion_tokens")
    if type(total) is int and type(after_close) is int and type(closed.tokens) is int:
        generated = closed.tokens + after_close
        if 0 <= generated <= total:
            extra.update(generated_tokens=generated, forced_tokens=total - generated)
            extra.update(_closed_classes(first_turn, cont_turn, closed.tokens, total - generated, total))
    try:
        merged = json.loads(line.split(" ", 1)[1]) | {key: _round(value) for key, value in extra.items()}
        return "crucible_turn " + json.dumps(merged, allow_nan=False)
    except (TypeError, ValueError):  # A log line must never stop the request.
        return line


def _opened(turn):
    """The restore that the batch engine named in the job's opened event, or None (serial backend)."""
    opened = turn.get("opened_cached")
    return opened if type(opened) is int else None


def _chunk_rows(turn):
    """The prompt rows that the job's chunks prefilled, or None when a chunk record is missing."""
    chunks = turn.get("prefill_chunks") or []
    if turn.get("chunk_count") and len(chunks) == turn["chunk_count"] and all(type(c[2]) is int for c in chunks):
        return sum(chunk[2] for chunk in chunks)
    return None


def _closed_classes(first_turn, cont_turn, closed_at, forced, total):
    """Token classes of a closed think turn: the first job's tokens up to the close and the text the server wrote
    are thinking; the continuation's tool call tokens are its own; the rest is answer."""
    if not any("think_tokens" in turn or "answer_tokens" in turn for turn in (first_turn, cont_turn)):
        return {}
    think, tool = closed_at + forced, cont_turn.get("tool_tokens", 0)
    if min(think, tool, total - think - tool) < 0: return {}
    return dict(think_tokens=think, tool_tokens=tool, answer_tokens=total - think - tool)


def prefill_line(turn, done, total, end, turn_request_id=None):
    # A closed think turn's continuation job has its own request_id; turn_request_id (the first job's id) lets a log reader
    # pair its prefill lines with the turn. Other lines keep their keys.
    try:
        fields = dict(request_id=turn.get("request_id"), done=done, total=total,
                      elapsed_s=_round(_span(end, turn.get("t_first_gpu"))))
        if turn_request_id is not None: fields["turn_request_id"] = turn_request_id
        return "crucible_prefill " + json.dumps(fields, allow_nan=False)
    except (TypeError, ValueError):
        return "crucible_prefill " + json.dumps(dict(request_id=turn.get("request_id"), error="unloggable metrics"))


def emit(line):
    try:
        # One write per line, so the line does not split against engine output on the same stderr.
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass  # A log write must never stop the worker.
