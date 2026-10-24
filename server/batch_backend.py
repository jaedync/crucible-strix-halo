"""One command worker controls prefill, packed decode, cancellation, and output backpressure."""
from collections import deque, namedtuple
from dataclasses import dataclass, replace
import json
import re
import os
import queue
import threading
import time
from .api import MAX_MTP_WINDOW, MTP_WINDOW, ApiError
from . import metrics
from .engine_messages import ENGINE_REJECTED, IMAGE_FAILED, ProtocolError
from .backend import EngineBackend, LOOKUP_DEPTH, MAX_CONVERSATION_CACHE_MIB, QUEUED_SECONDS
from .batch_scheduler import (ENCODER_SLICE_BAND, ENCODER_SLICE_SECONDS, INTERLEAVE_MIN_ROWS, PACK_ENTER, PASS_EVENTS, MODEL_LAYERS,
                              PendingPrefill, choose, encoder_rows, interleave_passes, packed_mode)

SMALL_ROW_CHUNK = 8
# Row caps of a decode pass, copied from engine/runtime/packed_rows_option.h for the crucible_turn row counter only.
PASS_ROWS, PACKED_ROWS, WIDE_DECODERS = 8, 16, 4


def pass_row_cap(setting, decoders, slots):
    """The most rows that a pass of this many decoders holds. setting is CRUCIBLE_PACKED_ROWS. Unset or empty: 16 rows
    from 4 decoders when the worker has 4 or more slots, else 8. A decimal 8 to 16: that many rows. Else the engine
    refuses to start, and the counter uses 8."""
    if not setting: return PACKED_ROWS if decoders >= WIDE_DECODERS and slots >= WIDE_DECODERS else PASS_ROWS
    if setting.isascii() and setting.isdigit() and PASS_ROWS <= int(setting) <= PACKED_ROWS: return int(setting)
    return PASS_ROWS
# The longest time a slot waits for its think-cap continuation (queued before the cancel, so normally none).
KEEP_SECONDS = 2.0
# The longest time new chunks wait for a think-cap continuation after its first job's cap token. The HTTP thread queues
# it within milliseconds; a peer chunk that starts first delays the turn by a whole chunk (about 3 s at 4096 rows).
CAP_HOLD_SECONDS = 0.25
# Bounds turns held outside the submit queue; submit still sees a full queue beyond this.
WAITING_LIMIT = 64
# The disk tier (engine/runtime/state_disk.h). The engine writes only states of at least
# 16384 tokens, so shorter prompts skip the stage command. A turn waits for its disk read (polled every
# STAGE_POLL_SECONDS) at most STAGE_WAIT_SECONDS, and STAGE_AHEAD waiting turns start their reads early.
STAGE_MIN_TOKENS = 16384
STAGE_POLL_SECONDS = 0.05
STAGE_WAIT_SECONDS = 30.0
STAGE_AHEAD = 4
# A reading turn that has no slot yet is polled less often: each poll costs the engine command thread about 8 ms
# (JSON out and in) while other sessions decode.
STAGE_AHEAD_POLL_SECONDS = 0.5
# The engine drops a ready read that no stage command touched for 60 s (engine/runtime/state_disk.h stage_seconds), so a
# turn that waits for a slot polls its ready read again after this long.
STAGE_REFRESH_SECONDS = 20.0
# A "none" reply is not final: a returning turn often arrives while the host cache still holds its state,
# and other sessions evict that entry to disk while the turn waits for a slot. A waiting turn asks again this often,
# and a turn asks once more when its slot frees.
STAGE_NONE_POLL_SECONDS = 5.0
STAGE_STATES = ("none", "reading", "ready", "failed")
# One stage reply (_stage). `tokens`: the disk state's length; `tag`: a read for a turn ahead took this turn's host
# state (the engine's host_evict tag), 0 when none. An older engine sends neither reason nor tag.
StageReply = namedtuple("StageReply", "state reason tokens tag", defaults=("-", 0, 0))
# The engine's stage reasons are fixed words (engine/serve/batched_serve.h Stage); anything else is logged as "other".
STAGE_REASON = re.compile(r"[a-z0-9_]{1,24}")
# A stopping engine writes its states to disk for up to 75 s (engine/serve/batched_serve.h kStateFlushSeconds). The server
# waits 85 s, then kills it: the whole stop fits in a systemd TimeoutStopSec=120 with a margin.
STATE_FLUSH_SECONDS = 85


@dataclass
class LiveRequest:
    job: object
    generation: int = 0
    ready: bool = False
    count: int = 0
    paused_since: object = None
    admitted_at: float = 0.0
    prefilled: int = 0
    slot_context: int = 0
    # The next chunk is the first after a cache restore; a restored tail runs it as an 8-row head chunk.
    restored: bool = False
    vision_slices: int = 0  # prefill commands that still run an encoder slice (they report no cursor)
    # When this decoder last got a step or a pass (time.monotonic), from its first token on. A decoder that a chunk's
    # passes leave out (no output room for all of them) keeps its own clock (train 13 audit L2).
    last_step: float = None
    # When this job published its think-cap token (time.monotonic): its continuation follows (CAP_HOLD_SECONDS).
    capped_at: float = None


class BatchedEngineBackend(EngineBackend):
    counts_stall = True
    bounds_think_cap = False  # the worker stops stepping a job at its think cap and keeps the slot for the continuation
    packed_rows = ""  # CRUCIBLE_PACKED_ROWS as the engine reads it; set in __init__
    prefill_interleave = 0
    encoder_slice_seconds = ENCODER_SLICE_SECONDS  # an average of the slice commands run; see encoder_rows
    interleave_min_rows = INTERLEAVE_MIN_ROWS
    kept = None  # set in __init__; None in tests that build the worker without it
    state_disk = False  # the disk tier; set from the engine's ready event
    stage_job = False  # the engine takes a stage "job" tag (fix 2); set from the ready event
    stage_jobs = 0  # the last stage job tag (_job_tag)
    def __init__(self, executable, checkpoint, context=32768, *, streams=8, output_timeout=30, prefill=None,
                 conversation_cache_mib=0, short_streams=0, short_ctx=0, packed_op="verify_step",
                 session_replay=False, slot_resume=None, slot_checkpoints=None, prefill_interleave=0,
                 interleave_min_rows=INTERLEAVE_MIN_ROWS, state_dir=None, state_disk_gib=None, cache_probation=None,
                 state_write_gib_per_day=None, cache_idle_minutes=None, engine_options=(), lookup=LOOKUP_DEPTH,
                 state_write_delay_ms=None,
                 mtp_window=MTP_WINDOW):
        if (type(streams) is not int or not 1 <= streams <= 8 or
                type(output_timeout) not in (int, float) or not 0 < output_timeout <= 300):
            raise ValueError("Use one to eight batch streams and an output timeout within 300 seconds.")
        if prefill is not None and (type(prefill) is not int or not 0 <= prefill <= 4096):
            raise ValueError('Use prefill rows within 0..4096.')
        if type(conversation_cache_mib) is not int or not 0 <= conversation_cache_mib <= MAX_CONVERSATION_CACHE_MIB:
            raise ValueError(f'Use a conversation cache budget within 0..{MAX_CONVERSATION_CACHE_MIB} MiB.')
        if (type(short_streams) is not int or type(short_ctx) is not int or not 0 <= short_streams < streams or
                (not 0 < short_ctx < context if short_streams else short_ctx != 0)):
            raise ValueError('Use 1..streams-1 short streams with a short context below the full context.')
        if session_replay is not False and (session_replay is not True or not conversation_cache_mib):
            raise ValueError('Session replay needs a conversation cache.')
        if state_dir is not None and (type(state_dir) is not str or not state_dir.startswith("/") or
                                      len(state_dir) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in state_dir) or
                                      session_replay is not True):
            raise ValueError('The state directory is an absolute path and needs session replay.')
        if state_disk_gib is not None and (type(state_disk_gib) is not int or not 1 <= state_disk_gib <= 65536 or
                                           state_dir is None):
            raise ValueError('The state disk size is 1..65536 GiB and needs a state directory.')
        if state_write_gib_per_day is not None and (type(state_write_gib_per_day) is not int or
                                                    not 0 <= state_write_gib_per_day <= 65536 or state_dir is None):
            raise ValueError('The daily state write budget is 0..65536 GiB and needs a state directory.')
        if state_write_delay_ms is not None and (type(state_write_delay_ms) is not int or
                                                 not 0 <= state_write_delay_ms <= 60000 or state_dir is None):
            raise ValueError('The state write delay is 0..60000 ms and needs a state directory.')
        if packed_op not in ("verify_step", "step"):
            raise ValueError('Use the packed op verify_step or step.')
        if type(prefill_interleave) is not int or not 0 <= prefill_interleave < MODEL_LAYERS:
            raise ValueError(f'Use a prefill interleave within 0..{MODEL_LAYERS - 1} layers.')
        if type(interleave_min_rows) is not int or not 0 <= interleave_min_rows <= 4096:
            raise ValueError('Use interleave minimum rows within 0..4096.')
        # Decode passes between the layers of a bulk chunk; 0 keeps whole chunks.
        self.prefill_interleave = prefill_interleave
        self.interleave_min_rows = interleave_min_rows
        if prefill_interleave:
            # Every pass of one command may publish for a decoder before its client reads.
            self.event_capacity = 32 + PASS_EVENTS * interleave_passes(prefill_interleave)
        # Slot classes (G4). The engine reports them at ready; this worker mirrors its slot choice.
        self.expected_contexts = [context] * (streams - short_streams) + [short_ctx] * short_streams
        self.free_contexts = []
        # Slots kept for think-cap continuations: id(first job) -> (first job, slot class, time). While one is kept, no other job
        # opens, so the continuation takes that engine slot (its checkpoint affinity) and resumes from the first job's
        # end state, not from an older restore point that a peer or the timing chose.
        self.kept = {}
        # The disk tier: on when the engine's ready event names it. id(job) -> its read: state, start, last poll.
        self.state_disk = False
        self.requested_state_dir = state_dir
        self.staging = {}
        self.queued_polls = {}  # id(job) -> the last read=0 stage of a turn behind the first (_stage_ahead)
        self.waiting = deque()
        self.requested_prefill = prefill
        self.streams = streams
        self.packed_rows = os.environ.get("CRUCIBLE_PACKED_ROWS", "")
        self.output_timeout = output_timeout
        self.active = {}
        self.next_id = 1
        self.pending_tokens = 0
        self.pending_token_limit = 1_000_000
        self.admission_lock = threading.Lock()
        # Scheduling estimates only; the engine owns the real chunk schedule.
        # None until the ready event reports the resolved chunk size.
        self.prefill_rows = None
        self.prefill_rate = 1000.0
        # The engine's prompt cursor after the last prefill command, when its batch_end names one.
        self.prefill_cursor = None
        self.last_step = float("-inf")
        # Above the slot count, packing never starts: every speculative decoder keeps solo steps.
        self.pack_enter = PACK_ENTER if packed_op == "verify_step" else 3
        self.packed = False
        # "verify_step" (G6d, default) also verifies each speculative request's drafts inside the packed pass.
        # "step" keeps the G6c packed spec=0 pass for one train, so integration can compare the two.
        self.packed_op = packed_op
        options = ('--streams', str(streams))
        if prefill is not None: options += ('--prefill', str(prefill))
        # Zero keeps the cache-free batch path; the engine never reads or writes snapshots.
        if conversation_cache_mib: options += ('--conversation-cache-mib', str(conversation_cache_mib))
        if short_streams: options += ('--short-streams', str(short_streams), '--short-ctx', str(short_ctx))
        # G15 opt-in A/B: a later turn resumes from the previous turn's decode-time state when its prompt repeats it.
        if session_replay: options += ('--session-replay', '1')
        # None keeps the engine default (on). Sent only when set, so older engines see the same options.
        if slot_resume is not None:
            if slot_resume not in (0, 1, False, True) or not session_replay:
                raise ValueError('Slot resume is 0 or 1 and needs session replay.')
            options += ('--slot-resume', str(int(slot_resume)))
        if slot_checkpoints is not None:
            if slot_checkpoints not in (0, 1, False, True) or not session_replay:
                raise ValueError('Slot checkpoints are 0 or 1 and need session replay.')
            options += ('--slot-checkpoints', str(int(slot_checkpoints)))
        # None keeps the engine default (on): a state that no later request used leaves first (speed only).
        if cache_probation is not None:
            if cache_probation not in (0, 1, False, True) or not conversation_cache_mib:
                raise ValueError('Cache probation is 0 or 1 and needs a conversation cache.')
            options += ('--cache-probation', str(int(cache_probation)))
        # None keeps the engine default (30): a session state unused for this long may leave the host cache.
        if cache_idle_minutes is not None:
            if type(cache_idle_minutes) is not int or not 1 <= cache_idle_minutes <= 10080 or not conversation_cache_mib:
                raise ValueError('The cache idle time is 1..10080 minutes and needs a conversation cache.')
            options += ('--cache-idle-minutes', str(cache_idle_minutes))
        # the engine sizes each slot's MTP cache by the server's default MTP window (0: the whole context). Sent
        # only when it is not the engine default, so older engines see the same options.
        if type(mtp_window) is not int or not 0 <= mtp_window <= MAX_MTP_WINDOW:
            raise ValueError(f'The MTP window is 0..{MAX_MTP_WINDOW}.')
        if mtp_window != MTP_WINDOW: options += ('--mtp-window', str(mtp_window))
        if state_dir is not None: options += ('--state-dir', state_dir)
        if state_disk_gib is not None: options += ('--state-disk-gib', str(state_disk_gib))
        if state_write_gib_per_day is not None:
            options += ('--state-disk-write-gib-per-day', str(state_write_gib_per_day))
        # The same-second proof only; 0 (the engine default) never goes out, so older engines see the same options.
        if state_write_delay_ms: options += ('--state-disk-write-delay-ms', str(state_write_delay_ms))
        super().__init__(executable, checkpoint, context, engine_options=options + tuple(engine_options),
                         lookup=lookup)

    def _validate_ready(self, ready):
        super()._validate_ready(ready)
        if (ready.get("protocol") != "batch-v1" or type(ready.get("context")) is not int or
                type(ready.get("streams")) is not int or ready["streams"] != self.streams):
            raise ProtocolError("The engine did not return the selected batch protocol.")
        rows = ready.get('prefill_rows')
        if (type(rows) is not int or not 0 <= rows <= 4096 or
                (self.requested_prefill is not None and rows != self.requested_prefill)):
            raise ProtocolError('The engine did not return the selected prefill grid.')
        self.prefill_rows = rows
        if ready.get('slot_contexts') != self.expected_contexts:
            raise ProtocolError('The engine did not return the selected slot contexts.')
        self.free_contexts = list(self.expected_contexts)
        # An engine can refuse the tier (another engine holds the directory); then it gets no stage command.
        self.state_disk = bool(getattr(self, "requested_state_dir", None)) and ready.get("state_disk") is True
        if self.state_disk: self.flush_seconds = STATE_FLUSH_SECONDS
        self.stage_job = self.state_disk and ready.get("stage_job") is True

    def submit(self, ids, request, cache_prefix_tokens=0, shared_prefix_tokens=0, log_turn=True, image_offsets=(),
               after=None, think_cap=None):
        if request.spec and request.max_tokens > 1:
            # G6a: solo speculative steps (greedy and sampled) with one fixed head.
            # Serial serve with the same policy and seed is the reference.
            # mtp_prefix 0 (the full MTP window) keeps the MTP rows in the cache snapshots.
            # at an equal restore point the window changes only the drafts, not the IDs, so clients may use it.
            serial_only = [(name, fix) for name, fix, bad in (
                ("draft_head full", "draft_head 64k", request.draft_head != "64k"),
                ("draft_adapt 1", "draft_adapt 0", request.draft_adapt == 1),
                ("draft_q 0", "draft_q above 0", not request.draft_q > 0)) if bad]
            if serial_only:
                raise ApiError("The batched server does not support " + ", ".join(n for n, _ in serial_only) +
                               ". Use " + ", ".join(f for _, f in serial_only) +
                               ", or start the server with --serial (rollback only).", "spec")
            request = replace(request, draft_adapt=0)
        with self.admission_lock:
            if self.jobs.full() or len(ids) > self.pending_token_limit - self.pending_tokens:
                raise ApiError("The batch request queue is full. Try again later.", status=503)
            job = super().submit(ids, request, cache_prefix_tokens, shared_prefix_tokens, log_turn, image_offsets, after,
                                 think_cap)
            self.pending_tokens += len(ids)
            return job

    def _deliver(self, job, event):
        if not self._try_deliver(job, event):
            raise ProtocolError("The batch output queue exceeded its reserved capacity.")

    def _validate_event(self, event, op, selected, tokens):
        kind = event.get("event")
        id = event.get("request_id")
        if type(id) is not int or id not in selected or id not in self.active:
            raise ProtocolError("The engine returned an event for an unselected request.")
        entry = self.active[id]
        generation = event.get("slot_generation")
        if type(generation) is not int or not 0 < generation < (1 << 64):
            raise ProtocolError("The engine returned an invalid sequence generation.")
        if kind == "opened":
            if op != "open" or entry.generation or event.get("slot_context") != entry.slot_context:
                raise ProtocolError("The engine returned an unexpected sequence lease.")
        elif generation != entry.generation:
            raise ProtocolError("The engine returned a stale sequence event.")
        elif kind == "token":
            token = event.get("id")
            seen = tokens.setdefault(id, [])
            # A speculative or verify step publishes up to eight tokens; nothing may follow a stop token.
            if (op not in ("step", "prefill", "spec_step", "verify_step") or
                    (seen and op not in ("spec_step", "verify_step")) or len(seen) >= 8 or
                    type(token) is not int or not 0 <= token < self.model.vocab or
                    entry.count + len(seen) >= entry.job.payload["max_tokens"] or
                    (seen and seen[-1] in entry.job.payload["stop_ids"])):
                raise ProtocolError("The engine returned an invalid batch token.")
            seen.append(token)
        elif kind == "done":
            seen = tokens.get(id, [])
            stopped = bool(seen) and seen[-1] in entry.job.payload["stop_ids"]
            if (not seen or type(event.get("prompt_tokens")) is not int or
                    type(event.get("completion_tokens")) is not int or
                    event["prompt_tokens"] != len(entry.job.payload["ids"]) or
                    event["completion_tokens"] != entry.count + len(seen) or
                    not (stopped or entry.count + len(seen) == entry.job.payload["max_tokens"]) or
                    event.get("finish_reason") != ("stop" if stopped else "length")):
                raise ProtocolError("The engine returned an invalid batch completion.")
        else:
            raise ProtocolError("The engine returned an unexpected batch event.")

    def _command(self, payload, rows=None):
        """Run one engine command and return its (start, end) server times.

        An interleaved prefill (request_ids and every) first returns verify passes for the decoders, each ended by a
        batch_part event; each part is validated and published like a verify_step before the chunk's own events.
        """
        op = payload["op"]
        selected = set(payload["request_ids"] if op in ("step", "verify_step") else [payload["request_id"]])
        decoders = [id for id in payload.get("request_ids", ()) if op == "prefill"]
        start = metrics.now()
        self.process.stdin.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        events, tokens, finished = [], {}, set()
        part, part_tokens, part_finished, part_start = [], {}, set(), start
        engine_s = None
        opened, rejection, passes = False, None, 0
        while True:
            event = self._next()
            if not isinstance(event, dict): raise ProtocolError("The engine returned a non-object batch event.")
            kind = event.get("event")
            if kind == "batch_end":
                if event.get("op") != op: raise ProtocolError("The engine returned the wrong batch boundary.")
                if part: raise ProtocolError("The engine left a decode pass without its boundary.")
                engine_s = event.get("engine_s") if isinstance(event.get("engine_s"), float) else None
                boundary = event
                break
            if kind == "batch_part":
                if not decoders or events or event.get("op") != "verify_step":
                    raise ProtocolError("The engine returned an unexpected decode pass boundary.")
                live = set(decoders)
                self._check_complete("verify_step", live, part_tokens, part_finished)
                stepped = self._count_rounds(part_tokens, "verify_step")
                self._publish(part, part_finished)
                self._decode_lines(stepped, part_finished)
                end = metrics.now()
                self.trace.add(part_start, end, "verify_step", sorted(live), None, None)
                self.last_step = time.monotonic()
                for id in part_tokens:
                    if id in self.active: self.active[id].last_step = self.last_step
                decoders = [id for id in decoders if id not in part_finished]
                part, part_tokens, part_finished, part_start = [], {}, set(), end
                passes += 1
                continue
            if rejection is not None: raise ProtocolError("The engine returned output after a rejected open.")
            if kind == "error":
                metrics.emit(metrics.engine_error_line(payload.get("request_id", payload.get("request_ids")), op,
                                                       event.get("error"), event.get("recoverable") is True))
                # An open may be refused before it leases a slot. A prefill may end its own request when an image
                # file fails its check after the open (the engine closed the request before any device work, so
                # before any decode pass of the command; review F1).
                own = (op == "prefill" and event.get("request_id") == payload["request_id"] and not part and
                       not passes and event.get("slot_generation") == self.active[payload["request_id"]].generation)
                if ((op != "open" and not own) or events or event.get("recoverable") is not True or
                        not isinstance(event.get("error"), str)):
                    raise ProtocolError("The engine rejected the batch command. Check its stderr log.")
                rejection = event
                continue
            if event.get("request_id") in decoders and not events:
                self._validate_event(event, "verify_step", set(decoders), part_tokens)
                if kind == "done":
                    if event["request_id"] in part_finished:
                        raise ProtocolError("The engine returned duplicate completion events.")
                    part_finished.add(event["request_id"])
                part.append(event)
                continue
            self._validate_event(event, op, selected, tokens)
            if kind == "opened":
                if opened: raise ProtocolError("The engine returned duplicate lease events.")
                opened = True
            if kind == "done":
                if event["request_id"] in finished: raise ProtocolError("The engine returned duplicate completion events.")
                finished.add(event["request_id"])
            events.append(event)
        end = metrics.now()
        if rejection is not None:
            # Native validation promises no lease or device write (an open), or the engine closed the request before
            # any device work (a prefill image check); consume the boundary before recycling the slot.
            if self.fatal: raise ProtocolError(self.fatal)
            self.trace.add(start, end, op, sorted(selected), rows, engine_s)
            id = payload["request_id"]
            self._deliver(self.active[id].job, dict(event="error", error=IMAGE_FAILED if op == "prefill" else ENGINE_REJECTED))
            self._release(id)
            return start, end
        if op == "open" and not opened: raise ProtocolError("The engine omitted a sequence lease.")
        self._check_complete(op, selected, tokens, finished)
        stepped = self._count_rounds(tokens, op) if op in ("step", "spec_step", "verify_step") else {}
        if op == "close" and events: raise ProtocolError("The engine returned output after cancellation.")
        if op == "prefill" and "cursor" in boundary:
            rows = self._reported_rows(self.active[payload["request_id"]], boundary["cursor"])
        elif op == "prefill" and self.active[payload["request_id"]].vision_slices:
            rows = 0  # an encoder slice
        self.trace.add(start, end, op, sorted(selected), rows, engine_s)
        if op == "prefill":
            turn = self.active[payload["request_id"]].job.turn
            metrics.add_chunk(turn, start, end, rows)
            if tokens: turn["t_prefill_end"] = end
        self._publish(events, finished)
        self._decode_lines(stepped, finished)
        return start, end

    def _count_rounds(self, tokens, op):
        """One decode round per request that a step or pass served, and the tokens it published (metrics.rounds).

        Also the pass counters of the crucible_turn line: the decoders in the pass and the rows that the server asked
        for. A plain step has one row per decoder. A speculative decoder asks for 1 plus its draft depth bound, and a
        pass holds at most pass_row_cap rows (the engine may draft fewer rows, and falls back to 8 where it cannot run a
        wide pass; the server cannot see that). Counters only.
        """
        try:
            asked = [1 + self.active[id].job.payload["spec"] if op != "step" and self._speculative(self.active[id]) else 1
                     for id in tokens]
        except (KeyError, TypeError):  # A counter must never fail a step; a payload without spec has one row.
            asked = [1] * len(tokens)
        rows = min(pass_row_cap(self.packed_rows, len(tokens), getattr(self, "streams", PASS_ROWS)), sum(asked))
        stepped = {}
        for id, seen in tokens.items():
            stepped[id] = self.active[id].job
            turn = stepped[id].turn
            turn["spec_rounds"] = turn.get("spec_rounds", 0) + 1
            turn["round_tokens"] = turn.get("round_tokens", 0) + len(seen)
            metrics.count_pass(turn, len(tokens), rows)
        return stepped

    @staticmethod
    def _decode_lines(stepped, finished):
        """The live crucible_decode lines after a step or pass has published (metrics.decode_line). A finished turn
        writes its final line with its crucible_turn line instead; a cancelled job writes none."""
        t = None  # read the clock only for a turn that can write a line
        for id, job in stepped.items():
            if id in finished or job.cancelled.is_set(): continue
            if not (getattr(getattr(job, "after", None) or job, "turn", None) or {}).get("speculative"): continue
            if t is None: t = metrics.now()
            line = metrics.decode_line(job, t)
            if line is not None: metrics.emit(line)

    def _check_complete(self, op, selected, tokens, finished):
        """Every selected decoder published, and exactly the turns that reached a stop or their limit finished."""
        if op in ("step", "spec_step", "verify_step") and set(tokens) != selected:
            raise ProtocolError("The engine omitted a batch token.")
        terminal = {id for id, seen in tokens.items() if seen[-1] in self.active[id].job.payload["stop_ids"] or
                    self.active[id].count + len(seen) == self.active[id].job.payload["max_tokens"]}
        if finished != terminal: raise ProtocolError("The engine omitted a required completion event.")

    def _publish(self, events, finished):
        """Deliver one command's (or one decode pass's) events and release its finished turns."""
        for event in events:
            entry = self.active[event["request_id"]]
            if event["event"] == "opened":
                entry.generation = event["slot_generation"]
                # The restore at open; a closed think turn cancels this job before its done event (http.py). The
                # whole prompt is an End resume (re-send fix A): the prefill command then runs no row.
                cached = event.get("cached_tokens")
                if type(cached) is int and 0 <= cached <= len(entry.job.payload["ids"]):
                    entry.job.turn["opened_cached"] = cached
                    # SRPT and aging rank the rows that remain after the restore, not the whole prompt.
                    entry.prefilled, entry.restored = cached, cached > 0
                slices = event.get("vision_slices")
                if type(slices) is int and 0 <= slices <= 1 << 20 and entry.job.payload.get("images"):
                    entry.vision_slices = slices
            elif event["event"] == "token":
                entry.ready = True; entry.count += 1
                if entry.last_step is None: entry.last_step = time.monotonic()
                if entry.count == entry.job.think_cap: entry.capped_at = time.monotonic()
                self._deliver(entry.job, self._stamp(entry.job, event))
            else: self._deliver(entry.job, self._stamp(entry.job, dict(event, slot_context=entry.slot_context)))
        for id in finished: self._release(id)

    def _reported_rows(self, entry, cursor):
        if type(cursor) is not int or not entry.prefilled < cursor <= len(entry.job.payload["ids"]):
            raise ProtocolError("The engine returned an invalid prefill cursor.")
        self.prefill_cursor = cursor
        return cursor - entry.prefilled

    def _release(self, id):
        entry = self.active.pop(id)
        self._forget_job(entry.job)
        self.free_contexts.append(entry.slot_context)

    def _slot_for(self, payload):
        """The engine's rule: the smallest free slot that holds the prompt and every requested token."""
        need = len(payload["ids"]) + payload["max_tokens"] - 1
        fitting = [context for context in self.free_contexts if context >= need]
        return min(fitting) if fitting else None

    def _admit(self):
        while len(self.waiting) < WAITING_LIMIT or (self.kept and not self._successors()):
            try: self.waiting.append(self.jobs.get_nowait())
            except queue.Empty: break
        if self.kept and self._admit_successor(): return True
        if self.state_disk: self._stage_ahead()
        # FIFO among turns that fit a free slot. A skipped turn fits no free slot, so a later
        # turn never takes a slot that an earlier one could use. A kept slot is not in free_contexts.
        for job in list(self.waiting):
            # continue_job queues the continuation before it cancels the first job: wait for that close, so the
            # continuation takes the first job's slot and not another free one.
            if job.after is not None and not job.cancelled.is_set() and self._running(job.after): return False
            if job.cancelled.is_set(): slot_context = None
            elif (slot_context := self._slot_for(job.payload)) is None: continue
            # The disk read runs before the open, so the open restores from memory (FIFO: later turns wait too).
            elif self.state_disk and not self._staged(job, opening=True, position=self._position(job)): return False
            if self.state_disk:
                self.staging.pop(id(job), None)
                self.queued_polls.pop(id(job), None)
            self.waiting.remove(job)
            with self.admission_lock: self.pending_tokens -= len(job.payload["ids"])
            if slot_context is None:
                self._forget_job(job)
                return True
            return self._open(job, slot_context)
        return False

    def _forget_job(self, job):
        if self.state_disk:
            self.staging.pop(id(job), None)
            self.queued_polls.pop(id(job), None)
        super()._forget_job(job)

    def _stageable(self, job):
        payload = job.payload
        return (not job.cancelled.is_set() and payload.get("cache_control") == "auto" and
                len(payload["ids"]) >= STAGE_MIN_TOKENS)

    def _stage_ahead(self):
        """Start the disk read of the turn next in line, so the read overlaps its wait for a slot. A turn with no
        free slot never reaches _staged in _admit, so this also polls its read and keeps a ready read alive.
        An earlier change: only the first long turn (FIFO: the next free slot takes it) starts a read. A read holds host
        memory inside the cache budget until its turn opens, so a read for a turn further back would hold that budget
        for minutes (x6 waits: 80-170 s) and evict host states for nothing if it expired. The later turns of the first
        STAGE_AHEAD poll only a read that they hold already, and each turn stages again when its slot frees. Each of
        them also says that it waits (read=0, every STAGE_REFRESH_SECONDS): a read for a turn ahead may take the host
        state of a session that waits behind it (the engine's read-ahead rule), never of a session that waits for
        nothing."""
        position, now = 0, time.monotonic()
        for job in list(self.waiting)[:STAGE_AHEAD]:
            if not self._stageable(job): continue
            record = self.staging.get(id(job))
            if position == 0 or (record is not None and record["state"] in ("reading", "ready")):
                self._staged(job, STAGE_AHEAD_POLL_SECONDS, position=position, read=position == 0)
            elif now - self.queued_polls.get(id(job), -STAGE_REFRESH_SECONDS) >= STAGE_REFRESH_SECONDS:
                if tag := self._stage(job.payload, position, False, self._job_tag(job)).tag:
                    job.turn["state_disk_tag"] = tag
                self.queued_polls[id(job)] = now
            position += 1

    def _position(self, job):
        """The turn's place among the waiting long turns (the order of _stage_ahead)."""
        position = 0
        for other in self.waiting:
            if other is job: return position
            if self._stageable(other): position += 1
        return position

    def _staged(self, job, poll_seconds=STAGE_POLL_SECONDS, opening=False, position=0, read=True):
        """True when `job` may open: no disk read, a finished read, or a read that took too long. A reading turn is
        polled every `poll_seconds` (a turn with a free slot: STAGE_POLL_SECONDS). `opening`: the turn has a free
        slot, so a "none" reply from an earlier poll is checked again. `position` and `read` go to the engine
        (_stage)."""
        if not self._stageable(job): return True
        now = time.monotonic()
        record = self.staging.get(id(job))
        none_poll = STAGE_POLL_SECONDS if opening else STAGE_NONE_POLL_SECONDS
        if (record is None or (record["state"] == "reading" and now - record["polled"] >= poll_seconds) or
                (record["state"] == "ready" and now - record["polled"] >= STAGE_REFRESH_SECONDS) or
                (record["state"] == "none" and now - record["polled"] >= none_poll)):
            state, reason, tokens, tag = self._stage(job.payload, position, read, self._job_tag(job))
            if record is None: record = self.staging[id(job)] = dict(state=state, since=now, polled=now)
            else:
                # The engine dropped the ready read and reads it again (a "reading" reply after "ready" or after a
                # gap in the polls): a new wait.
                if state == "reading" and (record["state"] == "ready" or
                                           now - record["polled"] >= STAGE_REFRESH_SECONDS):
                    record["since"] = now
                record.update(state=state, polled=now)
            job.turn["state_disk"] = state
            job.turn["state_disk_wait_s"] = now - record["since"]
            job.turn["state_disk_reason"] = reason
            job.turn["state_disk_polls"] = job.turn.get("state_disk_polls", 0) + 1
            job.turn["state_disk_tokens"] = tokens
            if tag: job.turn["state_disk_tag"] = tag
        return record["state"] != "reading" or now - record["since"] > STAGE_WAIT_SECONDS

    def _job_tag(self, job):
        """Fix 2 (cause.md item 10): the turn's tag on the engine's stage and host_evict lines and on its turn line
        (state_disk_job), so a log check links them. 0 for an engine without "stage_job" in its ready event: the train
        23 engine refuses unknown fields (HIGH-2)."""
        if not self.stage_job: return 0
        tag = job.turn.get("state_disk_job")
        if type(tag) is not int:
            self.stage_jobs += 1
            tag = job.turn["state_disk_job"] = self.stage_jobs
        return tag

    def _stage(self, payload, position=0, read=True, job=0):
        """One stage command: the engine starts or reports the disk read for this request (no device work).
        `position`: the turn's place among the waiting long turns; `read`: False for a turn that only says that it
        waits; `job`: the turn's tag (_job_tag). Returns a StageReply. A Reading reply with reason "writing" is a state
        that the engine's writer has not written yet: the turn waits for it as for a read (fix 2)."""
        command = dict(op="stage", request=payload)
        # Only a value that is not the engine default goes out: the train 23 engine refuses unknown fields (HIGH-2).
        if position: command["position"] = position
        if not read: command["read"] = 0
        if job: command["job"] = job
        self.process.stdin.write(json.dumps(command, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        staged = None
        while True:
            event = self._next()
            if not isinstance(event, dict): raise ProtocolError("The engine returned a non-object batch event.")
            if event.get("event") == "batch_end":
                if event.get("op") != "stage" or staged is None:
                    raise ProtocolError("The engine returned the wrong stage boundary.")
                return staged
            if (event.get("event") != "staged" or staged is not None or event.get("state") not in STAGE_STATES or
                    type(event.get("tokens")) is not int):
                raise ProtocolError("The engine returned an invalid stage event.")
            reason, tag = event.get("reason", "-"), event.get("tag", 0)
            if not (reason == "-" or isinstance(reason, str) and STAGE_REASON.fullmatch(reason)): reason = "other"
            if type(tag) is not int or tag < 0: raise ProtocolError("The engine returned an invalid stage event.")
            staged = StageReply(event["state"], reason, event["tokens"], tag)

    def _open(self, job, slot_context):
        if self.next_id >= (1 << 64): raise ProtocolError("The request ID limit was reached. Restart the worker.")
        id = self.next_id; self.next_id += 1
        self.free_contexts.remove(slot_context)
        self.active[id] = LiveRequest(job, admitted_at=time.monotonic(), slot_context=slot_context)
        turn = job.turn
        turn["request_id"] = id
        turn["t_admit"] = turn["t_first_gpu"] = metrics.now()
        turn["queued"] = turn["t_admit"] - turn["t_submit"] > QUEUED_SECONDS
        self._command(dict(op="open", request_id=id, request=job.payload))
        return True

    def _cancelled(self):
        progress = False
        for request_id, entry in list(self.active.items()):
            if entry.job.cancelled.is_set():
                self._command(dict(op="close", request_id=request_id))
                follow = entry.job.successor
                if follow is not None and not follow.cancelled.is_set() and not self._running(follow):
                    self.active.pop(request_id)
                    self._forget_job(entry.job)
                    if self.kept is None: self.kept = {}
                    self.kept[id(entry.job)] = (entry.job, entry.slot_context, time.monotonic())
                else:
                    self._release(request_id)
                progress = True
        return progress

    def _running(self, job):
        return any(entry.job is job for entry in self.active.values())

    def _successors(self):
        return [job for job in self.waiting if job.after is not None and id(job.after) in self.kept]

    def _admit_successor(self):
        """Open the continuation of a kept slot first. A cancelled continuation, or one that never arrives within
        KEEP_SECONDS (a client that left between the jobs), frees the slot. Other free slots admit meanwhile."""
        now = time.monotonic()
        for key, (first, slot_context, since) in list(self.kept.items()):
            follow = first.successor
            queued = any(job is follow for job in self.waiting)
            if follow.cancelled.is_set() or now - since > KEEP_SECONDS:
                del self.kept[key]
                self.free_contexts.append(slot_context)
                if follow.cancelled.is_set() and queued:
                    self.waiting = deque(job for job in self.waiting if job is not follow)
                    with self.admission_lock: self.pending_tokens -= len(follow.payload["ids"])
                    self._forget_job(follow)
                return True
            if queued:
                del self.kept[key]
                self.waiting = deque(job for job in self.waiting if job is not follow)
                with self.admission_lock: self.pending_tokens -= len(follow.payload["ids"])
                self.free_contexts.append(slot_context)
                return self._open(follow, slot_context)
        return False

    def _ready(self):
        ready = []
        now = time.monotonic()
        for id, entry in self.active.items():
            if entry.job.cancelled.is_set(): continue
            # Room for one step's tokens plus done; a speculative step can publish eight.
            if entry.job.events.qsize() <= entry.job.events.maxsize - (9 if self._speculative(entry) else 3):
                entry.paused_since = None
                if entry.ready: ready.append(id)
            elif entry.paused_since is None: entry.paused_since = now
            elif now - entry.paused_since >= self.output_timeout:
                # Two slots stay free while paused, so the terminal error needs no network wait.
                self._deliver(entry.job, dict(event="error", error="The output consumer timed out. Retry the request."))
                entry.job.cancelled.set()
        return ready

    @staticmethod
    def _speculative(entry):
        return entry.job.payload["spec"] > 0 and entry.job.payload["max_tokens"] > 1

    def _pending_prefills(self):
        return [PendingPrefill(id, len(entry.job.payload["ids"]) - entry.prefilled + self._encoder_rows(entry),
                               entry.admitted_at, entry.job.after is not None)
                for id, entry in self.active.items()
                if not entry.ready and not entry.job.cancelled.is_set()]

    def _encoder_rows(self, entry):
        if not entry.vision_slices: return 0
        return encoder_rows(entry.vision_slices, self.encoder_slice_seconds, self.prefill_rate)

    def _prefill(self, id):
        entry = self.active[id]
        rows = 0
        if self.prefill_rows is not None:
            # Rows 0 is the engine's small-row path, which uses kMaxRows chunks.
            rows = max(0, min(self.prefill_rows or SMALL_ROW_CHUNK, len(entry.job.payload["ids"]) - entry.prefilled))
        payload = dict(op="prefill", request_id=id)
        decoders = self._interleave_decoders(id, rows)
        if decoders:
            payload |= dict(request_ids=decoders, every=self.prefill_interleave)
        self.prefill_cursor = None
        # Decoders that this chunk holds back (not the ones that decode inside it): they wait for the whole chunk.
        waiting = [other.job.turn for other_id, other in self.active.items()
                   if other_id != id and other.ready and other_id not in decoders and not other.job.cancelled.is_set()]
        start, end = self._command(payload, rows if self.prefill_rows is not None else None)
        elapsed = end - start
        for turn in waiting: turn["stall_s"] = turn.get("stall_s", 0.0) + elapsed
        measured = self.prefill_cursor is not None
        if measured: rows = self.prefill_cursor - entry.prefilled
        elif entry.vision_slices:
            # No cursor from an engine that reports slices: the command ran one encoder slice and no prompt row.
            entry.vision_slices -= 1
            rows = 0
            # One slice, about 76 ms. A command with decoders attached or a failed slice is no slice time.
            if elapsed > 0 and not decoders and id in self.active:
                sample = min(max(elapsed, ENCODER_SLICE_BAND[0]), ENCODER_SLICE_BAND[1])
                self.encoder_slice_seconds = .7 * self.encoder_slice_seconds + .3 * sample
        entry.prefilled += rows
        head, entry.restored = entry.restored, False
        if id in self.active:
            # Without an engine cursor, rows are scheduled prompt rows: a restored tail's head chunk runs fewer.
            first = entry.job.after
            metrics.emit(metrics.prefill_line(entry.job.turn, entry.prefilled, len(entry.job.payload["ids"]), end,
                                              None if first is None else first.turn.get("request_id")))
        # The rate feeds aging deadlines for bulk chunks. A small head chunk, or a head chunk of unknown size,
        # would skew it (an 8-row head counted as 4096 rows reads as 40K tok/s).
        small = rows <= SMALL_ROW_CHUNK and self.prefill_rows
        if rows > 0 and elapsed > 0 and not small and (measured or not head):
            self.prefill_rate = .7 * self.prefill_rate + .3 * max(rows / elapsed, 1.0)

    def _step_clock(self, ready):
        """The last step of the ready decoder that waited longest: decode_max counts from there (train 13 audit L2).

        With every ready decoder in every step or pass, this is the last step, as before.
        """
        clocks = [self.active[id].last_step for id in ready if self.active[id].last_step is not None]
        return min(clocks) if clocks else self.last_step

    def _interleave_decoders(self, id, rows):
        """Ready decoders that can decode inside this chunk, each with output room for every pass of it.

        Only a bulk chunk (more than kMaxRows rows) runs passes; the engine also skips them for a tight H layout. A chunk
        below interleave_min_rows ends within the decode gap anyway, so its passes would only delay its own turn.
        """
        if not self.prefill_interleave or rows <= SMALL_ROW_CHUNK or rows < self.interleave_min_rows: return []
        need = PASS_EVENTS * interleave_passes(self.prefill_interleave)
        decoders = [ready for ready in self._ready() if ready != id and
                    self.active[ready].job.events.qsize() <= self.active[ready].job.events.maxsize - need]
        return decoders[:SMALL_ROW_CHUNK]

    def _schedule(self):
        ready = self._ready()
        action = choose(time.monotonic(), ready, self._pending_prefills(), self._step_clock(ready), self.prefill_rate)
        if action is None: return False
        kind, target = action
        if kind == "prefill" and self.active[target].job.after is None and self._awaiting_continuation():
            # A held chunk leaves the device to the ready decoders, not idle. The capped job's
            # own tokens past the cap are discarded, so it does not step.
            ready = [id for id in ready if not self._holding(self.active[id])]
            if not ready: return False
            kind, target = "step", ready
        if kind == "prefill": self._prefill(target)
        else:
            self.last_step = time.monotonic()
            for id in target: self.active[id].last_step = self.last_step
            decoders = sum(1 for entry in self.active.values() if entry.ready and not entry.job.cancelled.is_set())
            self.packed = packed_mode(self.packed, decoders, enter=self.pack_enter)
            if self.packed:
                self._command(dict(op=self.packed_op, request_ids=target))
                return True
            # Plain decoders share one packed pass; each speculative decoder runs its own solo step.
            plain = [id for id in target if not self._speculative(self.active[id])]
            if plain: self._command(dict(op="step", request_ids=plain))
            for id in target:
                if id in self.active and self._speculative(self.active[id]):
                    self._command(dict(op="spec_step", request_id=id))
        return True

    def _awaiting_continuation(self):
        """A first job published its cap token, is still open, and its cap is armed (the server disarms it when the
        model closes its own think block, so no continuation comes). continue_job queues the continuation before it
        cancels that job; the next _cancelled closes the job and opens the continuation on its slot, which then goes
        before every new turn (batch_scheduler._next_prefill). A cancel that lands during this loop's admissions
        still holds: in an agent test run, 16 of 106 continuations lost a chunk to that race."""
        return any(self._holding(entry) for entry in self.active.values())

    @staticmethod
    def _holding(entry):
        return (entry.capped_at is not None and time.monotonic() - entry.capped_at < CAP_HOLD_SECONDS and
                entry.job.think_cap is not None)

    def _fail_all(self):
        self.fatal = "The batch worker failed. Restart the server and check its stderr log."

    def _work(self):
        try:
            while not self.closed.is_set():
                progress = self._cancelled()
                while self._admit(): progress = True
                progress = self._schedule() or progress
                if not progress: self.closed.wait(.005)
        except Exception as error:
            if not self.closed.is_set():
                metrics.emit(metrics.worker_error_line(error))
                self._fail_all()
                self.close()
        finally:
            # Only this worker mutates leases and waiters; close() owns client terminal events.
            with self.admission_lock:
                self.active.clear()
                self.waiting.clear()
                while True:
                    try: self.jobs.get_nowait()
                    except queue.Empty: break
                self.pending_tokens = 0
                self.kept = {}
                self.free_contexts = list(self.expected_contexts)
