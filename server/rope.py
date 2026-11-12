"""The server's view of the engine's RoPE mode word (engine/runtime/rope_mode.h).

CRUCIBLE_ROPE_MODE goes to the engine unchanged (backend.ENGINE_KEEP_SWITCHES). The engine fixes each request's mode
from its own length; the server only checks --ctx against the word at start, so a wrong pair fails with a clear text.
"""
import json
import threading

ROPE_NATIVE_POSITIONS = 262144  # the model's native RoPE: every request at or below it is plain
ROPE_MAX_POSITIONS = 4 * ROPE_NATIVE_POSITIONS
ROPE_ENV = "CRUCIBLE_ROPE_MODE"
_FACTORS = {"2": 2, "2.0": 2, "4": 4, "4.0": 4}


def rope_context_cap(word):
    """The largest --ctx that the word allows, or None for a word that the engine refuses."""
    if word is None or word in ("", "plain"):
        return ROPE_NATIVE_POSITIONS
    for kind in ("yarn:", "auto:"):
        if word.startswith(kind) and word[len(kind):] in _FACTORS:
            return ROPE_NATIVE_POSITIONS * _FACTORS[word[len(kind):]]
    return None


# The largest context that --ctx alone selects (512K first). Factor 4 (1M) needs the explicit word.
ROPE_DEFAULT_LONG = 2 * ROPE_NATIVE_POSITIONS


def implied_word(ctx, environ):
    """The word that the server gives the engine for --ctx when the environment names none: auto:2 above 262,144
    (YaRN for the requests that need it, plain for the others), else None (today's engine)."""
    if environ.get(ROPE_ENV) or not ROPE_NATIVE_POSITIONS < ctx <= ROPE_DEFAULT_LONG:
        return None
    return "auto:2"


def context_error(ctx, environ):
    """Why --ctx does not fit the environment's word (or the word that --ctx implies), or None."""
    word = environ.get(ROPE_ENV) or implied_word(ctx, environ)
    cap = rope_context_cap(word)
    if cap is None:
        return f"{ROPE_ENV} must be plain, yarn:2, yarn:4, auto:2 or auto:4"
    if not 1 <= ctx <= ROPE_MAX_POSITIONS:
        return f"--ctx must be in [1,{ROPE_MAX_POSITIONS}]"
    if ctx > cap:
        need = "auto:2" if ctx <= 2 * ROPE_NATIVE_POSITIONS else "auto:4"
        return (f"--ctx {ctx} is above the native 262144 positions: it needs {ROPE_ENV}={need} in the environment "
                f"(YaRN above 262144; requests at or below it stay plain)")
    return None


def max_room(context, prompt_count):
    """--default-max-tokens max: the output room after the prompt. A prompt that fits the native positions gets the
    room up to them, so it stays plain (the same request as on a 262,144 context); a longer prompt gets the room up
    to the context (YaRN)."""
    if context > ROPE_NATIVE_POSITIONS and prompt_count <= ROPE_NATIVE_POSITIONS:
        return ROPE_NATIVE_POSITIONS - prompt_count + 1
    return context - prompt_count + 1


def state_dir_error(environ, ctx=0):
    """v1 (review M2): the disk tier does not record the RoPE mode, so a YaRN word
    (yarn:F or auto:F, also the one that --ctx implies) runs without --state-dir. None when the word is plain."""
    word = environ.get(ROPE_ENV) or implied_word(ctx, environ)
    if rope_context_cap(word) == ROPE_NATIVE_POSITIONS:
        return None
    return f"{ROPE_ENV}={word} keeps states out of the disk tier: remove --state-dir"


# The reply reserve: with an auto word, a request is plain
# when prompt + min(max_tokens, R) - 1 <= 262,144, else YaRN. A plain request's max_tokens is clamped to the plain
# limit, so a long reply ends there with finish_reason=length and the next request crosses to YaRN. A short prompt with
# a very large max_tokens (or the "max" default) stays plain. serve --yarn-reply-reserve sets R.
YARN_REPLY_RESERVE = 32768


def reserve_word(word):
    """True for a word whose requests pick their mode by length (auto:F): only then does the reply reserve apply. The
    forced test word puts every request in its mode, and no word (or plain) is today's engine."""
    return bool(word) and word.startswith("auto:")


def plain_max_tokens(prompt_count, max_tokens, reserve):
    """The max_tokens of a plain request (at most MAX_TOKENS), or None when the request runs YaRN."""
    if prompt_count + min(max_tokens, reserve) - 1 > ROPE_NATIVE_POSITIONS:
        return None
    return min(max_tokens, ROPE_NATIVE_POSITIONS - prompt_count + 1)


def request_mode(word, prompt_count, max_tokens):
    """The engine's mode word for a request of an auto word ("plain", "yarn2", "yarn4"), the same gate as the engine's
    RopeModeFor (engine/runtime/rope_policy.h), or None when no mode of the word holds it. The server decides it once
    for a request (Service.fit_reserve) and sends it with every job of that request (backend.submit), so a bounded
    first job and a continuation keep the request's mode."""
    fed = prompt_count + max(max_tokens, 1) - 1
    if fed <= ROPE_NATIVE_POSITIONS:
        return "plain"
    allow = rope_context_cap(word) or ROPE_NATIVE_POSITIONS
    for factor in (2, 4):
        if ROPE_NATIVE_POSITIONS * factor <= allow and fed <= ROPE_NATIVE_POSITIONS * factor:
            return f"yarn{factor}"
    return None


PLAIN_REPLY_CONTEXT = 200000  # the plain_reply counter: replies of plain jobs above this many prompt tokens


def _line(fields):
    return "crucible_rope " + json.dumps(fields)


def _count(value):
    return type(value) is int and value >= 0


class RopeCounters:
    """The crucible_rope lines: one for each clamp (a plain request's max_tokens lowered by the reply reserve), each cut
    (a clamped reply that reached its plain limit) and each crossing (the engine's rope_crossings count went up). Each
    line holds numbers and one closed word only, with the cumulative count of its kind."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clamps = self.cuts = self.crossings = self.plain_replies = 0

    def clamp_line(self, request_id, prompt_tokens, max_tokens, plain_max_tokens):
        with self.lock:
            self.clamps += 1
            count = self.clamps
        return _line(dict(event="clamp", request_id=request_id, prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                          plain_max_tokens=plain_max_tokens, clamps=count))

    def done_lines(self, turn, event):
        """The lines of a finished engine job (its turn and its done event), or None."""
        lines = []
        clamp = turn.get("rope_clamp")  # [prompt, max_tokens, plain max_tokens]; the request_id exists only now
        if type(clamp) is list and len(clamp) == 3 and all(map(_count, clamp)):
            lines.append(self.clamp_line(turn.get("request_id"), *clamp))
        cap = turn.get("rope_cap")
        if _count(cap) and event.get("finish_reason") == "length" and event.get("completion_tokens") == cap:
            with self.lock:
                self.cuts += 1
                count = self.cuts
            prompt = event.get("prompt_tokens")
            lines.append(_line(dict(event="cut", request_id=turn.get("request_id"),
                                    prompt_tokens=prompt if _count(prompt) else None, completion_tokens=cap, cuts=count)))
        # Review 0530 item 2: the length of each plain reply above 200K context (a long-profile job that ran plain), so the
        # reply reserve R can be checked against real replies. A continuation job gives its own line (same request_id).
        prompt, done = event.get("prompt_tokens"), event.get("completion_tokens")
        if turn.get("rope_mode") == "plain" and _count(prompt) and prompt > PLAIN_REPLY_CONTEXT and _count(done):
            with self.lock:
                self.plain_replies += 1
                count = self.plain_replies
            lines.append(_line(dict(event="plain_reply", request_id=turn.get("request_id"), prompt_tokens=prompt,
                                    completion_tokens=done, plain_replies=count)))
        crossings = event.get("rope_crossings")
        if _count(crossings):
            with self.lock:
                new, self.crossings = crossings > self.crossings, max(crossings, self.crossings)
            if new:
                tokens = event.get("rope_crossing_tokens")
                lines.append(_line(dict(event="crossing", request_id=turn.get("request_id"), crossings=crossings,
                                        crossing_tokens=tokens if _count(tokens) else None)))
        return lines or None


COUNTERS = RopeCounters()
