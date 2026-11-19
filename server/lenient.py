"""serve --lenient-fields (CRUCIBLE_LENIENT_FIELDS): an allow-list of request fields that cannot change the tokens.

The default is on. Off is the strict mode: the server refuses every unknown field (server/api.py
validate_request). With the mode on, only the top-level fields of a chat or completion body that validate_request does
not know are looked at:

- IGNORED: a field that cannot change the tokens at any value (an account, cache or routing hint). It is ignored.
- NEUTRAL: a sampling or output field of another server that Crucible cannot apply. Its neutral value (or null) is
  ignored. Any other value is a 400 that names the field: the server never changes sampling silently.
- DEPENDENT: a setting of a NEUTRAL sampler. The sampler is off (absent, or accepted only at its neutral value), so the
  setting changes nothing. A number or null is ignored; another type is a 400.
- Any other field is a 400 that names the field, as in the strict mode. REFUSED only gives a clearer message for the
  known fields that change the output. A request never gets 200 with a constraint that the server does not apply.

Nested objects (messages, stream_options, chat_template_kwargs) keep their strict checks. A field that validate_request
knows keeps its checks too: `response_format` (structured output) is such a field, so the mode does not touch it.
docs/OPENAI_CLIENTS.md ("Lenient request fields") lists each allowed field; a test checks that it names all of them.
"""
import json

from . import metrics
from .api import ApiError, REQUEST_FIELDS

ENV = "CRUCIBLE_LENIENT_FIELDS"
TRUE, FALSE = ("1", "on", "true", "yes"), ("0", "off", "false", "no")
DEFAULT = True  # no flag and no (or an empty) environment variable

# Field -> the reason that no value of it can change the tokens.
IGNORED = {
    "user": "an end-user ID for abuse checks of a hosted API",
    "safety_identifier": "an end-user ID for abuse checks (the newer OpenAI name of user)",
    "metadata": "tags of a stored completion; Crucible stores no completion",
    "service_tier": "a latency and price tier of a hosted API",
    "prompt_cache_key": "a prompt-cache routing hint; Crucible finds a cached prefix by itself",
    "cache_prompt": "OpenAI-compatible prefix reuse; Crucible finds a cached prefix by itself, and the tokens do not change",
}
# OpenAI-compatible, other engines and OpenAI fields with the value that changes nothing. Crucible samples with temperature, top_k,
# top_p, min_p and the penalties presence_penalty, frequency_penalty and repetition_penalty (validate_request). It has
# no Mirostat, typical, tail-free, XTC, DRY or logit bias sampler. OpenAI-compatible's repeat_penalty stays neutral only: its
# window is the last repeat_last_n tokens (64 by default), with prompt tokens, which is not the rule of
# repetition_penalty (the whole prompt and output, no window). The sign rule is the same.
NEUTRAL = {"repeat_penalty": 1, "typical_p": 1, "tfs_z": 1, "mirostat": 0, "dynatemp_range": 0, "xtc_probability": 0,
           "dry_multiplier": 0, "top_n_sigma": -1, "n_probs": 0, "best_of": 1, "logit_bias": {}, "ignore_eos": False,
           "echo": False, "store": False, "suffix": "", "modalities": ["text"]}
# A OpenAI-compatible sampler setting -> its sampler in NEUTRAL. OpenAI-compatible turns each of these samplers off by default.
# repeat_last_n is the window of OpenAI-compatible's own penalties, which this mode accepts at their neutral values only.
# It is also the window of OpenAI-compatible's presence and frequency penalties, and OpenAI-compatible's server puts the prompt tokens
# in that window (tools/server/server-context.cpp, common_sampler_accept of each prompt token; 0 turns them off). No
# value gives this server's rule, so a repeat_last_n that is not null is refused while a penalty is active (WINDOWED).
DEPENDENT = {"repeat_last_n": "repeat_penalty", "mirostat_tau": "mirostat", "mirostat_eta": "mirostat",
             "dynatemp_exponent": "dynatemp_range", "xtc_threshold": "xtc_probability", "dry_base": "dry_multiplier",
             "dry_allowed_length": "dry_multiplier", "dry_penalty_last_n": "dry_multiplier"}
# The penalties of validate_request with their neutral values (api.PENALTY_FIELDS): repeat_last_n is a window for them.
WINDOWED = {"presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1}
ALLOWED = frozenset(IGNORED) | frozenset(NEUTRAL) | frozenset(DEPENDENT)
# Known fields that change the output: refused with that reason (any other field outside ALLOWED is refused too).
REFUSED = ("functions", "function_call", "n_predict", "grammar", "json_schema", "samplers",
           "prediction", "audio", "web_search_options", "allowed_token_ids", "stop_token_ids", "min_tokens",
           "guided_regex", "guided_json", "guided_choice", "guided_grammar", "structured_outputs", "bad_words")


def resolve(flag, environ):
    """The mode: the flag ("on" or "off") wins over the environment variable; unset or empty is DEFAULT (on)."""
    if flag is not None:
        return flag == "on"
    value = environ.get(ENV, "").strip().lower()
    if not value:
        return DEFAULT
    if value in TRUE or value in FALSE:
        return value in TRUE
    raise ValueError(f"{ENV} must be 1 or 0 (on, off, true, false).")


def neutral(value, target):
    """True when value is null or the neutral value with its JSON type (true is not 1, 1.0 is 1)."""
    if value is None:
        return True
    if isinstance(target, bool) or isinstance(value, bool):
        return value is target
    if isinstance(target, (int, float)):
        return type(value) in (int, float) and value == target
    return type(value) is type(target) and value == target


def describe(target):
    return json.dumps(target)


def refusal(field, value):
    """The ApiError for one field outside the known set, or None when the mode ignores it."""
    if field in REFUSED:
        return ApiError(f"{field} is not supported, also with --lenient-fields: it changes the output.", field)
    if field in NEUTRAL and not neutral(value, NEUTRAL[field]):
        return ApiError(f"{field} must be {describe(NEUTRAL[field])} or null. Crucible cannot apply another value, "
                        "and it does not change sampling silently.", field)
    if field in DEPENDENT and value is not None and type(value) not in (int, float):
        return ApiError(f"{field} must be a number or null.", field)
    if field not in ALLOWED:
        return ApiError(f"{field} is not supported, also with --lenient-fields: the mode ignores only the fields that "
                        "cannot change the output (docs/OPENAI_CLIENTS.md).", field)
    return None


def window_refusal(body):
    """The ApiError for a repeat_last_n that is not null next to an active penalty, or None. A penalty value that is
    not a number is left to validate_request."""
    if body.get("repeat_last_n") is None:
        return None
    active = [f for f, n in WINDOWED.items() if type(body.get(f)) in (int, float) and body.get(f) != n]
    if not active:
        return None
    return ApiError(f"repeat_last_n must be null when {active[0]} is set: Crucible applies {active[0]} over the whole "
                    "output (repetition_penalty also over the prompt), with no window, and it does not change sampling "
                    "silently.", "repeat_last_n")


def lenient_body(body):
    """(body without the ignored fields, their names in order). Raises ApiError for a field that it must refuse."""
    if not isinstance(body, dict):
        return body, []
    window_error = window_refusal(body)
    if window_error is not None:
        raise window_error
    ignored = []
    for field in sorted(set(body) - REQUEST_FIELDS):
        error = refusal(field, body[field])
        if error is not None:
            raise error
        ignored.append(field)
    return {key: value for key, value in body.items() if key not in ignored}, ignored


def log_line(ignored, client_request_id=None):
    """One crucible_lenient line for the ignored field names, or None when no field was ignored. Each name is in
    ALLOWED, so the line never holds client text."""
    if not ignored:
        return None
    fields = dict(ignored=list(ignored))
    value = metrics.client_request_id(client_request_id)
    if value is not None:
        fields["client_request_id"] = value
    return "crucible_lenient " + json.dumps(fields)
