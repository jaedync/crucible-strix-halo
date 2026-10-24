"""Validate the supported OpenAI request subset before queueing GPU work."""
from dataclasses import dataclass
import math
import secrets

STOP_IDS = (248046, 248044)
# The MTP warm-up window (mtp_prefix) of a request that sets none; serve --mtp-window changes it (0: full history).
MTP_WINDOW = 256
MAX_MTP_WINDOW = 262144


@dataclass(frozen=True)
class ModelInfo:
    """The model fields of the engine's ready event (engine/serve/serve_protocol.h ReadyModelJson)."""
    arch: str
    vocab: int
    eos: int
    layers: int
    hidden: int


# Engines before an earlier change do not name their model; the backend keeps these values for them. Every other
# engine's ready event replaces them. VOCAB stays for offline tools that start no engine.
LEGACY_MODEL = ModelInfo(arch="qwen4_exp", vocab=248320, eos=248044, layers=48, hidden=2560)
VOCAB = LEGACY_MODEL.vocab

# serve --default-max-tokens max: a request without a count gets all the context room after its prompt (http.Service.fit).
MAX_ALL = "max"
MAX_COUNT = 262144


@dataclass(frozen=True)
class ServerDefaults:
    """What a request that omits a field gets. The built-in values are the ones the server always used; the serve
    flags --temperature, --top-k, --top-p, --min-p and --default-max-tokens replace them. A request field always wins."""
    temperature: float = 0
    top_k: int = 0
    top_p: float = 1
    min_p: float = 0
    max_tokens: int | str = 128  # a count, or MAX_ALL

    @classmethod
    def checked(cls, **fields):
        """The defaults with the request fields' ranges. It raises ApiError for a bad value, so serve can refuse it at start."""
        values = dict(fields)
        for name, high in (("temperature", 2), ("top_p", 1), ("min_p", 1)):
            if name in values:
                values[name] = number(values[name], name, 0, high)
        if "top_k" in values:
            values["top_k"] = integer(values["top_k"], "top_k", 0, VOCAB)
        if "max_tokens" in values and values["max_tokens"] != MAX_ALL:
            values["max_tokens"] = integer(values["max_tokens"], "max_tokens", 1, MAX_COUNT)
        return cls(**values)

    def describe(self):
        """The values that differ from the built-in ones, for the start line (empty when none)."""
        built_in = ServerDefaults()
        return ", ".join(f"{name} {getattr(self, name)}" for name in ("temperature", "top_k", "top_p", "min_p", "max_tokens")
                         if getattr(self, name) != getattr(built_in, name))


DEFAULTS = ServerDefaults()


class ApiError(ValueError):
    def __init__(self, message, param=None, status=400, code=None, counts=None):
        """code and counts are for the server log only (metrics.client_error_line); the client gets the message."""
        super().__init__(message)
        self.param = param
        self.status = status
        self.code = code
        self.counts = counts

@dataclass(frozen=True)
class Request:
    kind: str
    model: str
    max_tokens: int
    spec: int
    stream: bool
    include_usage: bool
    enable_thinking: bool
    reasoning_effort: str
    return_token_ids: bool
    prompt: object = None
    messages: tuple = ()
    temperature: float = 0
    top_k: int = 0
    top_p: float = 1
    min_p: float = 0
    seed: int = 0
    draft_q: float = 0.3
    draft_head: str = "64k"
    mtp_prefix: int = MTP_WINDOW
    draft_adapt: int | None = None
    cache_control: str = "auto"
    tools: tuple = ()
    tool_choice: object = "auto"
    parallel_tool_calls: bool = True
    stop: tuple = ()
    crucible_metrics: bool = False
    preserve_thinking: bool | None = None
    thinking_budget: int | None = None
    images: tuple = ()
    top_logprobs: int = 0  # 0 is off; N asks for the first token's top-N log-probabilities
    # True: the request sent no count and the server default is "max". max_tokens is then a placeholder until
    # http.Service.fit sets the context room after the prompt.
    max_tokens_all: bool = False
    # "min_thinking_tokens" when the server turned thinking off for this request (think_budget.skips_thinking).
    thinking_skipped_by: str | None = None

def integer(value, field, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ApiError(f"{field} must be an integer in [{low},{high}].", field)
    return value

def number(value, field, low, high):
    if type(value) not in (int, float) or not low <= value <= high:
        raise ApiError(f"{field} must be a finite number in [{low},{high}].", field)
    if not math.isfinite(value):
        raise ApiError(f"{field} must be a finite number in [{low},{high}].", field)
    return float(value)

# Some clients (SillyTavern) send both penalties in every request. The engine has no penalty sampler, so only
# "off" is accepted: 0 or null. The check keeps nothing, so the Request is the same as without the field.
PENALTY_FIELDS = ("presence_penalty", "frequency_penalty")

def penalty_off(value, field):
    if value is not None and not (type(value) in (int, float) and value == 0):  # NaN is not equal to 0
        raise ApiError(f"{field} must be 0. Crucible has no penalty sampler.", field)

def boolean(value, field):
    if type(value) is not bool:
        raise ApiError(f"{field} must be true or false.", field)
    return value

# OpenAI-style levels mapped to the template's three. "none" turns thinking off, as on Halogen and other engines.
EFFORTS = {"none": None, "minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh",
           "max": "xhigh"}
TEMPLATE_KWARGS = {"enable_thinking", "preserve_thinking", "reasoning_effort"}

def thinking_controls(body, default):
    """Resolve (enable_thinking, reasoning_effort, preserve_thinking) from every field clients use for thinking.

    An explicit enable_thinking (top level or chat_template_kwargs) wins. Otherwise a reasoning_effort turns
    thinking on, or off for "none". Otherwise the server default applies.
    """
    kwargs = body.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict) or set(kwargs) - TEMPLATE_KWARGS:
        raise ApiError("chat_template_kwargs supports enable_thinking, preserve_thinking, and reasoning_effort only.",
                       "chat_template_kwargs")
    switches = [boolean(source["enable_thinking"], "enable_thinking")
                for source in (body, kwargs) if "enable_thinking" in source]
    if len(set(switches)) > 1:
        raise ApiError("enable_thinking and chat_template_kwargs.enable_thinking disagree.", "enable_thinking")
    efforts = [source["reasoning_effort"] for source in (body, kwargs) if source.get("reasoning_effort") is not None]
    if any(not isinstance(effort, str) or effort not in EFFORTS for effort in efforts):
        raise ApiError("reasoning_effort must be none, minimal, low, medium, high, xhigh, or max.", "reasoning_effort")
    levels = {EFFORTS[effort] for effort in efforts}
    if len(levels) > 1:
        raise ApiError("reasoning_effort and chat_template_kwargs.reasoning_effort disagree.", "reasoning_effort")
    level = next(iter(levels), "xhigh")
    if switches:
        thinking = switches[0]
    elif efforts:
        thinking = level is not None
    else:
        thinking = default
    preserve = boolean(kwargs["preserve_thinking"], "preserve_thinking") if "preserve_thinking" in kwargs else None
    return thinking, level or "xhigh", preserve

BUDGET_FIELDS = ("max_thinking_tokens", "thinking_budget_tokens", "thinking_budget", "thinking_token_budget")

def thinking_budget(body):
    """The request's own thinking budget under any of the names clients send. Two different values are an error."""
    values = {name: integer(body[name], name, 1, 262144) for name in BUDGET_FIELDS if body.get(name) is not None}
    if len(set(values.values())) > 1:
        raise ApiError("The thinking budget fields disagree.", sorted(values)[0])
    return next(iter(values.values()), None)

def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if (not isinstance(item, dict) or set(item) - {"type", "text"} or
                    item.get("type") != "text" or not isinstance(item.get("text"), str)):
                raise ApiError("Only text message content is supported. Images and video are not supported.", "messages")
            pieces.append(item["text"])
        return "".join(pieces)
    raise ApiError("Message content must be text or a list of text blocks.", "messages")

def chat_messages(value, vision=None, images=None):
    from .tool_requests import chat_messages as normalize_messages
    return normalize_messages(value, vision, images)

def asks_scores(body):
    """True when the request asks for log-probabilities (logprobs or top_logprobs)."""
    return not (body.get("logprobs") in (None, False) and body.get("top_logprobs") is None)

def scoring_count(body, kind, count, temperature, stream):
    """The engine's top-N for a scoring request, or 0. Only one greedy token of a completion has a log-probability."""
    top = body.get("top_logprobs")
    if not asks_scores(body):
        return 0
    if kind != "completion" or count != 1 or temperature != 0 or stream:
        raise ApiError("logprobs needs a completion with max_tokens 1, temperature 0, and no stream.", "logprobs")
    if top is not None and type(top) is not int:
        raise ApiError("top_logprobs must be an integer in [0,20].", "top_logprobs")
    return max(integer(0 if top is None else top, "top_logprobs", 0, 20), 1)

def validate_request(body, kind, model, default_thinking=False, vision=None, vocab=VOCAB,
                     default_mtp_prefix=MTP_WINDOW, defaults=DEFAULTS):
    """vocab: the engine's vocabulary size (the backend's ModelInfo); the default suits engines before an earlier change.
    defaults: the server's values for the fields that the request leaves out (ServerDefaults)."""
    if not isinstance(body, dict):
        raise ApiError("The request must be a JSON object.")
    allowed = {"model", "messages", "prompt", "max_tokens", "max_completion_tokens", "temperature", "n",
               "stream", "stream_options", "spec", "enable_thinking", "reasoning_effort", "chat_template_kwargs", *BUDGET_FIELDS,
               "return_token_ids", "logprobs", "top_logprobs",
               "top_k", "top_p", "min_p", "seed", "draft_q", "draft_head", "mtp_prefix", "draft_adapt",
               "cache_control", "tools", "tool_choice", "parallel_tool_calls", "stop", "crucible_metrics", *PENALTY_FIELDS}
    unknown = set(body) - allowed
    if unknown:
        field = sorted(unknown)[0]
        raise ApiError(f"{field} is not supported. Use temperature, top_k, top_p, min_p, and seed for sampling.", field)
    for field in PENALTY_FIELDS:
        if field in body:
            penalty_off(body[field], field)
    from .stop_sequences import validate_stop
    stop = validate_stop(body.get("stop"))
    if body.get("model", model) != model:
        raise ApiError(f"The loaded model is {model}.", "model", 404)
    # A scoring request needs temperature 0 (scoring_count), so a server default never applies to it.
    temperature = number(body.get("temperature", 0 if asks_scores(body) else defaults.temperature), "temperature", 0, 2)
    # A default above the vocabulary filters nothing, so it is capped instead of failing every request.
    top_k = integer(body.get("top_k", min(defaults.top_k, vocab)), "top_k", 0, vocab)
    top_p = number(body.get("top_p", defaults.top_p), "top_p", 0, 1)
    min_p = number(body.get("min_p", defaults.min_p), "min_p", 0, 1)
    supplied_seed = body.get("seed")
    seed = secrets.randbits(64) if supplied_seed is None else integer(supplied_seed, "seed", 0, (1 << 64) - 1)
    metrics = boolean(body.get("crucible_metrics", False), "crucible_metrics")
    cache_control = body.get("cache_control", "auto")
    if cache_control not in ("auto", "bypass", "refresh"):
        raise ApiError("cache_control must be auto, bypass, or refresh.", "cache_control")
    integer(body.get("n", 1), "n", 1, 1)
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise ApiError("Use max_tokens or max_completion_tokens, not both.", "max_tokens")
    count_field = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
    fill = count_field not in body and defaults.max_tokens == MAX_ALL
    count = integer(MAX_COUNT if fill else body.get(count_field, defaults.max_tokens), count_field, 1, MAX_COUNT)
    spec = integer(body.get("spec", 7), "spec", 0, 7)
    draft_q = number(body.get("draft_q", 0.3), "draft_q", 0, 1)
    draft_head = body.get("draft_head", "64k")
    if draft_head not in ("full", "64k"):
        raise ApiError("draft_head must be full or 64k.", "draft_head")
    mtp_prefix = integer(body.get("mtp_prefix", default_mtp_prefix), "mtp_prefix", 0, MAX_MTP_WINDOW)
    draft_adapt = integer(body["draft_adapt"], "draft_adapt", 0, 1) if "draft_adapt" in body else None
    if draft_adapt == 1 and draft_head != "64k":
        raise ApiError("draft_adapt 1 requires the 64k head.", "draft_adapt")
    stream = boolean(body.get("stream", False), "stream")
    thinking, effort, preserve = thinking_controls(body, default_thinking)
    budget = thinking_budget(body)
    token_ids = boolean(body.get("return_token_ids", False), "return_token_ids")
    options = body.get("stream_options", {})
    if not isinstance(options, dict) or set(options) - {"include_usage"}:
        raise ApiError("stream_options supports include_usage only.", "stream_options")
    usage = boolean(options.get("include_usage", False), "include_usage")
    messages, prompt, tools, choice, parallel, images = (), None, (), "auto", True, []
    if kind == "chat":
        from .tool_requests import validate_tools, validate_choice
        tools = validate_tools(body.get("tools") if body.get("tools") is not None else [])
        choice = validate_choice(body.get("tool_choice") if body.get("tool_choice") is not None else "auto", tools)
        parallel = boolean(body.get("parallel_tool_calls", True), "parallel_tool_calls")
        if "prompt" in body:
            raise ApiError("Use messages for chat requests.", "prompt")
        messages = chat_messages(body.get("messages"), vision, images)
    else:
        if any(field in body for field in ("tools", "tool_choice", "parallel_tool_calls")):
            raise ApiError("Tool fields are supported by chat completions only.", "tools")
        if "messages" in body:
            raise ApiError("Use prompt for completion requests.", "messages")
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt:
            prompt = tuple(integer(token, "prompt", 0, vocab - 1) for token in prompt)
        elif not isinstance(prompt, str) or not prompt:
            raise ApiError("prompt must be nonempty text or one token-ID list. Batches are not supported.", "prompt")
    top = scoring_count(body, kind, count, temperature, stream)
    return Request(kind, model, count, spec, stream, usage, thinking, effort, token_ids, prompt, messages,
                   temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed, cache_control=cache_control,
                   draft_q=draft_q, draft_head=draft_head, mtp_prefix=mtp_prefix, draft_adapt=draft_adapt,
                   tools=tools, tool_choice=choice, parallel_tool_calls=parallel, stop=stop,
                   crucible_metrics=metrics, preserve_thinking=preserve,
                   thinking_budget=budget, images=tuple(images), top_logprobs=top, max_tokens_all=fill)
