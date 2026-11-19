"""Memory stage 2: the server side of the engine's start-time memory fit (engine/runtime/batch_admission.h
FitServeLayout, engine/serve/batched_serve.h BatchFitJson).

When the requested layout does not fit the free memory, the engine starts the largest layout that does: first a smaller
conversation cache budget, then fewer slots at the same --ctx, then (with one slot only) a lower --ctx. The ready
event then carries a "fit" object. This module checks that object against the request and writes the one start line
that names the old values, the new values, the cause and the programs that hold memory. It never stops a process.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:  # The holder scan of `./crucible doctor` (stage 1). A server copy without tools/ prints no holders.
    from tools.crucible_cli.memfit import MemoryHolder, memory_holders
except ImportError:  # pragma: no cover - every install has tools/
    MemoryHolder, memory_holders = None, None

STEPS = ("cache", "streams", "context")
FIT_KEYS = {"step", "requested", "cache_mib", "free_bytes", "need_bytes"}
HOLDERS_SHOWN = 3


@dataclass(frozen=True)
class MemoryFit:
    step: str
    requested_streams: int
    requested_context: int
    requested_cache_mib: int
    streams: int
    context: int
    cache_mib: int
    free_bytes: int
    need_bytes: int


def _count(value, low, high):
    return type(value) is int and low <= value <= high


def parse_fit(fit, streams, context, requested, short_streams):
    """The engine's fit object for a ready event that names `streams` slots at `context`, or None when it does not
    match the request (`requested` is the requested (streams, context, cache_mib)). Each step changes only its own
    values, in the engine's order."""
    want_streams, want_context, want_cache = requested
    if not isinstance(fit, dict) or set(fit) != FIT_KEYS or fit["step"] not in STEPS:
        return None
    if fit["requested"] != dict(streams=want_streams, context=want_context, cache_mib=want_cache):
        return None
    if not (_count(streams, short_streams + 1, want_streams) and _count(context, 1, want_context) and
            _count(fit["cache_mib"], 0, want_cache) and _count(fit["free_bytes"], 0, 1 << 60) and
            _count(fit["need_bytes"], 0, 1 << 60)):
        return None
    step = fit["step"]
    if step == "cache" and not (streams == want_streams and context == want_context and fit["cache_mib"] < want_cache):
        return None
    if step == "streams" and not (streams < want_streams and context == want_context):
        return None
    if step == "context" and not (streams == short_streams + 1 and context < want_context):
        return None
    return MemoryFit(step, want_streams, want_context, want_cache, streams, context, fit["cache_mib"],
                     fit["free_bytes"], fit["need_bytes"])


def memory_holders_now(exclude, proc=Path("/proc"), uid=None):
    """The user's processes that hold the most memory, without the server and its engine (`exclude`, PIDs). None when
    the scan is not available (a server copy without tools/)."""
    if memory_holders is None:
        return None
    host = SimpleNamespace(proc=proc, uid=os.getuid() if uid is None else uid)
    found = memory_holders(host, limit=HOLDERS_SHOWN + len(exclude))
    return [holder for holder in found if holder.pid not in exclude][:HOLDERS_SHOWN]


def _layout(streams, context, cache_mib):
    text = f"{streams} {'session' if streams == 1 else 'sessions'} of {context:,} tokens"
    return text + (f" with a {cache_mib:,} MiB conversation cache" if cache_mib else "")


def _gib(value):
    return f"{value / (1 << 30):.1f} GiB"


def fit_line(fit, holders):
    """One line: the requested layout and why it does not fit, the started layout, the holders and the choices.
    `holders` is None when the scan is not available."""
    if holders is None:
        names = "not known (the holder scan of tools/crucible_cli is not available)"
    else:
        names = "; ".join(f"PID {h.pid} {h.name}: {h.rss / 1e9:.1f} GB host, {h.gtt / 1e9:.1f} GB GPU" for h in holders)
    verb = "needs" if fit.requested_streams == 1 else "need"
    return (f"Memory fit: {_layout(fit.requested_streams, fit.requested_context, fit.requested_cache_mib)} {verb} "
            f"{_gib(fit.need_bytes)} free (with the start-time reserve), but only {_gib(fit.free_bytes)} is free. "
            f"Crucible starts {_layout(fit.streams, fit.context, fit.cache_mib)}. "
            f"Programs that hold the most memory: {names or 'no readable process memory values'}. "
            "Crucible does not stop other programs. For the requested layout, free memory and start again, or use "
            "the smaller 3.05 pack. Use --fit off to refuse a layout that does not fit.")


def backend_fit(backend):
    """The backend's memory fit, or None (no fit, or a backend without one)."""
    fit = getattr(backend, "memory_fit", None)
    return fit if isinstance(fit, MemoryFit) else None


def sessions_line(backend):
    """The start line of the real slot count: "Sessions: 3 of 262,144 tokens (memory fit; 4 requested).". None for a
    backend that does not report the counts."""
    counts = getattr(backend, "session_counts", None)
    context = getattr(backend, "context", None)
    sessions = counts()[1] if callable(counts) else None
    if type(sessions) is not int or type(context) is not int:
        return None
    fit = backend_fit(backend)
    text = f"Sessions: {sessions} of {context:,} tokens"
    if fit is not None:
        text += f" (memory fit; {fit.requested_streams} requested)"
    return text + "."
