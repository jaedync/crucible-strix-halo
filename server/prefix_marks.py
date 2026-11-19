"""Privacy-safe marks of the shared prompt head, for the crucible_turn line.

Two new sessions share a prompt prefix only up to their first different token. These marks show where that
difference is: in the reasoning line that the template renders before the tools, in the tools block, in one tool
definition, or after the tools block. The log gets only short keyed hashes and token counts, never text, tool names
or argument schemas.

Hash: BLAKE2b keyed with 16 random bytes that each process makes at import and keeps only in memory, cut to 10 hex
digits. Without the key, nobody can compute the hash of a guessed text, so a short text (a tool name, a level word)
cannot be found by trying candidates. A new process has a new key: hashes compare only within one server run.
"""
import hashlib
import json
import marshal
import os
import threading
from array import array
from collections import OrderedDict

HASH_DIGITS = 10
TOOL_HASH_CAP = 64  # the first 64 tool definitions get a hash each; prefix_tool_count gives the full count
CACHE_ENTRIES = 64  # heads of recent sessions; a repeat turn of a session skips the renders and the encode
# An entry keeps only hashes and counts (no text, no token IDs): one entry is under 4 KiB for any head size.
_KEY = os.urandom(16)
# A random part for each process, so client text cannot hold the sentinel (_head also takes the last match).
_NONCE = os.urandom(8).hex()
SYSTEM_SENTINEL = "\u2063crucible-system-mark-" + _NONCE + "\u2063"
USER_SENTINEL = "\u2063crucible-user-mark-" + _NONCE + "\u2063"
_LOCK = threading.Lock()


def keyed_hash(kind, text, digits=HASH_DIGITS):
    """digits hex digits of the keyed hash of text; kind keeps the hash families apart."""
    data = kind.encode() + b"\0" + text.encode("utf-8", "surrogatepass")
    return hashlib.blake2b(data, key=_KEY, digest_size=digits // 2).hexdigest()


def tool_text(tool):
    # The JSON that the template's tojson filter writes for each tool (server/template.py).
    return json.dumps(tool, ensure_ascii=False)


def _head(text):
    """The rendered text before the system sentinel, or None when the template did not render it."""
    end = text.rfind(SYSTEM_SENTINEL)
    return text[:end] if end > 0 else None


def prefix_marks(tokenizer, request, ids, system_end):
    """The prefix_* fields of a chat request's turn line ({} for other requests or on any error)."""
    try:
        return _marks(tokenizer, request, ids, system_end)
    except Exception:  # a diagnostic must never fail a request; the line then has no prefix_* fields
        return {}


def _tools(request):
    rendered = getattr(request, "render_tools", None)
    tools = list((request.tools if rendered is None else rendered) or ())
    return [] if request.tool_choice == "none" else tools  # the template renders no tools then (server/template.py)


def _cache_key(tools, request):
    """A keyed digest of everything that the head depends on. marshal version 0 writes equal values (same key order)
    as equal bytes, about 8 times faster than json.dumps for a Pi tool list (0.12 ms, not 0.8 ms, for 50 KB)."""
    setup = [tools, request.tool_choice, request.enable_thinking, request.reasoning_effort,
             getattr(request, "preserve_thinking", None)]
    try:
        data = marshal.dumps(setup, 0)
    except ValueError:  # a value that marshal cannot write: JSON is slower but takes every request value
        data = b"json\0" + json.dumps(setup, ensure_ascii=False).encode("utf-8", "surrogatepass")
    return hashlib.blake2b(data, key=_KEY, digest_size=16).digest()


def _marks(tokenizer, request, ids, system_end):
    if request.kind != "chat":
        return {}
    tools = _tools(request)
    key = _cache_key(tools, request)
    cache = tokenizer.__dict__.setdefault("_prefix_marks_cache", OrderedDict())
    with _LOCK:
        entry = cache.get(key)
        if entry is not None:
            cache.move_to_end(key)
    if entry is None:
        entry = _head_entry(tokenizer, request, tools)
        if entry is None:  # the template did not render the head: keep no slot for it
            return {}
        with _LOCK:
            cache[key] = entry
            while len(cache) > CACHE_ENTRIES:
                cache.popitem(last=False)
    fields = dict(entry["fields"])
    if system_end > 0:
        fields["prefix_system_end_tokens"] = system_end
    end = _exact_tokens(entry, ids) if tools else None
    if end is not None:
        fields["prefix_tools_end_tokens"] = end
    return fields


def _head_entry(tokenizer, request, tools):
    """The marks of one head (reasoning line and tools) and (count, digest) of each head encode; None if not rendered."""
    messages = (dict(role="system", content=SYSTEM_SENTINEL), dict(role="user", content=USER_SENTINEL))
    # The head without tools is the reasoning line (or none) that the template writes before the tools block.
    reasoning = _head(tokenizer.template.render(messages, request.enable_thinking, request.reasoning_effort, (), "auto",
                                                add_generation_prompt=False))
    text, restore = tokenizer.render_prompt(request, add_generation_prompt=False, messages=messages)
    head = _head(text)
    if reasoning is None or head is None:
        return None
    fields = dict(prefix_reasoning_hash=keyed_hash("reasoning", reasoning), prefix_tool_count=len(tools))
    if tools:
        block = head[len(reasoning):] if head.startswith(reasoning) else head
        fields.update(prefix_tools_hash=keyed_hash("tools", block),
                      prefix_tool_hashes=[keyed_hash("tool", tool_text(tool)) for tool in tools[:TOOL_HASH_CAP]])
    # The template writes a separator before the system text. The count stops before it (at the end of the tools
    # block text) when the tokens split there, as with the checkpoint tokenizer; else the full head is the second try.
    # Tokens can change at that edge, so a hit compares a digest of every prompt token before the count, not a length.
    prefixes = [tokenizer.encode_prompt(text, restore) for text in dict.fromkeys((head.rstrip(), head))]
    return dict(fields=fields, prefixes=tuple((len(prefix), _digest(prefix)) for prefix in prefixes if prefix))


def _digest(ids):
    """A keyed digest of token IDs, so that an entry keeps no IDs that decode back to the head text."""
    return hashlib.blake2b(array("i", ids), key=_KEY, digest_size=16).digest()


def _exact_tokens(entry, ids):
    """Token count of the first head encode that is an exact prefix of the prompt IDs, else None."""
    for count, digest in entry["prefixes"]:
        if len(ids) >= count and _digest(ids[:count]) == digest:
            return count
    return None
