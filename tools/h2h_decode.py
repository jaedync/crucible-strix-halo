#!/usr/bin/env python3
"""Head-to-head decode benchmark through an OpenAI-compatible server (Crucible serve or Halogen).

All content is synthetic: a Pi-shaped coding-agent conversation (system prompt, four tools, `read` tool calls
whose results are CPython standard-library files). The same prompt file serves both engines.

Each session is append-only, as an agent works: after the shared history it runs one turn per workload in order
(code, tool, code_sampled, think, prose). Each turn appends the engine's own reply (and a short tool result after
a tool call) and then the next user request. Sessions differ in the file their requests name.

  build:  h2h_decode.py build --out prompts.json [--contexts 4096,32768,102400]
  run:    h2h_decode.py run --prompts prompts.json --url http://127.0.0.1:8796/v1 --engine NAME --out rows.jsonl
          [--contexts ...] [--sessions 1,4] [--prefix {none,distinct,shared}] [--prefix-salt STR]
  table:  h2h_decode.py table rows.jsonl [rows.jsonl ...]

One session (session 0) runs alone. Then sessions 1..N run each turn at the same time (a barrier between turns).
Before its first turn, each session sends its first prompt once with max_tokens 1 (one at a time), so the first
measured turn starts from a warm prompt cache.

Prompt start (--prefix, rows with several sessions only): "none" sends the same long history in every session (the
prompt file as it is). "distinct" gives session k its own first system line, as agents have, so no session shares a
prompt start. "shared" gives all sessions of a row one first line that is new to every cache: the sessions share one
long prompt start, which an engine that keeps a prompt start only in the slot that owns it prefills again in each
other slot. One session (n = 1) always runs as "none". Each row records the mode it ran with in "prefix".

Decode rate of one request: (completion tokens - 1) / (last chunk time - first chunk time).
With several sessions, "total" counts only the window in which every session decodes: each stream's tokens are
spread over its chunks by text length, and the tokens inside the window are summed and divided by its length.
"""
import argparse
import collections
import hashlib
import json
import statistics
import sys
import sysconfig
import threading
import time
import urllib.request
from pathlib import Path

CPT = 3.6  # characters per token for Python source (checked against the server's prompt_tokens)
SYSTEM = """You are an expert coding assistant operating inside pi, a coding agent harness. You help users by \
reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement
- write: Create or overwrite files

Guidelines:
- Use bash for file operations like ls, rg, find
- Use read to examine files before editing. You must use this tool instead of cat or sed.
- Use edit for precise changes (edits[].oldText must match exactly)
- Each edits[].oldText is matched against the original file, not after earlier edits are applied.
- Keep edits[].oldText as small as possible while still being unique in the file.
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files

Current working directory: /work/project
"""
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read the contents of a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                    "offset": {"type": "number"}, "limit": {"type": "number"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Execute a bash command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                    "timeout": {"type": "number"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a file with exact text replacement.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array",
                    "items": {"type": "object", "properties": {"oldText": {"type": "string"},
                              "newText": {"type": "string"}}, "required": ["oldText", "newText"]}}},
                    "required": ["path", "edits"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write content to a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
]
# name: (thinking, sampling, max_tokens), in session order
WORKLOADS = {
    "code": (False, None, 512),
    "tool": (False, None, 512),
    "code_sampled": (False, dict(temperature=0.6, top_p=0.95, top_k=20, seed=7), 512),
    "think": (True, None, 768),
    "prose": (False, None, 512),
}
TOOL_RESULT = "Applied 1 edit."
MIN_WINDOW_S = 2.0  # a shorter all-decoding window gives no stable total


def source_files():
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in root.glob("*.py") if 2000 <= p.stat().st_size <= 60000)
    return [(f"src/{p.name}", p.read_text(errors="replace")) for p in files]


def first_function(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("def ") and "(" in s and not s.startswith("def _"):
            return s[4:s.index("(")]
    return "main"


def read_turn(i, path, text):
    call = {"id": f"call_{i}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": path})}}
    return [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": text}]


def user_request(workload, target):
    path, fn = target["path"], target["function"]
    if workload == "code":
        return (f"Write a new module src/{Path(path).stem}_cache.py that adds an LRU cache with a time-to-live to the "
                f"public functions of {path}. Include type hints, docstrings and a short unittest class. "
                "Do not call tools: answer with the complete file in one code block.")
    if workload == "tool":
        return (f"In {path}, rename the function `{fn}` to `{fn}_impl` and add a thin wrapper `{fn}` that calls it. "
                "Copy the full original function in edits[].oldText. Use the edit tool now.")
    if workload == "code_sampled":
        return (f"Now write src/test_{Path(path).stem}.py with pytest tests for every public function of {path}. "
                "Do not call tools: answer with the complete file in one code block.")
    if workload == "think":
        return (f"Find a real bug or edge case in {path} (for example in `{fn}`). Explain it and give the fixed "
                "function in a code block. Do not call tools.")
    return ("Write a short design note in plain prose (no code, no lists) for a new team member. Describe how the "
            f"module {path} is organized, what its main ideas are, and what to watch for when changing it. "
            "Do not call tools.")


def build(args):
    files = source_files()
    targets = [f for f in files if 2000 <= len(f[1]) < 4500 and "\ndef " in f[1]][-5:]  # files the session requests name
    rest = [f for f in files if f not in targets]
    out = {}
    for ctx in map(int, args.contexts.split(",")):
        budget = ctx * CPT - len(SYSTEM) - 3000 - sum(len(t) for _, t in targets)
        msgs, used, i = [{"role": "system", "content": SYSTEM}], 0, 0
        for path, text in rest + targets:
            if (path, text) not in targets and used + len(text) > budget:
                continue
            msgs += [{"role": "user", "content": f"Look at {path}."}] + read_turn(i, path, text)
            used += len(text)
            i += 1
        out[str(ctx)] = {"messages": msgs, "targets": [{"path": p, "function": first_function(t)} for p, t in targets]}
    Path(args.out).write_text(json.dumps(out))
    for ctx, v in out.items():
        chars = sum(len(m.get("content") or "") for m in v["messages"])
        print(f"ctx {ctx}: {len(v['messages'])} messages, {chars} chars (~{chars / CPT:.0f} tokens)")


def valid_json(text):
    """A call cut by max_tokens has incomplete arguments; a client would not send it back."""
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


class Session:
    """One append-only agent conversation."""

    def __init__(self, base, target, model, extra):
        self.messages, self.target, self.model, self.extra = list(base["messages"]), target, model, extra

    def body(self, workload, max_tokens=0):
        thinking, sampling, default_max = WORKLOADS[workload]
        msgs = self.messages + [{"role": "user", "content": user_request(workload, self.target)}]
        body = {"model": self.model, "messages": msgs, "tools": TOOLS, "stream": True,
                "stream_options": {"include_usage": True}, "max_tokens": max_tokens or default_max,
                "enable_thinking": thinking}
        if thinking:
            body["reasoning_effort"] = "medium"
        body.update(sampling or {"temperature": 0.0})
        body.update(self.extra)
        return body

    def append(self, workload, res):
        """Add the request and the engine's reply, as a client keeps its history."""
        self.messages.append({"role": "user", "content": user_request(workload, self.target)})
        calls = [{"id": f"call_t{len(self.messages)}_{k}", "type": "function",
                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                 for k, c in enumerate(res["calls"]) if valid_json(c["arguments"])]
        reply = {"role": "assistant", "content": res["content"]}
        if calls:
            reply["tool_calls"] = calls
        self.messages.append(reply)
        for c in calls:
            self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": TOOL_RESULT})


def stream(url, body, timeout=900):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    chunks, usage, finish, err = [], None, None, None
    content, calls = [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    finish = ch.get("finish_reason") or finish
                    content.append(d.get("content") or "")
                    n = len(d.get("content") or "") + len(d.get("reasoning_content") or d.get("reasoning") or "")
                    for tc in d.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
                        n += len(fn.get("arguments") or "") + len(fn.get("name") or "")
                    if n:
                        chunks.append((time.time(), n))
    except Exception as e:  # recorded in the row; the table skips it
        err = repr(e)[:300]
    return {"t0": t0, "chunks": chunks, "usage": usage or {}, "finish": finish, "error": err,
            "content": "".join(content), "calls": [calls[k] for k in sorted(calls)]}


def per_request(res):
    toks = res["usage"].get("completion_tokens") or 0
    ch = res["chunks"]
    if toks < 32 or len(ch) < 2 or ch[-1][0] <= ch[0][0]:
        return None
    return (toks - 1) / (ch[-1][0] - ch[0][0])


def max_gap(res):
    """Longest wait between two streamed chunks of one request (s): prefill of a peer shows up here."""
    ch = res["chunks"]
    return max((b[0] - a[0] for a, b in zip(ch, ch[1:])), default=None)


class Clocks:
    """Samples the platform power profile and the GPU clock (sclk) during a turn (read-only sysfs)."""
    PROFILE = Path("/sys/firmware/acpi/platform_profile")

    def __init__(self):
        self.cards = sorted(Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk"))
        self.samples, self.stop = [], threading.Event()

    def _read(self):
        for card in self.cards:
            for line in card.read_text().splitlines():
                if line.rstrip().endswith("*"):
                    return int(line.split()[1].lower().rstrip("mhz*").strip() or 0)
        return None

    def __enter__(self):
        def loop():
            while not self.stop.wait(0.5):
                try:
                    self.samples.append(self._read())
                except (OSError, ValueError):
                    pass
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        mhz = [x for x in self.samples if x]
        try:
            profile = self.PROFILE.read_text().strip()
        except OSError:
            profile = None
        return {"platform_profile": profile, "sclk_mhz_median": statistics.median(mhz) if mhz else None,
                "sclk_mhz_min": min(mhz) if mhz else None}


def command_trace(url, since):
    """Crucible only: decode command times from GET /v1/crucible/trace (server wall, engine time, gaps), in ms."""
    try:
        with urllib.request.urlopen(f"{url}/crucible/trace?since={since}", timeout=30) as r:
            entries = json.loads(r.read())["data"]
    except Exception:  # Halogen has no trace endpoint
        return None, since
    entries = [e for e in entries if e["t_start"] >= since]
    if not entries:
        return None, since
    decode = [e for e in entries if e["op"] in ("step", "spec_step", "verify_step")]
    gaps = [b["t_start"] - a["t_end"] for a, b in zip(entries, entries[1:])]
    engine = [e["engine_s"] for e in decode if e.get("engine_s") is not None]
    ms = lambda xs: round(1e3 * statistics.mean(xs), 3) if xs else None
    return {"commands": len(entries), "decode_commands": len(decode),
            "decode_wall_ms": ms([e["t_end"] - e["t_start"] for e in decode]), "decode_engine_ms": ms(engine),
            "gap_ms": ms(gaps), "ops": dict(collections.Counter(e["op"] for e in entries))}, max(e["t_end"] for e in entries)


def e2e(res):
    """Output tokens over the whole request time. Used for tool turns: a server may send a call in one chunk."""
    toks = res["usage"].get("completion_tokens") or 0
    return toks / (res["chunks"][-1][0] - res["t0"]) if toks and res["chunks"] else None


def window_total(results):
    """Tokens per second while every stream decodes (see the module docstring)."""
    spans = [(r["chunks"][0][0], r["chunks"][-1][0]) for r in results if len(r["chunks"]) > 1]
    if len(spans) != len(results):
        return None, 0.0
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    if hi - lo < MIN_WINDOW_S:
        return None, max(0.0, hi - lo)
    total = 0.0
    for r in results:
        chars = sum(n for _, n in r["chunks"]) or 1
        per_char = (r["usage"].get("completion_tokens") or 0) / chars
        total += sum(n * per_char for t, n in r["chunks"] if lo < t <= hi)
    return total / (hi - lo), hi - lo


def turn(url, sessions, workload, max_tokens, arrival=None):
    """Run one turn of every session at the same time and append the replies.

    arrival: (delay_s, body) of a fresh request (a new session's first prompt, no cache) that starts while the
    sessions decode. Its result is returned last and is not appended to any session."""
    bodies = [x.body(workload, max_tokens) for x in sessions]
    results = [None] * (len(sessions) + (arrival is not None))

    def go(k):
        results[k] = stream(url, bodies[k])

    def arrive():
        time.sleep(arrival[0])
        results[-1] = stream(url, arrival[1])
    threads = [threading.Thread(target=go, args=(k,)) for k in range(len(sessions))]
    if arrival is not None:
        threads.append(threading.Thread(target=arrive))
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for x, r in zip(sessions, results):
        x.append(workload, r)
    return results


def arrival_body(base, model, salt):
    """A new session's first turn: a unique first line defeats every prefix cache, so the whole prompt prefills."""
    msgs = [dict(m) for m in base["messages"]]
    msgs[0]["content"] = f"Session {salt}.\n" + msgs[0]["content"]
    msgs.append({"role": "user", "content": user_request("code", base["targets"][0])})
    return {"model": model, "messages": msgs, "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 64, "enable_thinking": False, "temperature": 0.0}


def with_first_line(base, salt):
    """The same history behind a new first line in the system message: no prefix cache has seen it."""
    msgs = [dict(m) for m in base["messages"]]
    msgs[0]["content"] = f"Session {salt}.\n" + msgs[0]["content"]
    return dict(base, messages=msgs)


def session_bases(base, n, mode, salt):
    """The prompt file entry of each of the n sessions of one row, for the --prefix mode (see the module docstring)."""
    if n == 1 or mode == "none":
        return [base] * n
    if mode == "shared":
        return [with_first_line(base, salt)] * n
    return [with_first_line(base, f"{salt}-{k}") for k in range(n)]


def run(args):
    prompts = json.loads(Path(args.prompts).read_text())
    url = args.url.rstrip("/")
    out = open(args.out, "a")
    deadline = time.time() + args.budget_s
    extras = json.loads(args.extra)
    extras = extras if isinstance(extras, list) else [extras]
    workloads = list(WORKLOADS)
    salt0 = args.prefix_salt or str(time.time_ns())
    rows = [(c, int(n), e) for c in args.contexts.split(",") for n in args.sessions.split(",") for e in extras]
    for row_no, (ctx, n, extra) in enumerate(rows):
        base = prompts[ctx]
        first = 0 if n == 1 else 1  # session 0 runs alone; sessions 1..n run together
        bases = session_bases(base, n, args.prefix, f"{salt0}-{row_no}")  # a new salt per row: no row reuses another's cache
        sessions = [Session(bases[k], base["targets"][(first + k) % len(base["targets"])], args.model, extra)
                    for k in range(n)]
        for k, x in enumerate(sessions):
            t = time.time()
            r = stream(url, x.body(workloads[0], 1))
            print(f"warm ctx {ctx} n{n} {extra} s{k}: prompt {r['usage'].get('prompt_tokens')} in {time.time() - t:.1f}s "
                  f"{r['error'] or ''}", flush=True)
        since = 0.0
        _, since = command_trace(url, since)
        for workload in workloads[:args.turns]:
            if time.time() > deadline:
                print("budget reached; stopping", flush=True)
                return
            arrival = None
            if args.arrival and n > 1:
                arrival = (args.arrival_delay, arrival_body(prompts[str(args.arrival)], args.model,
                                                            f"{args.arrival_salt}-{workload}" if args.arrival_salt
                                                            else f"{time.time_ns()}"))
            with Clocks() as clocks:
                results = turn(url, sessions, workload, args.max_tokens, arrival)
            if arrival is not None:
                new = results.pop()
                ends = [r["chunks"][-1][0] for r in results if r["chunks"]]
                firsts = [r["chunks"][0][0] for r in results if r["chunks"]]
                toks = sum(r["usage"].get("completion_tokens") or 0 for r in results)
                arrival_row = {"ctx": int(args.arrival), "prompt_tokens": new["usage"].get("prompt_tokens"),
                               "ttft_s": round(new["chunks"][0][0] - new["t0"], 3) if new["chunks"] else None,
                               "error": new["error"],
                               # every token of the turn over its whole span, prefill window included
                               "turn_total": toks / (max(ends) - min(firsts)) if ends and max(ends) > min(firsts) else None}
            rates = [per_request(r) for r in results]
            total, win = window_total(results) if n > 1 else (rates[0], 0.0)
            row = {"engine": args.engine, "ctx": int(ctx), "workload": workload, "sessions": n, "extra": extra,
                   "prefix": args.prefix if n > 1 else "none", "rates": rates, "total": total, "window_s": win, "e2e": [e2e(r) for r in results],
                   "max_gap_s": [max_gap(r) for r in results], **clocks.summary(),
                   "trace": None, "arrival": arrival_row if arrival is not None else None,
                   "prompt_tokens": [r["usage"].get("prompt_tokens") for r in results],
                   "completion_tokens": [r["usage"].get("completion_tokens") for r in results],
                   "ttft_s": [round(r["chunks"][0][0] - r["t0"], 3) if r["chunks"] else None for r in results],
                   "finish": [r["finish"] for r in results], "calls": [len(r["calls"]) for r in results],
                   "errors": [r["error"] for r in results if r["error"]],
                   "content_sha": [hashlib.sha256(r["content"].encode()).hexdigest()[:12] for r in results]}
            row["trace"], since = command_trace(url, since)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: row[k] for k in ("ctx", "workload", "sessions", "total", "completion_tokens",
                                                   "ttft_s", "calls", "errors")}), flush=True)


def table(args):
    rows = [json.loads(l) for f in args.files for l in open(f) if l.strip()]
    for r in rows:
        if r.get("extra"):
            r["engine"] += " " + json.dumps(r["extra"], separators=(",", ":"))
    keys = sorted({(r["ctx"], r["workload"], r["sessions"]) for r in rows})
    engines = sorted({r["engine"] for r in rows})
    print("| ctx | workload | n | " + " | ".join(f"{e} per-session / total" for e in engines) + " |")
    for k in keys:
        cells = []
        for e in engines:
            got = [r for r in rows if (r["ctx"], r["workload"], r["sessions"]) == k and r["engine"] == e]
            if not got:
                cells.append("-")
                continue
            r = got[-1]
            if k[1] == "tool":  # end-to-end rate (see e2e); the total is their sum
                per = [x for x in r["e2e"] if x]
                cells.append(f"{statistics.median(per):.1f} / {sum(per):.1f} e2e" if per else "err")
                continue
            per = [x for x in r["rates"] if x]
            med = statistics.median(per) if per else None
            total = f"{r['total']:.1f}" if r["total"] else "-"
            cells.append(f"{med:.1f} / {total}" if med else "err")
        print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--contexts", default="4096,32768,102400")
    b.add_argument("--sessions", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--engine", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="crucible")
    r.add_argument("--contexts", default="4096,32768,102400")
    r.add_argument("--workloads", default=",".join(WORKLOADS))
    r.add_argument("--sessions", default="1,4")
    r.add_argument("--max-tokens", type=int, default=0)
    r.add_argument("--budget-s", type=float, default=1e9)
    r.add_argument("--arrival", type=int, default=0, help="with several sessions: a fresh prompt of this context "
                   "(a prompts.json key) arrives during each turn")
    r.add_argument("--arrival-delay", type=float, default=3.0)
    r.add_argument("--arrival-salt", default="", help="fixed first-line salt for the arrivals (turn name appended), "
                   "so two runs prefill the same arrival prompts; empty uses the clock")
    r.add_argument("--prefix", choices=("none", "distinct", "shared"), default="none",
                   help="prompt start of the rows with several sessions: none = the same history in every session, "
                   "distinct = a unique first line per session, shared = one new first line for all sessions of a row")
    r.add_argument("--prefix-salt", default="", help="fixed salt of the first lines, for tests and reruns; empty uses the clock")
    r.add_argument("--turns", type=int, default=len(WORKLOADS), help="run only the first N turns of each session")
    r.add_argument("--extra", default="{}", help="JSON merged into every measured request, or a list of them "
                   "(each runs its own sessions, in order)")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"build": build, "run": run, "table": table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
