#!/usr/bin/env python3
"""Agent soak gate: synthetic Pi-shaped sessions through an OpenAI-compatible server (Oct 5 cache incident).

Every message is synthetic and seeded: about 12K tokens of system prompt and 23 tools, then tool results of 0.5K to
25K tokens after each assistant turn (tool calls passed back as the model wrote them; --thinking on), up to about
220K prompt tokens. `single` runs one such session; `multi` runs it beside two short sessions (three threads).

`agents` (Oct 6 multi-session failure): four sessions at once, thinking on, each with
its own 24K-token start, up to about 44K, on the h2h layout (4 x 131072 slots, 10 GiB cache). max_tokens 768 makes
the answer room close every think block after one token, so each turn runs a continuation job. The client sends
back the answer without the reasoning, as h2h_decode.py and many clients do, so the next prompt never repeats a
think turn's tail. `served3` runs three such sessions on the served layout. `agents6` runs six (the 6-slot candidate
B; the chain gives its layout with --server-args).

Rule: a later turn of a session re-prefills at most its appended tokens (this prompt minus the previous one) plus
--margin (4096, one prefill chunk). With --expect-replay it also re-prefills at most the tokens after the previous
turn's output plus --replay-margin. The served build broke the first rule from 94K (53248 cached forever).

--image-first WxH: one cold request with one synthetic screenshot before the sessions, so the encoder
scratch and the image buffers hold their memory for the whole soak. A failed image request is an ERROR.
--memory-floor-mib N (needs --tree): sample MemAvailable each second (this host) from server ready to the end
of the soak. A soak can end before its cache is full (S5-long fills about 20.6 of 25.0 GB), so the rule projects the
lowest value to a full cache: lowest MemAvailable - (cache budget - slot checkpoints - entry high-water), from the server
log. A projection under N makes a PASS a FAIL (rule "memory"). No sample or no cache record is an ERROR.

Crucible: --tree/--engine/--checkpoint start that tree's own server on 127.0.0.1:8794 (never 8790) with the served
flags. Any other server: --url. Exit 0 PASS, 1 FAIL, 2 ERROR.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PORT = 8794
CHARS_PER_TOKEN = 3.58  # synthetic lines, measured with the checkpoint tokenizer
SERVED = ("--streams 3 --ctx 262144 --short-streams 2 --short-ctx 131072 --prefill 4096 "
          "--conversation-cache-mib 10240 --clamp-max-tokens --default-thinking on --max-body-mib 16")
# The decode worker's h2h layout (Oct 6 failure).
H2H = ("--streams 4 --ctx 131072 --prefill 4096 --conversation-cache-mib 10240 --clamp-max-tokens "
       "--default-thinking on --max-body-mib 16")
# agents6 (train 22): the agents sessions, six at once, on the h2h layout with 6 slots.
H2H6 = H2H.replace("--streams 4 ", "--streams 6 ")
AGENT_SCENARIOS = {"agents": (4, H2H), "agents6": (6, H2H6), "served3": (3, SERVED)}
# The incident's appended tokens per turn (numbers only): small steps, then about 20K per turn.
STEPS = (690, 435, 750, 5550, 800, 1700, 4700, 5250, 720, 1500, 560, 1900, 1950, 1600, 600, 700, 1250, 740,
         20450, 19950, 19500, 19150, 21500, 20500, 24400, 4000, 3700, 18750, 2900, 9850)
WORDS = ("alpha beta gamma delta value index buffer config request handler stream module parser token cache "
         "state result error check update return import class function method field record table query path "
         "file line count limit offset length string number object array map list set queue worker thread").split()
TOOL_NAMES = ("read", "write", "edit", "bash", "grep", "find", "ls", "fetch", "search", "test", "build", "lint",
              "format", "diff", "commit", "status", "log", "blame", "open", "close", "rename", "move", "delete")


def lines(rng, chars):
    out, size = [], 0
    while size < chars:
        words = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12)))
        line = f"src/{rng.choice(WORDS)}_{rng.randint(0, 99)}/{rng.choice(WORDS)}.py:{rng.randint(1, 3000)}: {words}"
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)[:int(chars)]


def tool_result(seed, tokens):
    return lines(random.Random(seed), tokens * CHARS_PER_TOKEN)


def system_prompt(tokens):
    return "You are a coding agent. Use the tools to inspect and change files.\n" + lines(random.Random(-1), tokens * CHARS_PER_TOKEN)


def tool_specs():
    rng = random.Random(-2)
    return [dict(type="function", function=dict(
        name=name, description=f"{name}: " + " ".join(rng.choice(WORDS) for _ in range(40)),
        parameters=dict(type="object", required=["path"], properties=dict(
            path=dict(type="string", description="Target path. " + " ".join(rng.choice(WORDS) for _ in range(12))),
            options=dict(type="string", description="Options. " + " ".join(rng.choice(WORDS) for _ in range(12)))))))
        for name in TOOL_NAMES]


def schedule(kind, max_prompt, start):
    """Tool result sizes (tokens) until the prompt reaches max_prompt. Short sessions use the small steps only."""
    steps = [step for step in STEPS if step < 10000] if kind in ("short", "agent") else STEPS
    sizes, total = [], start
    for step in (steps * 4):
        if total + step > max_prompt:
            break
        sizes.append(max(500, step))
        total += sizes[-1]
    return sizes


def check(turns, margin=4096, replay_margin=64, expect_replay=False):
    """The soak rule over finished turns (dicts with session, turn, prompt/cached/completion tokens)."""
    sessions, failures, previous = {}, [], {}
    for row in sorted(turns, key=lambda r: (r["session"], r["turn"])):
        sessions[row["session"]] = sessions.get(row["session"], 0) + 1
        if row.get("error"):
            return dict(verdict="ERROR", error=f"turn {row['session']}/{row['turn']}: {row['error']}",
                        failures=failures, sessions=sessions)
        # A cold first turn has nothing to reuse; Halogen 0.16.2 omits its count.
        if row["turn"] == 0 and row.get("cached_tokens") is None:
            row = dict(row, cached_tokens=0)
        if any(type(row.get(key)) is not int for key in ("prompt_tokens", "cached_tokens", "completion_tokens")):
            return dict(verdict="ERROR", error=f"turn {row['session']}/{row['turn']} has no token counts",
                        failures=failures, sessions=sessions)
        prev, previous[row["session"]] = previous.get(row["session"]), row
        if prev is None or row["prompt_tokens"] < prev["prompt_tokens"]:
            continue
        prefilled = row["prompt_tokens"] - row["cached_tokens"]
        appended = row["prompt_tokens"] - prev["prompt_tokens"]
        after_output = max(0, appended - prev["completion_tokens"])
        rules = [("chunk", appended + margin)] + ([("replay", after_output + replay_margin)] if expect_replay else [])
        for rule, allowed in rules:
            if prefilled > allowed:
                failures.append(dict(session=row["session"], turn=row["turn"], prompt_tokens=row["prompt_tokens"],
                                     cached_tokens=row["cached_tokens"], prefilled=prefilled, appended=appended,
                                     allowed=allowed, rule=rule))
                break
    return dict(verdict="FAIL" if failures else "PASS", failures=failures, sessions=sessions)


def parse_stream(raw_lines, start=None):
    """One streamed chat completion: text, reasoning, tool calls, finish, usage, and a late error chunk."""
    reply = dict(content="", reasoning="", tool_calls=[], finish=None, prompt_tokens=None, cached_tokens=None,
                 completion_tokens=None, first_token_s=None, error=None, closed_at=None)
    calls, start = {}, time.monotonic() if start is None else start
    for raw in raw_lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if isinstance(chunk.get("error"), dict):
            reply["error"] = str(chunk["error"].get("message"))[:300]
        usage = chunk.get("usage")
        if usage:
            details = usage.get("prompt_tokens_details") or {}
            reply.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                         cached_tokens=details.get("cached_tokens", usage.get("cached_tokens")),
                         closed_at=(usage.get("completion_tokens_details") or {}).get("reasoning_closed_at"))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if reply["first_token_s"] is None and (delta.get("content") or delta.get("reasoning_content")
                                                   or delta.get("tool_calls")):
                reply["first_token_s"] = time.monotonic() - start
            reply["content"] += delta.get("content") or ""
            reply["reasoning"] += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call.get("index", 0), dict(id="", type="function",
                                                                    function=dict(name="", arguments="")))
                entry["id"] = call.get("id") or entry["id"]
                function = call.get("function") or {}
                entry["function"]["name"] += function.get("name") or ""
                entry["function"]["arguments"] += function.get("arguments") or ""
            reply["finish"] = choice.get("finish_reason") or reply["finish"]
    reply["tool_calls"] = [calls[index] for index in sorted(calls)]
    return reply


def content_hash(reply):
    """The reply's text, reasoning and tool calls, for output comparisons between runs (the slot-resume A/B).
    Call IDs are random server values, so they are left out."""
    calls = [call["function"] for call in reply["tool_calls"]]
    text = json.dumps([reply["content"], reply["reasoning"], calls], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stable_call_ids(reply, prefix):
    """The server's call IDs are random. Fixed IDs keep the next prompt equal between runs (the A/B gates)."""
    calls = [dict(call, id=f"call_{prefix}_{index}") for index, call in enumerate(reply["tool_calls"])]
    return dict(reply, tool_calls=calls)


def next_messages(reply, result, keep_reasoning=True):
    """The assistant turn as the client passes it back, then the synthetic tool result."""
    assistant = dict(role="assistant", content=reply["content"])
    if reply["reasoning"] and keep_reasoning:
        assistant["reasoning_content"] = reply["reasoning"]
    if reply["tool_calls"]:
        assistant["tool_calls"] = reply["tool_calls"]
        share = max(1, len(result) // len(reply["tool_calls"]))
        return [assistant] + [dict(role="tool", tool_call_id=call["id"], content=result[i * share:(i + 1) * share])
                              for i, call in enumerate(reply["tool_calls"])]
    return [assistant, dict(role="user", content="Tool output:\n" + result)]


def request_body(model, messages, tools, args):
    # With thinking on, max_tokens is Pi's 32768: a small max_tokens makes the answer room (max(1024, 15%)) close
    # the think block after one token. A thinking budget bounds the turn instead (Crucible and Halogen field).
    default = 32768 if args.thinking == "on" else 384
    body = dict(model=model, messages=messages, tools=tools, max_tokens=args.max_tokens or default, temperature=0,
                seed=7, stream=True, stream_options=dict(include_usage=True))
    if args.thinking == "off":
        body["chat_template_kwargs"] = dict(enable_thinking=False)
    else:
        body["reasoning_effort"] = args.reasoning_effort
        if args.thinking_budget:
            body["max_thinking_tokens"] = args.thinking_budget
    return body


def chat(url, model, messages, tools, args):
    body = request_body(model, messages, tools, args)
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        reply = parse_stream(response, start)
    reply["seconds"] = time.monotonic() - start
    return reply


def run_session(name, kind, args, model, rows, lock):
    rng_seed = sum(map(ord, name)) * 1000
    task = f"Session {name}: find and fix the failing check in the {name} module."
    start = args.system_tokens
    if kind == "agent":
        # Each agent session starts with its own large context (a file dump), as after a few long tool results.
        task += "\nCurrent files:\n" + tool_result(rng_seed - 1, args.agent_start)
        start += args.agent_start
    messages = [dict(role="system", content=system_prompt(args.system_tokens)), dict(role="user", content=task)]
    tools = tool_specs()
    limit = {"long": args.max_prompt, "agent": args.agent_prompt}.get(kind, args.short_prompt)
    sizes = schedule(kind, limit, start)
    for turn in range(len(sizes) + 1):
        reply = chat(args.url, model, messages, tools, args)
        row = dict(session=name, turn=turn, prompt_tokens=reply["prompt_tokens"], cached_tokens=reply["cached_tokens"],
                   completion_tokens=reply["completion_tokens"], finish=reply["finish"], seconds=round(reply["seconds"], 3),
                   ttft_s=None if reply["first_token_s"] is None else round(reply["first_token_s"], 3),
                   tool_calls=len(reply["tool_calls"]), think_closed_at=reply["closed_at"],
                   content_sha=content_hash(reply))
        if reply["error"]:
            row["error"] = reply["error"]
        with lock:
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Assistant turns add tokens too: stop before the next result would pass the limit.
        if (turn == len(sizes) or type(reply["prompt_tokens"]) is not int or reply["error"] or
                reply["prompt_tokens"] + sizes[turn] > limit):
            return
        if args.deadline and time.monotonic() - args.started > args.deadline:
            row["stopped_by_deadline"] = True
            return
        messages += next_messages(stable_call_ids(reply, f"{name}_{turn}"), tool_result(rng_seed + turn, sizes[turn]),
                                  args.reasoning_history == "keep")


LAUNCH = """import sys, http.server
tree, port = sys.argv[1], int(sys.argv[2]); sys.path.insert(0, tree)
import server.__main__ as entry
class Bound(http.server.ThreadingHTTPServer):
    def __init__(self, address, handler): super().__init__(("127.0.0.1", port), handler)
entry.ThreadingHTTPServer = Bound
sys.argv = ["server"] + sys.argv[3:]
raise SystemExit(entry.main())
"""


def start_server(args, log):
    command = [args.python, "-c", LAUNCH, str(args.tree), str(args.port), str(args.checkpoint), "--engine", str(args.engine)]
    command += shlex.split(args.server_args)
    process = subprocess.Popen(command, cwd=args.tree, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=5) as response:
                return process, json.loads(response.read())["data"][0]["id"]
        except OSError:
            time.sleep(3)
    process.terminate()
    raise RuntimeError("server not ready")


def file_sha256(path):
    """sha256 of the engine binary that this soak started, or None (no --engine, or no file)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except OSError:
        return None


MEMINFO = "/proc/meminfo"


def memavailable_kb(path=MEMINFO):
    """MemAvailable of this host in kB, or None when the file has no readable line."""
    try:
        with open(path) as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


class MemorySampler(threading.Thread):
    """The lowest MemAvailable while the soak runs (one read per interval)."""

    def __init__(self, interval=1.0, path=MEMINFO, read=memavailable_kb):
        super().__init__(daemon=True)
        self.interval, self.path, self.read, self.lowest_kb, self.samples = interval, path, read, None, 0
        self.stopped = threading.Event()

    def run(self):
        while True:
            value = self.read(self.path)
            if value is not None:
                self.samples += 1
                self.lowest_kb = value if self.lowest_kb is None else min(self.lowest_kb, value)
            if self.stopped.wait(self.interval):
                return

    def stop(self):
        self.stopped.set()
        if self.ident is not None:  # a server that did not start leaves the sampler unstarted
            self.join(timeout=5)


CACHE_BUDGET = re.compile(r"^conversation cache: (\d+) bytes budget", re.M)
CHECKPOINT_BYTES = re.compile(r"^slot checkpoints: .* (\d+) bytes from the cache budget$", re.M)
TURN_CACHE_BYTES = re.compile(r'^crucible_turn \{.*"cache_bytes": (\d+)', re.M)
TURN_PINNED_BYTES = re.compile(r'^crucible_turn \{.*"cache_pinned_bytes": (\d+)', re.M)


def cache_record(log_text):
    """The cache budget, the slot checkpoints in it and the cache high-water from a server log, or None.

    With cache_pinned_bytes the high-water is the pool's pinned memory,
    each chunk once: idle chunks already lower MemAvailable, and shared chunks count once. Older logs keep the entry
    bytes.
    """
    budget, used = CACHE_BUDGET.search(log_text), [int(value) for value in TURN_CACHE_BYTES.findall(log_text)]
    if budget is None or not used:
        return None
    checkpoints = CHECKPOINT_BYTES.search(log_text)
    record = dict(budget_bytes=int(budget[1]), checkpoint_bytes=int(checkpoints[1]) if checkpoints else 0,
                  high_water_bytes=max(used))
    pinned = [int(value) for value in TURN_PINNED_BYTES.findall(log_text)]
    return dict(record, high_water_bytes=max(pinned), pinned_counted=True) if pinned else record


def apply_memory_floor(report, lowest_kb, samples, floor_mib, cache=None):
    """The memory record, and a FAIL when a PASS, projected to a full cache, went under the floor.

    The cache part that the soak did not fill still costs host memory in daily use, so it comes off the lowest value.
    No sample or no cache record is an ERROR.
    """
    lowest = None if lowest_kb is None else lowest_kb // 1024
    memory = dict(lowest_memavailable_mib=lowest, samples=samples, floor_mib=floor_mib)
    if cache is not None:
        unfilled = max(0, cache["budget_bytes"] - cache["checkpoint_bytes"] - cache["high_water_bytes"]) >> 20
        memory.update(cache, unfilled_mib=unfilled, projected_memavailable_mib=None if lowest is None else lowest - unfilled)
    report = dict(report, memory=memory)
    if floor_mib is None:
        return report
    if lowest is None or cache is None:
        why = "no MemAvailable sample" if lowest is None else "no cache record in the server log"
        return dict(report, verdict="ERROR", error=report.get("error") or why)
    projected = memory["projected_memavailable_mib"]
    if projected < floor_mib and report["verdict"] == "PASS":
        failure = dict(rule="memory", lowest_memavailable_mib=lowest, unfilled_mib=memory["unfilled_mib"],
                       projected_memavailable_mib=projected, floor_mib=floor_mib)
        return dict(report, verdict="FAIL", failures=report["failures"] + [failure])
    return report


def image_first(url, model, size, timeout):
    """One cold request with one synthetic screenshot of SIZE (WxH). Returns the record; raises on an HTTP error."""
    from vision_session import screenshot
    width, height = (int(value) for value in size.split("x"))
    body = dict(model=model, max_tokens=1, temperature=0, stream=False, cache_control="bypass",
                chat_template_kwargs=dict(enable_thinking=False),
                messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url=screenshot(1, width, height))),
                                                      dict(type="text", text="What is the large number? Answer with it.")])])
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        usage = json.loads(response.read()).get("usage") or {}
        status = response.status
    return dict(size=size, status=status, prompt_tokens=usage.get("prompt_tokens"),
                seconds=round(time.monotonic() - start, 3))


def image_size(value):
    width, _, height = value.partition("x")
    if not (width.isdigit() and height.isdigit() and int(width) > 0 and int(height) > 0):
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, for example 2560x1440")
    return value


def switch_environment(environ):
    """The CRUCIBLE_* variables of this process; the server inherits them (the gate checks them)."""
    return {name: value for name, value in sorted(environ.items()) if name.startswith("CRUCIBLE_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=("single", "multi") + tuple(AGENT_SCENARIOS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", help="a running OpenAI-compatible server (for example Halogen)")
    parser.add_argument("--model", help="model id for --url (default: the first /v1/models entry)")
    parser.add_argument("--tree", type=Path, help="Crucible tree whose server/ to run")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=PORT, help="port of the tree's own server (never 8790)")
    parser.add_argument("--server-args", default=None,
                        help="server flags (default: the served layout; the h2h layout for agents)")
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--max-prompt", type=int, default=220000)
    parser.add_argument("--short-prompt", type=int, default=40000)
    parser.add_argument("--system-tokens", type=int, default=9000, help="plus about 3200 tokens of tools")
    parser.add_argument("--agent-start", type=int, default=24000, help="agents/served3: each session's own start")
    parser.add_argument("--agent-prompt", type=int, default=44000, help="agents/served3: prompt limit per session")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default 384, or 32768 with --thinking on, or 768 for agents/served3")
    parser.add_argument("--thinking", choices=("off", "on"), default=None, help="default off; on for agents/served3")
    parser.add_argument("--reasoning-history", choices=("keep", "drop"), default=None,
                        help="send reasoning back in history (default keep; drop for agents/served3)")
    parser.add_argument("--thinking-budget", type=int, default=None, help="max_thinking_tokens with --thinking on")
    parser.add_argument("--deadline", type=float, default=None,
                        help="seconds after the start; no session starts a new turn after it (the lease limit)")
    parser.add_argument("--reasoning-effort", default="high", help="with --thinking on")
    parser.add_argument("--request-timeout", type=float, default=400)
    parser.add_argument("--margin", type=int, default=4096)
    parser.add_argument("--replay-margin", type=int, default=64)
    parser.add_argument("--expect-replay", action="store_true")
    parser.add_argument("--image-first", type=image_size, default=None, metavar="WxH",
                        help="one cold image request before the sessions; the server needs --vision")
    parser.add_argument("--memory-floor-mib", type=int, default=None,
                        help="lowest MemAvailable (MiB, this host) for a PASS")
    args = parser.parse_args()
    agents = AGENT_SCENARIOS.get(args.scenario)
    if args.server_args is None:
        args.server_args = agents[1] if agents else SERVED
    if args.thinking is None:
        args.thinking = "on" if agents else "off"
    if args.reasoning_history is None:
        args.reasoning_history = "drop" if agents else "keep"
    if args.max_tokens is None and agents:
        args.max_tokens = 768  # the answer room (1024) closes each think block after one token
    args.started = time.monotonic()
    if bool(args.url) == bool(args.tree) or (args.tree and not (args.engine and args.checkpoint)):
        parser.error("give --url, or --tree with --engine and --checkpoint")
    if args.memory_floor_mib is not None and not args.tree:
        parser.error("--memory-floor-mib needs --tree (the cache record is in the server log)")
    rows, lock, process, model, error, first = [], threading.Lock(), None, args.model, None, None
    log = open(args.output.with_suffix(".server.log"), "w") if args.tree else None
    sampler = MemorySampler() if args.memory_floor_mib is not None else None
    try:
        if args.tree:
            process, model = start_server(args, log)
            args.url = f"http://127.0.0.1:{args.port}"
        elif not model:
            with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as response:
                model = json.loads(response.read())["data"][0]["id"]
        if sampler:  # from a ready server on: the soak's memory, not the checkpoint load
            sampler.start()
        if args.image_first:
            first = image_first(args.url, model, args.image_first, args.request_timeout)
        if agents:
            plan = [(f"agent-{index}", "agent") for index in range(agents[0])]
        else:
            plan = [("long", "long")] + ([("short-a", "short"), ("short-b", "short")] if args.scenario == "multi" else [])
        failed = []
        def worker(name, kind):
            try:
                run_session(name, kind, args, model, rows, lock)
            except Exception as exc:  # recorded; the verdict becomes ERROR
                failed.append(f"{name}: {exc}")
        threads = [threading.Thread(target=worker, args=item) for item in plan]
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads:
            thread.join()
        error = "; ".join(failed) or None
    except Exception as exc:
        error = str(exc)
    finally:
        if sampler:
            sampler.stop()
        if process is not None:
            process.terminate()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
    report = check(rows, args.margin, args.replay_margin, args.expect_replay)
    if error:
        report = dict(report, verdict="ERROR", error=error)
    if sampler:
        cache = cache_record(args.output.with_suffix(".server.log").read_text(errors="replace"))
        report = apply_memory_floor(report, sampler.lowest_kb, sampler.samples, args.memory_floor_mib, cache)
    if args.image_first:
        report = dict(report, image_first=first)
    report = dict(report, scenario=args.scenario, url=args.url, model=model, tree=str(args.tree or ""),
                  environment=switch_environment(os.environ), engine_sha256=file_sha256(args.engine),
                  engine=str(args.engine or ""), server_args=args.server_args if args.tree else None,
                  checkpoint=str(args.checkpoint or "") if args.tree else None,
                  max_prompt=max([r["prompt_tokens"] or 0 for r in rows] or [0]),
                  stopped_by_deadline=any(r.get("stopped_by_deadline") for r in rows), turns=rows)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(dict(verdict=report["verdict"], failures=report["failures"][:3], error=report.get("error"))))
    return {"PASS": 0, "FAIL": 1}.get(report["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
