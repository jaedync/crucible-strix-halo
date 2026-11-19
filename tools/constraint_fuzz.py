#!/usr/bin/env python3
"""CPU fuzz of the structured output boundary. No GPU, no network.

  constraint_fuzz.py run --runner build/tests/constraint_fuzz_runner --tokenizer TOKENIZER_JSON --out DIR
                         [--seed 1] [--count 5000] [--minutes 0] [--walk 64] [--seeds DIR]

Each case is a random chat request body with response_format or strict tools. Step 1 is the server: api.validate_request
and structured.engine_constraint, as the HTTP handler calls them. A refusal (ApiError) is a 400 for that client and
ends the case; any other exception is a finding (a 500). Step 2 is the engine: the accepted constraint goes to
constraint_fuzz_runner, which compiles it in the helper process and walks the matcher on the real vocabulary, as the
serve loop does. Findings (rules in `findings`): the runner process ends or stops answering (the engine would end or
hang), a server-accepted spec that reaches the helper's deadline or memory limit (the server limits let it through),
a compiler crash, a compile over SLOW_COMPILE_MS, an open or a token over SLOW_SERVE_MS (the serve loop would wait),
the grammar cache over its byte limit, and a legal token that the matcher refused. A finding case is written to
DIR/findings.jsonl with its body; DIR/summary.json holds the counts. Exit 0: no finding; 1: findings; 2: tool error.
"""
import argparse
import copy
import json
import os
import random
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import api, structured  # noqa: E402

SLOW_COMPILE_MS = 1000.0    # the struct-guard calibration: every shape at its limit compiles in 0.72 s or less
# A slow compile runs off the serve loop and stays under the helper's deadline: a note (notes.jsonl), not a finding.
NOTES = frozenset({"slow_compile"})
SLOW_SERVE_MS = 100.0       # an open of a compiled grammar or one masked token on the serve loop
CACHE_LIMIT = 512 << 20     # kGrammarCacheBytes (engine/runtime/constraint_helper.h)
ANSWER_S = 60.0             # the runner's longest answer: the helper start, one 5 s compile and the walk

# ---- the request generator (pure, seeded) ----------------------------------------------------------------------------

CHARS = ["a", "b", "z", "0", "9", "_", "-", ".", " ", "\\d", "\\w", "\\s", "\\.", "\\\\", "\\/", "é", "中", "\\u00e9",
         "\"", "'", "\\n", "\\t", "<", "/"]
CLASSES = ["[a-z]", "[A-Z0-9_]", "[^\"]", "[^\\\\]", "[ab]", "[0-9a-f]", "[\\w.-]", "[é-ü]", "[^a]"]
FORMATS = ["email", "date-time", "date", "time", "uri", "uri-reference", "uuid", "ipv4", "ipv6", "hostname",
           "duration", "idn-email", "byte", "regex", "unknown-format"]
KEYS = ["a", "name", "k_1", "é", "中文", "with space", "quote\"d", "back\\slash", "", "x" * 64, "</parameter>",
        "<tool_call>", "new\nline", "$ref", "type"]


def bound(rng, top):
    """A count near the interesting values: small, at a limit, one over it, or large."""
    return rng.choice([0, 1, 2, 7, 16, 17, 64, 255, 256, 999, 1000, 1001, 1024, 1025, 4096, 65536, 65537, top])


def random_pattern(rng, depth=0):
    parts = []
    for _ in range(rng.randint(1, 4)):
        roll = rng.random()
        if roll < 0.35:
            atom = rng.choice(CHARS)
        elif roll < 0.6:
            atom = rng.choice(CLASSES)
        elif roll < 0.8 and depth < 3:
            inner = "|".join(random_pattern(rng, depth + 1) for _ in range(rng.randint(1, 3)))
            atom = rng.choice(["(", "(?:", "(?=", "(?!"]) + inner + ")"
        elif roll < 0.85:
            # A chain of nested groups around one atom, near the group depth limit (16) of grammar_limits.py.
            levels = rng.choice([4, 15, 16, 17, 30])
            atom = "(" * levels + rng.choice(CHARS) + ")" * levels
        else:
            atom = rng.choice(["\\1", ".", "\\b", "^", "$", "\\p{L}"])
        quant = rng.choice(["", "", "*", "+", "?", "*?", "+?", "{%d}" % bound(rng, 3000),
                            "{%d,%d}" % (rng.randint(0, 5), bound(rng, 3000)), "{%d,}" % bound(rng, 3000)])
        parts.append(atom + quant)
    text = "".join(parts)
    if depth == 0 and rng.random() < 0.3:
        text = "^" + text + "$"
    return text


NODE_BUDGET = 600           # schema nodes of one generated schema: wide shapes get leaf children


def random_schema(rng, depth=0, defs=None, budget=None):
    budget = [NODE_BUDGET] if budget is None else budget
    budget[0] -= 1
    kinds = ["string", "pattern", "format", "integer", "number", "boolean", "null", "enum", "const", "types"]
    if depth < 8 and budget[0] > 0:
        kinds += ["object", "object", "array", "anyof"] + (["ref"] if defs else [])
    kind = rng.choice(kinds)
    if kind in ("boolean", "null"):
        return {"type": kind}
    if kind == "string":
        schema = {"type": "string"}
        if rng.random() < 0.5:
            schema["maxLength"] = bound(rng, 10 ** 9)
        if rng.random() < 0.3:
            schema["minLength"] = bound(rng, 100)
        return schema
    if kind == "pattern":
        return {"type": "string", "pattern": random_pattern(rng)}
    if kind == "format":
        return {"type": "string", "format": rng.choice(FORMATS)}
    if kind in ("integer", "number"):
        schema = {"type": kind}
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if rng.random() < 0.3:
                schema[key] = rng.choice([0, -1, 1, 10 ** 18, -(10 ** 18), 1.5, -0.25, 2 ** 63, 1e300])
        return schema
    if kind == "enum":
        n = rng.choice([1, 2, 5, 100, 1024, 1025])
        return {"type": "string", "enum": [rng.choice(KEYS) + str(i) for i in range(n)]}
    if kind == "const":
        return {"const": rng.choice(["x", "", "\u0000\u001f", "q\"\\", "</parameter>", 7, None, [1, 2], {"a": 1}])}
    if kind == "types":
        return {"type": rng.sample(["string", "integer", "number", "boolean", "null", "object", "array"],
                                   rng.randint(1, 4))}
    if kind == "object":
        n = rng.choice([0, 1, 3, 8, 129, 300])
        props = {f"{rng.choice(KEYS)}{i}": random_schema(rng, depth + 1, defs, budget) for i in range(n)}
        schema = {"type": "object", "properties": props}
        if rng.random() < 0.7:
            schema["required"] = [key for key in props if rng.random() < 0.5]
        schema["additionalProperties"] = rng.choice([False, True, {"type": "integer"}])
        if rng.random() < 0.1:
            schema["minProperties"], schema["maxProperties"] = 1, bound(rng, 100)
        return schema
    if kind == "array":
        schema = {"type": "array", "items": random_schema(rng, depth + 1, defs, budget)}
        if rng.random() < 0.5:
            schema["maxItems"] = bound(rng, 10 ** 6)
        if rng.random() < 0.3:
            schema["minItems"] = bound(rng, 100)
        if rng.random() < 0.15:
            schema["prefixItems"] = [random_schema(rng, depth + 1, defs, budget) for _ in range(rng.randint(1, 3))]
        return schema
    if kind == "anyof":
        return {"anyOf": [random_schema(rng, depth + 1, defs, budget) for _ in range(rng.choice([1, 2, 4, 9]))]}
    return {"$ref": "#/$defs/" + rng.choice(sorted(defs))}


def mutate(rng, schema):
    """A copy of `schema` with one change: a bound at a limit, a subtree twice, an anyOf around a node."""
    out = copy.deepcopy(schema)
    nodes = []

    def walk(node):
        if isinstance(node, dict):
            nodes.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(out)
    if not nodes:
        return out
    node = rng.choice(nodes)
    roll = rng.random()
    numeric = [key for key, value in node.items() if type(value) is int]
    if roll < 0.4 and numeric:
        node[rng.choice(numeric)] = bound(rng, 10 ** 6)
    elif roll < 0.6 and isinstance(node.get("pattern"), str):
        node["pattern"] = rng.choice([node["pattern"] * 2, "(" + node["pattern"] + ")+", random_pattern(rng)])
    elif roll < 0.8:
        inner = copy.deepcopy(node)
        node.clear()
        node["anyOf"] = [inner, copy.deepcopy(inner)]
    else:
        node.setdefault("properties", {})["m"] = random_schema(rng, 6)
        node["type"] = "object"
    return out


def random_tools(rng, schema_for):
    tools = []
    for t in range(rng.choice([1, 2, 3, 8])):
        props = {f"{rng.choice(KEYS)}{j}": schema_for() for j in range(rng.choice([0, 1, 3, 6]))}
        tools.append({"type": "function", "function": {
            "name": f"tool_{t}", "strict": rng.random() < 0.85,
            "parameters": {"type": "object", "properties": props, "additionalProperties": False,
                           "required": [key for key in props if rng.random() < 0.6]}}})
    return tools


def random_body(rng, index, seeds=()):
    defs = None
    if rng.random() < 0.25:
        defs = {"leaf": {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
                "item": {"type": "array", "items": {"$ref": "#/$defs/leaf"}, "maxItems": 3}}

    def schema_for():
        if seeds and rng.random() < 0.3:
            return mutate(rng, rng.choice(seeds))
        return random_schema(rng, 1, defs)
    body = {"messages": [{"role": "user", "content": f"case {index}"}], "max_tokens": 64,
            "enable_thinking": rng.random() < 0.3}
    if rng.random() < 0.6:
        schema = schema_for()
        if defs and isinstance(schema, dict):
            schema = dict(schema, **{"$defs": defs})
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "r", "schema": schema,
                                                                            "strict": rng.random() < 0.5}}
    else:
        body["tools"] = random_tools(rng, schema_for)
        body["tool_choice"] = rng.choice(["auto", "required", {"type": "function", "function": {"name": "tool_0"}}])
        body["parallel_tool_calls"] = rng.random() < 0.5
        if rng.random() < 0.1:
            body["response_format"] = {"type": "json_object"}
    return body


def body_kind(body):
    return "tools" if "tools" in body else "answer"


# ---- the server step and the rules (pure) ----------------------------------------------------------------------------

def server_step(body):
    """("ok", engine constraint), ("refused", api message) or ("fault", exception name): the HTTP handler's path."""
    try:
        request = api.validate_request(json.loads(json.dumps(body)), "chat", "crucible", structured=True)
        constraint = structured.engine_constraint(request) if request.constraint else None
    except api.ApiError as error:
        return "refused", str(error)[:120]
    except Exception as error:  # every other exception is a 500 of the server: a finding  # noqa: BLE001
        return "fault", type(error).__name__
    if constraint is None:
        return "refused", "no constraint"
    return "ok", constraint


FIELDS = ("index", "status", "make_ms", "child_rss_kb", "grammar_bytes", "cache_bytes", "open_ms", "tokens",
          "max_token_ms", "walk_ms")


def parse_result(line):
    words = line.split()
    if len(words) != 11 or words[0] != "R":
        raise ValueError("not a runner result line")
    values = dict(zip(FIELDS, words[1:]))
    out = dict(status=values["status"])
    for key in FIELDS:
        if key != "status":
            out[key] = float(values[key]) if "." in values[key] else int(values[key])
    return out


def findings(result):
    """The finding names of one runner result of a server-accepted spec (empty: a clean case)."""
    status, found = result["status"], []
    if status == "constraint_limit/time":
        found.append("accepted_spec_hits_deadline")
    elif status == "constraint_limit/memory":
        found.append("accepted_spec_hits_memory")
    elif status.endswith("/crash"):
        found.append("compiler_crash")
    elif status.endswith("/helper") or status.endswith("/load"):
        found.append("helper_" + status.rsplit("/", 1)[1])
    elif status == "ok" and result["make_ms"] > SLOW_COMPILE_MS:
        found.append("slow_compile")
    if result["open_ms"] > SLOW_SERVE_MS:
        found.append("slow_open")
    if result["max_token_ms"] > SLOW_SERVE_MS:
        found.append("slow_token")
    if result["cache_bytes"] > CACHE_LIMIT:
        found.append("cache_over_limit")
    return found


# ---- the run (CPU, the runner process) -------------------------------------------------------------------------------

class Runner:
    """constraint_fuzz_runner as a child process; one line in, one "S" and one "R" (or "E") line out."""

    def __init__(self, path, tokenizer, walk):
        self.argv = [str(path), str(tokenizer), str(walk)]
        self.proc = None
        self.starts = 0

    def start(self):
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.pending = b""
        self.starts += 1

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        if self.proc:
            self.proc.wait()
        self.proc = None

    def line(self, deadline):
        """The next output line, "" when the runner ended, None at the deadline. It reads the pipe with os.read into
        its own buffer: a buffered readline after select() can hold the next line while select() waits for more."""
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.pending:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                return None
            data = os.read(fd, 1 << 16)
            if not data:
                return ""
            self.pending += data
        text, self.pending = self.pending.split(b"\n", 1)
        return text.decode(errors="replace")

    def case(self, constraint):
        """(result dict or None, event): event is "ok", "refused_token", "ended" or "hang"."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        spec = constraint["spec"].replace("\t", " ").replace("\n", " ")
        self.proc.stdin.write(f"{constraint['kind']}\t{int(bool(constraint.get('reasoning')))}\t{spec}\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + ANSWER_S + 60  # the first case also loads the vocabulary
        event = "ok"
        while True:
            text = self.line(deadline)
            if text is None:
                self.stop()
                return None, "hang"
            if text == "":
                self.stop()
                return None, "ended"
            if text.startswith("E "):
                event = "refused_token"
            elif text.startswith("R "):
                return parse_result(text), event


SLOWEST = 8


def keep_slowest(slowest, key, case):
    """Keeps the SLOWEST cases with the largest key in `slowest` (a list of (key, case), largest first)."""
    slowest.append((key, case))
    slowest.sort(key=lambda item: -item[0])
    del slowest[SLOWEST:]


def json_depth(value):
    """The nesting depth of a JSON value, without recursion."""
    deepest, stack = 0, [(value, 1)]
    while stack:
        node, depth = stack.pop()
        deepest = max(deepest, depth)
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        stack.extend((child, depth + 1) for child in children)
    return deepest


def run_main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seeds = []
    if args.seeds:
        for path in sorted(Path(args.seeds).glob("*.json")):
            try:
                value = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            # The deep seeds (20,000 levels) are the server's depth limit already; a copy of one would recurse.
            if isinstance(value, dict) and len(path.read_bytes()) < 64 * 1024 and json_depth(value) <= 64:
                seeds.append(value)
    runner = Runner(args.runner, args.tokenizer, args.walk)
    counts = dict(cases=0, refused=0, accepted=0, findings=0, notes=0, statuses={}, kinds={}, worst_compile_ms=0,
                  worst_open_ms=0, worst_token_ms=0, max_cache_bytes=0)
    stop_at = time.monotonic() + args.minutes * 60 if args.minutes else None
    slowest = []
    with open(out / "findings.jsonl", "w") as log, open(out / "notes.jsonl", "w") as notes:
        for index in range(args.count):
            if stop_at and time.monotonic() > stop_at:
                break
            body = random_body(rng, index, seeds)
            counts["cases"] += 1
            verdict, value = server_step(body)
            found, result = [], None
            if verdict == "refused":
                counts["refused"] += 1
            elif verdict == "fault":
                found = ["server_fault_" + value]
            else:
                counts["accepted"] += 1
                result, event = runner.case(value)
                if result is None:
                    found = ["runner_" + event]
                else:
                    found = findings(result) + (["legal_token_refused"] if event == "refused_token" else [])
                    counts["statuses"][result["status"]] = counts["statuses"].get(result["status"], 0) + 1
                    if result["status"] == "ok":
                        counts["worst_compile_ms"] = max(counts["worst_compile_ms"], result["make_ms"])
                    counts["worst_open_ms"] = max(counts["worst_open_ms"], result["open_ms"])
                    counts["worst_token_ms"] = max(counts["worst_token_ms"], result["max_token_ms"])
                    counts["max_cache_bytes"] = max(counts["max_cache_bytes"], result["cache_bytes"])
                    keep_slowest(slowest, result["max_token_ms"], dict(index=index, result=result, body=body))
            for name in found:
                counts["kinds"][name] = counts["kinds"].get(name, 0) + 1
            if found:
                serious = [name for name in found if name not in NOTES]
                counts["findings" if serious else "notes"] += 1
                target = log if serious else notes
                target.write(json.dumps(dict(index=index, seed=args.seed, findings=found, result=result, body=body)) + "\n")
                target.flush()
            if index % 200 == 199:
                (out / "summary.json").write_text(json.dumps(dict(counts, runner_starts=runner.starts), indent=1))
    runner.stop()
    (out / "slowest.json").write_text(json.dumps([case for _, case in slowest], indent=1) + "\n")
    summary = dict(counts, runner_starts=runner.starts)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"FUZZ {'FINDINGS' if counts['findings'] else 'CLEAN'} {counts['cases']} cases, {counts['accepted']} accepted, "
          f"{counts['refused']} refused, {counts['findings']} findings, {counts['notes']} notes "
          f"{json.dumps(counts['kinds'], sort_keys=True)}; "
          f"worst compile {counts['worst_compile_ms']} ms, open {counts['worst_open_ms']} ms, "
          f"token {counts['worst_token_ms']} ms")
    return 1 if counts["findings"] else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--runner", default=str(ROOT / "build/tests/constraint_fuzz_runner"))
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--count", type=int, default=5000)
    run.add_argument("--minutes", type=float, default=0)
    run.add_argument("--walk", type=int, default=64)
    run.add_argument("--seeds", default=os.environ.get("CONSTRAINT_FUZZ_SEEDS", ""))
    args = p.parse_args(argv)
    try:
        return run_main(args)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"FUZZ ERROR {type(error).__name__}: {str(error)[:200]}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
