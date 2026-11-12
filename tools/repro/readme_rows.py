#!/usr/bin/env python3
"""The speed rows of the README tables, from the public tree only: one command per row.

  readme_rows.py serve-cmd --ckpt DIR --pack {4.05,3.05} [--engine build/engine/crucible] [--port N]
  readme_rows.py rows --ckpt DIR --pack {4.05,3.05} --out DIR [--engine BIN] [--row ROW ...] [--runs 2]
  readme_rows.py rows --url http://127.0.0.1:PORT/v1 --kind {crucible,halogen,gufo} --out DIR [--row ROW ...] [--runs 2]
  readme_rows.py table DIR/result.json [...]

`rows` with --ckpt starts `python -m server` in the settings of the README (4 sessions of 262,144 tokens, image input and
MTP on, CRUCIBLE_PF_LAUNCH_ROWS=8192+128 on the 4.05 pack), runs the rows, and stops the server it started. With --url it
measures a server that already runs (Halogen, Gufo or Crucible): start that server as `docs/RIVALS.md` describes.
Each run builds the prompts once (tools/h2h_decode.py build: Python standard-library files as tool results, no private
data), gives every long prompt its own first line, and measures the rows in this order. The value of a row is the median
of the runs (--runs, the README uses 2).

Rows (the README row each one gives):
  cold32_ttft     Cold prompt, 32K: first token (s)                       median of 5 cold prompts
  cold32_prefill  Cold prompt, 32K: prefill (tok/s)                       prompt tokens / first token time
  cold130_ttft    Cold prompt, 130K: first token (s)
  cold130_prefill Cold prompt, 130K: prefill (tok/s)
  resend32        Same 32K prompt sent again: first token (s)             median of 3
  dec32_code      Decode at 32K, 1 session, code (tok/s)                  (tokens - 1) / (last chunk - first chunk)
  dec32_prose     Decode at 32K, 1 session, prose (tok/s)
  dec4            Decode, 4 sessions, total (tok/s)                       tokens in the window in which all four decode
  d3              Serial decode, no speculation, short context (tok/s)    257 tokens, median of 2 (--kind picks the switch)
  d1_mixed        Decode, mixed text, 1 session, speculation (tok/s)      Halogen's ten prompt shapes, 3 reps, 256 tokens

A cold prompt restores at most 5% of its tokens from the cache, or it is not counted. A row with no valid sample has no value.
Check the GPU clock and the power profile yourself: the README rows use the balanced profile (--profile balanced stops the
run when /sys/firmware/acpi/platform_profile says another).
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import h2h_decode as h2h  # noqa: E402

CTX_SHORT, CTX_LONG = "32768", "130000"      # keys of the prompt file (h2h_decode.py build --contexts)
D3_PROMPT = ("Write a Python function fibonacci(n) that returns the first n Fibonacci numbers. Reject negative "
             "input with ValueError. Include type hints, a short docstring, and five unit tests. Return only code.")
D3_N, D3_REPS = 257, 2
COLD_MAX_FRAC = 0.05
RESEND_REPS = 3
HIT_MIN_FRAC = 0.9
SESSIONS = 4
MODEL = "crucible"
PROFILE_FILE = Path("/sys/firmware/acpi/platform_profile")

# Halogen's ten prompt shapes: tools/eval-prompts.json of https://github.com/peonist-ai/halogen-flash-server, commit e88be02e
# (release 0.18.14). D1 asks each shape 3 times over the chat endpoint (thinking off, temperature 0, 256 tokens).
D1_SHAPES = {
    "code_a": "Write a Python function that merges two sorted lists into one sorted list. Include a docstring.",
    "code_b": "Write a Python function parse_duration(s) that turns strings like 1h30m into seconds. Include a "
              "docstring and type hints.",
    "proof_a": "Prove that the square root of 2 is irrational.",
    "proof_b": "Prove that there are infinitely many prime numbers.",
    "proc_a": "List the steps to set up SSH key authentication on a new Linux server.",
    "proc_b": "Give me a numbered procedure for safely replacing a failed drive in a RAID 5 array.",
    "chat_a": "hey, how is your day going? I am procrastinating on laundry",
    "chat_b": "what do you think about people who put pineapple on pizza?",
    "prose_a": "Explain in a couple of paragraphs why unified memory changes the performance tradeoffs for LLM inference "
               "on an APU.",
    "prose_b": "Describe, in prose, what makes speculative decoding effective and when it stops helping.",
}
D1_REPS, D1_TOKENS = 3, 256

# row name -> (key of the values dict, unit, README row)
ROWS = {
    "cold32_ttft": ("cold32_ttft_s", "s", "Cold prompt, 32K: first token (s)"),
    "cold32_prefill": ("cold32_tok_s", "tok/s", "Cold prompt, 32K: prefill (tok/s)"),
    "cold130_ttft": ("cold130_ttft_s", "s", "Cold prompt, 130K: first token (s)"),
    "cold130_prefill": ("cold130_tok_s", "tok/s", "Cold prompt, 130K: prefill (tok/s)"),
    "resend32": ("resend32_ttft_s", "s", "Same 32K prompt sent again: first token (s)"),
    "dec32_code": ("dec32_code_tok_s", "tok/s", "Decode at 32K, 1 session, code (tok/s)"),
    "dec32_prose": ("dec32_prose_tok_s", "tok/s", "Decode at 32K, 1 session, prose (tok/s)"),
    "dec4": ("dec4_total_tok_s", "tok/s", "Decode, 4 sessions, total (tok/s)"),
    "d3": ("d3_tok_s", "tok/s", "Serial decode, no speculation, short context (tok/s)"),
    "d1_mixed": ("d1_mixed_tok_s", "tok/s", "Decode, mixed text, 1 session, speculation (tok/s)"),
}
# which measuring steps a row needs
NEEDS = {
    "cold32_ttft": {"s0cold", "four"}, "cold32_prefill": {"s0cold", "four"}, "cold130_ttft": {"cold130"},
    "cold130_prefill": {"cold130"}, "resend32": {"s0cold", "resend"}, "dec32_code": {"s0cold", "code"},
    "dec32_prose": {"s0cold", "prose"}, "dec4": {"four"}, "d3": {"d3"}, "d1_mixed": {"d1"},
}

SERVER_LAYOUT = ["--streams", "4", "--ctx", "262144", "--prefill", "4096", "--conversation-cache-mib", "9700",
                 "--vision", "--vision-max-images", "256"]
SERVER_FIXED = ["--model-id", MODEL, "--max-body-mib", "256", "--session-replay", "1", "--slot-resume", "1",
                "--slot-checkpoints", "1"]
SERVER_ENV = {"4.05": {"CRUCIBLE_PLE_HOST": "1", "CRUCIBLE_COMPACT_STATE": "1", "CRUCIBLE_PF_LAUNCH_ROWS": "8192+128"},
              "3.05": {"CRUCIBLE_PLE_HOST": "1", "CRUCIBLE_COMPACT_STATE": "1"}}   # the 3.05 pack keeps its default launch rows


def server_cmd(ckpt, pack, engine, port=8790):
    """(argv, env words) of the server in the README settings. No sidecar, no tuned draft head file: public files only."""
    argv = [sys.executable, "-m", "server", str(ckpt), "--engine", str(engine), "--port", str(port), *SERVER_LAYOUT, *SERVER_FIXED]
    return argv, dict(SERVER_ENV[pack])


# ---- requests ------------------------------------------------------------------------------------------------------

def timed(url, body):
    res = h2h.stream(url, body)
    res["t_end"] = time.time()
    return res


def first_token_s(res):
    if res["error"]:
        return None
    return res["chunks"][0][0] - res["t0"] if res["chunks"] else res["t_end"] - res["t0"]


def cached_tokens(res):
    return ((res.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def serial_fields(kind):
    """The per-request switch of serial decode. Gufo has none: its server flags decide (start it without speculation)."""
    return {"crucible": dict(spec=0), "halogen": dict(drafter="serial"), "gufo": {}}[kind]


def chat_body(kind, text, max_tokens, serial=False):
    body = dict(model=MODEL, messages=[dict(role="user", content=text)], max_tokens=max_tokens, temperature=0,
                stream=True, stream_options=dict(include_usage=True))
    if kind == "gufo":
        body["chat_template_kwargs"] = dict(enable_thinking=False)
    else:
        body["enable_thinking"] = False
    if serial:
        body.update(serial_fields(kind))
    return body


def row_d3(url, kind):
    timed(url, chat_body(kind, D3_PROMPT, 16, True))
    runs = [timed(url, chat_body(kind, D3_PROMPT, D3_N, True)) for _ in range(D3_REPS)]
    rates = [h2h.per_request(r) if not r["error"] else None for r in runs]
    return dict(value=med(rates), rates=rates, errors=[r["error"] for r in runs if r["error"]])


def row_d1(url, kind):
    """Mean of completion tokens / wall seconds over the ten shapes, 3 reps each (the way Halogen's own bench computes it)."""
    rates, errors = [], []
    for _ in range(D1_REPS):
        for text in D1_SHAPES.values():
            res = timed(url, chat_body(kind, text, D1_TOKENS))
            tokens, wall = (res["usage"] or {}).get("completion_tokens"), res["t_end"] - res["t0"]
            if res["error"] or not tokens:
                errors.append(res["error"] or "no tokens")
                continue
            rates.append(tokens / wall)
    return dict(value=sum(rates) / len(rates) if rates and not errors else None, requests=len(rates), errors=errors[:3])


def cold_sample(url, session, name):
    res = timed(url, session.body("code", 1))
    n, c, ttft = res["usage"].get("prompt_tokens"), cached_tokens(res), first_token_s(res)
    valid = bool(n and ttft) and (c or 0) <= COLD_MAX_FRAC * n
    return dict(name=name, valid=valid, prompt_tokens=n, cached_tokens=c, ttft_s=ttft,
                tok_s=n / ttft if valid else None, error=res["error"])


def row_resend(url, session):
    reps = []
    for _ in range(RESEND_REPS):
        res = timed(url, session.body("code", 1))
        n, c = res["usage"].get("prompt_tokens"), cached_tokens(res)
        hit = not res["error"] and (c is None or (bool(n) and c >= HIT_MIN_FRAC * n))
        reps.append(dict(valid=hit, prompt_tokens=n, cached_tokens=c, ttft_s=first_token_s(res), error=res["error"]))
    return dict(value=med([r["ttft_s"] for r in reps if r["valid"]]), reps=reps)


def turn_rate(url, session, workload):
    res = h2h.turn(url, [session], workload, 0)[0]
    return dict(value=h2h.per_request(res), completion_tokens=res["usage"].get("completion_tokens"), finish=res["finish"],
                error=res["error"])


def four_sessions(url, prompts, salt):
    base = prompts[CTX_SHORT]
    bases = h2h.session_bases(base, SESSIONS, "distinct", salt)
    sessions = [h2h.Session(bases[k], base["targets"][(1 + k) % len(base["targets"])], MODEL, {}) for k in range(SESSIONS)]
    colds = [cold_sample(url, s, f"s{k + 1}") for k, s in enumerate(sessions)]
    results = h2h.turn(url, sessions, "code", 0)
    total, window = h2h.window_total(results)
    return colds, dict(value=total, window_s=window, rates=[h2h.per_request(r) for r in results],
                       errors=[r["error"] for r in results if r["error"]])


def run_rows(url, prompts, salt, rows, kind="crucible"):
    """All wanted rows of one run against the server at `url`. Returns values (key of ROWS -> number or None), raw and errors."""
    need = set().union(*(NEEDS[r] for r in rows))
    errors, raw = [], {}

    def guarded(name, fn):
        try:
            return fn()
        except Exception as error:     # recorded, never swallowed: a run with errors gives no value for the row
            errors.append(f"{name}: {type(error).__name__}: {str(error)[:200]}")
            return None

    base = prompts[CTX_SHORT]
    s0 = h2h.Session(h2h.with_first_line(base, f"{salt}-solo"), base["targets"][0], MODEL, {})
    if "d3" in need:
        raw["d3"] = guarded("d3", lambda: row_d3(url, kind))
    if "d1" in need:
        raw["d1"] = guarded("d1_mixed", lambda: row_d1(url, kind))
    cold0 = guarded("cold32", lambda: cold_sample(url, s0, "s0")) if "s0cold" in need else None
    if "resend" in need:
        raw["resend"] = guarded("resend32", lambda: row_resend(url, s0))
    if "code" in need:
        raw["code"] = guarded("dec32_code", lambda: turn_rate(url, s0, "code"))
    if "prose" in need:
        raw["prose"] = guarded("dec32_prose", lambda: turn_rate(url, s0, "prose"))
    colds, raw["dec4"] = [], None
    if "four" in need:
        four = guarded("dec4", lambda: four_sessions(url, prompts, f"{salt}-four"))
        colds, raw["dec4"] = four if four else ([], None)
    if "cold130" in need:
        big = prompts[CTX_LONG]
        s130 = h2h.Session(h2h.with_first_line(big, f"{salt}-long"), big["targets"][0], MODEL, {})
        raw["cold130"] = guarded("cold130", lambda: cold_sample(url, s130, "s130"))
    cold32 = [c for c in [cold0, *colds] if c]
    ok32 = [c for c in cold32 if c["valid"]]
    c130 = raw.get("cold130") if raw.get("cold130") and raw["cold130"]["valid"] else None
    for c in cold32:
        if c["error"]:
            errors.append(f"cold32 {c['name']}: {c['error']}")
    for name, row in raw.items():
        if row and row.get("errors"):
            errors.append(f"{name}: {row['errors'][:2]}")
        if row and row.get("error"):
            errors.append(f"{name}: {row['error']}")
    got = lambda key: raw.get(key) and raw[key]["value"]    # noqa: E731
    values = {
        "cold32_ttft": med([c["ttft_s"] for c in ok32]), "cold32_prefill": med([c["tok_s"] for c in ok32]),
        "cold130_ttft": c130 and c130["ttft_s"], "cold130_prefill": c130 and c130["tok_s"],
        "resend32": got("resend"), "dec32_code": got("code"), "dec32_prose": got("prose"), "dec4": got("dec4"),
        "d3": got("d3"), "d1_mixed": got("d1")}
    values = {r: values[r] for r in rows}
    return dict(values=values, raw=raw, errors=errors, cold32_valid=len(ok32))


# ---- server, runs, table ------------------------------------------------------------------------------------------

def wait_ready(base, process, timeout_s=900):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"the server stopped with exit code {process.returncode}")
        try:
            with urllib.request.urlopen(base + "/models", timeout=5):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise TimeoutError("the server did not become ready")


class StartedServer:
    """`python -m server` in the README settings. Stops only the process group that it started."""

    def __init__(self, ckpt, pack, engine, log, port):
        self.argv, words = server_cmd(ckpt, pack, engine, port)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", **words)
        for gone in ("CRUCIBLE_MTP_SIDECAR", "CRUCIBLE_PLE_BF16", "CRUCIBLE_HC_WMMA"):
            self.env.pop(gone, None)          # public files only: no sidecar and no other file switch from the caller's shell
        self.log, self.port, self.process = log, port, None
        self.base = f"http://127.0.0.1:{port}/v1"

    def __enter__(self):
        self.logfile = open(self.log, "w")
        self.process = subprocess.Popen(self.argv, cwd=ROOT, env=self.env, stdout=self.logfile, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        wait_ready(self.base, self.process)
        return self

    def __exit__(self, *exc):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.logfile.close()


def profile_now():
    try:
        return PROFILE_FILE.read_text().strip()
    except OSError:
        return None


def median_over_runs(runs, rows):
    return {r: med([run["values"].get(r) for run in runs]) for r in rows}


def render(result):
    lines = ["| README row | Value | Unit | Runs with a value |", "|---|---:|---|---:|"]
    for row, value in result["median"].items():
        _, unit, label = ROWS[row]
        have = sum(1 for run in result["runs"] if run["values"].get(row) is not None)
        text = "no value" if value is None else (f"{value:.3f}" if unit == "s" and value < 1 else f"{value:,.1f}")
        lines.append(f"| {label} | {text} | {unit} | {have} of {len(result['runs'])} |")
    return "\n".join(lines)


def build_prompts(path):
    ns = argparse.Namespace(out=str(path), contexts=f"{CTX_SHORT},{CTX_LONG}", sessions=SESSIONS)
    h2h.build(ns)
    return json.loads(Path(path).read_text())


def do_rows(args):
    rows = args.row or list(ROWS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = profile_now()
    if args.profile and profile != args.profile:
        raise SystemExit(f"the platform profile is {profile!r}, not {args.profile!r}: stop")
    prompts = build_prompts(out / "prompts.json")
    runs = []
    for index in range(args.runs):
        salt = f"{int(time.time())}-run{index}"
        if args.url:
            runs.append(run_rows(args.url.rstrip("/"), prompts, salt, rows, args.kind))
        else:
            with StartedServer(args.ckpt, args.pack, args.engine, out / f"serve{index + 1}.log", args.port) as server:
                for warm in ("Say hello.",):      # first-use costs stay out of the rows
                    timed(server.base, chat_body("crucible", warm, 16))
                runs.append(run_rows(server.base, prompts, salt, rows, "crucible"))
        (out / "result.json").write_text(json.dumps(dict(rows=rows, runs=runs, median=median_over_runs(runs, rows),
                                                         kind=args.kind, pack=args.pack, platform_profile=profile), indent=1))
    result = json.loads((out / "result.json").read_text())
    print(render(result))
    bad = [e for run in runs for e in run["errors"]]
    if bad:
        print("\nErrors:\n" + "\n".join(bad[:10]), file=sys.stderr)
    return 1 if any(v is None for v in result["median"].values()) else 0


def do_serve_cmd(args):
    argv, words = server_cmd(args.ckpt, args.pack, args.engine, args.port)
    print(" ".join([*(f"{k}={shlex.quote(v)}" for k, v in sorted(words.items())), *map(shlex.quote, argv)]))
    return 0


def do_table(args):
    for path in args.files:
        print(render(json.loads(Path(path).read_text())))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve-cmd", help="print the server command of the README settings")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--pack", choices=sorted(SERVER_ENV), required=True)
    s.add_argument("--engine", default="build/engine/crucible")
    s.add_argument("--port", type=int, default=8790)
    r = sub.add_parser("rows", help="measure the rows")
    r.add_argument("--ckpt", help="pack folder: start the server here (the engine of the README settings)")
    r.add_argument("--pack", choices=sorted(SERVER_ENV), default="4.05")
    r.add_argument("--engine", default="build/engine/crucible")
    r.add_argument("--port", type=int, default=8790)
    r.add_argument("--url", help="measure a server that already runs (base URL with /v1)")
    r.add_argument("--kind", choices=("crucible", "halogen", "gufo"), default="crucible")
    r.add_argument("--out", required=True)
    r.add_argument("--row", action="append", choices=sorted(ROWS), help="a row (repeat for several); default all")
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--profile", choices=("balanced", "performance"), help="stop unless the platform profile is this")
    t = sub.add_parser("table", help="print the table of result.json files")
    t.add_argument("files", nargs="+")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "rows" and bool(args.url) == bool(args.ckpt):
        ap.error("rows needs exactly one of --ckpt (start the server) and --url (use a running server)")
    if args.cmd == "rows" and args.url and args.kind == "crucible" and args.pack != "4.05":
        pass
    return {"serve-cmd": do_serve_cmd, "rows": do_rows, "table": do_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
