#!/usr/bin/env python3
"""End-resume probe: does a request that resumes a whole prompt from its End point
give the same first token and the same top_logprobs as a cold prefill?

Each arm starts its own server (fresh cache) on its own port and sends, for each synthetic prompt, in this order:
  R1 chat, sampled, seed A, cache auto     (cold; saves the End point)
  R2 chat, sampled, seed A, cache auto     (a "regenerate"; must equal R1; it can resume a shorter session tail)
  R3 chat, sampled, seed B, cache auto     (a regenerate with a new seed)
  R4 completion of the prompt IDs, greedy, 64 tokens, auto (End resume; its decode leaves later logits on the slot)
  R5 scoring: completion, max_tokens 1, temperature 0, top_logprobs 5, auto  (End resume)
  R6 completion of the prompt IDs, sampled, seed B, 32 tokens, auto (End resume: the regenerate case)
  R7 chat, sampled, seed B, cache bypass   (cold; must equal R3)
  R8 scoring, bypass                       (cold; R5 must equal it)
  R9 completion, sampled, seed B, bypass   (cold; R6 must equal it)
The bypass requests go last: a bypass request can take the slot that holds the End point.

Lines (read by tools/end_logits_judge.py and by the night unit's pass rules). Token IDs, counts and log-probabilities
only; no text:
  ENDLOG arm=<a> prompt=<p> tokens=<n> regen_same_seed=<SAME|DIFF> (R2 = R1) regen_new_seed=<SAME|DIFF> (R7 = R3)
         sampled_end=<SAME|DIFF> (R6 = R9) sampled_first=<R6,R9> sampled_lens=<R6,R9> empty=<n>
         scoring=<SAME|DIFF> (R5 = R8) top1_same=<0|1> max_delta=<x> first=<R1,R2,R3,R7> lens=<R1,R2,R3,R7>
         cached=<R2,R3,R5,R6> end_resumes=<n>
Row keys of summary.json: r1, r2, r3, r7, auto (R5), cold (R8), s_auto (R6), s_cold (R9), cached.
  ENDLOG top5 arm=<a> prompt=<p> auto=<id:lp,...> bypass=<id:lp,...>
  ENDLOG DONE
  ENDLOG ERROR <text>

Usage:
  python3 tools/end_logits_probe.py run --arm daily=ENGINE --arm fixed=ENGINE --ckpt DIR --tree DIR --out DIR --port 8846
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# The cache budget must hold twice the slot checkpoint bytes, or the server turns the checkpoints (and End points) off
# (engine/serve/batched_serve.h BatchPrefixCache): 2,048 MiB did that on Nov 5. The daily layout uses 9,700.
SERVER_ARGS = ["--ctx", "65536", "--streams", "4", "--conversation-cache-mib", "6144",
               "--default-thinking", "on", "--default-max-tokens", "max", "--clamp-max-tokens",
               "--temperature", "1.0", "--top-k", "20", "--top-p", "0.95"]
# Synthetic prompts only: no user content.
PROMPTS = [
    [dict(role="system", content="You are a concise assistant."),
     dict(role="user", content="List three prime numbers between 10 and 30, then say which is largest.")],
    [dict(role="user", content="Write one sentence that describes a red bicycle in the rain.")],
]
END_RE = re.compile(r"slot_checkpoint end_resume tokens=(\d+) slot=\d+ request=\d+")
RUNNING = []


def say(text):
    print(text, flush=True)


def on_signal(signum, _frame):
    """Stop the servers before the lease's `timeout -k 10` sends SIGKILL, or one holds the GPU."""
    say(f"ENDLOG NOTE signal={signum}: stopping {len(RUNNING)} server(s)")
    for proc in list(RUNNING):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 6.0
    for proc in list(RUNNING):
        try:
            proc.wait(timeout=max(0.5, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    sys.exit(128 + signum)


def server_env(words):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CRUCIBLE_")}
    env.update(words)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def wait_health(port, timeout_s, proc):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return f"the server exited {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as answer:
                if answer.status == 200:
                    return None
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    return f"the server was not ready in {timeout_s} s"


def stop_server(proc, timeout_s=120):
    if proc.poll() is not None:
        return proc.returncode
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.returncode
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.wait(timeout=30)


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as answer:
        return json.load(answer)


def top5(response):
    content = (response["choices"][0].get("logprobs") or {}).get("content") or [{}]
    return [(int(e["token_id"]), float(e["logprob"])) for e in content[0].get("top_logprobs") or []]


def compare_scores(auto, cold):
    """(same, top1_same, max_delta): same is exact equality of the two lists (IDs and values)."""
    same = auto == cold and bool(auto)
    top1 = bool(auto) and bool(cold) and auto[0][0] == cold[0][0]
    a, b = dict(auto), dict(cold)
    shared = [t for t in a if t in b]
    delta = max((abs(a[t] - b[t]) for t in shared), default=None)
    return same, top1, delta


def chat(url, messages, seed, mode):
    r = post(url, "/v1/chat/completions", dict(messages=messages, max_tokens=160, seed=seed, cache_control=mode,
                                               return_token_ids=True))
    return r["choices"][0].get("token_ids") or [], r["usage"].get("prompt_tokens_details", {}).get("cached_tokens")


def one_prompt(url, index, messages):
    ids = post(url, "/tokenize", dict(messages=messages))["tokens"]
    seed_a, seed_b = 1100 + index, 2200 + index
    r1, _ = chat(url, messages, seed_a, "auto")
    r2, c2 = chat(url, messages, seed_a, "auto")
    r3, c3 = chat(url, messages, seed_b, "auto")
    post(url, "/v1/completions", dict(prompt=ids, temperature=0, max_tokens=64, cache_control="auto"))
    score = dict(prompt=ids, temperature=0, max_tokens=1, return_token_ids=True, top_logprobs=5)
    scored = post(url, "/v1/completions", score | dict(cache_control="auto"))
    auto, c5 = top5(scored), scored["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
    # A sampled completion of the prompt IDs (auto: End resume; then bypass): the regenerate case on the End point.
    sample = dict(prompt=ids, max_tokens=32, seed=seed_b, return_token_ids=True)
    resumed = post(url, "/v1/completions", sample | dict(cache_control="auto"))
    s_auto = resumed["choices"][0].get("token_ids") or []
    c8 = resumed["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
    r7, _ = chat(url, messages, seed_b, "bypass")
    cold = top5(post(url, "/v1/completions", score | dict(cache_control="bypass")))
    s_cold = post(url, "/v1/completions", sample | dict(cache_control="bypass"))["choices"][0].get("token_ids") or []
    return dict(prompt=len(ids), r1=r1, r2=r2, r3=r3, r7=r7, cached=[c2, c3, c5, c8], auto=auto, cold=cold,
                s_auto=s_auto, s_cold=s_cold)


def one_arm(args, name, binary, words, index):
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    serve_log = os.path.join(d, "serve.log")
    port = args.port + index
    argv = [args.python, "-m", "server", args.ckpt, "--engine", binary, "--host", "127.0.0.1", "--port", str(port),
            "--no-update-check", *SERVER_ARGS]
    say(f"ENDLOG NOTE arm={name} server={binary} port={port}")
    with open(serve_log, "wb") as log:
        proc = subprocess.Popen(argv, cwd=args.tree, env=server_env(words), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    RUNNING.append(proc)
    rows, error = [], None
    try:
        error = wait_health(port, args.ready_timeout, proc)
        if not error:
            for i, messages in enumerate(PROMPTS):
                rows.append(one_prompt(f"http://127.0.0.1:{port}", i, messages))
    except Exception as failure:  # the other arm's rows must stay
        error = f"{type(failure).__name__}: {failure}"
    finally:
        rc = stop_server(proc)
        RUNNING.remove(proc)
    with open(serve_log, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    resumed = [int(m.group(1)) for m in END_RE.finditer(text)]
    if "slot_checkpoint end_resume=0" in text:
        # No End points: the arm cannot test the claim (a setup fault, not data).
        say(f"ENDLOG ERROR arm={name}: the server runs without slot checkpoints (end_resume=0)")
    for i, row in enumerate(rows):
        # End resumes of this prompt: the serve log names the resumed length (the prompts differ in length).
        mine = [r for r in resumed if r == row["prompt"]]
        same, top1, delta = compare_scores(row["auto"], row["cold"])
        empty = sum(1 for key in ("r1", "r2", "r3", "r7") if not row[key])
        show = lambda pairs: ",".join(f"{t}:{v:.4f}" for t, v in pairs) or "-"
        say(f"ENDLOG arm={name} prompt={i} tokens={row['prompt']} "
            f"regen_same_seed={'SAME' if row['r1'] == row['r2'] else 'DIFF'} "
            f"regen_new_seed={'SAME' if row['r3'] == row['r7'] else 'DIFF'} "
            f"sampled_end={'SAME' if row['s_auto'] == row['s_cold'] and row['s_auto'] else 'DIFF'} "
            f"sampled_first={row['s_auto'][0] if row['s_auto'] else '-'},{row['s_cold'][0] if row['s_cold'] else '-'} "
            f"sampled_lens={len(row['s_auto'])},{len(row['s_cold'])} empty={empty} "
            f"scoring={'SAME' if same else 'DIFF'} top1_same={int(top1)} "
            f"max_delta={'none' if delta is None else f'{delta:.4f}'} "
            f"first={','.join(str(row[k][0]) if row[k] else '-' for k in ('r1', 'r2', 'r3', 'r7'))} "
            f"lens={','.join(str(len(row[k])) for k in ('r1', 'r2', 'r3', 'r7'))} "
            f"cached={','.join(str(c) for c in row['cached'])} end_resumes={len(mine)}")
        say(f"ENDLOG top5 arm={name} prompt={i} auto={show(row['auto'])} bypass={show(row['cold'])}")
    say(f"ENDLOG NOTE arm={name} server_rc={rc} rows={len(rows)} error={error}")
    return dict(arm=name, rows=rows, error=error, resumed=resumed, server_rc=rc)


def run(args):
    words = dict(w.partition("=")[::2] for w in args.server_env.split() if w.strip())
    os.makedirs(args.out, exist_ok=True)
    arms = []
    for spec in args.arm:
        name, _, binary = spec.partition("=")
        if not name or not os.path.isfile(binary):
            say(f"ENDLOG ERROR arm {spec!r}: give name=path to an engine binary")
            return 2
        arms.append((name, binary))
    summary = dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), server_args=SERVER_ARGS, words=words, arms=[])
    for index, (name, binary) in enumerate(arms):
        summary["arms"].append(one_arm(args, name, binary, words, index))
        with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
    say("ENDLOG DONE")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="start one server per arm and send the request list")
    go.add_argument("--arm", action="append", required=True, help="name=engine binary (repeat)")
    go.add_argument("--ckpt", required=True)
    go.add_argument("--tree", required=True, help="the server tree (cwd of python -m server)")
    go.add_argument("--out", required=True)
    go.add_argument("--port", type=int, default=8846, help="first port (each arm adds 1)")
    go.add_argument("--python", default=sys.executable)
    go.add_argument("--server-env", default="", help="CRUCIBLE_* words for every arm")
    go.add_argument("--ready-timeout", type=int, default=420)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
