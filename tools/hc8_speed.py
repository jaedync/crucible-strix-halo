#!/usr/bin/env python3
"""HC int8 speed A/B in ONE gpu-run lease (the test machine): one binary, CRUCIBLE_HC_WMMA unset (A) and 2 (B), interleaved.

  prep:  hc8_speed.py prep OUT [--pack 4.05]                   prompts (CPU; run before the lease)
  run:   gpu-run -p 1 -- timeout -k 10 900 PY hc8_speed.py run OUT BIN [--pairs 5] [--pack 4.05] [--bench CMD]

Each arm is one `crucible gen --jobs` load (measure_lib.run_gen, daily env): decode0 (D3: code prompt, spec 0,
257 tokens) and decode1 (spec 7, 257 tokens), plus prefill32k when --p32k is set. Pairs run A B, B A, A B ...
Every arm records its sclk median; the report drops pairs below 2200 MHz. Token IDs of each job go to OUT so the
tokens of A can be compared with trunk and B with itself. --bench runs a command first (the kernel proof).
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastgate_lib as lib  # noqa: E402
import measure_lib as ml  # noqa: E402

ARMS = {"A": {}, "B": {"CRUCIBLE_HC_WMMA": "2"}}


def prep(out, pack):
    ml.write_prompts(ml.build_prompts(ml.load_encoder(pack)), out / "prompts")


def arm(out, binary, pack, name, i, p32k):
    work = out / f"{i:02d}-{name}"
    work.mkdir(parents=True, exist_ok=True)
    for p in (out / "prompts").glob("*.ids"):
        (work / p.name).write_text(p.read_text())
    jobs = [dict(name="decode0", prompt="code_short", n=ml.DECODE_N, spec=0, reps=1),
            dict(name="decode1", prompt="code_short", n=ml.DECODE_N, spec=7, reps=1)]
    if p32k:
        jobs.append(dict(name="prefill32k", prompt="p32k", n=16, spec=0, reps=1))
    got = ml.run_gen(dict(binary=str(binary), pack=pack, env=ARMS[name], workdir=str(work), ctx=lib.GEN_CTX,
                          jobs=jobs))
    for p in work.glob("*.ids"):
        p.unlink()
    return dict(arm=name, i=i, **got)


def run(out, binary, pairs, pack, bench, p32k):
    if bench:
        t0 = time.time()
        proc = subprocess.run(bench, shell=True, capture_output=True, text=True)
        (out / "bench.log").write_text(proc.stdout + proc.stderr)
        print(f"bench rc={proc.returncode} {time.time() - t0:.0f} s", flush=True)
    order = []
    for p in range(pairs):
        order += ["A", "B"] if p % 2 == 0 else ["B", "A"]
    runs = []
    for i, name in enumerate(order):
        r = arm(out, binary, pack, name, i, p32k)
        runs.append(r)
        (out / "runs.json").write_text(json.dumps(runs, indent=1))
        res = {k: v[0].get("decode_tok_s") or v[0].get("prefill_tok_s") for k, v in r["results"].items()}
        print(f"{i:02d} {name} rc={r['rc']} {r['seconds']} s sclk {r['clock'].get('median_mhz')} {res}", flush=True)
    return runs


def report(out):
    runs = json.loads((out / "runs.json").read_text())
    lines = []
    for metric, key in (("decode0", "decode_tok_s"), ("decode1", "decode_tok_s"), ("prefill32k", "prefill_tok_s")):
        pairs, a_all, b_all = [], [], []
        for j in range(0, len(runs) - 1, 2):
            x, y = runs[j], runs[j + 1]
            ab = {x["arm"]: x, y["arm"]: y}
            if set(ab) != {"A", "B"}:
                continue
            clk = [ab[k]["clock"].get("median_mhz") or 0 for k in "AB"]
            va = (ab["A"]["results"].get(metric) or [{}])[0].get(key)
            vb = (ab["B"]["results"].get(metric) or [{}])[0].get(key)
            if va is None or vb is None:
                continue
            ok = min(clk) >= 2200
            lines.append(f"{metric} pair {j // 2}: A {va:.2f} B {vb:.2f} B/A {vb / va:.4f} sclk {clk} "
                         f"{'' if ok else 'DROPPED (sclk)'}")
            if ok:
                pairs.append(vb / va)
                a_all.append(va)
                b_all.append(vb)
        if pairs:
            lines.append(f"{metric}: n={len(pairs)} A median {statistics.median(a_all):.2f} B median "
                         f"{statistics.median(b_all):.2f} B/A median {statistics.median(pairs):.4f} "
                         f"(min {min(pairs):.4f} max {max(pairs):.4f})")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "report"])
    ap.add_argument("out")
    ap.add_argument("binary", nargs="?")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--pack", default="4.05")
    ap.add_argument("--bench")
    ap.add_argument("--p32k", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prep":
        prep(out, a.pack)
    elif a.cmd == "run":
        run(out, Path(a.binary), a.pairs, a.pack, a.bench, a.p32k)
        report(out)
    else:
        report(out)


if __name__ == "__main__":
    main()
