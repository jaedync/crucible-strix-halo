#!/usr/bin/env python3
"""P1c cost of CRUCIBLE_EXACT_RESUME: 2K new rows after a cached 32K, and a chat-turn-like
resume from an off-grid point, with the switch off (arm "off") and on (arm "on").

  resume_p1c.py run --base URL --prompts F --out F.jsonl --arm off|on --pack P [--reps 3]   inside a lease, against a server
  resume_p1c.py report DIR_OR_JSONL...                                                      paired table (ABAB leases)

Per case and repetition, two requests (max_tokens 1, temperature 0, cache_control auto) on one server:
  seed  P[:s]   fills the cache (and, with the switch on, the Pre point); its time is the cost of the save
  ext   P       resumes; its time is the cost of the resume (and `cached` says where it resumed)
Every repetition has its own first token, so no repetition can hit another one's state.
Cases: grid = (N 34,816, s 32,768): the End point is on the 2,048 grid, both arms resume at 32,768.
       turn = (N 35,816, s 33,768): off-grid End point; "off" resumes at 33,768 (with a small-row head), "on" at 33,768
              too (a multiple of 4 with 2,048 new rows, gdn-split-inv).
       off4 = (N 35,817, s 33,769): s is not a multiple of 4; "off" resumes at 33,769 (small-row head), "on" at the
              seed's Pre point 33,760 (b-fallback).
       short = (N 33,774, s 33,768): 6 new rows; "off" resumes at 33,768 (small-row head), "on" at the Pre point 33,756.
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_extend as ee  # noqa: E402

CASES = {"grid": (34816, 32768), "turn": (35816, 33768), "off4": (35817, 33769), "short": (33774, 33768)}
# Save points, run only when named in --cases: (N, s, d). The seed has s rows; the extension
# leaves it at row d (23 rows before its end: where a chat template closes the turn) and has N rows. p2c is the
# P2c shape: 2K new rows after a cached 130K. The end-64 Pre point is the only point at or below d, so the resume is
# there (new engine) or at the grid point 979 rows earlier (trunk). The text needs 133K tokens (prompts --chars 520000).
DIVERGE = {"p2c": (132050, 130003, 129980)}
MARKER_OFFSET = 8  # markers[8:] are free: any server starts empty, so a marker only has to differ inside one server


def body(prompt):
    return dict(prompt=prompt, max_tokens=1, temperature=0, spec=0, return_token_ids=True, cache_control="auto")


def run(base, prompts, out, arm, pack, reps, send=ee.post, cases=None):
    every = list(CASES) + list(DIVERGE)
    cases = {k: (CASES | DIVERGE)[k] for k in (cases or CASES)}
    markers, text = prompts["markers"], prompts["base"]
    with open(out, "a") as f:
        f.write(json.dumps(dict(meta=dict(arm=arm, pack=pack, reps=reps))) + "\n")
        send(base, dict(prompt=text[:512], max_tokens=4, temperature=0, cache_control="bypass"))  # warm-up
        for rep in range(reps):
            for name, (n, s, *diverge) in cases.items():
                prompt = ee.make_prompt(text, markers, MARKER_OFFSET + rep * len(every) + every.index(name), n)
                if diverge:  # the extension leaves the seed at row d and continues with other text
                    prompt = prompt[:diverge[0]] + text[len(text) - (n - diverge[0]):]
                    seed_prompt = ee.make_prompt(text, markers, MARKER_OFFSET + rep * len(every) + every.index(name), s)
                else:
                    seed_prompt = prompt[:s]
                seed = send(base, body(seed_prompt))
                ext = send(base, body(prompt)) if seed["status"] == 200 else dict(status=0)
                f.write(json.dumps(dict(case=name, n=n, s=s, rep=rep, seed=seed, ext=ext)) + "\n")
                f.flush()


def load(paths):
    """[(meta, [rows])] per file."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("**/*.jsonl")) if p.is_dir() else [p]
    runs = []
    for file in files:
        meta, rows = {}, []
        for line in file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "meta" in row:
                    meta = row["meta"] | dict(file=file.name)
                else:
                    rows.append(row)
        runs.append((meta, rows))
    return runs


def table(runs):
    """{(pack, case, arm): {"seed": [s], "ext": [s], "cached": {c}}} over all runs."""
    cells = {}
    for meta, rows in runs:
        for row in rows:
            if row["seed"].get("status") != 200 or row["ext"].get("status") != 200:
                continue
            cell = cells.setdefault((meta["pack"], row["case"], meta["arm"]), dict(seed=[], ext=[], cached=set()))
            cell["seed"].append(row["seed"]["seconds"])
            cell["ext"].append(row["ext"]["seconds"])
            cell["cached"].add(row["ext"].get("cached"))
    return cells


def report(runs):
    cells, lines = table(runs), [f"{'pack':5} {'case':5} {'arm':4} {'n':>3} {'resume at':>10} {'seed s':>8} {'ext s':>8} {'ext min':>8} {'ext max':>8}"]
    for key in sorted(cells):
        c = cells[key]
        lines.append(f"{key[0]:5} {key[1]:5} {key[2]:4} {len(c['ext']):3} {','.join(str(x) for x in sorted(c['cached'], key=str)):>10} "
                     f"{statistics.median(c['seed']):8.3f} {statistics.median(c['ext']):8.3f} {min(c['ext']):8.3f} {max(c['ext']):8.3f}")
    for pack, case in sorted({(k[0], k[1]) for k in cells}):
        off, on = cells.get((pack, case, "off")), cells.get((pack, case, "on"))
        if off and on:
            lines.append(f"{pack} {case}: on - off  ext {statistics.median(on['ext']) - statistics.median(off['ext']):+.3f} s  "
                         f"seed {statistics.median(on['seed']) - statistics.median(off['seed']):+.3f} s  (medians)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--arm", choices=("off", "on"), required=True)
    r.add_argument("--pack", required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--cases", default="", help="comma list of case names (default: all)")
    q = sub.add_parser("report")
    q.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.base, json.load(open(args.prompts)), args.out, args.arm, args.pack, args.reps,
            cases=[c for c in args.cases.split(",") if c] or None)
    else:
        print(report(load(args.paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
