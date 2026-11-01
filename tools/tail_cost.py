#!/usr/bin/env python3
"""Idle cold TTFT of prompts of 4,096 k + t rows (F3): what the small-row tail costs against the merged tail.

  tail_cost.py run BIN PACK OUT.json [--rounds 3]     one server, cold (bypass) spec 0 prompts, lengths interleaved
  tail_cost.py table A.json B.json                    medians of two runs (A = trunk, B = candidate)

Run it inside one gpu-run lease (refenv python: it needs `tokenizers`). The first request of a run is a warm-up.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastgate  # noqa: E402
import fastgate_lib as lib  # noqa: E402

# 4,096 k + t: t in 1..8 changes with F3 (4,100 and 12,292 are the odd-k cases of the load group); 4,105 and 12,297 (t = 9) and 8,192 + 4 are controls.
LENGTHS = [4100, 4104, 12292, 12296, 8196, 4105, 12297]
AFFECTED = lambda n: 1 <= n % 4096 <= 8 and n > 4096  # noqa: E731  8196 differs on trunk 3.05 too (load group)
GEN_TOKENS = 4


def prompt_ids(encode, n):
    return encode(lib.synth_code(10 * n, 31))[:n]


def run(args):
    encode = fastgate.load_encoder(args.pack)
    prompts = {n: prompt_ids(encode, n) for n in LENGTHS}
    out = dict(binary=str(args.bin), pack=args.pack, runs={str(n): [] for n in LENGTHS})
    workdir = Path(args.out).resolve().parent
    with lib.Server(args.bin, args.pack, workdir / f"tail-cost-serve-{Path(args.bin).name}.log") as server:
        def send(n):
            body = dict(prompt=prompts[n], max_tokens=GEN_TOKENS, spec=0, temperature=0, cache_control="bypass")
            return lib.stream_request(server.base, "/completions", body)
        send(LENGTHS[0])  # warm-up: first-use costs
        for _ in range(args.rounds):
            for n in LENGTHS:
                r = send(n)
                out["runs"][str(n)].append(round(r["times"][0], 4) if r["times"] else None)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(table([out]))


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def table(runs):
    lines = ["N      t  affected  " + "  ".join(f"{Path(r['binary']).name[-14:]:>14}" for r in runs)]
    for n in LENGTHS:
        cells = [median(r["runs"][str(n)]) for r in runs]
        lines.append(f"{n:<6} {n % 4096:<3}{'yes' if AFFECTED(n) else 'no':>8}  " + "  ".join(f"{c:>14.3f}" if c else f"{'-':>14}" for c in cells)
                     + ("" if len(cells) < 2 or not all(cells) else f"  delta {cells[1] - cells[0]:+.3f} s ({cells[1] / cells[0] - 1:+.1%})"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("bin"), r.add_argument("pack"), r.add_argument("out")
    r.add_argument("--rounds", type=int, default=3)
    t = sub.add_parser("table")
    t.add_argument("files", nargs=2)
    args = ap.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        print(table([json.loads(Path(f).read_text()) for f in args.files]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
