#!/usr/bin/env python3
"""Q1 table: KL(BF16 || engine) and top-1 agreement of several engine row files on their COMMON positions.

The scoring code is tools/bf16/score_kl.py of the private archive (row_kl, load_top20, score): the 21-bucket KL
over the engine's top 20 plus a rest bucket, a lower bound of the full-vocabulary KL. score_kl.py itself needs all
7,168 positions; an evaluation lease scores a subset (score.py --passes N, one pass = an even stride over every
chunk), so this wrapper keeps only the positions that every engine file has, and scores all engines on that set.

  kl_score.py --kl-tree DIR --engine LABEL=rows.jsonl [...] [--out-json F] [--out-md F] [--ci RESAMPLES [--seed N]]
--ci adds a 95% interval from a bootstrap over text chunks (packs/kl_boot.py), for each engine and for each pairwise difference.
DIR holds ref/bf16 and tools/bf16; the default is this tree, which tracks both.
Privacy: numbers only, no token text.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

DEFAULT_KL_TREE = Path(__file__).resolve().parents[2]   # tools/bf16/score_kl.py and ref/bf16/tokens.py are tracked in this tree

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-tree", default=str(DEFAULT_KL_TREE), help="folder with ref/bf16 and tools/bf16 (default: this tree)")
    ap.add_argument("--ref", required=True, help="folder with logprobs.f32.npy, reference.npz and meta.json (the BF16 reference)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the model (it is in every pack folder)")
    ap.add_argument("--engine", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--ci", type=int, default=0, metavar="RESAMPLES")
    ap.add_argument("--seed", type=int, default=20261103)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.kl_tree) / "tools/bf16"))
    sys.path.insert(0, str(Path(args.kl_tree) / "ref"))
    import score_kl as sk
    from bf16 import tokens
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(args.tokenizer)
    sk.TOKENIZER[0], sk.TOKENIZER[1] = tok, tokens.byte_table()
    hexmap, clash = tokens.hex_to_id(tok)
    if clash:
        raise SystemExit("ambiguous hex token map")
    meta = json.loads((Path(args.ref) / "meta.json").read_text())
    npz = np.load(Path(args.ref) / "reference.npz")
    ref = dict(logprobs=str(Path(args.ref) / "logprobs.f32.npy"), vocab=meta["vocab"], chunk=npz["chunk"], p=npz["p"],
               real_id=npz["real_id"], real_logprob=npz["real_logprob"], argmax_id=npz["argmax_id"])
    tables, counts = {}, {}
    for spec in args.engine:
        label, _, path = spec.partition("=")
        tables[label], counts[label] = sk.load_top20(path, hexmap)
    common = set.intersection(*(set(t) for t in tables.values()))
    tables = {k: {key: v for key, v in t.items() if key in common} for k, t in tables.items()}
    rec = None
    if args.ci:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kl_boot
        rec = kl_boot.Recorder()
        sk.row_kl = rec.wrap(sk.row_kl)
    out = sk.score(ref, tables)
    if rec:
        keys = sorted(common)
        index = {k: i for i, k in enumerate(zip(ref["chunk"].tolist(), ref["p"].tolist()))}
        arrays = rec.arrays(keys, list(tables), lambda k: int(ref["argmax_id"][index[k]]))
        out["ci"] = kl_boot.bootstrap(arrays, [k[0] for k in keys], resamples=args.ci, seed=args.seed)
    out["checks"]["common_positions"] = len(common)
    out["counts"] = counts
    kinds = {}
    for c, p in common:
        kinds[c] = kinds.get(c, 0) + 1
    out["checks"]["chunks"] = len(kinds)
    lines = [f"Common positions: {len(common)} over {len(kinds)} chunks (KL(BF16||engine), 21 buckets, lower bound)", "",
             "| Arm | Mean KL | Median | p99 | Max | Top-1 agreement | NLL (matched) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in tables:
        s = out[label]
        lines.append(f"| {label} | {s['mean']:.6f} | {s['median']:.6f} | {s['p99']:.6f} | {s['max']:.4f} | "
                     f"{100 * s['top1_agreement']:.2f}% | {s['nll_matched']:.4f} |")
    if "ci" in out:
        ci = out["ci"]
        lines += ["", f"95% interval, bootstrap over {ci['chunks']} chunks, {ci['resamples']} resamples, seed {ci['seed']}", "",
                  "| Arm | KL low | KL high | Top-1 low | Top-1 high |", "|---|---:|---:|---:|---:|"]
        for label, e in ci["engines"].items():
            lines.append(f"| {label} | {e['kl']['lo']:.6f} | {e['kl']['hi']:.6f} | {100 * e['top1']['lo']:.2f}% | {100 * e['top1']['hi']:.2f}% |")
        lines += ["", "| Difference (KL) | Mean | Low | High |", "|---|---:|---:|---:|"]
        for label, e in ci["diff_kl"].items():
            lines.append(f"| {label} | {e['mean']:.6f} | {e['lo']:.6f} | {e['hi']:.6f} |")
    text = "\n".join(lines)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))
    if args.out_md:
        Path(args.out_md).write_text(text + "\n")


if __name__ == "__main__":
    main()
