#!/usr/bin/env python3
"""Rank tokens and blocks by the expected gain of adding them to, or removing them from, the draft vocabulary.

Input: the day files draft-stats-YYYY-MM-DD.json of serve --draft-stats (section 6) and the
draft vocabulary as a block list (default tests/data/draft/blocks65536.txt, the built-in kDefaultDraftBlocks; use
--vocab-blocks FILE for a candidate set). Output: numbers only (token IDs, block numbers, counts, gains). It reads no
text and needs no tokenizer.

Export floor: at least 7 days (--allow-short for a test), classes with fewer than 20 turns are dropped, and a token or pair
with fewer than --min-count events is not ranked. Gains are upper bounds: the real value needs a head retrain.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.draft_capture import BLOCK  # noqa: E402
from server.leak_canary import CANARY  # noqa: E402
from server.draft_stats import K, read_doc  # noqa: E402

DEFAULT_VOCAB = ROOT / "tests" / "data" / "draft" / "blocks65536.txt"
MIN_DAYS, MIN_TURNS = 7, 20


def load_blocks(path):
    """The vocabulary as a set of block numbers, from a file of integers separated by commas, spaces or lines."""
    values = [int(v) for v in re.split(r"[\s,]+", Path(path).read_text().strip()) if v]
    if not values or min(values) < 0:
        raise ValueError("the vocabulary file must hold block numbers")
    return set(values)


def load_days(paths, allow_short=False):
    """Sum the class tables of the day files. A path is a file or a directory of draft-stats-*.json files."""
    files = sorted(f for p in map(Path, paths) for f in ([p] if p.is_file() else p.glob("draft-stats-*.json")))
    days = {}
    for file in files:
        m = re.fullmatch(r"draft-stats-(\d{4}-\d{2}-\d{2})\.json", file.name)
        classes = read_doc(file.read_text(), m.group(1)) if m else None
        if classes is None:
            raise ValueError(f"{file.name}: not a version 1 draft-stats day file")
        days[m.group(1)] = classes
    if len(days) < MIN_DAYS and not allow_short:
        raise ValueError(f"{len(days)} day files; the export floor is {MIN_DAYS} (--allow-short to override)")
    total = {}
    for classes in days.values():
        for name, entry in classes.items():
            into = total.setdefault(name, dict(turns=0, tokens=0, pos={k: [0] * K for k in ("proposed", "tested", "accepted")},
                                               tok={}, miss={}))
            into["turns"] += entry["turns"]
            into["tokens"] += entry["tokens"]
            for key in into["pos"]:
                into["pos"][key] = [a + b for a, b in zip(into["pos"][key], entry["pos"][key])]
            for d, row in entry["tok"].items():
                mine = into["tok"].setdefault(int(d), [0] * (2 * K))
                into["tok"][int(d)] = [a + b for a, b in zip(mine, row)]
            for pair, row in entry["miss"].items():
                mine = into["miss"].setdefault(pair, [0] * K)
                into["miss"][pair] = [a + b for a, b in zip(mine, row)]
    return {name: e for name, e in total.items() if e["turns"] >= MIN_TURNS}, len(days)


def head_totals(entry):
    """Per position: tested and accepted drafts of the head only. The pos arrays also count lookup rounds (copied
    prompt tokens), which the tok and miss tables leave out, so a head decision must not use them."""
    tested, accepted = [0] * K, [0] * K
    for row in entry["tok"].values():
        for i in range(K):
            tested[i] += row[i]
            accepted[i] += row[K + i]
    return tested, accepted


def rates(entry):
    """Per position: conditional accept rate c_i of the head, value w_i of one extra accept in tokens (it carries
    the later positions that follow it)."""
    tested, accepted = head_totals(entry)
    c = [a / t if t else 0.0 for a, t in zip(accepted, tested)]
    w = []
    for i in range(K):
        value, product = 1.0, 1.0
        for j in range(i + 1, K):
            product *= c[j]
            value += product
        w.append(value)
    return c, w


def rank(classes, vocab_blocks, min_count=20, row_cost=0.02):
    """Add gain per token outside the vocabulary, loss and waste per token inside it, and the block swap list."""
    inside = lambda t: t // BLOCK in vocab_blocks  # noqa: E731
    add, loss, waste, proposals = {}, {}, {}, {}
    total_tokens = sum(e["tokens"] for e in classes.values())
    for entry in classes.values():
        _, w = rates(entry)
        accepted = head_totals(entry)[1]
        miss_in = [sum(row[i] for (_, t), row in entry["miss"].items() if inside(t)) for i in range(K)]
        h = [a / (a + m) if a + m else 0.0 for a, m in zip(accepted, miss_in)]
        events = {}
        for (_, t), row in entry["miss"].items():
            if inside(t):
                continue
            events[t] = events.get(t, 0) + sum(row)
            add[t] = add.get(t, 0.0) + sum(h[i] * row[i] * w[i] for i in range(K))
        for t, n in events.items():
            proposals[("add", t)] = proposals.get(("add", t), 0) + n
        for d, row in entry["tok"].items():
            loss[d] = loss.get(d, 0.0) + sum(row[K + i] * w[i] for i in range(K))
            waste[d] = waste.get(d, 0.0) + sum((row[i] - row[K + i]) * row_cost for i in range(K))  # row[:K] = tested
            proposals[("in", d)] = proposals.get(("in", d), 0) + sum(row[:K])  # tested drafts
    scale = 1000.0 / total_tokens if total_tokens else 0.0
    adds = [dict(token=t, block=t // BLOCK, events=proposals[("add", t)], gain=g, gain_per_1k=g * scale)
            for t, g in add.items() if proposals[("add", t)] >= min_count]
    removes = [dict(token=d, block=d // BLOCK, tested=proposals[("in", d)], loss=loss[d], waste=waste[d],
                    net=loss[d] - waste[d], net_per_1k=(loss[d] - waste[d]) * scale)
               for d in loss if proposals[("in", d)] >= min_count and inside(d)]
    adds.sort(key=lambda r: -r["gain"])
    removes.sort(key=lambda r: r["net"])
    block_add, block_in = {}, {}
    for r in adds:
        block_add[r["block"]] = block_add.get(r["block"], 0.0) + r["gain"]
    for b in vocab_blocks:
        block_in[b] = 0.0
    for d in loss:
        if inside(d):
            block_in[d // BLOCK] = block_in.get(d // BLOCK, 0.0) + loss[d] - waste[d]
    ins = sorted(block_in.items(), key=lambda kv: kv[1])
    outs = sorted(block_add.items(), key=lambda kv: -kv[1])
    swaps = [dict(add_block=a, remove_block=r, gain=ga - lr, gain_per_1k=(ga - lr) * scale)
             for (a, ga), (r, lr) in zip(outs, ins) if ga > lr]
    return dict(total_tokens=total_tokens, classes=sorted(classes), add=adds, remove=removes, swaps=swaps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("days", nargs="+", help="day files or directories of draft-stats-*.json")
    parser.add_argument("--vocab-blocks", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--row-cost", type=float, default=0.02, help="cost of one verify row in tokens (default 0.02)")
    parser.add_argument("--class", dest="only", action="append", help="use only this turn class (repeatable)")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        classes, count = load_days(args.days, args.allow_short)
        if args.only:
            classes = {n: e for n, e in classes.items() if n in args.only}
        result = rank(classes, load_blocks(args.vocab_blocks), args.min_count, args.row_cost)
    except (OSError, ValueError) as error:
        print(f"draft_rank: {error}", file=sys.stderr)
        return 2
    result["days"] = count
    result["canary"] = CANARY   # leak guard: the numbers come from capture data, so the output carries the canary
    if args.json:
        print(json.dumps(result))
        return 0
    print("#" + CANARY)
    print(f"days {count}, classes {','.join(result['classes']) or '-'}, generated tokens {result['total_tokens']}")
    for title, rows, cols in (("ADD (not in vocabulary)", result["add"], ("token", "block", "events", "gain", "gain_per_1k")),
                              ("REMOVE (in vocabulary, lowest net first)", result["remove"],
                               ("token", "block", "tested", "loss", "waste", "net_per_1k")),
                              ("BLOCK SWAPS", result["swaps"], ("add_block", "remove_block", "gain", "gain_per_1k"))):
        print(f"\n{title}\n" + "  ".join(f"{c:>12}" for c in cols))
        for row in rows[:args.top]:
            print("  ".join(f"{row[c]:>12.3f}" if isinstance(row[c], float) else f"{row[c]:>12}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
