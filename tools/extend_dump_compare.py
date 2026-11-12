#!/usr/bin/env python3
"""Per-layer compare of gen --debug-schedule dumps.

  extend_dump_compare.py layers DIR_A DIR_B FROM [--json OUT]   h_XX.f32 of two runs: max |diff| per layer
  extend_dump_compare.py routes FILE_A SCHED_A FILE_B SCHED_B FROM   CRUCIBLE_ROUTE_DUMP files: expert-set flips
  extend_dump_compare.py state DIR_A DIR_B                        --dump-state dirs: bitwise per file
  extend_dump_compare.py logits FILE_A FILE_B                     --dump-step-logits: per step top-2 and gaps
  extend_dump_compare.py lp FILE RESULTS INDEX ROLE               step-0 top-20 log-probs against a server answer
  extend_dump_compare.py lpdiff FILE_A FILE_B                     step-0 top-20: max |diff| on shared ids (exact_extend)

Rows are prompt positions (each run dumps its passes in position order). FROM is the first position of the
"after the split" region (512 for s = 512).
"""
import json
import os
import sys

import numpy as np

WIDE = 10240
LAYERS = 48
TOPK = 10
VOCAB = 248320


def load_rows(path):
    return np.memmap(path, dtype=np.float32, mode="r").reshape(-1, WIDE)


def first_diff_row(a, b, start=0, block=256):
    for r in range(start, a.shape[0], block):
        x, y = a[r:r + block], b[r:r + block]
        ne = np.any(x.view(np.uint32) != y.view(np.uint32), axis=1)
        if ne.any():
            return r + int(np.argmax(ne))
    return -1


def layer_stats(a, b, start, block=512):
    """max |diff| over rows [0, start), over rows [start, end), on the last row; first differing row; and the
    max |diff| of each of the 8 rows from `start` (the resume head)."""
    rows = min(a.shape[0], b.shape[0])
    pre = post = 0.0
    sq = 0.0
    for r in range(0, rows, block):
        d = np.abs(a[r:r + block].astype(np.float64) - b[r:r + block])
        lo = max(0, start - r)
        if lo > 0:
            pre = max(pre, float(d[:lo].max()))
        if lo < d.shape[0]:
            post = max(post, float(d[lo:].max()))
            sq += float((d[lo:] ** 2).sum())
    ref = float(np.sqrt(sum(float((a[r:r + block].astype(np.float64)[max(0, start - r):] ** 2).sum())
                            for r in range(0, rows, block))))
    last = float(np.abs(a[rows - 1].astype(np.float64) - b[rows - 1]).max())
    head = [float(np.abs(a[start + j].astype(np.float64) - b[start + j]).max()) for j in range(8)]
    return dict(pre=pre, post=post, rel_rms=(np.sqrt(sq) / ref) if ref else 0.0, last=last,
                first=first_diff_row(a, b), head=head)


def cmd_layers(da, db, start, out=None):
    table = []
    for l in range(LAYERS + 1):
        name = f"h_{l:02d}.f32"
        a, b = load_rows(os.path.join(da, name)), load_rows(os.path.join(db, name))
        s = layer_stats(a, b, start)
        s["layer"] = l
        table.append(s)
        print(f"{l:3d} pre {s['pre']:.3e} post {s['post']:.3e} rel_rms {s['rel_rms']:.3e} last {s['last']:.3e} "
              f"first {s['first']} head {' '.join(f'{h:.1e}' for h in s['head'])}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(table, f, indent=1)
    return table


def parse_schedule(text):
    """Pass row counts in run order; 'j8' marks a pass whose rows are not prompt rows (the restore clobber)."""
    passes = []
    for item in text.split(","):
        skip = item.startswith("j")
        rows = int(item.lstrip("fj").rstrip("!"))
        passes.append((rows, skip))
    return passes


def load_routes(path, schedule):
    """routes[layer] = int array (positions, TOPK) of sorted expert ids. The engine warm-up writes records before the
    prompt, so the schedule's records are the last ones of the file."""
    raw = np.fromfile(path, dtype=np.int32)
    records = []
    at = 0
    while at < raw.size:
        li, r = int(raw[at]), int(raw[at + 1])
        records.append((li, r, raw[at + 2:at + 2 + r * TOPK].reshape(r, TOPK)))
        at += 2 + r * TOPK
    passes = parse_schedule(schedule)
    records = records[len(records) - len(passes) * LAYERS:]
    out = [[] for _ in range(LAYERS)]
    for p, (rows, skip) in enumerate(passes):
        for layer in range(LAYERS):
            li, r, ids = records[p * LAYERS + layer]
            if li != layer or r != rows:
                raise SystemExit(f"route record mismatch in pass {p}: layer {li} rows {r}, want {layer} {rows}")
            if not skip:
                out[layer].append(np.sort(ids, axis=1))
    return [np.concatenate(x) for x in out]


def cmd_routes(fa, sa, fb, sb, start):
    a, b = load_routes(fa, sa), load_routes(fb, sb)
    table = []
    for l in range(LAYERS):
        ne = np.any(a[l] != b[l], axis=1)
        first = int(np.argmax(ne)) if ne.any() else -1
        table.append(dict(layer=l, flips=int(ne[start:].sum()), flips_pre=int(ne[:start].sum()), first=first))
        print(f"{l:3d} flips {table[-1]['flips']} pre {table[-1]['flips_pre']} first {first}")
    return table


def cmd_state(da, db):
    names = sorted(set(os.listdir(da)) | set(os.listdir(db)))
    bad = 0
    for n in names:
        pa, pb = os.path.join(da, n), os.path.join(db, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{n}: only in one"); bad += 1; continue
        x, y = np.fromfile(pa, dtype=np.uint8), np.fromfile(pb, dtype=np.uint8)
        if x.size != y.size or (x != y).any():
            bad += 1
            first = int(np.argmax(x[:min(x.size, y.size)] != y[:min(x.size, y.size)]))
            print(f"{n}: DIFFERS (sizes {x.size} {y.size}, first byte {first})")
    print(f"state files {len(names)}, differing {bad}")
    return bad


def cmd_logits(fa, fb, tokens=None):
    a = np.fromfile(fa, dtype=np.float32).reshape(-1, VOCAB)
    b = np.fromfile(fb, dtype=np.float32).reshape(-1, VOCAB)
    for k in range(min(a.shape[0], b.shape[0])):
        rows = []
        for x in (a[k], b[k]):
            top = np.argsort(-x, kind="stable")[:2]
            lse = np.log(np.exp(x.astype(np.float64) - x.max()).sum()) + x.max()
            rows.append((int(top[0]), int(top[1]), float(x[top[0]] - x[top[1]]), float(x[top[0]] - lse)))
        d = float(np.abs(a[k].astype(np.float64) - b[k]).max())
        print(f"step {k:2d} A top {rows[0][0]} 2nd {rows[0][1]} gap {rows[0][2]:.4f} lp1 {rows[0][3]:.6g} | "
              f"B top {rows[1][0]} 2nd {rows[1][1]} gap {rows[1][2]:.4f} lp1 {rows[1][3]:.6g} | max|dlogit| {d:.4g}"
              f"{' bitwise' if (a[k].view(np.uint32) == b[k].view(np.uint32)).all() else ''}")


def top_logprobs(x, n=20):
    """engine/serve/logprobs.h TopLogprobs: lse in double, best first, ties by lower id."""
    peak = float(x.max())
    lse = peak + np.log(np.exp(x.astype(np.float64) - peak).sum())
    order = np.lexsort((np.arange(x.size), -x))[:n]
    return [(int(i), float(np.float32(float(x[i]) - lse))) for i in order]


def cmd_lp(path, results, index, role):
    x = np.fromfile(path, dtype=np.float32, count=VOCAB)
    mine = top_logprobs(x)
    for line in open(results):
        r = json.loads(line)
        if r.get("case", {}).get("index") == index:
            theirs = [(int(i), float(np.float32(v))) for i, v in r[role]["scores"]]
            ids = [i for i, _ in mine] == [i for i, _ in theirs]
            ulp = max(abs(int(np.float32(a).view(np.int32)) - int(np.float32(b).view(np.int32)))
                      for (_, a), (_, b) in zip(mine, theirs))
            print(f"lp {path} vs {role} index {index}: ids equal {ids}, max float32 ulp {ulp}, "
                  f"top1 {mine[0]} server {theirs[0]}")
            return ids and ulp == 0
    raise SystemExit(f"no case index {index}")


def cmd_lpdiff(fa, fb):
    a = top_logprobs(np.fromfile(fa, dtype=np.float32, count=VOCAB))
    b = top_logprobs(np.fromfile(fb, dtype=np.float32, count=VOCAB))
    values = dict(a)
    diffs = [abs(values[i] - v) for i, v in b if i in values]
    equal = a == b
    print(f"lpdiff {fa} {fb}: bitwise {equal}, max |diff| {max(diffs):.6g}, top1 {a[0]} {b[0]}")


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cmd = argv[1]
    if cmd == "layers":
        out = argv[argv.index("--json") + 1] if "--json" in argv else None
        cmd_layers(argv[2], argv[3], int(argv[4]), out)
    elif cmd == "routes":
        cmd_routes(argv[2], argv[3], argv[4], argv[5], int(argv[6]))
    elif cmd == "state":
        cmd_state(argv[2], argv[3])
    elif cmd == "lp":
        cmd_lp(argv[2], argv[3], int(argv[4]), argv[5])
    elif cmd == "lpdiff":
        cmd_lpdiff(argv[2], argv[3])
    elif cmd == "logits":
        cmd_logits(argv[2], argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
