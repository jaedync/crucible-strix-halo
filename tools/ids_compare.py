#!/usr/bin/env python3
"""Token identity of an abab.py out dir: per pair and shape, the hash and count of A against B. Prints hashes and counts only.
usage: ids_compare.py ABAB_DIR   (run on the test machine with the refenv python; imports fastgate_lib from its own folder)"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastgate_lib as lib

def h(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]

root = Path(sys.argv[1])
rows, bad = [], 0
pairs = sorted({int(p.name[:-1]) for p in root.iterdir() if p.is_dir() and p.name[:-1].isdigit() and p.name[-1] in "AB"})
for pair in pairs:
    got = {}
    for arm in "AB":
        tries = sorted((root / f"{pair}{arm}").glob("try*"), key=lambda p: int(p.name[3:]))
        d = tries[-1]
        for shape in ("decode1", "decode0", "prefill32k"):
            out = d / f"{shape}.0.out"
            ids, _ = lib.parse_gen_output(out.read_text()) if out.exists() else (None, {})
            got[(shape, arm)] = (len(ids), h(ids)) if ids else None
    # streams: from the lease dir arms-out.json
    for sub in [root / str(pair), *root.glob(f"{pair}-try*"), *root.glob(f"lease-{pair}-*")]:
        f = sub / "arms-out.json"
        if not f.exists():
            continue
        specs = json.loads((sub / "arms-spec.json").read_text())["arms"]
        outs = json.loads(f.read_text())
        for spec, out in zip(specs, outs):
            wd = Path(spec["gen"]["workdir"]).parent.name
            arm = wd[-1]
            if str(pair) != wd[:-1]:
                continue
            rounds = [r for r in ((out.get("serve") or {}).get("streams") or {}).get("rounds", [])]
            hs = [r.get("ids_sha") for r in rounds if r.get("ids_sha")]
            got[("decode4", arm)] = (sum(sum(r["tokens"]) for r in rounds), hs) if hs else None
    for shape in ("decode1", "decode0", "prefill32k", "decode4"):
        if got.get((shape, "A")) is None and got.get((shape, "B")) is None:
            continue  # pair did not finish
        a, b = got.get((shape, "A")), got.get((shape, "B"))
        same = a is not None and a == b
        if shape == "decode4" and a and b:   # streams finish in the same order per round: compare hash lists
            same = a[1] == b[1]
        rows.append((pair, shape, a, b, same))
for pair, shape, a, b, same in rows:
    print(f"pair {pair} {shape}: {'EQUAL' if same else 'DIFFERENT'}  A={a}  B={b}")
tot = {}
for _, shape, _, _, same in rows:
    t = tot.setdefault(shape, [0, 0]); t[0] += same; t[1] += 1
print({k: f"{v[0]}/{v[1]} equal" for k, v in tot.items()})

# acceptance of the spec 7 gen run (decode1): verify forwards and drafts accepted per forward, per arm
import re
for pair in pairs:
    line = {}
    for arm in "AB":
        d = sorted((root / f"{pair}{arm}").glob("try*"), key=lambda p: int(p.name[3:]))[-1]
        out = d / "decode1.0.out"
        m = re.search(r"(\d+) verify forwards, ([\d.]+) drafts accepted per forward \(([\d.]+)%\)", out.read_text()) if out.exists() else None
        line[arm] = m.groups() if m else None
    if line['A'] is None and line['B'] is None:
        continue
    print(f"pair {pair} decode1 verify forwards / accepted per forward / pct: A={line['A']} B={line['B']}")
