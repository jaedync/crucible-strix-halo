"""Turn spec_sweep generation logs into mtp_dump LIST shards (prompt + greedy output).

  python -m training.mtp_tune.gen_to_list GENDIR [--heldout 0.15]

Reads GENDIR/gen-*.log ("ids NAME: ..." lines) and GENDIR/ids/NAME.txt. Writes
GENDIR/gen-train-000.list and GENDIR/gen-heldout-000.list; a prompt goes to the
held-out set by a hash of its name.
"""
import argparse
import glob
import hashlib
import json

from . import taint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gendir")
    ap.add_argument("--heldout", type=float, default=0.15)
    ap.add_argument("--max", type=int, default=4096)
    ap.add_argument("--public", action="store_true",
                    help="a person states that every prompt is public text: the lists get the public line")
    a = ap.parse_args()
    out = {"train": [], "heldout": []}
    tainted = False     # leak guard: a prompt file that carries the canary taints every list of this folder
    for log in sorted(glob.glob(f"{a.gendir}/gen-*.log")):
        for line in open(log):
            if not line.startswith("ids "):
                continue
            name, rest = line[4:].split(":", 1)
            gen = [int(v) for v in rest.split()]
            prompt_file = f"{a.gendir}/ids/{name}.txt"
            tainted = tainted or taint.list_tainted(prompt_file)
            prompt = taint.read_ids(prompt_file)
            ids = (prompt + gen)[-a.max:]
            h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 2**32
            out["heldout" if h < a.heldout else "train"].append((name, ids, len(gen)))
    taint.guard_output(a.gendir, tainted or not a.public)   # H13: lists that are not stated public (unknown counts as tainted) never go into a git work tree
    meta = {}
    for split, items in out.items():
        rows = [(f"g{''.join(c for c in name if c.isalpha())}-{name}", ids) for name, ids, _ in items]
        taint.write_list(f"{a.gendir}/gen-{split}-000.list", rows, tainted, public=a.public and not tainted)
        meta[split] = {"seqs": len(items), "generated": sum(g for _, _, g in items),
                       "tokens": sum(len(i) for _, i, _ in items)}
    json.dump(taint.tag_doc(meta, tainted), open(f"{a.gendir}/lists.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
