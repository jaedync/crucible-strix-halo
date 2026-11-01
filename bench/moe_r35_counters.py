#!/usr/bin/env python3
"""Summary of a bench/moe_r35_counters.sh folder: per MoeSgKernel shape (R, D, NP) the mean of each counter per
dispatch, and WAVE_ISSUE_WAIT = SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES (rocprofv3 counter_defs.yaml), in %.

  python3 -I moe_r35_counters.py DIR
"""
import collections
import csv
import re
import sys
from pathlib import Path


def shape(name):
    m = re.search(r"MoeSgKernel<([^>]*)>", name)
    if not m:
        return None
    a = [x.strip() for x in m.group(1).split(",")]
    return f"R{a[0]} D{a[1]} NP{a[8]}"


def main(d):
    vals = collections.defaultdict(list)  # (shape, counter) -> values per dispatch
    for f in sorted(Path(d).rglob("*counter_collection.csv")):
        per = collections.defaultdict(float)  # (dispatch, shape, counter) -> sum over dimensions
        for r in csv.DictReader(open(f)):
            s = shape(r.get("Kernel_Name", ""))
            if s:
                per[(r.get("Dispatch_Id"), s, r["Counter_Name"])] += float(r["Counter_Value"])
        for (_, s, c), v in per.items():
            vals[(s, c)].append(v)
    shapes = sorted({s for s, _ in vals})
    counters = sorted({c for _, c in vals})
    print("| shape | " + " | ".join(counters) + " | WAVE_ISSUE_WAIT % |")
    print("|---|" + "---:|" * (len(counters) + 1))
    for s in shapes:
        m = {c: sum(vals[(s, c)]) / len(vals[(s, c)]) for c in counters if vals.get((s, c))}
        w = 100 * m["SQ_WAIT_INST_ANY"] / m["SQ_WAVE_CYCLES"] if "SQ_WAIT_INST_ANY" in m and m.get("SQ_WAVE_CYCLES") else None
        print(f"| {s} | " + " | ".join(f"{m[c]:.4g}" if c in m else "" for c in counters) + f" | {'' if w is None else f'{w:.1f}'} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
