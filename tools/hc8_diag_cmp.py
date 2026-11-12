#!/usr/bin/env python3
"""Compare the token IDs of fastgate B phases case by case (hc8 diagnostic of the B=S p8300-ext invariant).

  hc8_diag_cmp.py LABEL=phase.json[:PART] ...   (PART = S or B for an SB phase file; none for a B-only file)
Prints, for each case, the first index where each arm differs from the first arm. Numbers only.
"""
import json
import sys


def load(spec):
    label, _, rest = spec.partition("=")
    path, _, part = rest.partition(":")
    d = json.load(open(path))
    d = d[part] if part else d
    return label, d.get("results", d)  # a fastgate cache file holds the results without the wrapper


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    arms = [load(s) for s in sys.argv[1:]]
    base_label, base = arms[0]
    for case in base:
        row = []
        for label, res in arms[1:]:
            if case not in res or res[case].get("ids") is None or base[case].get("ids") is None:
                row.append(f"{label}: -")
                continue
            d = first_diff(base[case]["ids"], res[case]["ids"])
            row.append(f"{label}: {'same' if d is None else f'diff at {d}'}")
        print(f"{case:12} vs {base_label}: " + ", ".join(row))


if __name__ == "__main__":
    main()
