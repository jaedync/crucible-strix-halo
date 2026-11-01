#!/usr/bin/env python3
"""Summary of a moe-b128 lease (bench1.txt, bench2.txt of bench/moe_b128_lease.sh) and the kill decision.

  moe_b128_eval.py DIR [--bar 336]

Prints one table (us per MoE layer, median of each pass, mean of the passes) and, on the last line,
`MODE <n>` for the engine switch CRUCIBLE_MOE_B128 to gate (3: stream grid = v130 passes the bar at R = 4,
1: B64 D4 = v118 passes, 2: Q4 W5 D4 = v110 passes), or `MODE 0` (no engine variant passes: KILL). A variant passes only if it is
"bitwise equal" in every pass and its mean R = 4 time is at or below the bar.
"""
import re
import sys
from pathlib import Path

LINE = re.compile(r"^R=(\d+) v(\d+) (.*?)\s+MB [\d.]+ floor ([\d.]+) us \| fused med ([\d.]+) us .*\| (\S.*)$")


def load(d):
    t = {}
    for f in sorted(Path(d).glob("bench*.txt")):
        for line in f.read_text().splitlines():
            m = LINE.match(line)
            if m:
                R, v = int(m[1]), int(m[2])
                e = t.setdefault((R, v), {"name": m[3].strip(), "floor": float(m[4]), "us": [], "check": []})
                e["us"].append(float(m[5]))
                e["check"].append(m[6].strip())
    return t


def main():
    d = sys.argv[1]
    bar = float(sys.argv[sys.argv.index("--bar") + 1]) if "--bar" in sys.argv else 336.0
    t = load(d)
    if not t:
        print("no results\nMODE 0")
        return 1
    rs = sorted({R for R, _ in t})
    vs = sorted({v for _, v in t}, key=lambda v: (v != 0, v))
    print("| v | name | " + " | ".join(f"R={R}" for R in rs) + " | check |")
    print("|---|---|" + "---:|" * len(rs) + "---|")
    for v in vs:
        cells, checks = [], set()
        for R in rs:
            e = t.get((R, v))
            if not e:
                cells.append("-")
                continue
            mean = sum(e["us"]) / len(e["us"])
            cells.append(f"{mean:.1f} ({100 * e['floor'] / mean:.0f}%)")
            checks.update(e["check"])
        name = next(t[(R, v)]["name"] for R in rs if (R, v) in t)
        print(f"| {v} | {name} | " + " | ".join(cells) + f" | {', '.join(sorted(checks))} |")

    def passes(v):
        e = t.get((4, v))
        if not e:
            return False
        exact = all(c == "bitwise equal" for R in rs if (R, v) in t for c in t[(R, v)]["check"])
        return exact and sum(e["us"]) / len(e["us"]) <= bar

    mode = 3 if passes(130) else 1 if passes(118) else 2 if passes(110) else 0
    others = [v for v in (111, 112, 113, 114, 119, 131, 132, 133, 134, 138, 140, 141, 142, 143, 144, 145, 146, 160, 161, 162) if passes(v)]
    # Mutants (135-137) must be DIFFERENT in every pass: proof that the byte check sees a changed order.
    for v in (135, 136, 137, 147, 163):
        if any((R, v) in t for R in rs):
            bad = [R for R in rs if (R, v) in t and any(c != "DIFFERENT" for c in t[(R, v)]["check"])]
            print(f"mutant {v}: " + ("DIFFERENT at every R" if not bad else f"NOT DIFFERENT at R={bad}"))
    if others:
        print(f"also at or below {bar} us at R=4 (no engine path yet): {others}")
    print(f"MODE {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
