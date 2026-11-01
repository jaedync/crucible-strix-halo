#!/usr/bin/env python3
"""moe-r35b timing table of bench/moe_sg_engine.hip leases: one row per variant.

  python3 -I moe_r35b_table.py [--base V] DIR [DIR ...]

Per variant: the median over the passes of the per-pass medians (us) at each R, the D1-weighted saving against the
base (default v170, CRUCIBLE_MOE_B128=3) over R = 2..8 (us per layer, weights of moe_r35_eval.py), the fit
t = intercept + bytes / rate over R = 2..8, and the passes. Last line: BYTES PASS|FAIL (moe_r35_eval.py rule).
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_r35_eval import WEIGHT, fit, parse  # noqa: E402


def main(argv):
    base = 170
    if "--base" in argv:
        i = argv.index("--base")
        base = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    meds, mb, bad = parse(argv)
    m = {k: st.median(x) for k, x in meds.items()}
    rows = sorted({r for r, _ in m})
    vs = sorted({v for _, v in m}, key=lambda v: (v % 1000, v))
    print("| v | " + " | ".join(f"R{r}" for r in rows) + " | D1 save us | intercept | GB/s | passes |")
    print("|---|" + "---:|" * (len(rows) + 4))
    for v in vs:
        cells = [f"{m[(r, v)]:.1f}" if (r, v) in m else "" for r in rows]
        rs = [r for r in WEIGHT if (r, v) in m and (r, base) in m]
        save = sum(WEIGHT[r] * (m[(r, base)] - m[(r, v)]) for r in rs) / sum(WEIGHT[r] for r in rs) if rs else 0
        f = fit([(mb[(r, v)], m[(r, v)]) for r in rows if 2 <= r <= 8 and (r, v) in m])
        n = min(len(meds[(r, v)]) for r in rows if (r, v) in meds)
        fs = f"{f[0]:.1f} | {1e3 / f[1]:.1f}" if f and f[1] > 0 else "- | -"
        print(f"| v{v} | " + " | ".join(cells) + f" | {save:+.1f} | {fs} | {n} |")
    print()
    for b in bad:
        print(b)
    print(f"BYTES {'FAIL' if bad else 'PASS'}")


if __name__ == "__main__":
    main(sys.argv[1:])
