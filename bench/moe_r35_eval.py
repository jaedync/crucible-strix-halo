#!/usr/bin/env python3
"""moe-r35 timing eval of bench/moe_sg_engine.hip leases.

  python3 -I moe_r35_eval.py DIR [DIR ...]

Reads DIR/p*.txt. Per (R, variant): the median of the per-pass medians (us) and the ratio to v170 (the default
stream-grid launch, CRUCIBLE_MOE_B128=3). Per variant: a least-squares fit of the median against the call bytes over
R = 2..8 (intercept us, rate GB/s). Last lines:
  BYTES PASS|FAIL   every non-mutant line clean and every mutant caught
  PICK <mode> v<variant> ...  or  PICK none ...   the engine mode for the gate (rule: PICK_RULE)
"""
import re
import statistics as st
import sys
from pathlib import Path

LINE = re.compile(r"^R=(\d+) v(\d+) med ([\d.]+) us .*?\| ([\d.]+) MB [\d.]+ GB/s \|.*\| (clean|FAIL|mutant caught|MUTANT PASSED.*)$")
MODES = {180: 4, 181: 5, 182: 6}  # bench variant -> CRUCIBLE_MOE_B128 value (MoeSgModeLaunch)
BASE = 170
# Verify rounds per R in the D1 traces of moe-sg-fix (prose + code) and the
# round time (verify + draft, ms) at that R.
WEIGHT = {2: 61, 3: 57, 4: 13, 5: 10, 6: 7, 7: 2, 8: 22}
ROUND_MS = {2: 36.66, 3: 41.61, 4: 48.07, 5: 52.25, 6: 58.64, 7: 60.46, 8: 66.27}
LAYERS = 48
PICK_RULE = "faster than v170 at each R = 3, 4, 5, mean ratio there <= 0.99; best D1-weighted saving"


def parse(dirs):
    """{(R, v): [medians]}, {(R, v): MB}, [bad lines] from the lease folders."""
    med, mb, bad = {}, {}, []
    for d in dirs:
        for f in sorted(Path(d).glob("p*.txt")):
            for line in f.read_text(errors="replace").splitlines():
                m = LINE.match(line)
                if not m:
                    continue
                r, v, t, b, status = int(m[1]), int(m[2]), float(m[3]), float(m[4]), m[5]
                med.setdefault((r, v), []).append(t)
                mb[(r, v)] = b
                if status == "FAIL" or status.startswith("MUTANT PASSED"):
                    bad.append(f"{f.name}: {line[:120]}")
    return med, mb, bad


def fit(points):
    """Least squares t = a + b x: (a, b), or None with fewer than 2 distinct x."""
    if len({x for x, _ in points}) < 2:
        return None
    mx = st.mean(x for x, _ in points)
    mt = st.mean(t for _, t in points)
    b = sum((x - mx) * (t - mt) for x, t in points) / sum((x - mx) ** 2 for x, _ in points)
    return mt - b * mx, b


def pick(m):
    """(mode, variant, saving us per layer, ratios at R 3..5) of the best candidate, or None."""
    best = None
    for v, mode in MODES.items():
        if any((r, v) not in m or (r, BASE) not in m for r in (3, 4, 5)):
            continue
        ratios = [m[(r, v)] / m[(r, BASE)] for r in (3, 4, 5)]
        if max(ratios) >= 1.0 or st.mean(ratios) > 0.99:
            continue
        rs = [r for r in WEIGHT if (r, v) in m and (r, BASE) in m]
        wsum = sum(WEIGHT[r] for r in rs)
        saving = sum(WEIGHT[r] * (m[(r, BASE)] - m[(r, v)]) for r in rs) / wsum
        if saving > 0 and (best is None or saving > best[2]):
            best = (mode, v, saving, ratios)
    return best


def report(dirs):
    meds, mb, bad = parse(dirs)
    m = {k: st.median(x) for k, x in meds.items()}
    rows = sorted({r for r, _ in m})
    vs = sorted({v for _, v in m})
    out = ["| R | " + " | ".join(f"v{v}" for v in vs) + " |", "|---:|" + "---:|" * len(vs)]
    for r in rows:
        cells = []
        for v in vs:
            if (r, v) not in m:
                cells.append("")
                continue
            ratio = f" ({m[(r, v)] / m[(r, BASE)]:.3f})" if (r, BASE) in m and v != BASE else ""
            cells.append(f"{m[(r, v)]:.1f}{ratio}")
        out.append(f"| {r} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("| variant | intercept us | rate GB/s | passes |")
    out.append("|---|---:|---:|---:|")
    for v in vs:
        f = fit([(mb[(r, v)], m[(r, v)]) for r in rows if 2 <= r <= 8 and (r, v) in m])
        n = min((len(meds[(r, v)]) for r in rows if (r, v) in meds), default=0)
        out.append(f"| v{v} | {f[0]:.1f} | {1e3 / f[1]:.1f} | {n} |" if f and f[1] > 0 else f"| v{v} | - | - | {n} |")
    out.append("")
    out.extend(bad)
    out.append(f"BYTES {'FAIL' if bad else 'PASS'}")
    p = pick(m)
    rs = [r for r in WEIGHT]
    round_ms = sum(WEIGHT[r] * ROUND_MS[r] for r in rs) / sum(WEIGHT.values())
    if p:
        mode, v, saving, ratios = p
        d_ms = LAYERS * saving / 1e3
        out.append(f"PICK {mode} v{v} saving {saving:.1f} us per layer (D1-weighted), {d_ms:.2f} ms per round of "
                   f"{round_ms:.1f} ms, expected D1 x{round_ms / (round_ms - d_ms):.3f}; ratios R3..5 "
                   + " ".join(f"{x:.3f}" for x in ratios))
    else:
        out.append(f"PICK none (rule: {PICK_RULE})")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(report(sys.argv[1:]))
