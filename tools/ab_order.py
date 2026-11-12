#!/usr/bin/env python3
"""The A/B order rule, shared by every A/B tool (stdlib plus fastgate_lib; no GPU).

Oct 25: the second arm of a pair runs about 0.7% faster, so a run needs equal numbers of AB and BA pairs and an
order-corrected estimate. This module holds the pure parts; the ABAB tools and the
shell scripts use them.

  python3 ab_order.py pairs N      the pair count rounded up to an even number
  python3 ab_order.py order I      the order of pair I (0-based): AB or BA, in ABBA blocks
  python3 ab_order.py summary FILE sclk median and energy of a sample file (lines: epoch_s sclk_hz power_uW busy_pct)
"""
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastgate_lib as lib  # noqa: E402

SAMPLE_PERIOD = 0.5   # s; the gap cap of lib.power_summary is two periods


def pair_order(pair):
    """ABBA blocks of pairs: A B, B A, B A, A B, then again. Any even pair count has equal AB and BA counts, and a
    linear drift of the GPU cancels inside each block."""
    return ("A", "B") if pair % 4 in (0, 3) else ("B", "A")


def order_string(pair):
    return "".join(pair_order(pair))


def even_pairs(pairs):
    """(pair count rounded up to an even number, whether it was rounded)."""
    return (pairs + 1, True) if pairs % 2 else (pairs, False)


def sequence(first, second, pairs):
    """The run list of two configs over `pairs` pairs (rounded up to even): first second, second first, ..."""
    out = []
    for i in range(even_pairs(pairs)[0]):
        out += [first, second] if pair_order(i) == ("A", "B") else [second, first]
    return out


def order_stats(a, b, orders):
    """Order-corrected comparison of one metric. a and b are the values of the A and B arm per pair, orders the run
    order per pair ("AB": A ran first). Per pair r = ln(b / a). In an AB pair B runs second, so r = true + s where s is
    the second-arm effect; in a BA pair r = true - s. The corrected ln-ratio is the mean of the AB mean and the BA mean
    (s cancels), the order term is s (half the difference of the two means)."""
    ab, ba = [], []
    for x, y, order in zip(a, b, orders):
        if x and y:
            (ab if order == "AB" else ba).append(math.log(y / x))
    out = dict(n_ab=len(ab), n_ba=len(ba), corrected_ratio=None, order_term_pct=None, corrected_se_pct=None,
               pairs_faster=None)
    if not ab or not ba:
        return out
    mean_ab, mean_ba = sum(ab) / len(ab), sum(ba) / len(ba)
    corrected, term = (mean_ab + mean_ba) / 2, (mean_ab - mean_ba) / 2
    adjusted = [r - term for r in ab] + [r + term for r in ba]  # each pair with the order effect taken out
    n = len(adjusted)
    mean_adj = sum(adjusted) / n
    se = math.sqrt(sum((r - mean_adj) ** 2 for r in adjusted) / (n - 1) / n) if n > 1 else None
    out.update(corrected_ratio=math.exp(corrected), order_term_pct=(math.exp(term) - 1) * 100,
               corrected_se_pct=None if se is None else (math.exp(se) - 1) * 100,
               pairs_faster=sum(1 for r in adjusted if r > 0))
    return out


def corrected_row(a, b, orders, noise, up="FASTER", down="SLOWER"):
    """order_stats plus the verdict on the corrected ratio: NOISE when it is within `noise` percent of 1 (the spread
    rule of abab.py), UNBALANCED when one order is missing."""
    row = order_stats(a, b, orders)
    corrected = row["corrected_ratio"]
    row["verdict_corrected"] = ("UNBALANCED" if corrected is None else "NOISE" if abs(corrected - 1) * 100 <= noise
                                else up if corrected > 1 else down)
    return row


def parse_samples(lines):
    """(sclk samples (t, MHz, busy), power samples (t, W, busy)) from lines `epoch_s sclk_hz power_uW busy_pct`.
    A line that does not parse is skipped."""
    clock, power = [], []
    for line in lines:
        try:
            t, hz, uw, busy = line.split()
            t, hz, uw, busy = float(t), int(hz), int(uw), int(busy)
        except ValueError:
            continue
        clock.append((t, hz // 1_000_000, busy))
        power.append((t, uw / 1e6, busy))
    return clock, power


def sample_summary(lines, busy_min=50, period=SAMPLE_PERIOD):
    """sclk_median_mhz, energy_j, mean_w, busy_s of the busy samples; {} when no line parses. Same integration as
    fastgate_lib.ClockSampler (power_summary)."""
    clock, power = parse_samples(lines)
    if not clock:
        return {}
    busy = [s[1] for s in clock if s[2] >= busy_min] or [s[1] for s in clock]
    return {"sclk_median_mhz": int(statistics.median(busy)), **lib.power_summary(power, busy_min, period)}


def main(argv):
    if len(argv) == 2 and argv[0] == "pairs":
        print(even_pairs(int(argv[1]))[0])
    elif len(argv) == 2 and argv[0] == "order":
        print(order_string(int(argv[1])))
    elif len(argv) == 2 and argv[0] == "summary":
        got = sample_summary(Path(argv[1]).read_text().splitlines())
        print(" ".join(f"{k}={got[k]}" for k in ("sclk_median_mhz", "energy_j", "mean_w") if k in got))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
