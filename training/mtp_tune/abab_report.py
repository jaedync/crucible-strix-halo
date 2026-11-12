"""Summarize ABAB leases (abab.sh): per-pair group ratios, their spread, and the GPU clock of each run.

  python3 training/mtp_tune/abab_report.py PREFIX [PREFIX ...]

PREFIX.<i>.<arm>.log are spec_sweep logs, PREFIX.<i>.<arm>.samples the clock and power samples of abab.sh ("epoch sclk_hz
power_uW busy_pct"; an old run has PREFIX.<i>.<arm>.sclk with the clock only), PREFIX.order the run order of each pair
and PREFIX.env the platform profile. Stock is arm A, tuned is arm B. The second arm of a pair runs about 0.7% faster
(Oct 25), so the report ends with the order-corrected ratio (tools/ab_order.py) and the order term: quote the corrected
ratio. It also gives the energy per arm run (socket power x busy time) and the platform profile.
"""
import glob
from pathlib import Path
import re
import statistics
import sys
from collections import defaultdict

from ab_report import group, parse  # noqa: E402  (run from this directory or with it on sys.path)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import ab_order  # noqa: E402

SAMPLE_PERIOD = 0.25   # abab.sh samples every 0.2 s plus the time of the reads


def sclk(path):
    v = [int(x) / 1e6 for x in open(path).read().split() if x.isdigit()]
    return (statistics.median(v), min(v)) if v else (None, None)


def clock(base):
    """(sclk median MHz, sclk min MHz, energy J) of one arm run, from BASE.samples, else from the old BASE.sclk."""
    try:
        lines = open(base + ".samples").read().splitlines()
    except OSError:
        return (*sclk(base + ".sclk"), None)
    samples, _ = ab_order.parse_samples(lines)
    busy = [mhz for _, mhz, b in samples if b >= 50] or [mhz for _, mhz, _ in samples]
    return (statistics.median(busy), min(busy), ab_order.sample_summary(lines, period=SAMPLE_PERIOD).get("energy_j")) \
        if busy else (None, None, None)


def read_orders(prefix):
    """{pair number: "AB" | "BA"} from PREFIX.order ("<i> AB" per line); {} when the file is missing."""
    try:
        words = open(prefix + ".order").read().split()
    except OSError:
        return {}
    return {int(i): o for i, o in zip(words[::2], words[1::2])}


def read_profile(prefixes):
    seen = set()
    for prefix in prefixes:
        try:
            seen |= {line.split()[1] for line in open(prefix + ".env") if line.startswith("platform_profile ")}
        except (OSError, IndexError):
            pass
    return ", ".join(sorted(seen)) or "unknown"


def corrected_lines(rows, energy):
    """rows: {group: [(order, ratio)]}; energy: [(order, tuned J / stock J)]. The order-corrected lines."""
    lines = []
    for g, items in sorted(rows.items()):
        orders = [o for o, _ in items]
        st = ab_order.order_stats([1.0] * len(items), [r for _, r in items], orders)
        if st["corrected_ratio"] is None:
            lines.append(f"  {g:6} unbalanced ({st['n_ab']} AB + {st['n_ba']} BA): no order correction")
        else:
            lines.append(f"  {g:6} {st['corrected_ratio'] - 1:+.1%}  order term {st['order_term_pct']:+.1f}%"
                         f"  ({st['n_ab']} AB + {st['n_ba']} BA)")
    if energy:
        st = ab_order.order_stats([1.0] * len(energy), [r for _, r in energy], [o for o, _ in energy])
        if st["corrected_ratio"] is not None:
            lines.append(f"energy tuned/stock {st['corrected_ratio']:.3f} (order-corrected, socket power x busy time; "
                         f"lower is better)")
    return lines


def main():
    pairs, orders = [], {}
    for prefix in sys.argv[1:]:
        prefix_orders = read_orders(prefix)
        for f in sorted(glob.glob(prefix + ".*.stock.log"), key=lambda p: int(re.findall(r"\.(\d+)\.stock", p)[0])):
            i = re.findall(r"\.(\d+)\.stock", f)[0]
            t = f.replace(".stock.log", ".tuned.log")
            pairs.append((f"{prefix.rsplit('/', 1)[-1]}.{i}", parse(f), parse(t), clock(f[:-4]), clock(t[:-4])))
            orders[len(pairs) - 1] = prefix_orders.get(int(i))
    per_group = defaultdict(list)
    hashes_ok = True
    first = {}
    by_group, energy = defaultdict(list), []
    for n, (name, s, t, cs, ct) in enumerate(pairs):
        ratios = defaultdict(list)
        for k in s:
            if k in t:
                ratios[group(k[0])].append(t[k]["tps"] / s[k]["tps"])
                for arm in (s, t):
                    hashes_ok &= first.setdefault(k[0], arm[k]["hash"]) == arm[k]["hash"]
        allr = [r for v in ratios.values() for r in v]
        means = {g: sum(v) / len(v) for g, v in ratios.items()}
        for g, x in list(means.items()) + [("mean", sum(allr) / len(allr))]:
            if g != "mean":
                per_group[g].append(x)
            if orders[n]:
                by_group[g].append((orders[n], x))
        per_group["mean"].append(sum(allr) / len(allr))
        if orders[n] and cs[2] and ct[2]:
            energy.append((orders[n], ct[2] / cs[2]))
        print(f"{name}: " + " ".join(f"{g} {x - 1:+.1%}" for g, x in sorted(means.items())) +
              f" mean {sum(allr) / len(allr) - 1:+.1%}  sclk median/min MHz stock {cs[0]:.0f}/{cs[1]:.0f}"
              f" tuned {ct[0]:.0f}/{ct[1]:.0f}")
    print(f"over {len(pairs)} pairs (mean, min, max of the per-pair ratio):")
    for g, v in sorted(per_group.items()):
        print(f"  {g:6} {statistics.mean(v) - 1:+.1%}  [{min(v) - 1:+.1%}, {max(v) - 1:+.1%}]")
    print("same output hash per prompt in every run:", hashes_ok)
    if by_group:
        print(f"order-corrected (tuned / stock tok/s; stock is arm A), {len(pairs)} pairs:")
        print("\n".join(corrected_lines(by_group, energy)))
    else:
        print("no order file (PREFIX.order): the pair order is unknown, so no order correction (the second arm of a pair "
              "runs about 0.7% faster)")
    print("platform profile:", read_profile(sys.argv[1:]))


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    main()
