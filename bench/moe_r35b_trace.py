#!/usr/bin/env python3
"""moe-r35b: read the MoeSgKernel timeline traces of bench/moe_sg_engine.hip (variants 190 to 199, MUT 40).

  moe_r35b_trace.py DIR [--bins US]

Per trace file tr-R<R>-v<v>.bin (16 calls) it prints the medians over the calls of:
  evt, ovh  event time of the launch (us) and evt - span (launch and end cost outside the waves)
  span      first wave entry to last wave exit (us)
  entry     last wave entry - first wave entry (launch ramp)
  first     first unit body start - first entry (claim, job read, pipeline start)
  gu_end    last gate/up body end; dn0 = first down body start
  lastclaim last unit start (claim known); tail = span end - lastclaim
  exit50    median wave exit; drain = span end - exit50 (waves idle at the end)
  spin      sum over down units of the mid wait (wave us) and the longest wait
  gu_us, dn_us  mean body time of one gate/up and one down unit (us)
  busy      sum of unit time (claim known to mid done) / (waves x span)
  bw        effective GB/s over the span (bytes / span) and the peak 10 us bin
and one line of GB/s per time bin of the median call (bytes spread over each body interval).
"""
import glob
import os
import struct
import sys
from statistics import median

KTOPK = 10
UNIT = 8
HDR = 16  # header words per call (8 in the traces of bench 8215967b: --hdr8)


def load(fn):
    calls = []
    with open(fn, "rb") as f:
        while True:
            h = f.read(8 * HDR)
            if len(h) < 8 * HDR:
                break
            R, v, i, layer, nbytes, wclk, nu, nw = struct.unpack("<8Q", h[:64])
            evt = struct.unpack("<Q", h[64:72])[0] if HDR > 8 else 0
            words = struct.unpack("<%dQ" % (UNIT + 6 * nu), f.read(8 * (UNIT + 6 * nu)))
            waves = struct.unpack("<%dQ" % nw, f.read(8 * nw))
            calls.append(dict(R=R, v=v, i=i, nbytes=nbytes, wclk=wclk, nu=nu, words=words, waves=waves, evt=evt / 1e3))
    return calls


def analyze(c, bin_us):
    tick = 1e3 / c["wclk"]  # us per tick (wclk in kHz)
    R = c["R"]
    w = c["words"]
    total, n_gu, nj = w[0], w[1], w[2]
    U = min(R, 16) * KTOPK
    units = []
    for k in range(total):
        meta, t0, t1, t2, t3, _ = w[UNIT + 6 * k: UNIT + 6 * k + 6]
        if t0 == 0:
            continue
        gid = meta & 0xFFFF
        is_gu = (meta >> 16) & 1
        job = (meta >> 20) & 0xFFF
        shared = job >= U
        nb = (30720 if shared else 20480) if is_gu else (7680 if shared else 5120)
        units.append((k, gid, is_gu, job, t0, t1, t2, t3, nb))
    ent = [(c["waves"][2 * g], c["waves"][2 * g + 1]) for g in range(len(c["waves"]) // 2) if c["waves"][2 * g]]
    t_s = min(e for e, _ in ent)
    t_e = max(x for _, x in ent)
    us = lambda t: (t - t_s) * tick
    span = us(t_e)
    gu = [u for u in units if u[2]]
    dn = [u for u in units if not u[2]]
    r = {}
    r["evt"] = c["evt"]
    r["ovh"] = c["evt"] - span
    r["units"] = len(units)
    r["nj"] = nj
    r["waves"] = len(ent)
    r["span"] = span
    r["entry"] = us(max(e for e, _ in ent))
    r["first"] = min(us(u[5]) for u in units)
    r["gu_end"] = max(us(u[6]) for u in gu)
    r["gu_mid_end"] = max(us(u[7]) for u in gu)
    r["dn0"] = min(us(u[5]) for u in dn)
    r["lastclaim"] = max(us(u[4]) for u in units)
    r["tail"] = span - r["lastclaim"]
    ex = sorted(us(x) for _, x in ent)
    r["exit50"] = ex[len(ex) // 2]
    r["exit10"] = ex[len(ex) // 10]
    r["drain"] = span - r["exit50"]
    r["spin"] = sum((u[5] - u[4]) * tick for u in dn)
    r["spinmax"] = max((u[5] - u[4]) * tick for u in dn)
    r["gu_us"] = sum((u[6] - u[5]) * tick for u in gu) / len(gu)
    r["dn_us"] = sum((u[6] - u[5]) * tick for u in dn) / len(dn)
    r["mid_us"] = sum((u[7] - u[6]) * tick for u in gu)
    r["busy"] = sum((u[7] - u[4]) * tick for u in units) / (len(ent) * span)
    # Gaps between units of one wave (previous mid done to this claim known) and wave start to first unit.
    byw = {}
    for u in units:
        byw.setdefault(u[1], []).append(u)
    gap = 0.0
    for g, lst in byw.items():
        lst.sort(key=lambda u: u[4])
        for a, b in zip(lst, lst[1:]):
            gap += (b[4] - a[7]) * tick
    r["gap"] = gap
    nb = sum(u[8] for u in units)
    r["MB"] = nb / 1e6
    r["bw"] = nb / span / 1e3
    nbins = int(span / bin_us) + 1
    bins = [0.0] * nbins
    act = [0.0] * nbins
    for u in units:
        a, b = us(u[5]), us(u[6])
        d = max(b - a, 1e-6)
        x = a
        while x < b:
            k = int(x / bin_us)
            y = min(b, (k + 1) * bin_us)
            bins[k] += u[8] * (y - x) / d
            act[k] += (y - x) / bin_us
            x = y
    r["bins"] = [bb / bin_us / 1e3 for bb in bins]
    r["act"] = act
    r["peak"] = max(r["bins"])
    # Per-job: last gate/up mid done and the job's last down end (critical path of the last job).
    lastjob = max(gu, key=lambda u: u[7])[3]
    r["lastjob_mid"] = max(us(u[7]) for u in gu if u[3] == lastjob)
    r["lastjob_dn_end"] = max((us(u[6]) for u in dn if u[3] == lastjob), default=0)
    return r


KEYS = ["evt", "ovh", "span", "entry", "first", "gu_end", "dn0", "lastclaim", "tail", "exit10", "exit50", "drain", "spin",
        "spinmax", "gap", "gu_us", "dn_us", "busy", "MB", "bw", "peak", "lastjob_mid"]


def main():
    global HDR
    if "--hdr8" in sys.argv:
        HDR = 8
    d = sys.argv[1]
    bin_us = float(sys.argv[sys.argv.index("--bins") + 1]) if "--bins" in sys.argv else 10.0
    files = sorted(glob.glob(os.path.join(d, "tr-R*-v*.bin")),
                   key=lambda f: (int(f.split("-R")[1].split("-")[0]), int(f.split("-v")[1].split(".")[0])))
    print("| R | v | waves | units | " + " | ".join(KEYS) + " |")
    print("|---" * (4 + len(KEYS)) + "|")
    tl = []
    for fn in files:
        rs = [analyze(c, bin_us) for c in load(fn)]
        if not rs:
            continue
        m = {k: median(r[k] for r in rs) for k in KEYS}
        c0 = load(fn)[0]
        print("| %d | %d | %d | %d | " % (c0["R"], c0["v"], rs[0]["waves"], median(r["units"] for r in rs)) +
              " | ".join("%.2f" % m[k] if k in ("busy",) else "%.1f" % m[k] for k in KEYS) + " |")
        mid = sorted(rs, key=lambda r: r["span"])[len(rs) // 2]
        tl.append((c0["R"], c0["v"], mid))
    print()
    print("GB/s per %.0f us bin (median call), then active waves per bin:" % bin_us)
    for R, v, r in tl:
        print("R%d v%d bw  " % (R, v) + " ".join("%3.0f" % x for x in r["bins"]))
        print("R%d v%d act " % (R, v) + " ".join("%3.0f" % x for x in r["act"]))


if __name__ == "__main__":
    main()
