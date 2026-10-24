"""Measure the error of the quantized n-gram tables against the 16-bit rows.

Usage:
  error_check.py BF16_FILE SRC_DIR --table NAME=PATH [--table NAME=PATH ...]
                 [--blocks 1024] [--block-rows 4096] [--seed 20261018]
                 [--json OUT.json] [--log LOG] [--stop-at "YYYY-MM-DD HH:MM"]

BF16_FILE is the output of convert.py. SRC_DIR holds the source shards; a
subset of rows of BF16_FILE is compared bit for bit with them. Each --table is
an exl3_ngram_trellis file. All reads use O_DIRECT, so the page cache (and the
pages of a live table) is not changed. Only the sampled blocks are read.

Sampling: blocks // 16 random blocks of block-rows contiguous rows in each of
the 16 head regions (fixed seed). Decode: codec.dequant_rows with the table's
head_bias, in float32. Control: the same decoded rows against the 16-bit rows
shifted by one row; a correct row order gives a small error, the shifted
order an error near 140%.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec import ROW_DIM, bf16_to_f32, dequant_rows, head_of_rows, mul1_codebook, words_per_row  # noqa: E402
from common import Deadline, DirectReader, Guard, read_header  # noqa: E402
from convert import find_shards  # noqa: E402

BITWISE_EVERY = 16  # compare every 16th block bit for bit with the source shards


class QuantTable:
    def __init__(self, name: str, path: str, max_rows: int):
        header, base = read_header(path)
        meta = header.pop("__metadata__", {})
        if meta.get("format") != "exl3_ngram_trellis" or meta.get("codebook") != "mul1":
            raise ValueError(f"{path}: not an exl3_ngram_trellis mul1 table: {meta}")
        self.name, self.path, self.meta = name, path, meta
        self.K = int(meta["K"])
        self.words = words_per_row(self.K)
        self.row_bytes = self.words * 2
        keys = [k for k in header if k.endswith(".trellis")]
        shard = sorted((k for k in keys if ".shard_" in k),
                       key=lambda k: int(k.rsplit(".shard_", 1)[1].split(".")[0]))
        infos = [header[k] for k in shard] if shard else [header[keys[0]]]
        if not shard and len(keys) != 1:
            raise ValueError(f"{path}: expected one trellis tensor")
        for a, b in zip(infos, infos[1:]):
            if a["data_offsets"][1] != b["data_offsets"][0]:
                raise ValueError(f"{path}: trellis shards are not contiguous")
        if any(i["dtype"] != "I16" or i["shape"][1] != self.words for i in infos):
            raise ValueError(f"{path}: trellis shape does not match K={self.K}")
        self.rows = sum(i["shape"][0] for i in infos)
        self.data = base + infos[0]["data_offsets"][0]
        self.reader = DirectReader(path, max_rows * self.row_bytes)

        def small(suffix, dtype):
            info = header[next(k for k in header if k.endswith(suffix))]
            lo, hi = info["data_offsets"]
            raw = self.reader.read(base + lo, hi - lo)
            return np.frombuffer(raw, dtype=dtype).reshape(info["shape"]).copy()

        self.head_bias = small(".head_bias", np.float16)
        self.head_offsets = small(".head_offsets", np.int64)
        self.head_sizes = small(".head_vocab_sizes", np.int64)

    def packed(self, start: int, n: int) -> np.ndarray:
        raw = self.reader.read(self.data + start * self.row_bytes, n * self.row_bytes)
        return np.frombuffer(raw, dtype="<i2").reshape(n, self.words)


class Bf16Table:
    def __init__(self, path: str, max_rows: int):
        header, base = read_header(path)
        meta = header.pop("__metadata__", {})
        if meta.get("format") != "ngram_bf16_concat" or len(header) != 1:
            raise ValueError(f"{path}: not a convert.py output")
        (self.key, info), = header.items()
        self.meta = meta
        self.rows, dim = info["shape"]
        if info["dtype"] != "BF16" or dim != ROW_DIM:
            raise ValueError(f"{path}: want BF16 [rows, {ROW_DIM}]")
        self.data = base + info["data_offsets"][0]
        self.reader = DirectReader(path, max_rows * ROW_DIM * 2)

    def raw(self, start: int, n: int) -> bytes:
        return self.reader.read(self.data + start * ROW_DIM * 2, n * ROW_DIM * 2)


class ShardRows:
    """Rows read straight from the source shard files, for the bitwise check."""

    def __init__(self, src_dir: str, max_rows: int):
        _, self.shards = find_shards(src_dir)
        self.shard_rows = self.shards[0]["rows"]
        self.readers = {}
        self.max_bytes = max_rows * ROW_DIM * 2

    def raw(self, start: int, n: int) -> bytes:
        out = []
        while n > 0:
            s, lo = divmod(start, self.shard_rows)
            take = min(n, self.shard_rows - lo)
            sh = self.shards[s]
            if sh["path"] not in self.readers:
                self.readers[sh["path"]] = DirectReader(sh["path"], self.max_bytes)
            out.append(self.readers[sh["path"]].read(sh["offset"] + lo * ROW_DIM * 2, take * ROW_DIM * 2))
            start += take
            n -= take
        return b"".join(out)


def sample_blocks(head_offsets, head_sizes, total_rows, blocks, block_rows, seed) -> np.ndarray:
    """blocks // H block starts per head; each block plus one row stays in its head region."""
    rng = np.random.default_rng(seed)
    H = len(head_offsets)
    if blocks % H:
        raise ValueError(f"blocks must be a multiple of {H}")
    starts = []
    for h in range(H):
        lo = int(head_offsets[h])
        end = lo + int(head_sizes[h]) if h + 1 < H else total_rows
        hi = min(end, total_rows) - block_rows - 1
        starts.append(rng.integers(lo, hi + 1, size=blocks // H))
    return np.sort(np.concatenate(starts))


def row_stats(q: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, ...]:
    d = q - x
    return ((d * d).sum(1, dtype=np.float64), (x * x).sum(1, dtype=np.float64),
            (q * q).sum(1, dtype=np.float64), (q * x).sum(1, dtype=np.float64))


def summarize(err2, x2, q2, dot) -> dict:
    ok = x2 > 0
    rel = np.sqrt(err2[ok] / x2[ok])
    cos = dot[ok] / np.sqrt(np.maximum(q2[ok] * x2[ok], 1e-300))
    pct = np.percentile(rel, [50, 90, 99])
    return {
        "rows": int(err2.size), "zero_rows": int((~ok).sum()),
        "rel_rms": float(np.sqrt(err2.sum() / x2.sum())),
        "row_rel_mean": float(rel.mean()), "row_rel_median": float(pct[0]),
        "row_rel_p90": float(pct[1]), "row_rel_p99": float(pct[2]), "row_rel_max": float(rel.max()),
        "cos_mean": float(cos.mean()), "cos_p1": float(np.percentile(cos, 1)), "cos_min": float(cos.min()),
    }


def run(a) -> dict:
    guard = Guard(a.log, a.stop_at)
    t0 = time.monotonic()
    x_tab = Bf16Table(a.bf16, a.block_rows + 1)
    tables = []
    for spec in a.table:
        name, path = spec.split("=", 1)
        tables.append(QuantTable(name, path, a.block_rows))
    for t in tables:
        if t.rows != x_tab.rows:
            raise ValueError(f"{t.name}: {t.rows} rows, the 16-bit table has {x_tab.rows}")
        if not (np.array_equal(t.head_offsets, tables[0].head_offsets)
                and np.array_equal(t.head_sizes, tables[0].head_sizes)):
            raise ValueError(f"{t.name}: head regions differ from {tables[0].name}")
    offsets, sizes = tables[0].head_offsets, tables[0].head_sizes
    starts = sample_blocks(offsets, sizes, x_tab.rows, a.blocks, a.block_rows, a.seed)
    src = ShardRows(a.src_dir, a.block_rows)
    if src.shard_rows * len(src.shards) != x_tab.rows:
        raise ValueError("source shards and the 16-bit file disagree on the row count")

    n = len(starts) * a.block_rows
    heads = np.empty(n, dtype=np.int8)
    acc = {t.name: {k: np.empty(n) for k in ("err2", "x2", "q2", "dot", "err2_shift", "x2_shift", "err2_f16")}
           for t in tables}
    nonfinite = 0
    bitwise = {"rows": 0, "mismatch_rows": 0}
    guard.wait_start()
    for b, start in enumerate(starts):
        guard.tick(f"block {b}/{len(starts)}")
        start = int(start)
        raw = x_tab.raw(start, a.block_rows + 1)
        if b % BITWISE_EVERY == 0:
            same = np.frombuffer(raw, np.uint16).reshape(-1, ROW_DIM) == \
                np.frombuffer(src.raw(start, a.block_rows + 1), np.uint16).reshape(-1, ROW_DIM)
            bitwise["rows"] += same.shape[0]
            bitwise["mismatch_rows"] += int((~same.all(1)).sum())
        x_all = bf16_to_f32(np.frombuffer(raw, np.uint16).reshape(-1, ROW_DIM))
        nonfinite += int((~np.isfinite(x_all[:-1])).sum())
        # the quantizer replaced non-finite source values with zero
        x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
        x, x_next = x_all[:-1], x_all[1:]
        rows = np.arange(start, start + a.block_rows)
        hd = head_of_rows(rows, offsets)
        sl = slice(b * a.block_rows, (b + 1) * a.block_rows)
        heads[sl] = hd
        for t in tables:
            q = dequant_rows(t.packed(start, a.block_rows), t.K, a.codebook, t.head_bias[hd])
            st = acc[t.name]
            st["err2"][sl], st["x2"][sl], st["q2"][sl], st["dot"][sl] = row_stats(q, x)
            st["err2_shift"][sl], st["x2_shift"][sl], _, _ = row_stats(q, x_next)
            # The engine rounds each decoded value to fp16 (RoundH in PleGatherKernel).
            st["err2_f16"][sl] = row_stats(q.astype(np.float16).astype(np.float32), x)[0]
    bitwise.update(boundary_check(x_tab, src, guard))

    result = {"seed": a.seed, "blocks": len(starts), "block_rows": a.block_rows, "rows": n,
              "nonfinite_values": nonfinite, "bitwise": bitwise,
              "head_offsets": offsets.tolist(), "head_sizes": sizes.tolist(),
              "bf16": {"path": a.bf16, "rows": x_tab.rows, "meta": x_tab.meta}, "tables": {}}
    for t in tables:
        st = acc[t.name]
        r = summarize(st["err2"], st["x2"], st["q2"], st["dot"])
        r["rel_rms_engine_f16_out"] = float(np.sqrt(st["err2_f16"].sum() / st["x2"].sum()))
        rs = np.sqrt(st["err2_shift"] / np.maximum(st["x2_shift"], 1e-300))
        r["shift_control"] = {"rel_rms": float(np.sqrt(st["err2_shift"].sum() / st["x2_shift"].sum())),
                              "row_rel_mean": float(rs.mean())}
        r["per_head"] = []
        for h in range(len(offsets)):
            m = heads == h
            ph = summarize(st["err2"][m], st["x2"][m], st["q2"][m], st["dot"][m])
            ph["head"] = h
            r["per_head"].append(ph)
        result["tables"][t.name] = {"path": t.path, "K": t.K, "meta": t.meta, "stats": r}
    result["seconds"] = round(time.monotonic() - t0, 1)
    result["min_mem_available_kb"] = guard.min_mem
    return result


def boundary_check(x_tab: Bf16Table, src: ShardRows, guard: Guard) -> dict:
    """First and last row of every source shard, bit for bit."""
    bad = 0
    for s in range(len(src.shards)):
        guard.tick(f"boundary shard {s}")
        for r in (s * src.shard_rows, (s + 1) * src.shard_rows - 1):
            bad += x_tab.raw(r, 1) != src.raw(r, 1)
    return {"boundary_rows": 2 * len(src.shards), "boundary_mismatch_rows": int(bad)}


def markdown(res: dict) -> str:
    names = list(res["tables"])
    lines = ["| table | K | rel RMS | row rel mean | median | p90 | p99 | max | cos mean | cos p1 | shift control rel RMS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for nm in names:
        t = res["tables"][nm]
        s = t["stats"]
        lines.append(f"| {nm} | {t['K']} | {s['rel_rms']:.4%} | {s['row_rel_mean']:.4%} | {s['row_rel_median']:.4%} | "
                     f"{s['row_rel_p90']:.4%} | {s['row_rel_p99']:.4%} | {s['row_rel_max']:.4%} | "
                     f"{s['cos_mean']:.6f} | {s['cos_p1']:.6f} | {s['shift_control']['rel_rms']:.2%} |")
    lines += ["", "| head | rows | " + " | ".join(f"{nm} rel RMS | {nm} row rel mean | {nm} cos p1" for nm in names) + " |",
              "|---:|---:|" + "---:|" * (3 * len(names))]
    for h in range(len(res["head_offsets"])):
        cells = []
        for nm in names:
            ph = res["tables"][nm]["stats"]["per_head"][h]
            cells += [f"{ph['rel_rms']:.4%}", f"{ph['row_rel_mean']:.4%}", f"{ph['cos_p1']:.6f}"]
        rows = res["tables"][names[0]]["stats"]["per_head"][h]["rows"]
        lines.append(f"| {h} | {rows} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bf16")
    ap.add_argument("src_dir")
    ap.add_argument("--table", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--block-rows", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--json")
    ap.add_argument("--log")
    ap.add_argument("--stop-at")
    a = ap.parse_args(argv)
    a.codebook = mul1_codebook()
    try:
        res = run(a)
    except Deadline as e:
        print(f"stopped: {e}; run again", file=sys.stderr)
        return 3
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    print(markdown(res))
    print(f"nonfinite source values: {res['nonfinite_values']}; bitwise: {res['bitwise']}; "
          f"{res['seconds']} s; lowest MemAvailable {res['min_mem_available_kb']} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
