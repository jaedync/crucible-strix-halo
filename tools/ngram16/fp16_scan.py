"""Scan every value of the BF16 n-gram table for the engine's fp16 range.

Usage:
  fp16_scan.py BF16_FILE [--json OUT.jsonl] [--chunk-mib 64] [--threads 4]
               [--log LOG] [--stop-at "YYYY-MM-DD HH:MM"]

The engine's BF16 row path (CRUCIBLE_PLE_BF16) rounds each value to fp16
(codec.engine_value_bf16). A value with |v| > 65504 gives inf there, then NaN.
bench/ple_bf16_check.hip checks only sampled rows; this tool reads all of them.

It reads the data of each BF16 tensor in chunks with pread, and drops each
chunk from the page cache (posix_fadvise DONTNEED), so a 102 GB scan does not
push the pages of a live table out of memory. It writes one JSON line (stdout,
and appended to --json) and one line
  RESULT ng16-fp16-scan PASS max_abs=<v> nonfinite=0
(FAIL when a value is not finite or is above the fp16 limit).
Exit code: 0 PASS, 1 FAIL, 2 bad input, 3 the time window closed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec import bf16_to_f32  # noqa: E402
from common import Deadline, Guard, read_header  # noqa: E402

FP16_MAX = 65504.0
ABS_MASK = 0x7FFF
# |bf16| bit patterns: at or above this one, the value is inf or NaN.
NONFINITE_BITS = 0x7F80
# The smallest |bf16| above FP16_MAX (65536): it and every larger finite value round to inf in fp16.
FP16_OVER_BITS = 0x4780


def bf16_tensors(header: dict, data_start: int, file_size: int) -> list[tuple[str, int, int]]:
    """(name, absolute start, bytes) of each BF16 tensor; refuses offsets outside the file."""
    out = []
    for name, info in header.items():
        if name == "__metadata__" or info.get("dtype") != "BF16":
            continue
        lo, hi = info["data_offsets"]
        count = 1
        for dim in info["shape"]:
            count *= dim
        if not 0 <= lo <= hi or hi - lo != 2 * count or data_start + hi > file_size:
            raise ValueError(f"{name}: data_offsets {lo}..{hi} do not fit shape {info['shape']} in the file")
        out.append((name, data_start + lo, hi - lo))
    if not out:
        raise ValueError("no BF16 tensor in the file")
    return sorted(out, key=lambda item: item[1])


def chunk_stats(bits: np.ndarray, scratch: np.ndarray) -> tuple[int, int, int]:
    """(largest finite |bits|, non-finite count, count above the fp16 limit) of uint16 bf16 bit patterns.
    For non-negative floats the bit pattern orders like the value, so the max of |bits| is the max |v|."""
    np.bitwise_and(bits, ABS_MASK, out=scratch)
    top = int(scratch.max()) if scratch.size else 0
    if top < FP16_OVER_BITS:  # the usual case: two passes over the chunk
        return top, 0, 0
    finite = scratch[scratch < NONFINITE_BITS]
    nonfinite = scratch.size - finite.size
    over = int(np.count_nonzero(finite >= FP16_OVER_BITS))
    return (int(finite.max()) if finite.size else 0), nonfinite, over


class Scanner:
    def __init__(self, path: str, chunk_bytes: int):
        self.fd = os.open(path, os.O_RDONLY)
        self.chunk = chunk_bytes
        self.buffers: dict[int, tuple[bytearray, np.ndarray]] = {}

    def scan(self, slot: int, offset: int, nbytes: int) -> tuple[int, int, int]:
        """Read [offset, offset + nbytes) into the buffer of one worker slot and return its stats."""
        if slot not in self.buffers:
            self.buffers[slot] = (bytearray(self.chunk), np.empty(self.chunk // 2, np.uint16))
        buf, scratch = self.buffers[slot]
        view = memoryview(buf)[:nbytes]
        done = 0
        while done < nbytes:
            got = os.preadv(self.fd, [view[done:]], offset + done)
            if got <= 0:
                raise IOError(f"short read at byte {offset + done}")
            done += got
        view.release()
        os.posix_fadvise(self.fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
        bits = np.frombuffer(buf, np.uint16, nbytes // 2)
        return chunk_stats(bits, scratch[: nbytes // 2])

    def close(self) -> None:
        os.close(self.fd)


def run(a) -> dict:
    started = time.monotonic()
    guard = Guard(a.log, a.stop_at)
    header, data_start = read_header(a.bf16)
    file_size = os.stat(a.bf16).st_size
    tensors = bf16_tensors(header, data_start, file_size)
    chunk = max(1, a.chunk_mib) << 20
    jobs = [(off, min(chunk, start + size - off)) for _, start, size in tensors
            for off in range(start, start + size, chunk)]
    guard.wait_start()
    scanner = Scanner(a.bf16, chunk)
    top = nonfinite = over = 0
    scanned = 0
    try:
        os.posix_fadvise(scanner.fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        with ThreadPoolExecutor(a.threads) as pool:
            # Each worker slot owns one buffer; a wave holds at most one job per slot.
            for wave in range(0, len(jobs), a.threads):
                guard.tick(f"scanned {scanned} of {sum(n for _, n in jobs)} bytes")
                batch = jobs[wave: wave + a.threads]
                for (_, nbytes), (t, n, o) in zip(batch, pool.map(lambda j: scanner.scan(*j), [
                        (slot, off, nbytes) for slot, (off, nbytes) in enumerate(batch)])):
                    top, nonfinite, over, scanned = max(top, t), nonfinite + n, over + o, scanned + nbytes
    finally:
        scanner.close()
    max_abs = float(bf16_to_f32(np.array([top], np.uint16))[0])
    status = "PASS" if nonfinite == 0 and over == 0 and max_abs <= FP16_MAX else "FAIL"
    return {"tool": "ng16-fp16-scan", "status": status, "file": os.path.basename(a.bf16),
            "file_bytes": file_size, "tensors": [name for name, _, _ in tensors], "values": scanned // 2,
            "max_abs": max_abs, "nonfinite": nonfinite, "above_fp16_max": over, "fp16_max": FP16_MAX,
            "seconds": round(time.monotonic() - started, 1), "min_mem_available_kb": guard.min_mem}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bf16")
    ap.add_argument("--json", help="append the JSON line to this file")
    ap.add_argument("--chunk-mib", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log")
    ap.add_argument("--stop-at")
    a = ap.parse_args(argv)
    if a.threads < 1:
        ap.error("--threads must be at least 1")
    try:
        res = run(a)
    except Deadline as e:
        print(f"stopped: {e}; run again", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        print("RESULT ng16-fp16-scan FAIL error", flush=True)
        return 2
    line = json.dumps(res, separators=(",", ":"))
    if a.json:
        with open(a.json, "a") as f:
            f.write(line + "\n")
    print(line)
    print(f"RESULT ng16-fp16-scan {res['status']} max_abs={res['max_abs']:g} nonfinite={res['nonfinite']}"
          f"{'' if res['status'] == 'PASS' else ' above_fp16_max=' + str(res['above_fp16_max'])}", flush=True)
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
