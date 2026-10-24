"""Write the 16-bit n-gram table as one contiguous BF16 tensor.

Usage:
  convert.py SRC_DIR OUT [--revision REV] [--log LOG] [--stop-at "YYYY-MM-DD HH:MM"]
             [--rate-mb 100] [--chunk-mb 64]

SRC_DIR holds safetensors files with the tensors
  <prefix>.ngram_embedding.shard_{0..N-1}.weight   BF16 [R, 160]
(other tensors in the same files are ignored). OUT gets one tensor
  <prefix>.ngram_embedding.weight                  BF16 [N*R, 160]
with global row = shard * R + local row. This is the row order of the
exl3_ngram_trellis tables (exllamav3 NgramSource.read_rows); error_check.py
proves it against the quantized tables.

The copy goes to OUT.part. After each chunk the script calls fdatasync, drops
the pages of both files from the page cache and records the synced length in
OUT.part.progress, so an interrupted run resumes. At the end OUT.part is
hard-linked to OUT (this fails if OUT exists, so the script never overwrites)
and the sha256 of OUT is printed and written to OUT.sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Deadline, Guard, encode_header, read_header  # noqa: E402

ROW_DIM = 160
FORMAT = "ngram_bf16_concat"
SHARD_RE = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")


def find_shards(src_dir: str) -> tuple[str, list[dict]]:
    """Return (key prefix, shards in index order) with file and byte range of each."""
    found = {}
    prefixes = set()
    names = sorted(n for n in os.listdir(src_dir) if n.endswith(".safetensors"))
    for name in names:
        path = os.path.join(src_dir, name)
        header, base = read_header(path)
        size = os.path.getsize(path)
        for key, info in header.items():
            m = SHARD_RE.match(key)
            if not m:
                continue
            idx = int(m.group(2))
            if idx in found:
                raise ValueError(f"shard {idx} is in two files: {found[idx]['file']} and {name}")
            lo, hi = info["data_offsets"]
            if info["dtype"] != "BF16" or len(info["shape"]) != 2 or info["shape"][1] != ROW_DIM:
                raise ValueError(f"{key} in {name}: want BF16 [R, {ROW_DIM}], got {info}")
            if hi - lo != info["shape"][0] * ROW_DIM * 2 or base + hi > size:
                raise ValueError(f"{key} in {name}: byte range disagrees with shape or file size")
            prefixes.add(m.group(1))
            found[idx] = {"index": idx, "key": key, "file": name, "path": path,
                          "offset": base + lo, "nbytes": hi - lo, "rows": info["shape"][0]}
    if not found:
        raise ValueError(f"no *.ngram_embedding.shard_N.weight tensors in {src_dir}")
    if len(prefixes) != 1:
        raise ValueError(f"more than one table prefix: {sorted(prefixes)}")
    if sorted(found) != list(range(len(found))):
        missing = sorted(set(range(max(found) + 1)) - set(found))
        raise ValueError(f"shard indices are not 0..N-1; missing {missing[:10]}")
    shards = [found[i] for i in range(len(found))]
    if len({s["rows"] for s in shards}) != 1:
        raise ValueError("shards have different row counts")
    return prefixes.pop(), shards


def build_header(prefix: str, shards: list[dict], revision: str) -> bytes:
    rows = sum(s["rows"] for s in shards)
    meta = {
        "format": FORMAT,
        "version": "1",
        "row_dim": str(ROW_DIM),
        "rows": str(rows),
        "shard_rows": str(shards[0]["rows"]),
        "num_shards": str(len(shards)),
        "row_order": "global row = shard * shard_rows + row in shard (order of exl3_ngram_trellis tables)",
        "source_revision": revision,
        "source_tensors": f"{prefix}.shard_{{0..{len(shards) - 1}}}.weight",
        "shard_files": json.dumps([s["file"] for s in shards], separators=(",", ":")),
    }
    header = {
        "__metadata__": meta,
        f"{prefix}.weight": {"dtype": "BF16", "shape": [rows, ROW_DIM],
                             "data_offsets": [0, rows * ROW_DIM * 2]},
    }
    return encode_header(header)


def _drop(fd: int, offset: int, nbytes: int) -> None:
    os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)


def _write_progress(path: str, synced: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{synced}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _resume(part: str, progress: str, head: bytes, chunk: int, guard: Guard) -> tuple[int, "hashlib._Hash"]:
    """Return (synced bytes, sha256 state over them) of an earlier partial run, or (0, new)."""
    h = hashlib.sha256()
    if not os.path.exists(part):
        return 0, h
    if not os.path.exists(progress):
        raise ValueError(f"{part} exists without {progress}; delete it by hand to start again")
    synced = int(open(progress).read().strip())
    with open(part, "rb") as f:
        if f.read(len(head)) != head:
            raise ValueError(f"{part}: header differs from this run; delete it by hand")
    if os.path.getsize(part) < synced:
        raise ValueError(f"{part} is shorter than its progress record")
    fd = os.open(part, os.O_RDONLY)
    try:
        pos = 0
        while pos < synced:
            n = min(chunk, synced - pos)
            data = os.pread(fd, n, pos)
            if len(data) != n:
                raise IOError(f"short read in {part} at {pos}")
            h.update(data)
            _drop(fd, pos, n)
            pos += n
            guard.tick(f"rehash {pos}/{synced} bytes")
    finally:
        os.close(fd)
    guard.log(f"resume at {synced} bytes")
    return synced, h


def copy(shards: list[dict], head: bytes, out: str, rate_mb: float, chunk: int, guard: Guard) -> str:
    """Copy the header and all shard bytes to out.part, then link it to out. Returns sha256."""
    part, progress = out + ".part", out + ".part.progress"
    total = len(head) + sum(s["nbytes"] for s in shards)
    synced, h = _resume(part, progress, head, chunk, guard)
    fd = os.open(part, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, synced)
        if synced == 0:
            os.write(fd, head)
            os.fdatasync(fd)
            h.update(head)
            synced = len(head)
            _write_progress(progress, synced)
        buf = bytearray(chunk)
        t_run, done_run = time.monotonic(), 0
        pos = len(head)
        for s in shards:
            if pos + s["nbytes"] <= synced:
                pos += s["nbytes"]
                continue
            src = os.open(s["path"], os.O_RDONLY)
            try:
                skip = max(0, synced - pos)
                off = skip
                while off < s["nbytes"]:
                    n = min(chunk, s["nbytes"] - off)
                    guard.tick(f"copy shard {s['index']} ({s['file']}), {synced}/{total} bytes")
                    t0 = time.monotonic()
                    view = memoryview(buf)[:n]
                    if os.preadv(src, [view], s["offset"] + off) != n:
                        raise IOError(f"short read in {s['file']}")
                    h.update(view)
                    w = 0
                    while w < n:
                        w += os.pwrite(fd, view[w:], synced + w)
                    view.release()
                    os.fdatasync(fd)
                    _drop(fd, synced, n)
                    _drop(src, s["offset"] + off, n)
                    synced += n
                    off += n
                    done_run += n
                    _write_progress(progress, synced)
                    # Cap every chunk at the write rate, so no burst follows a pause.
                    wait = n / (rate_mb * 1e6) - (time.monotonic() - t0)
                    if wait > 0:
                        time.sleep(wait)
            finally:
                os.close(src)
            pos += s["nbytes"]
        os.fsync(fd)
    finally:
        os.close(fd)
    if os.path.getsize(part) != total:
        raise IOError(f"{part}: size {os.path.getsize(part)} != {total}")
    os.link(part, out)  # raises FileExistsError: never overwrite
    os.unlink(part)
    os.unlink(progress)
    dfd = os.open(os.path.dirname(os.path.abspath(out)), os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)
    elapsed = time.monotonic() - t_run
    guard.log(f"copied {done_run} bytes in {elapsed:.0f} s ({done_run / max(elapsed, 1e-9) / 1e6:.1f} MB/s)")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src_dir")
    ap.add_argument("out")
    ap.add_argument("--revision", default="unknown")
    ap.add_argument("--log")
    ap.add_argument("--stop-at", help="stop at this local time, YYYY-MM-DD HH:MM")
    ap.add_argument("--rate-mb", type=float, default=100.0)
    ap.add_argument("--chunk-mb", type=float, default=64.0)
    ap.add_argument("--dry-run", action="store_true", help="check the shards and print the plan")
    a = ap.parse_args(argv)
    guard = Guard(a.log, a.stop_at)
    if os.path.exists(a.out):
        print(f"error: {a.out} exists; refusing to overwrite", file=sys.stderr)
        return 2
    try:
        prefix, shards = find_shards(a.src_dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    head = build_header(prefix, shards, a.revision)
    total = len(head) + sum(s["nbytes"] for s in shards)
    guard.log(f"{len(shards)} shards x {shards[0]['rows']} rows from "
              f"{len({s['file'] for s in shards})} files -> {a.out} ({total} bytes)")
    if a.dry_run:
        for s in shards:
            print(f"shard {s['index']:3d} {s['file']} offset {s['offset']} bytes {s['nbytes']}")
        return 0
    try:
        guard.wait_start()
        digest = copy(shards, head, a.out, a.rate_mb, int(a.chunk_mb * (1 << 20)), guard)
    except Deadline as e:
        guard.log(f"stopped: {e}; run again to resume")
        return 3
    with open(a.out + ".sha256", "x") as f:
        f.write(f"{digest}  {os.path.basename(a.out)}\n")
    guard.log(f"done: {a.out} {total} bytes sha256 {digest}; lowest MemAvailable {guard.min_mem} kB")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
