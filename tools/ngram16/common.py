"""Shared helpers: safetensors headers, the memory guard, the time window, aligned reads."""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys
import time
from datetime import datetime

# Memory guard for a shared machine: start only with this much MemAvailable, pause below the second value.
MEM_START_KB = 15_625_000
MEM_PAUSE_KB = 13_600_000
MEM_POLL_S = 10
LOG_EVERY_S = 60
ALIGN = 4096
# Tests on tmpfs turn this off; tmpfs has no O_DIRECT. Real runs keep it on.
DIRECT = True


def read_header(path: str) -> tuple[dict, int]:
    """Return (header dict, absolute byte offset of the data area)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short for a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if n > 100 << 20:
            raise ValueError(f"{path}: header length {n} is not plausible")
        header = json.loads(f.read(n))
    return header, 8 + n


def encode_header(header: dict) -> bytes:
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    return struct.pack("<Q", len(hj)) + hj


def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable not in /proc/meminfo")


class Deadline(Exception):
    """The time window closed; the job must stop now."""


class Guard:
    """Memory guard, time window and a log line per minute with MemAvailable."""

    def __init__(self, log_path: str | None, stop_at: str | None):
        self.log_path = log_path
        self.stop_at = datetime.strptime(stop_at, "%Y-%m-%d %H:%M") if stop_at else None
        self.min_mem = None
        self.last_log = 0.0

    def log(self, text: str) -> None:
        line = f"{datetime.now():%F %T} {text}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def _mem(self) -> int:
        m = mem_available_kb()
        self.min_mem = m if self.min_mem is None else min(self.min_mem, m)
        return m

    def check_time(self) -> None:
        if self.stop_at and datetime.now() >= self.stop_at:
            raise Deadline(f"time window closed at {self.stop_at:%F %H:%M}")

    def wait_start(self) -> None:
        while (m := self._mem()) < MEM_START_KB:
            self.check_time()
            self.log(f"wait to start: MemAvailable {m} kB < {MEM_START_KB} kB")
            time.sleep(MEM_POLL_S)
        self.log(f"start: MemAvailable {m} kB")

    def tick(self, progress: str) -> None:
        """Call between work units: pauses on low memory, logs once a minute."""
        self.check_time()
        while (m := self._mem()) < MEM_PAUSE_KB:
            self.log(f"pause: MemAvailable {m} kB < {MEM_PAUSE_KB} kB; {progress}")
            time.sleep(MEM_POLL_S)
            self.check_time()
        now = time.monotonic()
        if now - self.last_log >= LOG_EVERY_S:
            self.last_log = now
            self.log(f"{progress}; MemAvailable {m} kB (min {self.min_mem} kB)")


class DirectReader:
    """O_DIRECT reads into a page-aligned buffer: the page cache stays untouched."""

    def __init__(self, path: str, max_bytes: int):
        self.fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if DIRECT else 0))
        self.size = os.fstat(self.fd).st_size
        self.cap = (max_bytes + 2 * ALIGN + ALIGN - 1) // ALIGN * ALIGN
        self.buf = mmap.mmap(-1, self.cap)

    def read(self, offset: int, nbytes: int) -> bytes:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError(f"read [{offset}, {offset + nbytes}) outside the file")
        lo = offset // ALIGN * ALIGN
        hi = (offset + nbytes + ALIGN - 1) // ALIGN * ALIGN
        if hi - lo > self.cap:
            raise ValueError("read is larger than the buffer")
        view = memoryview(self.buf)[: hi - lo]
        got = os.preadv(self.fd, [view], lo)
        view.release()
        if got < offset + nbytes - lo:  # a short read at the end of the file is fine
            raise IOError(f"short O_DIRECT read: {got} bytes at {lo}")
        return self.buf[offset - lo: offset - lo + nbytes]

    def close(self) -> None:
        os.close(self.fd)
        self.buf.close()


def die(text: str, code: int = 2) -> None:
    print(f"error: {text}", file=sys.stderr, flush=True)
    sys.exit(code)
