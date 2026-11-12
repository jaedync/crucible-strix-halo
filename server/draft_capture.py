"""Draft vocabulary data from served turns. Off unless the server gets --draft-capture DIR.

counts: per day and turn class, how often each 128-token block occurs in the generated output. The file holds
  numbers only (draft-counts-YYYY-MM-DD.json). Prompts are not counted.
tokens: per day, the prompt and output token IDs of each finished turn in a binary spool
  (draft-tokens-YYYY-MM-DD.bin) for the machine-only training pipeline. It needs the user's explicit OK.

Both modes write no text, no request ID and no time finer than the UTC day (file times are set to the day's
00:00 UTC). The directory must be the server user's own and private (0700, no symlink); every file is 0600. One
server at a time writes a directory (DIR/.lock). The spool stops at max_bytes over all its files; later turns are
dropped and counted.

Leak guard (WP part 1): every file here carries the canary (server/leak_canary.py): the JSON files as the key
"canary", the spool as its header, the lock file as its content. The export scan and the repository guard fail on any
file that holds it. The directory must lie outside every git work tree: the server refuses to start otherwise.

Export floor: the per-day files never leave the serving machine. Any export sums 7 or more days and drops each class with
fewer than 20 sequences.
"""
import calendar
import fcntl
import json
import os
import stat
import struct
import tempfile
import threading
import time
from pathlib import Path

from .leak_canary import CANARY, CANARY_BYTES, refuse_inside_git_tree, shield_folder

VOCAB, BLOCK = 248320, 128
BLOCKS = VOCAB // BLOCK
MODES = ("counts", "tokens")
WRITE_SECONDS = 600
SPOOL_MAGIC = b"DCT1"                      # the old header, without the canary: read, and upgraded on the next write
SPOOL_HEADER = b"DCT2" + CANARY_BYTES      # the header of every new spool file
RECORD = struct.Struct("<BII")             # class index, prompt length, output length; then uint32 IDs
CLASSES = ("chat", "chat+think", "tool", "tool+think")


def turn_class(tools, thinking):
    return ("tool" if tools else "chat") + ("+think" if thinking else "")


def _check_ids(ids):
    for token in ids:
        if type(token) is not int or not 0 <= token < VOCAB:
            raise ValueError("draft capture needs token IDs in [0, 248320)")


def _refuse_git_tree(directory):
    refuse_inside_git_tree(directory, "private capture text data")


def _open_private(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "wb")


def _private_dir(directory):
    """Make or check the capture directory: a real directory of this user, closed to group and other."""
    if not directory.exists() and not directory.is_symlink():
        directory.mkdir(mode=0o700, parents=True)
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("the draft capture directory must be a directory, not a symlink")
    if info.st_uid != os.geteuid():
        raise ValueError("the draft capture directory must belong to the server user")
    if info.st_mode & 0o077:
        raise ValueError("the draft capture directory must be closed to group and other (0700)")


def without_canary(doc):
    """DOC as the reader sees it: a day file that the writer made (write_day_file adds the "canary" key at every flush) loses
    that one key; anything else is returned as it is. The input is not changed. The server (_load) and the daily-switch
    rehearsal both read a counts file through this function, so they accept the same key and no other."""
    if isinstance(doc, dict):
        return {key: value for key, value in doc.items() if key != "canary"}
    return doc


def _valid_counts(doc, day):
    """True when doc is a counts document of this day with the expected shape (numbers only)."""
    def count(v):
        return type(v) is int and v >= 0
    if not isinstance(doc, dict) or set(doc) != {"day", "granularity", "vocab", "block", "categories"}:
        return False
    if (doc["day"], doc["granularity"], doc["vocab"], doc["block"]) != (day, "block", VOCAB, BLOCK):
        return False
    cats = doc["categories"]
    return isinstance(cats, dict) and all(
        name in CLASSES and isinstance(entry, dict) and set(entry) == {"sequences", "tokens", "counts"}
        and count(entry["sequences"]) and count(entry["tokens"]) and isinstance(entry["counts"], list)
        and len(entry["counts"]) == BLOCKS and all(count(c) for c in entry["counts"])
        for name, entry in cats.items())


def write_day_file(directory, path, data, day_start):
    """Write DATA (a JSON document) to PATH as a private file with the canary key: a temp file in DIRECTORY, fsync,
    then rename, then the day's time."""
    shield_folder(directory)
    doc = json.loads(data)
    doc["canary"] = CANARY
    data = json.dumps(doc, separators=(",", ":")).encode()
    fd, temp = tempfile.mkstemp(dir=directory, prefix=".draft-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
    os.utime(path, (day_start, day_start), follow_symlinks=False)


class DraftCapture:
    def __init__(self, directory, *, mode, max_bytes=8 << 30, clock=time.time, draft_stats=False):
        # mode has no default: only the exact value "tokens" writes token IDs (the user's approval).
        directory = Path(directory)
        if not directory.is_absolute():
            raise ValueError("the draft capture directory must be an absolute path")
        if mode not in MODES:
            raise ValueError("the draft capture mode is counts or tokens")
        self.dir, self.mode, self.max_bytes, self.clock = directory, mode, int(max_bytes), clock
        _refuse_git_tree(self.dir)
        _private_dir(self.dir)
        shield_folder(self.dir)         # no `git add` takes this folder, even in a work tree that .git does not mark (H10)
        self.lock_fd = os.open(self.dir / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.lock_fd)
            raise RuntimeError("another server writes this draft capture directory") from None
        if os.fstat(self.lock_fd).st_size != len(CANARY_BYTES):   # the lock file carries the canary too
            os.ftruncate(self.lock_fd, 0)
            os.pwrite(self.lock_fd, CANARY_BYTES, 0)
        self._migrate()
        self.lock = threading.Lock()
        self.day, self.doc, self.last_write, self.dropped = None, None, 0.0, 0
        self.stats = None
        if draft_stats:   # per-token acceptance counters (server/draft_stats.py); counts files only
            from .draft_stats import DraftStats
            self.stats = DraftStats(self)

    def _rewrite(self, path, produce):
        """Replace PATH by a temp file that PRODUCE(handle) fills, keep its modification time and make it 0600."""
        times = os.stat(path, follow_symlinks=False)
        fd, temp = tempfile.mkstemp(dir=self.dir, prefix=".draft-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                produce(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except BaseException:
            if os.path.exists(temp):
                os.unlink(temp)
            raise
        os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns), follow_symlinks=False)

    def _migrate(self):
        """Once at the start: every file of an older server gets the canary (outside check H16). A spool with the old
        header gets the new one, a JSON object gets the key, any other file (a .bad file, a left-over temp file) gets a
        trailer line. Modification times stay. Run again, it changes nothing. The lock is held, so no other server writes."""
        for path in sorted(self.dir.iterdir()):
            name = path.name
            if name in (".lock", ".gitignore") or path.is_symlink() or not path.is_file():
                continue
            if name.startswith("draft-tokens-") and name.endswith(".bin"):
                with open(path, "rb") as handle:
                    head = handle.read(len(SPOOL_HEADER))
                if head[: len(SPOOL_MAGIC)] != SPOOL_MAGIC:
                    if CANARY_BYTES not in head:
                        self._trailer(path)          # not a spool of any known header: still no file without the canary
                    continue

                def produce(out, path=path):
                    out.write(SPOOL_HEADER)
                    with open(path, "rb") as src:
                        src.seek(len(SPOOL_MAGIC))
                        while chunk := src.read(1 << 22):
                            out.write(chunk)
                self._rewrite(path, produce)
                continue
            data = path.read_bytes()
            if CANARY_BYTES in data:
                continue
            try:
                doc = json.loads(data)
            except ValueError:
                doc = None
            if isinstance(doc, dict):
                doc["canary"] = CANARY
                text = json.dumps(doc, separators=(",", ":")).encode()
                self._rewrite(path, lambda out, text=text: out.write(text))
            else:
                self._trailer(path, data)

    def _trailer(self, path, data=None):
        data = path.read_bytes() if data is None else data
        tail = b"" if data.endswith(b"\n") or not data else b"\n"
        self._rewrite(path, lambda out: out.write(data + tail + CANARY_BYTES + b"\n"))

    def record_draft(self, *, tools, thinking, events):
        """Add the per-token draft counters of a finished turn (the done events of its jobs), if the flag is on."""
        if self.stats is not None:
            with self.lock:
                self.stats.record(turn_class(tools, thinking), events)

    def _today(self):
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def record(self, *, tools, thinking, prompt_ids, output_ids):
        output_ids = list(output_ids)
        _check_ids(output_ids)
        if self.mode == "tokens":       # counts mode never reads or copies the prompt
            prompt_ids = list(prompt_ids)
            _check_ids(prompt_ids)
        with self.lock:
            day = self._today()
            if self.mode == "tokens":
                self._spool(day, CLASSES.index(turn_class(tools, thinking)), prompt_ids, output_ids)
                return
            if day != self.day:
                self._flush()
                self.day, self.doc = day, self._load(day)
            entry = self.doc["categories"].setdefault(
                turn_class(tools, thinking), {"sequences": 0, "tokens": 0, "counts": [0] * BLOCKS})
            entry["sequences"] += 1
            entry["tokens"] += len(output_ids)
            for token in output_ids:
                entry["counts"][token // BLOCK] += 1
            if self.clock() - self.last_write >= WRITE_SECONDS:
                self._flush()

    def _counts_path(self, day):
        return self.dir / f"draft-counts-{day}.json"

    def _day_start(self, day):
        return calendar.timegm(time.strptime(day, "%Y-%m-%d"))

    def _load(self, day):
        """The day's sums so far (a restart adds to them), or an empty document. A file of another shape moves
        to NAME.bad and the day starts empty."""
        path = self._counts_path(day)
        if path.exists():
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError):
                doc = None
            doc = without_canary(doc)        # the writer adds the canary again at every flush
            if _valid_counts(doc, day):
                return doc
            os.replace(path, path.with_name(path.name + ".bad"))
        return {"day": day, "granularity": "block", "vocab": VOCAB, "block": BLOCK, "categories": {}}

    def _flush(self):
        if self.doc is None:
            return
        write_day_file(self.dir, self._counts_path(self.day), json.dumps(self.doc, separators=(",", ":")).encode(),
                       self._day_start(self.day))
        self.last_write = self.clock()

    def _spool_bytes(self):
        return sum(p.stat().st_size for p in self.dir.glob("draft-tokens-*.bin"))

    def _spool(self, day, cls, prompt_ids, output_ids):
        if self.mode != "tokens":
            raise RuntimeError("only the tokens mode writes token IDs")
        shield_folder(self.dir)
        path = self.dir / f"draft-tokens-{day}.bin"
        new = not path.exists()
        if not new:
            self._upgrade_spool(path, day)
        size = (len(SPOOL_HEADER) if new else 0) + RECORD.size + 4 * (len(prompt_ids) + len(output_ids))
        if self._spool_bytes() + size > self.max_bytes:
            self.dropped += 1
            return
        data = (SPOOL_HEADER if new else b"") + RECORD.pack(cls, len(prompt_ids), len(output_ids))
        data += struct.pack(f"<{len(prompt_ids) + len(output_ids)}I", *prompt_ids, *output_ids)
        with _open_private(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        start = self._day_start(day)
        os.utime(path, (start, start), follow_symlinks=False)

    def _upgrade_spool(self, path, day):
        """A spool file of an older server has no canary. Rewrite it once with the new header, in place of the old."""
        with open(path, "rb") as handle:
            head = handle.read(len(SPOOL_MAGIC))
        if head != SPOOL_MAGIC:
            return
        old = path.read_bytes()
        fd, temp = tempfile.mkstemp(dir=self.dir, prefix=".draft-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(SPOOL_HEADER + old[len(SPOOL_MAGIC):])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except BaseException:
            if os.path.exists(temp):
                os.unlink(temp)
            raise
        start = self._day_start(day)
        os.utime(path, (start, start), follow_symlinks=False)

    def close(self):
        with self.lock:
            try:
                try:
                    if self.stats is not None:
                        self.stats.flush()
                finally:
                    self._flush()
            finally:
                if self.lock_fd is not None:
                    os.close(self.lock_fd)       # closing the descriptor releases the flock
                    self.lock_fd = None


def read_spool(path):
    """Yield (class, prompt IDs, output IDs) of a spool file. For the test machine pipeline only: never print them."""
    with open(path, "rb") as handle:
        head = handle.read(len(SPOOL_MAGIC))
        if head == b"DCT2":
            if handle.read(len(CANARY_BYTES)) != CANARY_BYTES:
                raise ValueError("not a draft token spool of this version (the canary is missing)")
        elif head != SPOOL_MAGIC:
            raise ValueError("not a draft token spool")
        while header := handle.read(RECORD.size):
            if len(header) != RECORD.size:
                raise ValueError("truncated draft token spool")
            cls, n_prompt, n_output = RECORD.unpack(header)
            body = handle.read(4 * (n_prompt + n_output))
            if len(body) != 4 * (n_prompt + n_output) or cls >= len(CLASSES):
                raise ValueError("truncated draft token spool")
            ids = list(struct.unpack(f"<{n_prompt + n_output}I", body))
            yield CLASSES[cls], ids[:n_prompt], ids[n_prompt:]
