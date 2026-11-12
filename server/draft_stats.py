"""Draft acceptance counters by position and by token. Part of --draft-capture
counts mode and off unless the server gets --draft-stats.

A day file draft-stats-YYYY-MM-DD.json beside the counts file holds, per turn class, counts of draft tokens by position
(proposed, tested, accepted), of drafts rejected for a given target token, and the draft length histogram. The
engine merges each request into counts by (token, position) before the done event, and this class adds them into the
class tables and drops them. The file holds counts of single token IDs: no text, no sequence, no request ID, no time
finer than the UTC day. DraftCapture owns the directory, the writer lock and the file rules.
"""
import json
import os

from .draft_capture import BLOCK, CLASSES, VOCAB, write_day_file

VERSION = 1
K = 7                          # draft positions 1..K
TOKEN_IDS_MAX = 65536          # draft tokens per class (the draft head vocabulary)
MISS_PAIRS_MAX = 262144        # (draft, target) pairs per class; more are counted in "overflow"
ROWS_MAX = 2048                # rows of each per-token list of one done event (engine kDraftDetailMax)
LISTS = (("proposed", K), ("tested", K), ("accepted", K))


def _count(v):
    return type(v) is int and v >= 0


def _counts(values, size):
    return type(values) is list and len(values) == size and all(_count(v) for v in values)


def parse_detail(event):
    """The validated per-token counters of one done event, or None: a bad or missing field drops the whole event."""
    try:
        rows, misses = event["draft_tally"], event["draft_miss"]
        if not (type(rows) is list and type(misses) is list and len(rows) <= ROWS_MAX and len(misses) <= ROWS_MAX):
            return None
        for d, i, p, a in rows:
            if not (all(type(v) is int for v in (d, i, p, a)) and 0 <= d < VOCAB and 1 <= i <= K and 0 <= a <= p):
                return None
        for d, i, t, c in misses:
            if not (all(type(v) is int for v in (d, i, t, c)) and 0 <= d < VOCAB and 0 <= t < VOCAB and 1 <= i <= K and c >= 1):
                return None
        lists = {name: event["draft_pos_" + name] for name, _ in LISTS}
        if not (all(_counts(lists[name], size) for name, size in LISTS) and _counts(event["draft_len_hist"], K + 1) and
                _count(event["draft_detail_cut"]) and _count(event["draft_lookup_rounds"]) and
                _count(event["completion_tokens"])):
            return None
        return dict(rows=rows, misses=misses, pos=lists, len_hist=event["draft_len_hist"], cut=event["draft_detail_cut"],
                    lookup_rounds=event["draft_lookup_rounds"], tokens=event["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


def _empty_class():
    return dict(turns=0, rounds=0, lookup_rounds=0, tokens=0, pos={name: [0] * K for name, _ in LISTS},
                len_hist=[0] * (K + 1), tok={}, miss={}, overflow=[0] * K, detail_cut=0)


def _valid_class(entry):
    """True for a class record of the file: counts only, with the table limits."""
    if not (isinstance(entry, dict) and set(entry) == {"turns", "rounds", "lookup_rounds", "tokens", "pos", "len_hist",
                                                       "tok", "miss", "overflow", "detail_cut"}):
        return False
    pos = entry["pos"]
    return (all(_count(entry[k]) for k in ("turns", "rounds", "lookup_rounds", "tokens", "detail_cut")) and
            isinstance(pos, dict) and set(pos) == {name for name, _ in LISTS} and
            all(_counts(pos[name], size) for name, size in LISTS) and _counts(entry["len_hist"], K + 1) and
            _counts(entry["overflow"], K) and isinstance(entry["tok"], dict) and len(entry["tok"]) <= TOKEN_IDS_MAX and
            all(type(d) is str and d.isdigit() and int(d) < VOCAB and _counts(v, 2 * K) for d, v in entry["tok"].items()) and
            type(entry["miss"]) is list and len(entry["miss"]) <= MISS_PAIRS_MAX and
            all(type(r) is list and len(r) == 2 + K and all(_count(v) for v in r) and r[0] < VOCAB and r[1] < VOCAB
                for r in entry["miss"]))


def read_doc(text, day):
    """The class tables of a day file's text (miss pairs as a dict), or None when the text is not a valid version-1 file."""
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    if isinstance(doc, dict):
        doc.pop("canary", None)      # the writer adds it again at every flush (server/leak_canary.py)
    if not (isinstance(doc, dict) and set(doc) == {"version", "day", "vocab", "block", "k", "classes"} and
            (doc["version"], doc["day"], doc["vocab"], doc["block"], doc["k"]) == (VERSION, day, VOCAB, BLOCK, K) and
            isinstance(doc["classes"], dict) and all(n in CLASSES and _valid_class(e) for n, e in doc["classes"].items())):
        return None
    return {name: dict(entry, tok=dict(entry["tok"]), miss={(r[0], r[1]): r[2:] for r in entry["miss"]})
            for name, entry in doc["classes"].items()}


class DraftStats:
    """The day's per-class tables. DraftCapture calls it under its own lock, so this class takes none."""

    def __init__(self, capture):
        self.capture = capture
        self.day, self.classes, self.last_write = None, {}, 0.0

    def _path(self, day):
        return self.capture.dir / f"draft-stats-{day}.json"

    def _load(self, day):
        """The day's sums so far (a restart adds to them). A file of another shape or version moves to NAME.bad."""
        path = self._path(day)
        classes = {}
        if path.exists():
            try:
                classes = read_doc(path.read_text(), day)
            except (OSError, ValueError):  # unreadable text (UnicodeDecodeError is a ValueError) is a bad file
                classes = None
            if classes is None:
                os.replace(path, path.with_name(path.name + ".bad"))
                return {}
        return classes

    def record(self, cls, events):
        """Add the counters of one turn's done events to its class. Events without valid detail are skipped."""
        details = [d for d in (parse_detail(event) for event in events) if d]
        if not details:
            return
        day = self.capture._today()
        if day != self.day:
            self.flush()
            self.day, self.classes = day, self._load(day)
        entry = self.classes.setdefault(cls, _empty_class())
        entry["turns"] += 1
        for detail in details:
            self._add(entry, detail)
        if self.capture.clock() - self.last_write >= 600:
            self.flush()

    @staticmethod
    def _add(entry, detail):
        entry["rounds"] += sum(detail["len_hist"])
        entry["lookup_rounds"] += detail["lookup_rounds"]
        entry["tokens"] += detail["tokens"]
        entry["detail_cut"] += detail["cut"]
        for name, _ in LISTS:
            entry["pos"][name] = [a + b for a, b in zip(entry["pos"][name], detail["pos"][name])]
        entry["len_hist"] = [a + b for a, b in zip(entry["len_hist"], detail["len_hist"])]
        for d, i, p, a in detail["rows"]:
            row = entry["tok"].get(str(d))
            if row is None:
                if len(entry["tok"]) >= TOKEN_IDS_MAX:
                    entry["detail_cut"] += 1
                    continue
                row = entry["tok"][str(d)] = [0] * (2 * K)
            row[i - 1] += p
            row[K + i - 1] += a
        for d, i, t, c in detail["misses"]:
            row = entry["miss"].get((d, t))
            if row is None:
                if len(entry["miss"]) >= MISS_PAIRS_MAX:
                    entry["overflow"][i - 1] += c
                    continue
                row = entry["miss"][(d, t)] = [0] * K
            row[i - 1] += c

    def flush(self):
        if self.day is None:
            return
        self.last_write = self.capture.clock()  # also on a failed write: a full disk is not retried at every turn
        classes = {name: dict(entry, miss=[[d, t, *row] for (d, t), row in sorted(entry["miss"].items())])
                   for name, entry in self.classes.items()}
        doc = dict(version=VERSION, day=self.day, vocab=VOCAB, block=BLOCK, k=K, classes=classes)
        write_day_file(self.capture.dir, self._path(self.day), json.dumps(doc, separators=(",", ":")).encode(),
                       self.capture._day_start(self.day))
        self.last_write = self.capture.clock()
