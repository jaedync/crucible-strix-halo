"""The start fuzz at the server level (start plan B1b step 1): the real batch backend (_admit, _stage_ahead, _staged,
_cancelled, _admit_successor) with the fake engine in FAKE_TIERS mode (server/tests/fake_batch_engine.py).

A reference model says where a saved point must exist: the prompt end of each turn whose prefill ended (End, and Pre
one token below it), the block of each prompt, the rows that a cancelled prefill computed, and the shared prefix of
a request that is in prefill when a request with the same prefix arrives. It does not copy the rules of the server
or the fake (plan B1b, "Do not").

Properties (plan B1b; the C++ fuzz adds F3, F4 and F6):
  F1 no silent loss: a turn resumes at or above the longest expected point that its prompt repeats, less MARGIN.
     The base has no `lost_cause` field (B1a adds it), so a short resume is a failure there.
  F2 exactness: a turn's tokens equal the tokens of a cold run of the same prompt (the fake's tokens depend only on
     the prompt).
  F5 closed fields: every crucible_turn line holds numbers, booleans, and words of the server's closed sets only.

Named seeds (orders of plan A.5 that the server decides): L9 (two equal new prefixes together), L12 (a think-cap
continuation that arrives after KEEP_SECONDS, when another turn took the slot), L13 (a client gives up in a cold
prefill and sends the prompt again), L15 (a slow disk read: the server opens cold after STAGE_WAIT_SECONDS). The
server's time constants are scaled down for the test (patched), with the same rules.
RED on the base: L9, L13 and L15 (expectedFailure: the package that fixes an order removes its mark; B3 removed the
mark of L15: a turn with a free slot waits for its read up to the cold prefill time of the read's tokens). GREEN on the
base: L12 (with a fake that keeps a handed-over point; the loss needs L2 as well).

Random seeds: CRUCIBLE_FUZZ_SEEDS (default 4) seeded event lists; they check F2 and F5 and count F1 failures by
event. Record: `python3 -m server.tests.test_start_fuzz --record OUT/fuzz_seeds.json`.
"""
import json
import os
import re
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from server import batch_backend, metrics
from server.api import validate_request
from server.batch_backend import BatchedEngineBackend

FAKE = Path(__file__).parent / "fake_batch_engine.py"
CONTEXT = 4096
BLOCK = list(range(1000, 1064))          # the shared system and tool block (64 tokens)
MARGIN = 4
# STAGE_COLD_TOKENS_PER_S (start plan B3, L15): the prefill rate that sets how long a turn with a free slot waits for
# its read (the cold prefill time of the read's tokens); 100 tok/s gives the 364-token disk state of the L15 seed 3.6 s.
SCALE = dict(STAGE_MIN_TOKENS=64, STAGE_WAIT_SECONDS=0.3, KEEP_SECONDS=0.2, STAGE_COLD_TOKENS_PER_S=100.0)
READ_DELAY_S = 1.5                        # above 2 * STAGE_WAIT_SECONDS: the bound of a turn that holds a free slot
WORD = r"[a-z0-9_]{1,32}|-"          # "-": the stage reply named no reason (server/metrics.py)
TEXT_FIELDS = dict(miss_reason=metrics.MISS_REASONS, closest_outcome=metrics.CLOSEST_OUTCOMES,
                   state_disk=frozenset(batch_backend.STAGE_STATES))


def request(max_tokens=4):
    return validate_request(dict(prompt="x", max_tokens=max_tokens, spec=0), "completion", "crucible")


class Harness:
    """One backend on the fake engine, restartable on the same state folder; the turn lines it wrote."""

    def __init__(self, folder, slots=2, **env):
        self.folder, self.slots = Path(folder), slots
        self.env = dict(FAKE_TIERS="1", FAKE_STATE_DISK="1", FAKE_TIER_FILE=str(self.folder / "points.json"),
                        FAKE_SLOT_CONTEXTS=",".join([str(CONTEXT)] * slots), FAKE_PREFILL_ROWS="16",
                        FAKE_DELAY="0.002", FAKE_READ_DELAY_S=str(READ_DELAY_S)) | env
        self.lines, self.backend = [], None
        self.expected = set()             # the reference model: point ID tuples that must exist
        self.failures, self.turns = [], []

    def start(self):
        popen = subprocess.Popen
        with patch.dict(os.environ, self.env), patch(
                "server.backend.subprocess.Popen",
                side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = BatchedEngineBackend("unused", "unused", context=CONTEXT, streams=self.slots,
                                                conversation_cache_mib=64, session_replay=True,
                                                state_dir=str(self.folder / "states"))
        return self

    def stop(self):
        if self.backend is not None:
            self.backend.flush_seconds = 30   # the engine writes its points at the end of input
            self.backend.close()
            self.backend = None

    def submit(self, ids, max_tokens=4, think_cap=None, after=None):
        options = dict(cache_prefix_tokens=len(BLOCK)) | (dict(think_cap=think_cap) if think_cap else {})
        return self.backend.submit(list(ids), request(max_tokens), after=after, **options)

    def expect_before(self, ids):
        key = tuple(ids)
        return max((len(point) for point in self.expected if len(point) < len(key) and key[:len(point)] == point), default=0)

    def finish(self, job, ids, expected, order, timeout=30):
        """Wait for the job, then check F1 and F2; the reference model takes its points."""
        tokens, done, deadline = [], None, time.monotonic() + timeout
        while done is None:
            event = job.events.get(timeout=max(0.1, deadline - time.monotonic()))
            if event["event"] == "token": tokens.append(event["id"])
            elif event["event"] == "done": done = event
            elif event["event"] == "error": raise AssertionError(f"{order}: {event}")
        cached = done.get("cached_tokens", job.turn.get("opened_cached", 0))
        turn = dict(order=order, prompt=len(ids), cached=cached, expected=expected)
        self.turns.append(turn)
        if cached < expected - MARGIN:
            self.failures.append(dict(turn, prop="F1"))
        if tokens != [ids[-1] + 1 + index for index in range(len(tokens))]:
            self.failures.append(dict(turn, prop="F2"))
        self.took(ids)
        return turn

    def took(self, ids):
        ids = tuple(ids)
        self.expected |= {ids, ids[:-1], ids[:len(BLOCK)]}

    def run(self, ids, order, max_tokens=4):
        expected = self.expect_before(ids)
        return self.finish(self.submit(ids, max_tokens), ids, expected, order)

    def closed_fields(self):
        """F5 over every crucible_turn line."""
        bad = []
        for line in self.lines:
            if not line.startswith("crucible_turn "): continue
            for key, value in json.loads(line[len("crucible_turn "):]).items():
                if value is None or isinstance(value, (bool, int, float)): continue
                if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value): continue
                if isinstance(value, str) and key in TEXT_FIELDS and value in TEXT_FIELDS[key]: continue
                if isinstance(value, str) and key not in TEXT_FIELDS and re.fullmatch(WORD, value): continue
                bad.append((key, str(value)[:40]))
        return bad


def ids_of(seed, size):
    rng = random.Random(repr(seed))   # tuples allowed; repr keeps the seed fixed across processes
    return [rng.randrange(2000, 200000) for _ in range(size)]


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline: raise AssertionError("timed out")
        time.sleep(0.005)


# ---- named seeds ----------------------------------------------------------------------------------------------------

def seed_l9(h):
    """Two sessions with the same new prefix of 400 tokens arrive together (two free slots, slow prefill)."""
    h.run(BLOCK + ids_of(90, 20), "L9-warm")
    shared = ids_of(91, 400)
    first, second = BLOCK + shared + ids_of(92, 30), BLOCK + shared + ids_of(93, 30)
    one = h.submit(first)
    wait_for(lambda: one.turn.get("chunk_count", 0) >= 1)
    two = h.submit(second)
    h.finish(one, first, h.expect_before(first), "L9-first")
    # The reference: the shared prefix is in prefill when the second arrives, so it is computed once.
    h.finish(two, second, len(BLOCK) + len(shared), "L9")


def seed_l12(h):
    """One slot pair: A's think cap closes, its continuation comes after KEEP_SECONDS, and B takes the slot."""
    a = BLOCK + ids_of(120, 300)
    first = h.submit(a, max_tokens=8, think_cap=3)
    tokens = []
    while len(tokens) < 3:
        event = first.events.get(timeout=10)
        if event["event"] == "token": tokens.append(event["id"])
    h.took(a)                                   # A's prompt ended its prefill: End and Pre exist
    blocker = BLOCK + ids_of(121, 200)
    other = h.submit(blocker, max_tokens=200)   # decodes on the second slot for the whole seed
    first.cancelled.set()
    b = BLOCK + ids_of(122, 200)
    taker = h.submit(b)                          # waits; it takes A's slot when the keep time ends
    time.sleep(2 * SCALE["KEEP_SECONDS"])
    continuation = a + tokens
    follow = h.submit(continuation, max_tokens=5)
    h.finish(taker, b, h.expect_before(b), "L12-taker")
    h.finish(follow, continuation, h.expect_before(continuation), "L12")
    h.finish(other, blocker, h.expect_before(blocker), "L12-blocker")


def seed_l13(h):
    """A client gives up during a cold prefill of 600 tokens and sends the same prompt again."""
    ids = BLOCK + ids_of(130, 600)
    job = h.submit(ids)
    wait_for(lambda: job.turn.get("chunk_count", 0) >= 10)
    job.cancelled.set()
    wait_for(lambda: not h.backend.active and not h.backend.waiting)
    computed = sum(rows for _, _, rows in job.turn["prefill_chunks"] if rows) + (job.turn.get("opened_cached") or 0)
    # The reference: the rows that the cancelled prefill computed stay (a point at its last chunk end).
    h.expected.add(tuple(ids[:computed]))
    h.run(ids, "L13")


def seed_l15(h):
    """A long turn, a stop and a start on the same folder, then the next turn waits for a slow disk read."""
    ids = BLOCK + ids_of(150, 300)
    h.run(ids, "L15-first")
    h.stop()
    h.start()
    h.run(ids + ids_of(151, 40), "L15")


SEEDS = dict(L9=(seed_l9, 2), L12=(seed_l12, 2), L13=(seed_l13, 2), L15=(seed_l15, 2))
CONTROL_READ_DELAY_S = 0.05               # L15 with a read inside the bound: the same seed resumes from the disk
BASE = dict(L9="RED", L12="GREEN", L13="RED", L15="RED")


def run_seed(name, folder, **env):
    """The named seed on a new harness: (F1 and F2 failures, F5 bad fields, the turns)."""
    seed, slots = SEEDS[name]
    harness = Harness(folder, slots, **env)
    with patch.object(metrics, "emit", side_effect=harness.lines.append), \
            patch.multiple(batch_backend, create=True, **SCALE):
        harness.start()
        try:
            seed(harness)
        finally:
            harness.stop()
    return [f for f in harness.failures if f["order"] == name], harness.closed_fields(), harness.turns


# ---- random seeds ---------------------------------------------------------------------------------------------------

def random_events(seed, count=14):
    rng = random.Random(seed)
    return [rng.choice(("turn", "turn", "turn", "undo", "new", "together", "cancel", "restart")) for _ in range(count)]


def run_random(seed, folder):
    """A seeded event list on 2 to 4 slots with 1 to 2N sessions; F1 failures are counted by event, F2 and F5 must hold."""
    rng = random.Random(seed)
    harness = Harness(folder, slots=rng.choice((2, 3, 4)), FAKE_READ_DELAY_S="0")
    sessions, f1 = [], {}
    with patch.object(metrics, "emit", side_effect=harness.lines.append), patch.multiple(batch_backend, create=True, **SCALE):
        harness.start()
        try:
            for index, event in enumerate(random_events(seed)):
                before = len([f for f in harness.failures if f["prop"] == "F1"])
                if event == "new" or not sessions:
                    sessions.append([BLOCK + ids_of((seed, index), rng.randrange(50, 400))])
                    harness.run(sessions[-1][-1], "new")
                elif event == "turn":
                    history = rng.choice(sessions)
                    history.append(history[-1] + ids_of((seed, index, 1), rng.randrange(5, 200)))
                    harness.run(history[-1], "turn")
                elif event == "undo":
                    history = rng.choice(sessions)
                    back = history[max(0, len(history) - 1 - rng.choice((1, 2)))]
                    history.append(back + ids_of((seed, index, 2), 20))
                    harness.run(history[-1], "undo")
                elif event == "together":
                    picked = [rng.choice(sessions) for _ in range(min(len(sessions), harness.slots + 1))]
                    jobs = []
                    for k, history in enumerate(picked):
                        ids = history[-1] + ids_of((seed, index, 3, k), rng.randrange(5, 100))
                        history.append(ids)
                        jobs.append((harness.submit(ids), ids, harness.expect_before(ids)))
                    for job, ids, expected in jobs:
                        harness.finish(job, ids, expected, "together")
                elif event == "cancel":
                    ids = BLOCK + ids_of((seed, index, 4), 300)
                    job = harness.submit(ids)
                    wait_for(lambda: job.turn.get("chunk_count", 0) >= 3 or job.turn.get("t_done"))
                    job.cancelled.set()
                    wait_for(lambda: not harness.backend.active and not harness.backend.waiting)
                elif event == "restart":
                    harness.stop()
                    harness.start()
                after = len([f for f in harness.failures if f["prop"] == "F1"])
                if after > before: f1[event] = f1.get(event, 0) + after - before
        finally:
            harness.stop()
    other = [f for f in harness.failures if f["prop"] != "F1"]
    return other, harness.closed_fields(), f1


class NamedSeedTest(unittest.TestCase):
    def check(self, name):
        with tempfile.TemporaryDirectory() as folder:
            failures, bad, turns = run_seed(name, folder)
        self.assertEqual(bad, [])
        self.assertTrue(any(turn["order"] == name for turn in turns), turns)
        self.assertEqual(failures, [], f"{name} on this tree: {failures}")

    @unittest.expectedFailure
    def test_l9_two_equal_new_prefixes_compute_once(self):   # RED on the base (B4 step 3 owns it)
        self.check("L9")

    def test_l12_a_late_continuation_resumes_from_its_point(self):   # GREEN on the base: a regression test
        self.check("L12")

    @unittest.expectedFailure
    def test_l13_a_cancelled_cold_prefill_keeps_its_rows(self):  # RED on the base
        self.check("L13")

    def test_l15_a_slow_disk_read_is_not_dropped(self):      # RED on the base; GREEN with B3 (the cold bound)
        self.check("L15")

    def test_l15_control_a_read_inside_the_bound_resumes_from_the_disk(self):
        # The L15 seed fails only by the read time: the same events with a fast read resume from the disk state.
        with tempfile.TemporaryDirectory() as folder:
            failures, bad, turns = run_seed("L15", folder, FAKE_READ_DELAY_S=str(CONTROL_READ_DELAY_S))
        self.assertEqual((failures, bad), ([], []))
        self.assertEqual([t["cached"] for t in turns if t["order"] == "L15"], [len(BLOCK) + 300])


class RandomSeedTest(unittest.TestCase):
    def test_random_seeds_keep_tokens_and_closed_fields(self):
        for seed in range(int(os.environ.get("CRUCIBLE_FUZZ_SEEDS", "4"))):
            with tempfile.TemporaryDirectory() as folder:
                other, bad, _ = run_random(seed, folder)
            self.assertEqual((other, bad), ([], []), f"seed {seed}")


def record(path):
    """fuzz_seeds.json records of the named seeds on this tree (plan B1b gate table)."""
    rows = []
    for name in SEEDS:
        with tempfile.TemporaryDirectory() as folder:
            failures, bad, turns = run_seed(name, folder)
        state = "RED" if failures or bad else "GREEN"
        rows.append(dict(seed=name, order=name, slots=SEEDS[name][1], level="server", base=BASE[name], head=state,
                         failed=sorted({f["prop"] for f in failures}) + (["F5"] if bad else []),
                         turns=[t for t in turns if t["order"] == name]))
    Path(path).write_text(json.dumps(rows, indent=1) + "\n")
    return rows


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--record":
        for row in record(sys.argv[2]):
            print(row["seed"], "base", row["base"], "head", row["head"], row["failed"])
    else:
        unittest.main()
