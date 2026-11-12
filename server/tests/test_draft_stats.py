"""serve --draft-stats: counts of draft tokens by position, turn class and rejected target; nothing else."""
import json
import os
import sys
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from server import draft_capture as dc
from server import draft_stats as ds
from server.leak_canary import CANARY
from server import metrics

DAY = 1_760_400_000.0          # 2025-10-14 00:00 UTC


def event(rows=(), misses=(), pos=None, hist=None, tokens=40, cut=0, lookup=0):
    pos = pos or dict(proposed=[11, 9, 7, 5, 4, 2, 2], tested=[11, 9, 6, 4, 2, 1, 0], accepted=[9, 6, 4, 2, 1, 0, 0])
    return dict(event="done", completion_tokens=tokens, draft_tally=[list(r) for r in rows],
                draft_miss=[list(m) for m in misses], draft_detail_cut=cut, draft_lookup_rounds=lookup,
                draft_pos_proposed=pos["proposed"], draft_pos_tested=pos["tested"], draft_pos_accepted=pos["accepted"],
                draft_len_hist=hist or [0, 2, 2, 2, 1, 2, 0, 2])


def make(tmp, now=DAY + 5):
    return dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: now, draft_stats=True)


def read(tmp):
    return json.loads(Path(tmp, "cap", "draft-stats-2025-10-14.json").read_text())


class ParseTest(unittest.TestCase):
    def test_a_valid_event_parses_and_every_bad_field_drops_it(self):
        self.assertEqual(ds.parse_detail(event([(70, 1, 3, 2)], [(70, 2, 99, 1)]))["rows"], [[70, 1, 3, 2]])
        bad = [dict(event(), draft_tally=[[248320, 1, 1, 1]]), dict(event(), draft_tally=[[1, 0, 1, 1]]),
               dict(event(), draft_tally=[[1, 8, 1, 1]]), dict(event(), draft_tally=[[1, 1, 1, 2]]),
               dict(event(), draft_tally=[[1, 1, -1, 0]]), dict(event(), draft_tally=[[True, 1, 1, 1]]),
               dict(event(), draft_miss=[[1, 1, 248320, 1]]), dict(event(), draft_miss=[[1, 1, 2, 0]]),
               dict(event(), draft_tally=[[1, 1, 1]]), dict(event(), draft_tally="x"),
               dict(event(), draft_tally=[[1, 1, 1, 1]] * (ds.ROWS_MAX + 1)),
               dict(event(), draft_len_hist=[1] * 7), dict(event(), draft_detail_cut=-1),
               dict(event(), completion_tokens="4"), {k: v for k, v in event().items() if k != "draft_miss"}]
        for item in bad:
            self.assertIsNone(ds.parse_detail(item), item)
        self.assertIsNone(ds.parse_detail(None))


class FileTest(unittest.TestCase):
    def test_counts_by_class_position_token_and_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = make(tmp)
            cap.record_draft(tools=True, thinking=False, events=[event([(70, 1, 3, 2), (71, 2, 1, 0)], [(71, 2, 99, 1)])])
            cap.record_draft(tools=True, thinking=False, events=[event([(70, 1, 1, 1)], [(70, 3, 5, 2)], cut=1, lookup=3)])
            cap.record_draft(tools=False, thinking=True, events=[event()])
            cap.close()
            doc = read(tmp)
            mode = stat.S_IMODE(Path(tmp, "cap", "draft-stats-2025-10-14.json").stat().st_mode)
            self.assertEqual(Path(tmp, "cap", "draft-stats-2025-10-14.json").stat().st_mtime, DAY)
        self.assertEqual(mode, 0o600)
        self.assertEqual((doc["version"], doc["k"], doc["vocab"], doc["block"]), (1, 7, 248320, 128))
        self.assertEqual(sorted(doc["classes"]), ["chat+think", "tool"])
        tool = doc["classes"]["tool"]
        self.assertEqual((tool["turns"], tool["rounds"], tool["tokens"], tool["lookup_rounds"], tool["detail_cut"]),
                         (2, 22, 80, 3, 1))
        self.assertEqual(tool["pos"]["proposed"], [22, 18, 14, 10, 8, 4, 4])
        self.assertEqual(tool["len_hist"], [0, 4, 4, 4, 2, 4, 0, 4])
        self.assertEqual(tool["tok"]["70"], [4, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0])
        self.assertEqual(tool["tok"]["71"], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(tool["miss"], [[70, 5, 0, 0, 2, 0, 0, 0, 0], [71, 99, 0, 1, 0, 0, 0, 0, 0]])
        self.assertEqual(tool["overflow"], [0] * 7)

    def test_the_file_has_counts_only_and_no_list_grows_with_the_request(self):
        def numbers(value, key=None):
            if isinstance(value, dict):
                return all(isinstance(k, str) and numbers(v, k) for k, v in value.items())
            if isinstance(value, list):
                return all(numbers(v, key) for v in value)
            if isinstance(value, str):
                return key == "day" or (key == "canary" and value == CANARY)   # the leak-guard marker is the one string
            return type(value) is int
        outputs = []
        for tokens in (40, 40000):  # a short and a long request with the same counts give the same file
            with tempfile.TemporaryDirectory() as tmp:
                cap = make(tmp)
                cap.record_draft(tools=True, thinking=False, events=[event([(70, 1, 3, 2)], [(70, 2, 9, 1)], tokens=tokens)])
                cap.close()
                doc = read(tmp)
                self.assertTrue(numbers(doc))
                outputs.append(doc["classes"]["tool"].pop("tokens"))  # the one scalar that holds the request length
                outputs.append(doc)
        self.assertEqual(outputs[0::2], [40, 40000])
        self.assertEqual(outputs[1], outputs[3])
        two = event([(70, 1, 3, 2)], [(70, 2, 9, 1)])
        with tempfile.TemporaryDirectory() as tmp:  # two requests are the sum of the two: they cannot be told apart
            cap = make(tmp)
            cap.record_draft(tools=True, thinking=False, events=[two])
            cap.record_draft(tools=True, thinking=False, events=[two])
            cap.close()
            self.assertEqual(read(tmp)["classes"]["tool"]["tok"]["70"][0], 6)

    def test_a_file_that_is_not_utf8_moves_to_bad_and_the_turn_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = make(tmp)
            Path(tmp, "cap", "draft-stats-2025-10-14.json").write_bytes(b"\xff")
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 3, 2)])])
            cap.close()
            self.assertEqual(read(tmp)["classes"]["chat"]["tok"]["70"][0], 3)
            self.assertEqual(Path(tmp, "cap", "draft-stats-2025-10-14.json.bad").read_bytes(), b"\xff")

    def test_a_failed_write_is_not_retried_at_every_turn(self):
        now = [DAY + 5]
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: now[0], draft_stats=True)
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            now[0] += 700
            with patch("os.replace", side_effect=OSError("disk full")) as replace:
                with self.assertRaises(OSError):
                    cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
                now[0] += 5
                cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            self.assertEqual(replace.call_count, 1)
            cap.close()

    def test_a_restart_adds_to_the_day_and_a_bad_file_moves_to_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                cap = make(tmp)
                cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 3, 2)])])
                cap.close()
            self.assertEqual(read(tmp)["classes"]["chat"]["tok"]["70"][0], 6)
            for text in ("not json", json.dumps(dict(read(tmp), version=2)),
                         json.dumps(dict(read(tmp), classes={"chat": {"turns": -1}}))):
                path = Path(tmp, "cap", "draft-stats-2025-10-14.json")
                path.write_text(text)
                cap = make(tmp)
                cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 3, 2)])])
                cap.close()
                self.assertEqual(read(tmp)["classes"]["chat"]["tok"]["70"][0], 3)
                self.assertTrue(Path(tmp, "cap", "draft-stats-2025-10-14.json.bad").exists())

    def test_a_new_day_starts_a_new_file(self):
        now = [DAY + 5]
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: now[0], draft_stats=True)
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            now[0] = DAY + 86400 + 5
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            cap.close()
            self.assertEqual(sorted(p.name for p in Path(tmp, "cap").glob("draft-stats-*")),
                             ["draft-stats-2025-10-14.json", "draft-stats-2025-10-15.json"])

    def test_without_the_flag_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: DAY)
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            cap.close()
            self.assertEqual(list(Path(tmp, "cap").glob("draft-stats-*")), [])

    def test_the_tables_are_bounded_and_the_overflow_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = make(tmp)
            with patch.object(ds, "MISS_PAIRS_MAX", 2), patch.object(ds, "TOKEN_IDS_MAX", 2):
                cap.record_draft(tools=True, thinking=False, events=[event(
                    [(1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1)], [(1, 1, 9, 1), (2, 2, 9, 1), (3, 3, 9, 4), (1, 1, 9, 2)])])
            cap.close()
            tool = read(tmp)["classes"]["tool"]
        self.assertEqual(sorted(tool["tok"]), ["1", "2"])
        self.assertEqual(tool["detail_cut"], 1)
        self.assertEqual(len(tool["miss"]), 2)
        self.assertEqual(tool["overflow"], [0, 0, 4, 0, 0, 0, 0])
        self.assertEqual(tool["miss"][0][2], 3)  # a known pair still adds after the limit

    def test_the_writer_syncs_before_the_rename_and_flushes_on_close(self):
        order = []
        real_fsync, real_replace = os.fsync, os.replace
        with tempfile.TemporaryDirectory() as tmp, \
                patch("os.fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1]), \
                patch("os.replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1]):
            cap = make(tmp, now=DAY + 5)
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            order.clear()
            cap.stats.last_write = cap.clock()  # inside the interval: record does not write, close does
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            self.assertEqual(order, [])
            cap.close()
            self.assertEqual(order[:2], ["fsync", "replace"])
            self.assertEqual(read(tmp)["classes"]["chat"]["tok"]["70"][0], 2)

    def test_a_failed_write_leaves_no_temp_file_and_closing_still_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = make(tmp)
            cap.record_draft(tools=False, thinking=False, events=[event([(70, 1, 1, 1)])])
            with patch("os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    cap.stats.flush()
                with self.assertRaises(OSError):
                    cap.close()
            self.assertEqual([p.name for p in Path(tmp, "cap").iterdir() if p.name.endswith(".tmp")], [])
            make(tmp).close()  # the lock is free again

    def test_one_server_per_directory_and_only_the_counts_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make(tmp)
            with self.assertRaises(RuntimeError):
                make(tmp)
            first.close()


class FlagTest(unittest.TestCase):
    def start(self, *args):
        import server.__main__ as startup
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", *args]), \
                patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                patch("server.batch_backend.BatchedEngineBackend") as backend, \
                patch.object(startup, "ThreadingHTTPServer", return_value=listener), patch.object(startup, "Service") as service:
            return startup.main(), backend, service

    def test_the_flag_needs_the_counts_capture_and_the_batch_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (["--draft-stats"], ["--draft-stats", "--draft-capture", tmp, "--draft-capture-mode", "tokens"],
                         ["--draft-stats", "--draft-capture", tmp, "--serial"]):
                with self.assertRaises(SystemExit) as stop:
                    self.start(*args)
                self.assertEqual(stop.exception.code, 2, args)

    def test_the_flag_turns_on_the_file_and_the_engine_request_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, backend, service = self.start("--draft-stats", "--draft-capture", tmp)
            capture = service.call_args.kwargs["capture"]
            self.assertEqual((code, backend.return_value.draft_detail), (0, True))
            self.assertIsNotNone(capture.stats)
            capture.close()
            code, backend, service = self.start("--draft-capture", tmp)   # default off
            self.assertEqual((code, backend.return_value.draft_detail), (0, False))
            self.assertIsNone(service.call_args.kwargs["capture"].stats)
            service.call_args.kwargs["capture"].close()


class FieldsTest(unittest.TestCase):
    EVENT = dict(spec_steps=11, draft_proposed=40, draft_accepted=22, draft_passes=14, draft_ms=21.5,
                 draft_pos_proposed=[11, 9, 7, 5, 4, 2, 2], draft_pos_tested=[11, 9, 6, 4, 2, 1, 0],
                 draft_pos_accepted=[9, 6, 4, 2, 1, 0, 0], draft_len_hist=[0, 2, 2, 2, 1, 2, 0, 2])

    def line(self, *events):
        turn = metrics.new_turn(1)
        fields = metrics.draft_fields(*[(turn, e) for e in events])
        return fields

    def test_the_turn_line_gets_the_position_fields_and_they_keep_their_invariants(self):
        f = self.line(self.EVENT)
        self.assertEqual((f["draft_passes"], f["draft_ms"]), (14, 21.5))
        self.assertEqual(sum(f["draft_pos_proposed"]), f["draft_proposed"])
        self.assertEqual(sum(f["draft_pos_accepted"]), f["draft_accepted"])
        self.assertEqual(sum(f["draft_len_hist"]), f["spec_rounds"])
        for p, t, a in zip(f["draft_pos_proposed"], f["draft_pos_tested"], f["draft_pos_accepted"]):
            self.assertTrue(a <= t <= p)

    def test_two_jobs_add_and_a_bad_job_drops_the_new_fields_only(self):
        f = self.line(self.EVENT, self.EVENT)
        self.assertEqual((f["draft_pos_proposed"][0], f["draft_len_hist"][7], f["draft_passes"], f["draft_ms"]), (22, 4, 28, 43.0))
        for key, value in (("draft_pos_tested", [1] * 6), ("draft_len_hist", "x"), ("draft_pos_accepted", [1] * 6 + [-1]),
                           ("draft_passes", True), ("draft_ms", float("nan")), ("draft_ms", None)):
            f = self.line(dict(self.EVENT, **{key: value}))
            self.assertEqual((f["draft_proposed"], f["spec_rounds"]), (40, 11))
            self.assertNotIn("draft_pos_proposed", f)
            self.assertNotIn("draft_ms", f)
        f = self.line({k: self.EVENT[k] for k in ("spec_steps", "draft_proposed", "draft_accepted")})
        self.assertEqual(f["draft_accepted"], 22)  # an older engine: only the three scalars

    def test_the_line_serialises(self):
        turn = metrics.new_turn(2)
        done = dict(completion_tokens=34, **self.EVENT)
        line = metrics.turn_line(metrics.finalize(turn, done), done, metrics.draft_fields((turn, done)))
        fields = json.loads(line.split(" ", 1)[1])
        self.assertEqual(fields["draft_pos_accepted"], [9, 6, 4, 2, 1, 0, 0])
        self.assertEqual(fields["draft_len_hist"], [0, 2, 2, 2, 1, 2, 0, 2])


if __name__ == "__main__":
    unittest.main()
