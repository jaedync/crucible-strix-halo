"""serve --draft-capture: counts only, or a token spool; never text, request IDs or times finer than a day."""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from server import draft_capture as dc
from server.leak_canary import CANARY

DAY = 1_760_400_000.0          # 2025-10-14 00:00 UTC
SECRET = [248045, 77, 88, 99, 123456, 248046]


def doc_numbers_only(value, key=None):
    """True when every leaf is a number, except the fixed day and granularity strings."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and doc_numbers_only(v, k) for k, v in value.items())
    if isinstance(value, list):
        return all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    if isinstance(value, str):
        return key in ("day", "granularity") or (key == "canary" and value == CANARY)   # the leak-guard marker
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CountsTest(unittest.TestCase):
    def test_counts_are_numbers_only_by_class_and_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "cap")
            cap = dc.DraftCapture(root, mode="counts", clock=lambda: DAY + 60)
            cap.record(tools=True, thinking=False, prompt_ids=SECRET, output_ids=[248058, 1, 129, 248059])
            cap.record(tools=False, thinking=True, prompt_ids=SECRET, output_ids=[5, 6])
            cap.close()
            files = [f for f in root.iterdir() if f.name not in (".lock", ".gitignore")]
            self.assertEqual([f.name for f in files], ["draft-counts-2025-10-14.json"])
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
            text = files[0].read_text()
            doc = json.loads(text)
        self.assertTrue(doc_numbers_only(doc))
        self.assertEqual(sorted(doc), ["block", "canary", "categories", "day", "granularity", "vocab"])
        tool = doc["categories"]["tool"]
        self.assertEqual((tool["sequences"], tool["tokens"]), (1, 4))
        self.assertEqual(len(tool["counts"]), 1940)
        self.assertEqual((tool["counts"][0], tool["counts"][1], tool["counts"][1937]), (1, 1, 2))
        self.assertEqual(doc["categories"]["chat+think"]["tokens"], 2)
        # Prompts are never counted, and no ID list from a request appears in the file.
        self.assertEqual(sum(sum(c["counts"]) for c in doc["categories"].values()), 6)
        self.assertNotIn("123456", text)

    def test_a_new_day_starts_a_new_file_and_a_restart_adds_to_the_day(self):
        now = [DAY + 10]
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: now[0])
                cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1, 2, 3])
                cap.close()
            now[0] = DAY + 86400 + 10
            cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: now[0])
            cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1])
            cap.close()
            days = {f.name: json.loads(f.read_text()) for f in Path(tmp).iterdir() if f.name not in (".lock", ".gitignore")}
        self.assertEqual(days["draft-counts-2025-10-14.json"]["categories"]["chat"]["tokens"], 6)
        self.assertEqual(days["draft-counts-2025-10-15.json"]["categories"]["chat"]["tokens"], 1)

    def test_bad_ids_and_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
            for bad in ([248320], [-1], [True], ["7"]):
                with self.assertRaises(ValueError):
                    cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=bad)
            with self.assertRaises(ValueError):
                dc.DraftCapture(Path("relative"), mode="counts")
            with self.assertRaises(ValueError):
                dc.DraftCapture(Path(tmp), mode="text")


class ReviewDcTest(unittest.TestCase):
    """The review conditions of the draft capture."""

    def test_a_bad_day_file_moves_aside_and_the_day_starts_empty(self):
        for bad in ("{not json", json.dumps({"day": "2025-10-14", "granularity": "block", "categories": {
                "chat": {"sequences": 1, "tokens": 1, "counts": [1, 2]}}}),
                json.dumps({"day": "2025-10-14", "granularity": "block", "vocab": 248320, "block": 128,
                            "categories": {"text": {"sequences": 1, "tokens": 1, "counts": [0] * 1940}}})):
            with tempfile.TemporaryDirectory() as tmp:
                os.chmod(tmp, 0o700)
                Path(tmp, "draft-counts-2025-10-14.json").write_text(bad)
                cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
                cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1])
                cap.close()
                doc = json.loads(Path(tmp, "draft-counts-2025-10-14.json").read_text())
                self.assertEqual(doc["categories"]["chat"]["tokens"], 1)
                aside = Path(tmp, "draft-counts-2025-10-14.json.bad").read_text()
                self.assertIn(CANARY, aside)      # H16: the start of the server marks every old file, a .bad file too
                try:
                    kept = json.loads(aside)
                    kept.pop("canary")
                    self.assertEqual(kept, json.loads(bad))
                except ValueError:
                    self.assertTrue(aside.startswith(bad))

    def test_without_canary_drops_only_the_canary_key(self):
        """The reader and the daily-switch rehearsal both read a day file through without_canary, then _valid_counts."""
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: DAY + 60)
            cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1])
            cap.close()
            doc = json.loads(Path(tmp, "cap", "draft-counts-2025-10-14.json").read_text())
        self.assertIn("canary", doc)
        self.assertFalse(dc._valid_counts(doc, "2025-10-14"))        # the strict shape check does not take the key
        body = dc.without_canary(doc)
        self.assertIn("canary", doc)                                 # the input is not changed
        self.assertNotIn("canary", body)
        self.assertTrue(dc._valid_counts(body, "2025-10-14"))
        self.assertFalse(dc._valid_counts(dc.without_canary(dict(doc, extra=1)), "2025-10-14"))
        self.assertEqual(dc.without_canary([1]), [1])
        self.assertIsNone(dc.without_canary(None))

    def test_one_writer_per_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
            with self.assertRaises(RuntimeError):
                dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
            first.close()
            dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY).close()

    def test_unsafe_directories_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            open_dir = Path(tmp, "open")
            open_dir.mkdir(mode=0o755)
            os.chmod(open_dir, 0o750)
            with self.assertRaises(ValueError):
                dc.DraftCapture(open_dir, mode="counts")
            private = Path(tmp, "private")
            private.mkdir(mode=0o700)
            link = Path(tmp, "link")
            link.symlink_to(private)
            with self.assertRaises(ValueError):
                dc.DraftCapture(link, mode="counts")
            with patch.object(dc.os, "geteuid", return_value=os.geteuid() + 1):
                with self.assertRaises(ValueError):
                    dc.DraftCapture(private, mode="counts")

    def test_file_times_are_the_day_start_and_the_write_is_synced(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY + 3 * 3600 + 17)
            cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1])
            with patch.object(dc.os, "fsync", wraps=os.fsync) as sync:
                cap.close()
            self.assertTrue(sync.called)
            info = Path(tmp, "draft-counts-2025-10-14.json").stat()
            self.assertEqual((info.st_mtime, info.st_atime), (DAY, DAY))
            self.assertEqual([p.name for p in Path(tmp).iterdir() if not p.name.startswith(".")],
                             ["draft-counts-2025-10-14.json"])

    def test_spool_times_are_the_day_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp), mode="tokens", clock=lambda: DAY + 7000)
            cap.record(tools=False, thinking=False, prompt_ids=[1], output_ids=[2])
            cap.close()
            self.assertEqual(Path(tmp, "draft-tokens-2025-10-14.bin").stat().st_mtime, DAY)

    def test_counts_mode_never_reads_the_prompt(self):
        class Untouchable:
            def __iter__(self):
                raise AssertionError("counts mode read the prompt")
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
            cap.record(tools=False, thinking=False, prompt_ids=Untouchable(), output_ids=[3])
            cap.close()

    def test_close_flushes_before_the_backend_closes(self):
        import server.__main__ as startup
        order = []
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        backend = MagicMock()
        backend.close.side_effect = lambda: order.append("backend")
        with tempfile.TemporaryDirectory() as tmp:
            real = dc.DraftCapture.close
            def close(self):
                order.append("capture")
                real(self)
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", "--draft-capture", tmp]), \
                    patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                    patch("server.batch_backend.BatchedEngineBackend", return_value=backend), \
                    patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                    patch.object(startup, "Service"), patch.object(dc.DraftCapture, "close", close):
                self.assertEqual(startup.main(), 0)
        self.assertEqual(order, ["capture", "backend"])

    def test_a_held_directory_stops_the_start(self):
        import server.__main__ as startup
        with tempfile.TemporaryDirectory() as tmp:
            held = dc.DraftCapture(Path(tmp), mode="counts")
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", "--draft-capture", tmp]), \
                    patch.object(startup, "ModelTokenizer") as load:
                self.assertEqual(startup.main(), 2)
            load.assert_not_called()
            held.close()


class TokensModeGuardTest(unittest.TestCase):
    def test_only_the_exact_tokens_value_writes_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                dc.DraftCapture(Path(tmp))                     # no default mode
            for bad in ("token", "Tokens", "tokens ", "", None, True):
                with self.assertRaises(ValueError):
                    dc.DraftCapture(Path(tmp), mode=bad)
            cap = dc.DraftCapture(Path(tmp), mode="counts", clock=lambda: DAY)
            with self.assertRaises(RuntimeError):
                cap._spool("2025-10-14", 0, [1], [2])

    def test_the_server_default_never_writes_token_ids(self):
        import server.__main__ as startup
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        prompt, output = [111111, 222222, 7], [123457, 234567, 12]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", "--draft-capture", tmp]), \
                    patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                    patch("server.batch_backend.BatchedEngineBackend"), \
                    patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                    patch.object(startup, "Service") as service:
                self.assertEqual(startup.main(), 0)
            capture = service.call_args.kwargs["capture"]
            self.assertEqual(capture.mode, "counts")
            capture.record(tools=True, thinking=True, prompt_ids=prompt, output_ids=output)
            capture.close()
            files = sorted(p.name for p in Path(tmp).iterdir() if p.name not in (".lock", ".gitignore"))
            data = b"".join(p.read_bytes() for p in Path(tmp).iterdir() if p.name != ".gitignore")
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("draft-counts-") and files[0].endswith(".json"))
        for token in prompt[:2] + output[:2]:
            self.assertNotIn(str(token).encode(), data)
        import struct
        self.assertNotIn(struct.pack("<I", output[0]), data)


class TokensTest(unittest.TestCase):
    def test_spool_holds_ids_only_with_a_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "spool")
            cap = dc.DraftCapture(root, mode="tokens", max_bytes=200, clock=lambda: DAY + 5)
            cap.record(tools=True, thinking=True, prompt_ids=SECRET, output_ids=[1, 2])
            cap.record(tools=False, thinking=False, prompt_ids=[3] * 60, output_ids=[4])  # over the cap: dropped
            cap.close()
            files = [f for f in root.iterdir() if f.name not in (".lock", ".gitignore")]
            self.assertEqual([f.name for f in files], ["draft-tokens-2025-10-14.bin"])
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
            data = files[0].read_bytes()
            records = list(dc.read_spool(files[0]))
        # Header, one record header and 8 uint32 IDs: nothing else is in the file.
        self.assertEqual(len(data), len(dc.SPOOL_HEADER) + dc.RECORD.size + 4 * 8)
        self.assertEqual(records, [("tool+think", SECRET, [1, 2])])
        self.assertEqual(cap.dropped, 1)

    def test_cap_counts_every_spool_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "draft-tokens-2025-10-13.bin").write_bytes(b"x" * 500)
            cap = dc.DraftCapture(Path(tmp), mode="tokens", max_bytes=600, clock=lambda: DAY)
            cap.record(tools=False, thinking=False, prompt_ids=[1] * 40, output_ids=[2])
            cap.close()
            self.assertEqual(cap.dropped, 1)


class ServeWiringTest(unittest.TestCase):
    def test_turn_class_from_the_request(self):
        seen = []
        capture = SimpleNamespace(record=lambda **k: seen.append(k))
        from server.http import Service
        service = Service(MagicMock(), MagicMock(), capture=capture)
        service.capture_turn(SimpleNamespace(tools=({"type": "function"},), enable_thinking=True), [1], [2])
        service.capture_turn(SimpleNamespace(tools=(), enable_thinking=False), [1], [2])
        self.assertEqual([(k["tools"], k["thinking"]) for k in seen], [(True, True), (False, False)])
        Service(MagicMock(), MagicMock()).capture_turn(SimpleNamespace(tools=(), enable_thinking=False), [1], [2])

    def test_a_capture_fault_never_fails_the_turn(self):
        from server.http import Service

        def broken(**kwargs):
            raise OSError("disk full at /secret/path")
        service = Service(MagicMock(), MagicMock(), capture=SimpleNamespace(record=broken))
        with patch("sys.stderr") as err:
            service.capture_turn(SimpleNamespace(tools=(), enable_thinking=False), [1], [2])
        written = "".join(call.args[0] for call in err.write.call_args_list)
        self.assertIn("draft capture: turn not recorded (OSError)", written)
        self.assertNotIn("secret", written)

    def test_flag_builds_the_capture_and_is_off_by_default(self):
        import server.__main__ as startup
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt

        def run(argv):
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", *argv]), \
                    patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                    patch("server.batch_backend.BatchedEngineBackend"), \
                    patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                    patch.object(startup, "Service") as service:
                self.assertEqual(startup.main(), 0)
            return service.call_args.kwargs.get("capture")

        self.assertIsNone(run([]))
        with tempfile.TemporaryDirectory() as tmp:
            capture = run(["--draft-capture", tmp])
            self.assertEqual(capture.mode, "counts")
            for bad in (["--draft-capture", "rel"], ["--draft-capture-mode", "tokens"],
                        ["--draft-capture", tmp, "--draft-capture-mode", "text"]):
                with patch.object(sys, "argv", ["crucible", "checkpoint", *bad]), \
                        patch.object(startup, "ModelTokenizer") as load, self.assertRaises(SystemExit):
                    startup.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
