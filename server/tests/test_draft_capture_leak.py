"""Leak guard, WP part 1: each file that the server writes holds the canary, and no folder of recordings sits inside a
checkout of a repository."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from server import draft_capture as dc
from server.leak_canary import CANARY, CANARY_BYTES

DAY = 1_760_400_000.0          # 2025-10-14 00:00 UTC


class CanaryInEveryFileTest(unittest.TestCase):
    def files(self, root):
        return sorted(p for p in Path(root).iterdir() if p.is_file())

    def test_counts_mode_files_carry_the_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="counts", clock=lambda: DAY + 5, draft_stats=True)
            cap.record(tools=False, thinking=False, prompt_ids=[1], output_ids=[2, 3])
            cap.close()
            files = self.files(Path(tmp, "cap"))
            self.assertGreaterEqual(len(files), 2)
            for path in files:
                self.assertIn(CANARY_BYTES, path.read_bytes(), path.name)

    def test_tokens_mode_files_carry_the_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap = dc.DraftCapture(Path(tmp, "cap"), mode="tokens", clock=lambda: DAY + 5)
            cap.record(tools=True, thinking=False, prompt_ids=[1, 2], output_ids=[3])
            cap.record(tools=False, thinking=False, prompt_ids=[4], output_ids=[5, 6])
            cap.close()
            files = self.files(Path(tmp, "cap"))
            self.assertTrue(any(p.name.startswith("draft-tokens-") for p in files))
            for path in files:
                self.assertIn(CANARY_BYTES, path.read_bytes(), path.name)
            spool = Path(tmp, "cap", "draft-tokens-2025-10-14.bin")
            self.assertEqual(list(dc.read_spool(spool)),
                             [("tool", [1, 2], [3]), ("chat", [4], [5, 6])])

    def test_a_counts_file_has_a_canary_key_and_a_day_file_of_an_old_server_still_loads(self):
        chat = {"sequences": 1, "tokens": 2, "counts": [0] * 1940}     # a day file of an older server: no canary key
        old = {"vocab": 248320, "block": 128, "day": "2025-10-14", "granularity": "block"}
        old["categories"] = {"chat": chat}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "cap")
            root.mkdir(mode=0o700)
            Path(root, "draft-counts-2025-10-14.json").write_text(json.dumps(old))
            cap = dc.DraftCapture(root, mode="counts", clock=lambda: DAY + 5)
            cap.record(tools=False, thinking=False, prompt_ids=[], output_ids=[1])
            cap.close()
            doc = json.loads(Path(root, "draft-counts-2025-10-14.json").read_text())
        self.assertEqual(doc["canary"], CANARY)
        self.assertEqual(doc["categories"]["chat"]["tokens"], 3)
        self.assertFalse(Path(root.parent, "cap", "draft-counts-2025-10-14.json.bad").exists())

    def test_an_old_spool_without_the_canary_reads_and_is_upgraded_on_the_next_turn(self):
        record = struct.pack("<BII", 0, 1, 1) + struct.pack("<2I", 7, 8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "cap")
            root.mkdir(mode=0o700)
            spool = Path(root, "draft-tokens-2025-10-14.bin")
            spool.write_bytes(b"DCT1" + record)
            os.chmod(spool, 0o600)
            self.assertEqual(list(dc.read_spool(spool)), [("chat", [7], [8])])
            cap = dc.DraftCapture(root, mode="tokens", clock=lambda: DAY + 5)
            cap.record(tools=False, thinking=False, prompt_ids=[9], output_ids=[10])
            cap.close()
            self.assertIn(CANARY_BYTES, spool.read_bytes())
            self.assertEqual(list(dc.read_spool(spool)), [("chat", [7], [8]), ("chat", [9], [10])])
            self.assertEqual(oct(spool.stat().st_mode & 0o777), "0o600")

    def test_a_spool_with_a_wrong_canary_is_refused(self):
        # A full spool body: only the wrong canary can make this read fail.
        record = struct.pack("<BII", 0, 1, 1) + struct.pack("<2I", 7, 8)
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp, "s.bin")
            spool.write_bytes(b"DCT2" + b"x" * len(CANARY_BYTES) + record)
            with self.assertRaises(ValueError):
                list(dc.read_spool(spool))


class NotInAGitTreeTest(unittest.TestCase):
    def test_a_directory_inside_a_git_work_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "repo", ".git").mkdir(parents=True)
            for sub in ("cap", "a/b/cap", "."):
                target = Path(tmp, "repo", sub)
                with self.assertRaises(ValueError) as caught:
                    dc.DraftCapture(target, mode="counts")
                self.assertIn("git", str(caught.exception).lower())
            # The repository still holds only its own .git folder.
            self.assertEqual(sorted(p.name for p in Path(tmp, "repo").iterdir()), [".git"])

    def test_a_git_file_of_a_linked_work_tree_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "wt").mkdir()
            Path(tmp, "wt", ".git").write_text("gitdir: /elsewhere\n")
            with self.assertRaises(ValueError):
                dc.DraftCapture(Path(tmp, "wt", "deep", "cap"), mode="tokens")
            self.assertFalse(Path(tmp, "wt", "deep").exists())

    def test_a_symlinked_parent_into_a_git_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "repo", ".git").mkdir(parents=True)
            Path(tmp, "link").symlink_to(Path(tmp, "repo"))
            with self.assertRaises(ValueError):
                dc.DraftCapture(Path(tmp, "link", "cap"), mode="counts")

    def test_a_directory_outside_any_git_tree_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            dc.DraftCapture(Path(tmp, "plain", "cap"), mode="counts").close()

    def test_the_server_stops_with_a_friendly_message(self):
        import server.__main__ as startup
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "repo", ".git").mkdir(parents=True)
            target = Path(tmp, "repo", "cap")
            err = io.StringIO()
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--streams", "2", "--draft-capture", str(target)]), \
                    patch.object(startup, "ModelTokenizer") as load, redirect_stderr(err):
                self.assertEqual(startup.main(), 2)
            load.assert_not_called()
            text = err.getvalue()
            self.assertIn("git", text)
            self.assertIn("outside", text)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
