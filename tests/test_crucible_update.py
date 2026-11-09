"""Tests of `./crucible update` and `./crucible uninstall` (wp-t5b-update). A fake release folder, a local git repo, a fake
systemd and a stub ./crucible in each tree. No network, no GPU, no real systemd, no sudo.

Exit codes of update: 0 updated or up to date, 1 failed before any change, 2 usage, 4 rolled back (the old release runs),
5 the rollback failed. A failed download, a bad sha256 or a partial file changes nothing: the old tree is byte-equal to what it
was, the service was not touched, and nothing from the download ran.
"""
import errno
import fcntl
import gzip
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from crucible_update_fakes import CANARY, EMAIL, GIT, GIT_ENV, NEW, OLD, STEM, World, git, sha256_bytes, sha256_file, snapshot  # noqa: E402

from crucible_cli import cli, release, update  # noqa: E402


class Crash(BaseException):
    """A kill: it is not an Exception, so no cleanup handler of the update may catch it."""


class Base(unittest.TestCase):
    out = ""

    def world(self, **kw):
        self.w = World(self, **kw)
        self.before = snapshot(self.w.dir)
        return self.w

    def update(self, *args, answers=None, tty=True, w=None, **machine):
        w = w or self.w
        host = w.host(answers=answers, tty=tty, **machine)
        code = cli.main(["update", *args], host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        self.host = host
        return code

    def assertUntouched(self, msg=""):
        """No change: the tree is byte-equal, the service was not touched, nothing ran from the download, nothing is left."""
        self.assertEqual(snapshot(self.w.dir), self.before, msg or self.out)
        self.assertEqual(git(self.w.dir, "rev-parse", "HEAD"), self.w.old_commit)
        stops = [v for v in self.w.machine.verbs if v in ("stop", "start", "restart")]
        self.assertEqual(stops, [], f"the service was touched: {stops}")
        self.assertEqual(self.w.stub_calls(), [], "code of the new release ran")
        self.assertEqual(self.w.leftovers(), [])
        self.assertEqual(self.w.unit.read_text(), "unit 0.1.0\n")

    def assertOldRelease(self):
        self.assertEqual((self.w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertEqual(git(self.w.dir, "rev-parse", "HEAD"), self.w.old_commit)
        self.assertEqual(self.w.unit.read_text(), "unit 0.1.0\n")
        self.assertEqual((self.w.dir / ".crucible/serve.sh").read_text(), "serve 0.1.0\n")
        self.assertEqual(sha256_file(self.w.dir / "build/engine/crucible"), self.w.old_engine_sha)
        self.assertEqual((self.w.pack / "weights.safetensors").read_text(), "WEIGHTS\n")
        self.assertEqual(self.w.leftovers(), [])


class UpToDateTest(Base):
    def test_same_release_is_up_to_date_and_changes_nothing(self):
        w = self.world()
        git(w.src, "tag", "-d", NEW)
        w.make_release(tag=OLD, commit=w.old_src_commit)
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertIn("up to date", self.out)
        self.assertIn(OLD, self.out)
        self.assertUntouched()

    def test_a_checkout_ahead_of_the_newest_release_is_left_alone(self):
        w = self.world()
        git(w.dir, "config", "user.email", EMAIL)
        git(w.dir, "config", "user.name", "t")
        (w.dir / "extra.txt").write_text("local\n")
        git(w.dir, "add", "extra.txt")
        git(w.dir, "commit", "-q", "-m", "local")
        # the local commit is on top of the NEW release: ahead of the newest release
        git(w.dir, "fetch", "-q", "--tags", str(w.src))
        git(w.dir, "reset", "-q", "--hard", NEW)
        git(w.dir, "commit", "-q", "--allow-empty", "-m", "local on top")
        self.before = snapshot(w.dir)
        w.old_commit = git(w.dir, "rev-parse", "HEAD")
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertRegex(self.out, "ahead|up to date")
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])

    def test_check_reports_a_new_release_and_changes_nothing(self):
        self.world()
        self.assertEqual(self.update("--check"), 0, self.out)
        self.assertIn(NEW, self.out)
        self.assertRegex(self.out, "(?i)update available|newer release")
        self.assertUntouched()

    def test_a_held_release_is_skipped_until_you_name_it(self):
        w = self.world()
        (w.dir / ".crucible/update-hold").write_text(NEW + "\n")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertIn("hold", self.out)
        self.assertUntouched()
        self.assertEqual(self.update("--yes", "--tag", NEW), 0, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")


class UpdateSuccessTest(Base):
    def test_updates_a_release_install(self):
        w = self.world()
        code = self.update("--yes")
        self.assertEqual(code, 0, self.out)
        d = w.dir
        self.assertEqual((d / "VERSION").read_text(), "0.2.0\n")
        self.assertEqual(git(d, "rev-parse", "HEAD"), w.new_commit)
        self.assertEqual(git(d, "describe", "--tags", "--exact-match"), NEW)
        self.assertEqual(sha256_file(d / "build/engine/crucible"), w.new_engine_sha)
        self.assertTrue(os.access(d / "build/engine/crucible", os.X_OK))
        rec = release.engine_release(d)
        self.assertIsNotNone(rec, "release.json must prove the engine")
        self.assertEqual((rec["tag"], rec["commit"]), (NEW, w.new_commit))
        self.assertTrue((d / ".git/crucible-install").is_file())

    def test_keeps_the_models_the_work_folder_and_the_venv(self):
        w = self.world()
        (w.dir / "notes.txt").write_text("my note\n")
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual((w.pack / "weights.safetensors").read_text(), "WEIGHTS\n")
        self.assertEqual((w.states / "conv.state").read_text(), "STATE\n")
        self.assertTrue((w.dir / ".venv/bin/python").is_file())
        self.assertEqual(json.loads((w.dir / ".crucible/config.json").read_text())["port"], 8790)
        self.assertEqual((w.dir / "notes.txt").read_text(), "my note\n")

    def test_the_new_code_writes_the_unit_and_the_service_restarts(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(w.unit.read_text(), "unit 0.2.0\n")
        self.assertEqual((w.dir / ".crucible/serve.sh").read_text(), "serve 0.2.0\n")
        events = w.machine.events
        stop = events.index(("ctl", "stop"))
        stub = [i for i, e in enumerate(events) if e[0] == "stub"][0]
        start = max(i for i, e in enumerate(events) if e[1] in ("start", "restart"))
        self.assertLess(stop, stub, "the new code runs after the service stopped")
        self.assertLess(stub, start)
        self.assertTrue(w.machine.active)
        self.assertIn("service install --no-start", w.stub_calls()[0])

    def test_a_smoke_request_proves_the_new_release(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(len(self.host.http.posts), 1)
        body = self.host.http.posts[0][1]
        self.assertEqual((body["max_tokens"], body["temperature"]), (8, 0))

    def test_leaves_no_work_folder_no_journal_and_no_old_tree(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(w.leftovers(), [])
        self.assertIn(NEW, self.out)

    def test_the_message_says_how_to_leave_the_old_folder(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertIn(f"cd {w.dir.resolve()}", self.out)

    def test_the_hold_file_is_removed_when_a_newer_release_installs(self):
        w = self.world()
        (w.dir / ".crucible/update-hold").write_text("v0.0.9\n")
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertFalse((w.dir / ".crucible/update-hold").exists())

    def test_asks_once_and_a_no_changes_nothing(self):
        self.world()
        self.assertEqual(self.update(answers=["n"]), 0, self.out)
        self.assertEqual(len(self.host.prompt.asked), 1)
        self.assertIn("Nothing was changed", self.out)
        self.assertUntouched()

    def test_a_yes_answer_updates(self):
        w = self.world()
        self.assertEqual(self.update(answers=["y"]), 0, self.out)
        self.assertEqual(len(self.host.prompt.asked), 1)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")

    def test_no_terminal_and_no_yes_is_a_usage_error(self):
        self.world()
        self.assertEqual(self.update(tty=False), 2, self.out)
        self.assertIn("--yes", self.out)
        self.assertUntouched()

    def test_a_service_that_was_stopped_stays_stopped(self):
        w = self.world()
        w.machine.active = False
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertFalse(w.machine.active)
        self.assertNotIn("start", w.machine.verbs)
        self.assertNotIn("restart", w.machine.verbs)
        self.assertEqual(w.unit.read_text(), "unit 0.2.0\n")
        self.assertEqual(self.host.http.posts, [])

    def test_a_server_started_by_crucible_start_is_stopped_and_started_again(self):
        w = self.world()
        w.unit.unlink()
        w.machine.unit_present = False
        w.machine.active = False
        w.machine.live.add(40001)
        (w.dir / ".crucible/server.pid").write_text("40001 0\n")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(w.machine.terminated, [40001])
        self.assertIn("start", [e[1] for e in w.machine.events if e[0] == "stub"])
        self.assertIn(40001, w.machine.live)
        self.assertFalse(w.unit.exists())

    def test_nothing_running_nothing_started(self):
        w = self.world()
        w.unit.unlink()
        w.machine.active = False
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(w.machine.verbs.count("start") + w.machine.verbs.count("restart"), 0)
        self.assertIn("not running", self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")

    def test_the_install_may_be_reached_through_a_symlink(self):
        w = self.world()
        link = w.t / "link"
        link.symlink_to(w.dir)
        host = w.host()
        self.assertEqual(cli.main(["update", "--yes"], host=host, repo=link), 0, "\n".join(host.lines))
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")
        self.assertTrue(link.is_symlink())

    def test_new_python_requirements_are_installed_before_the_service_stops(self):
        w = self.world(new_requirements="numpy\nrequests\n")
        self.assertEqual(self.update("--yes"), 0, self.out)
        events = w.machine.events
        pip = [i for i, e in enumerate(events) if e[0] == "pip" and e[1].startswith("install")]
        self.assertEqual(len(pip), 1)
        self.assertIn("-r", events[pip[0]][1])
        self.assertLess(pip[0], events.index(("ctl", "stop")), "pip runs while the old server still serves")

    def test_unchanged_requirements_run_no_pip(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual([e for e in w.machine.events if e[0] == "pip"], [])

    def test_a_pip_failure_changes_nothing_and_keeps_the_service(self):
        w = self.world(new_requirements="numpy\nrequests\n")
        w.machine.fail_pip = True
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)pip")
        self.assertUntouched()


class RollbackFlagTest(Base):
    """`--rollback` was the flag of tools/update.sh. It gets a clear answer and exit 2, not an argparse error."""

    def check_message(self, text):
        self.assertRegex(text, r"rolls? back by itself|rolled back by itself")
        self.assertIn("./crucible update --tag vX.Y.Z", text)

    def test_the_flag_gives_a_message_and_exit_2_and_changes_nothing(self):
        self.world()
        self.assertEqual(self.update("--rollback"), 2, self.out)
        self.check_message(self.out)
        self.assertUntouched()

    def test_the_flag_with_yes_or_check_gives_the_same_answer(self):
        self.world()
        for extra in (("--yes",), ("--check",)):
            with self.subTest(extra=extra):
                self.assertEqual(self.update("--rollback", *extra), 2, self.out)
                self.check_message(self.out)
        self.assertUntouched()

    def test_the_real_launcher_gives_the_message(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run([sys.executable, "-I", str(root / "crucible"), "update", "--rollback"], capture_output=True,
                              text=True, timeout=60, cwd=root)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("usage:", proc.stdout + proc.stderr)
        self.check_message(proc.stdout)


class DowngradeTest(Base):
    """A downgrade by name: --tag with an older release. Pinned, sha-checked, same rollback path as an update."""

    def installed_at_new(self):
        w = self.world()
        self.assertEqual(self.update("--yes"), 0, self.out)
        w.make_release(tag=OLD, commit=w.old_src_commit)
        w.machine.calls.clear()
        w.machine.events.clear()
        w.log.write_text("")
        self.before = snapshot(w.dir)
        w.old_commit = git(w.dir, "rev-parse", "HEAD")
        return w

    def test_an_older_tag_by_name_is_installed(self):
        w = self.installed_at_new()
        self.assertEqual(self.update("--yes", "--tag", OLD), 0, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertEqual(git(w.dir, "rev-parse", "HEAD"), w.old_src_commit)
        rec = release.engine_release(w.dir)
        self.assertEqual((rec["tag"], rec["commit"]), (OLD, w.old_src_commit))
        self.assertEqual(w.leftovers(), [])
        self.assertTrue(w.machine.active)

    def test_without_the_name_an_older_release_is_never_chosen(self):
        w = self.installed_at_new()
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertIn("up to date", self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")

    def test_a_downgrade_is_checked_like_an_update(self):
        w = self.installed_at_new()
        gz = w.asset("gz", OLD)
        data = bytearray(gz.read_bytes())
        data[-10] ^= 0xFF
        gz.write_bytes(bytes(data))
        self.assertEqual(self.update("--yes", "--tag", OLD), 1, self.out)
        self.assertRegex(self.out, r"does not match")
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(git(w.dir, "rev-parse", "HEAD"), w.new_commit)
        self.assertEqual(w.machine.verbs, [])
        self.assertEqual(w.stub_calls(), [])
        self.assertEqual(w.leftovers(), [])

    def test_a_downgrade_that_does_not_start_rolls_back(self):
        w = self.installed_at_new()
        (w.dir / "VERSION").write_text("0.2.0\n")   # same content; the old release is the one that runs now
        w.machine.starts = [False]
        self.assertEqual(self.update("--yes", "--tag", OLD), 4, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")
        self.assertEqual(git(w.dir, "rev-parse", "HEAD"), w.new_commit)
        self.assertTrue(w.machine.active)

    def test_a_checkout_that_is_not_a_release_is_not_downgraded(self):
        w = self.world()
        git(w.dir, "config", "user.email", EMAIL)
        git(w.dir, "config", "user.name", "t")
        git(w.dir, "fetch", "-q", "--tags", str(w.src))
        git(w.dir, "reset", "-q", "--hard", NEW)
        git(w.dir, "commit", "-q", "--allow-empty", "-m", "local on top")
        w.old_commit = git(w.dir, "rev-parse", "HEAD")
        w.make_release(tag=OLD, commit=w.old_src_commit)
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes", "--tag", OLD), 1, self.out)
        self.assertRegex(self.out, r"(?i)release line")
        self.assertEqual(snapshot(w.dir), self.before)


class BadReleaseTest(Base):
    """A release that fails a check changes nothing, and nothing from it runs."""

    def check_refused(self, pattern):
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, pattern)
        self.assertUntouched()

    def test_a_missing_engine_file(self):
        w = self.world()
        w.asset("gz").unlink()
        self.check_refused(r"(?s)--source-build")

    def test_a_missing_sums_file(self):
        w = self.world()
        (w.release / "SHA256SUMS").unlink()
        self.check_refused(r"(?i)SHA256SUMS")

    def test_a_bad_sha_of_the_engine_file(self):
        w = self.world()
        gz = w.asset("gz")
        data = bytearray(gz.read_bytes())
        data[-10] ^= 0xFF
        gz.write_bytes(bytes(data))
        self.check_refused(r"does not match")

    def test_a_partial_engine_file(self):
        w = self.world()
        gz = w.asset("gz")
        gz.write_bytes(gz.read_bytes()[: gz.stat().st_size // 2])
        self.check_refused(r"does not match")

    def test_a_partial_manifest(self):
        w = self.world()
        n = w.asset("needs")
        n.write_text(n.read_text()[:30])
        self.check_refused(r"does not match")

    def test_an_empty_download(self):
        w = self.world()
        w.asset("gz").write_bytes(b"")
        self.check_refused(r"does not match")

    def test_a_file_that_the_sums_do_not_list(self):
        w = self.world()
        w.write_sums(skip=(STEM.format(tag=NEW) + ".gz",))
        self.check_refused(r"not listed")

    def test_two_different_hashes_for_one_file(self):
        w = self.world()
        w.write_sums(extra=[f"{'0' * 64}  {STEM.format(tag=NEW)}.gz"])
        self.check_refused(r"(?i)different hashes")

    def test_a_malformed_hash_line(self):
        w = self.world()
        w.write_sums(skip=(STEM.format(tag=NEW) + ".gz",), extra=[f"zzzz  {STEM.format(tag=NEW)}.gz"])
        self.check_refused(r"(?i)bad line|not listed")

    def test_the_unpacked_engine_must_match_the_manifest(self):
        w = self.world()
        w.make_release(binary_sha="1" * 64)
        self.check_refused(r"(?i)unpacked engine")

    def test_the_manifest_commit_must_be_the_tag_commit(self):
        w = self.world()
        w.make_release(commit="2" * 40)
        self.check_refused(r"(?i)commit")

    def test_the_manifest_must_be_for_this_tag(self):
        w = self.world()
        w.make_release(tag=NEW)
        text = w.asset("needs").read_text().replace(f"tag {NEW}", "tag v9.9.9")
        w.asset("needs").write_text(text)
        w.write_sums()
        self.check_refused(r"(?i)another release")

    def test_the_manifest_must_be_for_this_gpu_and_rocm(self):
        for kw, pattern in (({"arch": "gfx1100"}, r"(?i)GPU"), ({"rocm": "6.4"}, r"(?i)ROCm")):
            with self.subTest(kw=kw):
                w = self.world()
                w.make_release(**kw)
                self.check_refused(pattern)

    def test_a_pinned_hash_of_the_sums_file(self):
        self.world()
        host = self.w.host(env={"CRUCIBLE_SUMS_SHA256": "3" * 64})
        code = cli.main(["update", "--yes"], host=host, repo=self.w.dir)
        self.out = "\n".join(host.lines)
        self.assertEqual(code, 1, self.out)
        self.assertRegex(self.out, r"(?i)pinned")
        self.assertUntouched()

    def test_a_missing_library_is_named_and_sudo_is_never_run(self):
        w = self.world()
        w.make_release(needs=("libnotthere.so.1",))
        self.check_refused(r"libnotthere\.so\.1")
        self.assertNotIn("sudo ", self.out)

    def test_a_glibc_that_is_too_old(self):
        w = self.world()
        w.make_release(glibc_min="99.0")
        self.check_refused(r"(?i)glibc")

    def test_a_tag_that_moves_during_the_run_is_refused(self):
        w = self.world()
        orig = update.crash_point

        def move(name):
            if name == "resolved":
                git(w.src, "tag", "-f", NEW, w.old_src_commit)
            return orig(name)
        with mock.patch.object(update, "crash_point", move):
            self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)moved|not the commit")
        self.assertUntouched()

    def test_a_branch_with_the_name_of_the_tag_is_not_used(self):
        w = self.world()
        git(w.src, "branch", NEW, w.old_src_commit)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)not the commit")
        self.assertUntouched()

    def test_the_engine_is_never_run(self):
        w = self.world()
        flag = w.t / "engine.ran"
        w.make_release(engine=f"#!/bin/sh\ntouch {flag}\n".encode())
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertFalse(flag.exists(), "update must not run the engine")

    def test_a_decompression_bomb_is_refused(self):
        w = self.world()
        bomb = gzip.compress(b"\0" * (1 << 20))
        w.make_release(engine=b"x")
        w.asset("gz").write_bytes(bomb)
        w.write_sums()
        with mock.patch.object(update, "MAX_ENGINE_BYTES", 1 << 10):
            self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)too (big|large)")
        self.assertUntouched()


class RefusalTest(Base):
    def test_local_changes_to_tracked_files_stop_the_update(self):
        w = self.world()
        (w.dir / "VERSION").write_text("hacked\n")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)local changes")
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])

    def test_a_commit_that_is_not_on_the_release_line_stops_the_update(self):
        w = self.world()
        git(w.dir, "config", "user.email", EMAIL)
        git(w.dir, "config", "user.name", "t")
        git(w.dir, "commit", "-q", "--allow-empty", "-m", "local work")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)release line|not on a release")
        self.assertEqual(w.machine.verbs, [])

    def test_a_folder_that_is_not_a_git_checkout(self):
        w = self.world()
        shutil_rm = __import__("shutil").rmtree
        shutil_rm(w.dir / ".git")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)git checkout")

    def test_an_old_untracked_name_that_the_new_release_tracks_is_refused(self):
        w = self.world()
        (w.src / "notes.txt").write_text("tracked now\n")
        git(w.src, "add", "-A")
        git(w.src, "commit", "-q", "-m", "tracks notes.txt")
        git(w.src, "tag", "-f", NEW)
        w.new_commit = git(w.src, "rev-parse", "HEAD")
        w.make_release()
        (w.dir / "notes.txt").write_text("my own file\n")
        self.before = snapshot(w.dir)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertIn("notes.txt", self.out)
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])
        self.assertEqual(w.leftovers(), [])

    def test_not_enough_disk_space(self):
        self.world()
        self.assertEqual(self.update("--yes", free=100 * 1024 * 1024), 1, self.out)
        self.assertRegex(self.out, r"(?i)disk")
        self.assertUntouched()

    def test_a_folder_that_cannot_be_written_next_to_the_install(self):
        w = self.world()
        os.chmod(w.parent, 0o555)
        self.addCleanup(os.chmod, w.parent, 0o755)
        if os.access(w.parent, os.W_OK):
            self.skipTest("running as a user that ignores the mode (root)")
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)next to")
        self.assertEqual(w.machine.verbs, [])

    def test_a_second_update_at_the_same_time_is_refused(self):
        w = self.world()
        fd = os.open(w.parent, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertIn("Another update is running", self.out)
        self.assertUntouched()

    def test_a_bad_tag_name_is_refused_before_anything_runs(self):
        self.world()
        for bad in ("../x", "v1.2", "main", "v1.2.3-rc1", "v1.2.3; rm", ""):
            with self.subTest(tag=bad):
                with self.assertRaises(SystemExit):
                    self.update("--yes", "--tag", bad)
        self.assertUntouched()

    def test_an_unknown_tag(self):
        self.world()
        self.assertEqual(self.update("--yes", "--tag", "v7.7.7"), 1, self.out)
        self.assertRegex(self.out, r"(?i)no release")
        self.assertUntouched()

    def test_a_release_base_must_be_https_or_file(self):
        w = self.world()
        host = w.host(env={"CRUCIBLE_RELEASE_BASE": "http://example.invalid/releases"})
        code = cli.main(["update", "--yes"], host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        self.assertEqual(code, 2, self.out)
        self.assertIn("https://", self.out)
        self.assertUntouched()


class SourceBuildTest(Base):
    def test_a_source_build_install_builds_in_the_staged_tree(self):
        w = self.world(release_engine=False)
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(len(w.machine.make_calls), 1)
        argv, cwd = w.machine.make_calls[0]
        self.assertNotEqual(Path(cwd).resolve(), w.dir.resolve(), "the build must not touch the live tree")
        self.assertEqual((w.dir / "build/engine/crucible").read_text(), "#!/bin/sh\necho built\n")
        self.assertFalse((w.dir / "build/engine/release.json").exists())
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")

    def test_a_failed_build_changes_nothing(self):
        w = self.world(release_engine=False)
        w.machine.fail_make = True
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)build")
        self.assertUntouched()

    def test_source_build_flag_builds_even_for_a_release_install(self):
        w = self.world()
        w.asset("gz").unlink()
        self.assertEqual(self.update("--yes", "--source-build"), 0, self.out)
        self.assertEqual(len(w.machine.make_calls), 1)
        self.assertFalse((w.dir / "build/engine/release.json").exists())


class RollbackTest(Base):
    def check_rolled_back(self, text):
        w = self.w
        self.assertEqual(self.update("--yes"), 4, self.out)
        self.assertOldRelease()
        self.assertTrue(w.machine.active, "the old release must run again")
        self.assertRegex(self.out, text)
        self.assertEqual((w.dir / ".crucible/update-hold").read_text().strip(), NEW)
        self.assertFalse((w.parent / ".crucible.update.json").exists())

    def test_the_service_does_not_start(self):
        self.world(bad_start=True)
        self.check_rolled_back(r"(?i)rolled back")

    def test_the_service_starts_but_health_fails(self):
        self.world(bad_health=True)
        self.check_rolled_back(r"(?i)health")

    def test_the_smoke_request_fails(self):
        self.world(bad_smoke=True)
        self.check_rolled_back(r"(?i)test request")

    def test_the_new_code_cannot_write_the_unit(self):
        self.world(bad_install=True)
        self.check_rolled_back(r"(?i)service install")

    def test_the_old_unit_and_files_come_back_byte_for_byte(self):
        w = self.world(bad_start=True)
        self.assertEqual(self.update("--yes"), 4, self.out)
        snap = snapshot(w.dir)
        snap.pop(".crucible/update-hold")
        self.assertEqual(snap, self.before)

    def test_a_rollback_says_how_to_leave_the_old_folder(self):
        w = self.world(bad_start=True)
        self.assertEqual(self.update("--yes"), 4, self.out)
        self.assertIn(f"cd {w.dir.resolve()}", self.out)

    def test_the_old_release_is_named_in_the_message(self):
        self.world(bad_start=True)
        self.assertEqual(self.update("--yes"), 4, self.out)
        self.assertIn(OLD, self.out)
        self.assertIn(NEW, self.out)

    def test_a_held_release_is_not_tried_again_by_the_next_update(self):
        w = self.world(bad_start=True)
        self.assertEqual(self.update("--yes"), 4, self.out)
        w.log.write_text("")
        n = len(w.machine.events)
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertIn("hold", self.out)
        self.assertEqual(w.machine.events[n:], [])

    def test_a_server_started_by_crucible_start_comes_back_after_a_failed_update(self):
        w = self.world(bad_start=True)
        w.unit.unlink()
        w.machine.active = False
        w.machine.live.add(40001)
        (w.dir / ".crucible/server.pid").write_text("40001 0\n")
        self.assertEqual(self.update("--yes"), 4, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertIn(40001, w.machine.live)

    def test_the_rollback_fails_too(self):
        w = self.world(bad_start=True)
        w.machine.starts = [False, False]
        self.assertEqual(self.update("--yes"), 5, self.out)
        self.assertRegex(self.out, r"service status")
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n", "the old tree is back even when it does not start")
        self.assertEqual(w.unit.read_text(), "unit 0.1.0\n")
        self.assertEqual(w.leftovers(), [])

    def test_the_rollback_fails_when_the_old_release_is_unhealthy(self):
        w = self.world(bad_start=True)
        w.machine.old_unhealthy = True
        self.assertEqual(self.update("--yes"), 5, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")

    def test_ctrl_c_after_the_stop_puts_the_old_release_back(self):
        w = self.world()

        def interrupt(name):
            if name == "swapped":
                raise KeyboardInterrupt
        with mock.patch.object(update, "crash_point", interrupt):
            code = self.update("--yes")
        self.assertEqual(code, 4, self.out)
        self.assertOldRelease()
        self.assertTrue(w.machine.active)


class CrashRecoveryTest(Base):
    """A kill (power loss, kill -9) at any point leaves a journal. The next update puts the old release back first."""

    POINTS = ("stopped", "carry:1", "carry:2", "carried", "swap:1", "swap:2", "swapped", "refreshed", "started")

    def crash_at(self, point, fallback=False):
        w = self.w

        def hook(name):
            if name == point:
                raise Crash(name)
        patches = [mock.patch.object(update, "crash_point", hook)]
        if fallback:
            patches.append(mock.patch.object(update, "exchange", side_effect=OSError(errno.EINVAL, "no exchange")))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        host = w.host()
        with self.assertRaises(Crash):
            cli.main(["update", "--yes"], host=host, repo=w.dir)
        for p in patches:
            p.stop()
            self.addCleanup(lambda: None)

    def recover(self):
        w = self.w
        code = self.update("--yes")
        return code

    def test_every_crash_point_recovers_to_the_old_release(self):
        for fallback in (False, True):
            for point in self.POINTS:
                if point.startswith("swap:") and not fallback:
                    continue
                with self.subTest(point=point, fallback=fallback):
                    self.world()
                    w = self.w
                    self.crash_at(point, fallback)
                    self.assertTrue((w.parent / ".crucible.update.json").exists(), "a journal must exist after a kill")
                    code = self.recover()
                    self.assertEqual(code, 4, self.out)
                    self.assertRegex(self.out, r"(?i)earlier update")
                    self.assertOldRelease()
                    self.assertTrue(w.machine.active)
                    self.assertFalse((w.parent / ".crucible.update.json").exists())
                    snap = snapshot(w.dir)
                    snap.pop(".crucible/update-hold", None)
                    self.assertEqual(snap, self.before)

    def test_a_crash_before_the_stop_leaves_no_journal_and_nothing_changed(self):
        self.world()
        w = self.w

        def hook(name):
            if name == "staged":
                raise Crash(name)
        with mock.patch.object(update, "crash_point", hook):
            with self.assertRaises(Crash):
                cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])
        # the next run cleans the staging folder that the kill left
        self.assertEqual(self.update("--yes"), 0, self.out)
        self.assertEqual(w.leftovers(), [])

    def test_a_journal_that_is_not_valid_is_refused_and_nothing_is_guessed(self):
        w = self.world()
        (w.parent / ".crucible.update.json").write_text("{not json")
        self.assertEqual(self.update("--yes"), 1, self.out)
        self.assertRegex(self.out, r"(?i)journal")
        self.assertEqual(w.machine.verbs, [])


class ExchangeTest(unittest.TestCase):
    def test_exchange_swaps_two_folders_in_one_step(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            a.mkdir()
            b.mkdir()
            (a / "x").write_text("A")
            (b / "y").write_text("B")
            try:
                update.exchange(a, b)
            except OSError as exc:
                if exc.errno in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
                    self.skipTest("this file system has no RENAME_EXCHANGE")
                raise
            self.assertTrue((a / "y").exists() and (b / "x").exists())

    def test_the_swap_works_without_exchange_support(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            a.mkdir()
            b.mkdir()
            (a / "x").write_text("A")
            (b / "y").write_text("B")
            with mock.patch.object(update, "exchange", side_effect=OSError(errno.EINVAL, "x")):
                update.swap_dirs(a, b)
            self.assertTrue((a / "y").exists() and (b / "x").exists())


class SumsParserTest(unittest.TestCase):
    def good(self, name="f.gz"):
        return f"{'a' * 64}  {name}\n"

    def test_a_star_before_the_name_is_allowed(self):
        self.assertEqual(update.sums_hash(f"{'a' * 64} *f.gz\n", "f.gz"), "a" * 64)

    def test_a_name_with_a_folder_part_does_not_match(self):
        self.assertRaises(update.CheckFailed, update.sums_hash, self.good("../f.gz"), "f.gz")
        self.assertRaises(update.CheckFailed, update.sums_hash, self.good("sub/f.gz"), "f.gz")

    def test_the_same_hash_twice_is_fine_and_two_hashes_are_not(self):
        self.assertEqual(update.sums_hash(self.good() * 2, "f.gz"), "a" * 64)
        self.assertRaises(update.CheckFailed, update.sums_hash, self.good() + f"{'b' * 64}  f.gz\n", "f.gz")

    def test_an_upper_case_or_short_hash_is_refused(self):
        self.assertRaises(update.CheckFailed, update.sums_hash, f"{'A' * 64}  f.gz\n", "f.gz")
        self.assertRaises(update.CheckFailed, update.sums_hash, f"{'a' * 63}  f.gz\n", "f.gz")

    def test_the_names_of_the_asset_files_match_the_release_tool(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import release_engine
        self.assertEqual((update.ARCH, update.ROCM_SERIES), (release_engine.ARCH, release_engine.ROCM_SERIES))
        self.assertEqual(update.asset_stem("v1.2.3"), STEM.format(tag="v1.2.3"))


class TagTest(unittest.TestCase):
    def test_the_newest_tag_is_the_highest_version_not_the_last_word(self):
        out = "\n".join(f"{'1' * 40}\trefs/tags/{t}" for t in ("v0.9.0", "v0.10.0", "v0.2.0", "v1.0.0-rc1", "latest", "v0.10.0x"))
        self.assertEqual(update.parse_tags(out)["newest"], "v0.10.0")

    def test_a_peeled_line_wins_for_an_annotated_tag(self):
        out = f"{'1' * 40}\trefs/tags/v0.3.0\n{'2' * 40}\trefs/tags/v0.3.0^{{}}\n"
        self.assertEqual(update.parse_tags(out)["tags"]["v0.3.0"], "2" * 40)


class UninstallTest(Base):
    def uninstall(self, *args, answers=None, tty=True, **machine):
        w = self.w
        host = w.host(answers=answers, tty=tty, **machine)
        code = cli.main(["uninstall", *args], host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        self.host = host
        return code

    def test_removes_the_service_the_unit_and_the_tree_and_keeps_the_models(self):
        w = self.world()
        self.assertEqual(self.uninstall("--yes"), 0, self.out)
        self.assertFalse(w.unit.exists())
        self.assertIn("stop", w.machine.verbs)
        self.assertIn("disable", w.machine.verbs)
        self.assertFalse(w.machine.active)
        self.assertFalse((w.dir / ".git").exists())
        self.assertFalse((w.dir / "crucible").exists())
        self.assertFalse((w.dir / ".venv").exists())
        self.assertFalse((w.dir / ".crucible").exists())
        self.assertEqual((w.pack / "weights.safetensors").read_text(), "WEIGHTS\n", "the models stay")
        self.assertEqual((w.states / "conv.state").read_text(), "STATE\n", "user data stays")
        self.assertIn("models", self.out)

    def test_the_remove_models_flag_removes_the_pack_and_the_state_folder(self):
        w = self.world()
        (w.states / "conv.state").unlink()   # the state folder holds only what the engine writes there
        (w.states / "write-ledger").mkdir()
        self.assertEqual(self.uninstall("--yes", "--remove-models"), 0, self.out)
        self.assertFalse(w.pack.exists())
        self.assertFalse(w.states.exists())
        self.assertFalse(w.dir.exists())

    def test_a_models_folder_outside_the_install_keeps_other_models(self):
        w = self.world()
        other = w.home / "models"
        other.mkdir()
        pack = other / w.pack.name
        pack.mkdir()
        (pack / "weights.safetensors").write_text("W\n")
        (other / "some-other-model").mkdir()
        (other / "some-other-model/w").write_text("OTHER\n")
        cfg = json.loads((w.dir / ".crucible/config.json").read_text())
        cfg["models_dir"] = str(other)
        (w.dir / ".crucible/config.json").write_text(json.dumps(cfg))
        self.assertEqual(self.uninstall("--yes", "--remove-models"), 0, self.out)
        self.assertFalse(pack.exists())
        self.assertEqual((other / "some-other-model/w").read_text(), "OTHER\n", "only the pack folder goes")
        self.assertFalse(w.dir.exists())

    def test_asks_one_question(self):
        w = self.world()
        self.assertEqual(self.uninstall(answers=["y"]), 0, self.out)
        self.assertEqual(len(self.host.prompt.asked), 1)
        self.assertFalse((w.dir / "crucible").exists())

    def test_no_nothing_is_removed(self):
        w = self.world()
        self.assertEqual(self.uninstall(answers=["n"]), 0, self.out)
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])
        self.assertTrue(w.unit.exists())
        self.assertIn("Nothing was removed", self.out)

    def test_no_terminal_and_no_yes_is_a_usage_error(self):
        w = self.world()
        self.assertEqual(self.uninstall(tty=False), 2, self.out)
        self.assertEqual(snapshot(w.dir), self.before)

    def test_the_plan_names_what_goes_and_what_stays(self):
        self.world()
        self.uninstall(answers=["n"])
        self.assertRegex(self.out, r"(?i)remove")
        self.assertRegex(self.out, r"(?i)keep")

    def test_a_folder_that_install_sh_did_not_make_is_left_alone(self):
        w = self.world()
        (w.dir / ".git/crucible-install").unlink()
        self.before = snapshot(w.dir)
        self.assertEqual(self.uninstall("--yes"), 2, self.out)
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertEqual(w.machine.verbs, [])

    def test_the_home_folder_is_never_removed(self):
        w = self.world()
        host = w.host()
        (w.home / ".git").mkdir()
        (w.home / ".git/crucible-install").write_text("x\n")
        (w.home / "crucible").write_text("x\n")
        code = cli.main(["uninstall", "--yes"], host=host, repo=w.home)
        self.assertEqual(code, 2, "\n".join(host.lines))
        self.assertTrue((w.home / "crucible").exists())

    def test_a_server_started_by_crucible_start_is_stopped(self):
        w = self.world()
        w.unit.unlink()
        w.machine.active = False
        w.machine.live.add(40001)
        (w.dir / ".crucible/server.pid").write_text("40001 0\n")
        self.assertEqual(self.uninstall("--yes"), 0, self.out)
        self.assertEqual(w.machine.terminated, [40001])
        self.assertFalse((w.dir / "crucible").exists())

    def test_a_service_that_does_not_stop_keeps_everything(self):
        w = self.world()
        w.machine.stop_fails = True
        self.assertEqual(self.uninstall("--yes"), 1, self.out)
        self.assertEqual(snapshot(w.dir), self.before)
        self.assertTrue(w.unit.exists())

    def test_a_link_in_the_tree_is_removed_and_not_followed(self):
        w = self.world()
        outside = w.t / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("K\n")
        (w.dir / "linked").symlink_to(outside)
        self.assertEqual(self.uninstall("--yes"), 0, self.out)
        self.assertEqual((outside / "keep.txt").read_text(), "K\n")
        self.assertFalse((w.dir / "linked").exists() or (w.dir / "linked").is_symlink())

    def test_it_never_removes_the_api_key_or_the_install_log(self):
        w = self.world()
        key = w.home / ".config/crucible/api-key"
        key.parent.mkdir(parents=True)
        key.write_text("crucible-secret\n")
        self.assertEqual(self.uninstall("--yes", "--remove-models"), 0, self.out)
        self.assertTrue(key.exists())
        self.assertNotIn("crucible-secret", self.out)

    def test_a_cut_update_is_put_back_first(self):
        w = self.world()

        def hook(name):
            if name == "carried":
                raise Crash(name)
        with mock.patch.object(update, "crash_point", hook):
            with self.assertRaises(Crash):
                cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        self.assertEqual(self.uninstall("--yes"), 0, self.out)
        self.assertFalse((w.parent / ".crucible.update.json").exists())
        self.assertFalse((w.dir / "crucible").exists())
        self.assertEqual((w.pack / "weights.safetensors").read_text(), "WEIGHTS\n")


if __name__ == "__main__":
    unittest.main()
