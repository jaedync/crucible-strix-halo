"""Security fixes of wp-sec-install (findings 1 to 5, 9).

The first block is the review's own tests (InstallShTest, UpdateTest), copied unchanged from the review file
test_sec_review_wps.py (md5 31bc00cb...). The test_f* tests state the fixed behavior. Further tests below the copy
(class names ending in More) cover the details of the fixes.
No network, no GPU, no systemd, no sudo. Synthetic data only.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import test_install_sh as ish
from crucible_update_fakes import World

from crucible_cli import cli, packs, update  # noqa: E402

PACK = packs.PACKS["4.05"].folder


class Crash(BaseException):
    """A kill: not an Exception, so no handler of the update catches it."""


class InstallShTest(unittest.TestCase):
    def setUp(self):
        self.w = ish.Env(self)

    def installed(self):
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def config(self, **values):
        (self.w.dir / ".crucible").mkdir(exist_ok=True)
        (self.w.dir / ".crucible/config.json").write_text(json.dumps(values))

    def test_ok_a_bad_download_leaves_no_temp_folder(self):
        tmp = self.w.t / "tmpdir"
        tmp.mkdir()
        (self.w.release / f"{ish.STEM}.gz").write_bytes(b"not the engine")
        r = self.w.install("--yes", TMPDIR=str(tmp))
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertEqual(list(tmp.iterdir()), [])
        self.assertFalse(self.w.dir.exists())

    def test_f1_remove_models_keeps_other_files_of_the_models_folder(self):
        """install.sh:841-850: --remove-models runs rm -rf on the whole models_dir of config.json. ./crucible uninstall removes
        only the pack folders (update.py:946, test_a_models_folder_outside_the_install_keeps_other_models)."""
        self.installed()
        models = self.w.home / "models"
        (models / PACK).mkdir(parents=True)
        (models / PACK / "weights").write_text("W")
        (models / "my-other-model.gguf").write_text("OTHER")
        self.config(models_dir=str(models))
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((models / PACK).exists())
        self.assertTrue((models / "my-other-model.gguf").exists(), "a file that Crucible did not make was removed")

    def test_f2a_uninstall_keeps_a_pack_in_the_install_folder_itself(self):
        """install.sh:816-840: models_dir equal to the install folder matches no "$real_dir"/* case, so rm -rf takes the pack.
        The plan line said Keep."""
        self.installed()
        (self.w.dir / PACK).mkdir()
        (self.w.dir / PACK / "weights").write_text("W")
        self.config(models_dir=str(self.w.dir))
        r = self.w.install("uninstall", "--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.w.dir / PACK / "weights").exists(), "the pack was removed without --remove-models")

    def test_f2b_uninstall_keeps_models_named_through_a_symlinked_home(self):
        """install.sh:820-821: rel=${p#"$real_dir"/} strips a resolved prefix from a path that is not resolved. The kept name is
        empty and the loop removes every entry, the models too."""
        self.installed()
        link = self.w.t / "homelink"
        link.symlink_to(self.w.home)
        (self.w.dir / "models/pack").mkdir(parents=True)
        (self.w.dir / "models/pack/weights").write_text("W")
        self.config(models_dir=str(link / "crucible/models"))
        r = self.w.install("uninstall", "--yes", HOME=str(link), CRUCIBLE_DIR=str(link / "crucible"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.w.dir / "models/pack/weights").exists(), "the models were removed without --remove-models")

    def test_f4_a_branch_with_the_name_of_the_tag_is_not_installed(self):
        """install.sh:596-601: git clone --branch takes refs/heads/TAG before refs/tags/TAG. Only the tag commit is compared with
        the manifest; HEAD is not, so the branch code is checked out and run. update.py:507-510 checks HEAD."""
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
        g = lambda *a: ish.run([ish.REAL_GIT, "-C", str(self.w.src), *a], env=env)  # noqa: E731
        g("checkout", "-q", "-b", "other")
        (self.w.src / "crucible").write_text('#!/bin/sh\necho ran > "$HOME/branch.ran"\n')
        g("commit", "-qam", "not the release")
        g("branch", ish.TAG)
        g("checkout", "-q", "main")
        r = self.w.install("--yes")
        self.assertFalse((self.w.home / "branch.ran").exists(), "code that is not the tagged release ran")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_f11_the_installer_refuses_to_run_as_root(self):
        """install.sh: `curl ... | sudo bash` runs git, setup, pip and the user service as root. Nothing refuses it."""
        (self.w.shims / "id").write_text('#!/bin/sh\n[ "$1" = "-u" ] && { echo 0; exit 0; }\nexec /usr/bin/id "$@"\n')
        (self.w.shims / "id").chmod(0o755)
        r = self.w.install("--yes")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((self.w.dir / "handover.args").exists())

    def test_f12_downloads_refuse_a_redirect_away_from_https(self):
        """install.sh:401-405: curl -L follows a redirect to http://, and SHA256SUMS comes from the same place."""
        body = re.search(r"^fetch\(\) \{\n(.*?)^\}", ish.INSTALL.read_text(), re.S | re.M).group(1)
        self.assertRegex(body, r"--proto-redir")

    def test_f13_uninstall_removes_nothing_when_the_server_does_not_stop(self):
        """install.sh:811-814: the service uninstall and stop results are ignored (|| true); the tree goes while the server may
        still run. ./crucible uninstall stops with exit 1 then (update.py:968-977)."""
        self.installed()
        (self.w.dir / "crucible").write_text('#!/bin/sh\nexit 1\n')
        (self.w.dir / "crucible").chmod(0o755)
        r = self.w.install("uninstall", "--yes")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.w.dir / ".git").exists())

    def test_f16_sudo_is_asked_once_per_run(self):
        """install.sh:279 and 536: a missing tool and a missing library give two questions and two sudo runs."""
        w = self.w
        bin_ = w.t / "hostbin"
        bin_.mkdir()
        for tool in ish.HOST_TOOLS + ["sha256sum", "ldd"]:
            path = shutil.which(tool)
            if path:
                (bin_ / tool).symlink_to(path)
        (w.shims / "gzip").unlink()
        (w.shims / "sudo").write_text(
            f'#!/bin/sh\nprintf "sudo %s\\n" "$*" >> "{w.log}"\ncase "$*" in\n  *" gzip"*) ln -sf {shutil.which("gzip")} {bin_}/gzip ;;\n'
            f'  *libnuma1*) printf "\\tlibnuma.so.1 (libc6,x86-64) => /usr/lib/libnuma.so.1\\n" >> "{w.ld_add}" ;;\nesac\nexit 0\n')
        w.make_release(needs=["libnuma.so.1"])
        w.ld_hide.write_text("libnuma.so.1\n")
        r = w.install("--yes", answers="y\n", PATH=f"{w.shims}:{bin_}", SHIM_REAL_PATH=f"{bin_}:/usr/bin")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((r.stdout + r.stderr).count("(y/N)"), 1, w.calls("sudo"))


def _crash_at(point):
    def hook(name):
        if name == point:
            raise Crash(name)
    return hook


class UpdateTest(unittest.TestCase):
    def cli(self, w, *args, answers=None, **machine):
        host = w.host(answers=answers, **machine)
        code = cli.main(list(args), host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        return code

    def config(self, w, **values):
        path = w.dir / ".crucible/config.json"
        cfg = json.loads(path.read_text())
        cfg.update(values)
        path.write_text(json.dumps(cfg))

    def test_ok_a_kill_during_the_recovery_still_recovers(self):
        w = World(self)
        with mock.patch.object(update, "crash_point", _crash_at("swapped")), self.assertRaises(Crash):
            cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        with mock.patch.object(update.Update, "restore_files", side_effect=Crash("in the recovery")), self.assertRaises(Crash):
            cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        self.assertEqual(self.cli(w, "update", "--yes"), 4, self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertEqual(w.unit.read_text(), "unit 0.1.0\n")
        self.assertEqual((w.pack / "weights.safetensors").read_text(), "WEIGHTS\n")
        self.assertFalse((w.parent / ".crucible.update.json").exists())
        self.assertEqual(w.leftovers(), [])

    def test_f5_update_keeps_your_files_inside_tracked_folders(self):
        """update.py:313 and 534-543: the clean check uses --untracked-files=no and only top-level names move to the new tree.
        A file of yours in docs/ or tools/ stays in the old tree, and commit() removes that tree. No warning."""
        w = World(self)
        (w.dir / "docs" / "my-notes.md").write_text("MY NOTES\n")
        code = self.cli(w, "update", "--yes")
        self.assertTrue((w.dir / "docs" / "my-notes.md").exists() or code == 1, self.out)
        if code == 1:
            self.assertIn("my-notes.md", self.out)

    def test_f8_a_no_answer_changes_no_python_package(self):
        """update.py:356-361 and 576-588: pip installs the new requirements in the live venv before the question. A "no" then
        says "Nothing was changed.", and a rollback (exit 4) does not undo it."""
        w = World(self, new_requirements="numpy\nrequests\n")
        self.assertEqual(self.cli(w, "update", answers=["n"]), 0, self.out)
        self.assertIn("Nothing was changed", self.out)
        self.assertEqual([e for e in w.machine.events if e[0] == "pip"], [])

    def test_f2c_crucible_uninstall_keeps_a_pack_in_the_install_folder_itself(self):
        """update.py:990-998: _within(models_dir, real) is False when they are equal, so nothing is kept and the pack goes.
        The plan line said Keep."""
        w = World(self)
        pack = w.dir / PACK
        os.rename(w.pack, pack)
        self.config(w, models_dir=str(w.dir))
        self.assertEqual(self.cli(w, "uninstall", "--yes"), 0, self.out)
        self.assertRegex(self.out, r"Keep: +" + re.escape(str(pack)))
        self.assertTrue((pack / "weights.safetensors").exists(), "the pack was removed without --remove-models")

    def test_f3_remove_models_does_not_take_a_folder_that_holds_other_files(self):
        """cli.py:140-147 accepts any folder inside home as the state folder (for example ~/.cache). update.py:980-986 and
        install.sh:841-850 then run rmtree on it with --remove-models."""
        w = World(self)
        cache = w.home / ".cache"
        (cache / "pip").mkdir(parents=True)
        (cache / "pip/keep").write_text("x")
        self.config(w, state_dir=str(cache))
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertTrue((cache / "pip/keep").exists(), "a file that the engine did not write was removed")


# ---- the details of the fixes (green from the start of the fix commit; see the handoff for the choices) ------------------

from types import SimpleNamespace  # noqa: E402

from crucible_cli import settings  # noqa: E402


class StateFolderMore(unittest.TestCase):
    """Finding 3: a state folder is a folder that setup made or took empty. The proof is a record in the work folder."""

    def test_new_empty_and_engine_written_folders_are_accepted(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            d = Path(d)
            self.assertIsNone(settings.state_folder_problem(d / "new"))
            (d / "empty").mkdir()
            self.assertIsNone(settings.state_folder_problem(d / "empty"))
            (d / "engine").mkdir()
            (d / "engine/write-ledger").write_text("x")
            (d / "engine" / ("0123456789abcdef-" * 2 + "0123456789abcdef")).mkdir()
            self.assertIsNone(settings.state_folder_problem(d / "engine"))

    def test_a_folder_with_other_files_a_link_and_a_file_are_refused(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            d = Path(d)
            (d / "cache/pip").mkdir(parents=True)
            self.assertIn("pip", settings.state_folder_problem(d / "cache"))
            (d / "link").symlink_to(d / "cache")
            self.assertIn("link", settings.state_folder_problem(d / "link"))
            (d / "file").write_text("x")
            self.assertIn("not a folder", settings.state_folder_problem(d / "file"))

    def test_an_owned_folder_is_accepted_as_it_is(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            d = Path(d)
            (d / "cache/pip").mkdir(parents=True)
            self.assertIsNone(settings.state_folder_problem(d / "cache", owned=True))

    def test_the_record_names_one_real_path_and_a_link_is_not_a_record(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            d = Path(d)
            repo = d / "repo"
            (repo / ".crucible").mkdir(parents=True)
            state = d / "states"
            state.mkdir()
            self.assertFalse(settings.state_owned(repo, state))
            settings.state_marker_path(repo).write_text(settings.state_marker_text(state))
            self.assertTrue(settings.state_owned(repo, state))
            (d / "other").mkdir()
            self.assertFalse(settings.state_owned(repo, d / "other"))
            settings.state_marker_path(repo).unlink()
            (d / "record").write_text(settings.state_marker_text(state))
            settings.state_marker_path(repo).symlink_to(d / "record")
            self.assertFalse(settings.state_owned(repo, state))

    def test_setup_refuses_a_state_folder_that_holds_other_files(self):
        with __import__("tempfile").TemporaryDirectory() as d:
            d = Path(d)
            (d / "home/.cache/pip").mkdir(parents=True)
            host = SimpleNamespace(home=d / "home", out=lambda line: lines.append(line))
            lines = []
            s = SimpleNamespace(state_dir=d / "home/.cache", repo=d / "repo")
            self.assertFalse(cli._validate(s, host))
            self.assertIn("pip", " ".join(lines))
            s = SimpleNamespace(state_dir=d / "home/.cache/crucible/states", repo=d / "repo")
            self.assertTrue(cli._validate(s, host))


class UninstallMore(unittest.TestCase):
    def cli(self, w, *args, answers=None):
        host = w.host(answers=answers)
        code = cli.main(list(args), host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        return code

    def test_remove_models_removes_the_state_folder_that_setup_made(self):
        w = World(self)
        self.engine_layout(w)
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertFalse(w.states.exists())
        self.assertFalse(w.pack.exists())

    @staticmethod
    def engine_layout(w):
        """The state folder holds only what the engine writes there (--remove-models looks inside before it removes it)."""
        (w.states / "conv.state").unlink()
        (w.states / "write-ledger").mkdir()
        (w.states / ("a" * 16 + "-" + "b" * 16 + "-" + "c" * 16)).mkdir()

    def test_remove_models_keeps_a_state_folder_that_setup_did_not_make(self):
        w = World(self)
        (w.dir / ".crucible/state-dir.json").unlink()
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertTrue((w.states / "conv.state").exists())
        self.assertIn("setup did not make", self.out)

    def test_a_kept_state_folder_that_holds_the_install_folder_stops_the_command(self):
        w = World(self)
        config = json.loads((w.dir / ".crucible/config.json").read_text())
        config["state_dir"] = str(w.parent)
        (w.dir / ".crucible/config.json").write_text(json.dumps(config))
        self.assertEqual(self.cli(w, "uninstall", "--yes"), 1, self.out)
        self.assertIn("holds it", self.out)
        self.assertTrue((w.dir / ".git").exists())
        self.assertTrue(w.unit.exists())

    def test_a_pack_in_the_install_folder_is_kept_by_its_real_path(self):
        w = World(self)
        link = w.t / "link"
        link.symlink_to(w.dir)
        config = json.loads((w.dir / ".crucible/config.json").read_text())
        config["models_dir"] = str(link / "models")
        (w.dir / ".crucible/config.json").write_text(json.dumps(config))
        self.assertEqual(self.cli(w, "uninstall", "--yes"), 0, self.out)
        self.assertTrue((w.pack / "weights.safetensors").exists())


class OwnFilesMore(unittest.TestCase):
    """Finding 5: the update refuses when it would delete a file of yours, and carries what it can carry."""

    def update(self, w):
        host = w.host()
        code = cli.main(["update", "--yes"], host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        return code

    def test_a_tool_cache_is_not_a_file_of_yours(self):
        w = World(self)
        (w.dir / "docs" / "__pycache__").mkdir()
        (w.dir / "docs" / "__pycache__" / "x.cpython-314.pyc").write_bytes(b"x")
        self.assertEqual(self.update(w), 0, self.out)

    def test_an_ignored_file_in_a_tracked_folder_is_refused_and_nothing_changes(self):
        w = World(self)
        (w.dir / ".git/info/exclude").write_text("*.secret\n")
        (w.dir / "docs" / "key.secret").write_text("MINE\n")
        self.assertEqual(self.update(w), 1, self.out)
        self.assertIn("key.secret", self.out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertEqual((w.dir / "docs" / "key.secret").read_text(), "MINE\n")
        self.assertEqual(w.leftovers(), [])

    def test_a_file_with_a_glob_character_in_its_name_is_listed_and_refused(self):
        w = World(self)
        (w.dir / "docs/[a]*.md").write_text("MINE\n")
        self.assertEqual(self.update(w), 1, self.out)
        self.assertIn("[a]*.md", self.out)

    def test_a_file_at_the_top_level_is_carried_to_the_new_tree(self):
        w = World(self)
        (w.dir / "my-notes.txt").write_text("MINE\n")
        self.assertEqual(self.update(w), 0, self.out)
        self.assertEqual((w.dir / "my-notes.txt").read_text(), "MINE\n")

    def test_a_file_that_appears_after_the_check_is_refused_before_the_swap(self):
        w = World(self)
        host = w.host()

        def hook(name):
            if name == "staged":
                (w.dir / "docs" / "late.md").write_text("LATE\n")

        with mock.patch.object(update, "crash_point", hook):
            code = cli.main(["update", "--yes"], host=host, repo=w.dir)
        self.assertEqual(code, 1, "\n".join(host.lines))
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")
        self.assertTrue((w.dir / "docs" / "late.md").exists())


class PipMore(unittest.TestCase):
    """Finding 9: pip runs after the yes, with the old pins recorded first and put back by a rollback."""

    def run_update(self, w, **machine):
        host = w.host(**machine)
        code = cli.main(["update", "--yes"], host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        return code

    def test_the_pins_are_recorded_and_put_back_when_the_new_service_does_not_start(self):
        w = World(self, new_requirements="numpy\nrequests\n", bad_start=True)
        self.assertEqual(self.run_update(w), 4, self.out)
        self.assertEqual(w.machine.put_back, [w.machine.freeze])
        self.assertFalse((w.dir / ".crucible/update-pip-before.txt").exists())

    def test_a_pip_failure_puts_the_pins_back_and_keeps_no_journal(self):
        w = World(self, new_requirements="numpy\nrequests\n")
        w.machine.fail_pip = True
        self.assertEqual(self.run_update(w), 1, self.out)
        self.assertEqual(w.machine.put_back, [w.machine.freeze])
        self.assertFalse((w.parent / ".crucible.update.json").exists())
        self.assertEqual(w.leftovers(), [])

    def test_a_kill_after_pip_is_undone_by_the_next_run(self):
        w = World(self, new_requirements="numpy\nrequests\n")
        with mock.patch.object(update, "crash_point", _crash_at("pipped")), self.assertRaises(Crash):
            cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        journal = json.loads((w.parent / ".crucible.update.json").read_text())
        self.assertEqual(journal["pip_freeze"], w.machine.freeze)
        self.assertEqual(w.machine.put_back, [])
        self.assertEqual(self.run_update(w), 4, self.out)
        self.assertEqual(w.machine.put_back, [w.machine.freeze])
        self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")

    def test_a_put_back_that_fails_shows_the_command_to_run(self):
        w = World(self, new_requirements="numpy\nrequests\n", bad_start=True)
        w.machine.fail_put_back = True
        self.assertEqual(self.run_update(w), 4, self.out)
        self.assertIn("pip install --no-deps -r", self.out)
        self.assertEqual((w.dir / ".crucible/update-pip-before.txt").read_text(), w.machine.freeze)

    def test_no_change_of_requirements_runs_no_pip_and_records_nothing(self):
        w = World(self)
        self.assertEqual(self.run_update(w), 0, self.out)
        self.assertEqual([e for e in w.machine.events if e[0] == "pip"], [])

    def test_the_question_says_that_the_packages_change_after_the_yes(self):
        w = World(self, new_requirements="numpy\nrequests\n")
        host = w.host(answers=["n"])
        self.assertEqual(cli.main(["update"], host=host, repo=w.dir), 0)
        self.assertIn("Python packages", "\n".join(host.lines))


class TagMore(unittest.TestCase):
    def test_the_tag_pattern_takes_ascii_digits_and_no_trailing_newline(self):
        self.assertTrue(update.TAG_RE.match("v0.34.0"))
        self.assertIsNone(update.TAG_RE.match("v0.34.0\n"))
        self.assertIsNone(update.TAG_RE.match("v\u0660.\u0661.\u0662"))   # Arabic-Indic digits
        self.assertIsNone(update.HEX40.match("a" * 40 + "\n"))
        self.assertIsNone(update.HEX64.match("b" * 64 + "\n"))
        self.assertTrue(update.HEX40.match("a" * 40) and update.HEX64.match("b" * 64))

    def test_install_sh_lists_the_pack_folders_of_packs_py(self):
        text = ish.INSTALL.read_text()
        listed = re.search(r"^PACK_FOLDERS=\((.*?)\)", text, re.M).group(1).split()
        self.assertEqual(sorted(listed), sorted(p.folder for p in packs.PACKS.values()))


class OldInstallUninstallMore(unittest.TestCase):
    """The shell path (an install without tools/crucible_cli/update.py)."""

    def setUp(self):
        self.w = ish.Env(self)
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def config(self, **values):
        (self.w.dir / ".crucible").mkdir(exist_ok=True)
        (self.w.dir / ".crucible/config.json").write_text(json.dumps(values))

    def test_an_update_that_was_cut_off_stops_the_uninstall(self):
        (self.w.home / ".crucible.update.json").write_text("{}")
        r = self.w.install("uninstall", "--yes")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertTrue((self.w.dir / ".git").exists())

    @unittest.skipUnless(shutil.which("flock"), "flock is not installed")
    def test_a_running_update_stops_the_uninstall(self):
        import fcntl
        fd = os.open(self.w.home, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = self.w.install("uninstall", "--yes")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("update is running", r.stdout + r.stderr)
        self.assertTrue((self.w.dir / ".git").exists())

    def test_remove_models_in_the_install_folder_removes_the_pack_and_keeps_the_other_files(self):
        models = self.w.dir / "models"
        (models / PACK).mkdir(parents=True)
        (models / "mine.gguf").write_text("MINE")
        self.config(models_dir=str(models))
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((models / PACK).exists())
        self.assertTrue((models / "mine.gguf").exists())
        self.assertFalse((self.w.dir / ".git").exists())

    def test_the_state_folder_is_never_removed_by_the_shell_path(self):
        state = self.w.home / "states"
        state.mkdir()
        (state / "conv").write_text("S")
        self.config(models_dir=str(self.w.home / "models"), state_dir=str(state))
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((state / "conv").exists())

    def test_a_kept_item_that_holds_the_install_folder_stops_the_command(self):
        self.config(models_dir=str(self.w.home / "models"), state_dir=str(self.w.home))
        r = self.w.install("uninstall", "--yes")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.w.dir / ".git").exists())

    def test_a_new_install_hands_the_uninstall_to_crucible(self):
        (self.w.dir / "tools/crucible_cli").mkdir(parents=True)
        (self.w.dir / "tools/crucible_cli/update.py").write_text("")
        (self.w.dir / "crucible").write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$HOME/handover.uninstall"\n')
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.w.home / "handover.uninstall").read_text().strip(), "uninstall --yes --remove-models")


class InstallShMore(unittest.TestCase):
    def setUp(self):
        self.w = ish.Env(self)

    def test_root_is_refused_for_every_command_that_changes_something(self):
        (self.w.shims / "id").write_text('#!/bin/sh\n[ "$1" = "-u" ] && { echo 0; exit 0; }\nexec /usr/bin/id "$@"\n')
        (self.w.shims / "id").chmod(0o755)
        for cmd in ("install", "update", "uninstall"):
            with self.subTest(cmd):
                r = self.w.install(cmd, "--yes")
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("root", r.stdout + r.stderr)

    def test_a_branch_with_the_name_of_the_tag_is_refused_with_the_source_build_too(self):
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
        g = lambda *a: ish.run([ish.REAL_GIT, "-C", str(self.w.src), *a], env=env)  # noqa: E731
        g("checkout", "-q", "-b", "other")
        (self.w.src / "crucible").write_text('#!/bin/sh\necho ran > "$HOME/branch.ran"\n')
        g("commit", "-qam", "not the release")
        g("branch", ish.TAG)
        g("checkout", "-q", "main")
        r = self.w.install("--yes", "--source-build")
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertFalse((self.w.home / "branch.ran").exists())
        self.assertFalse(self.w.dir.exists())

    def test_the_download_command_limits_the_protocols(self):
        body = re.search(r"^fetch\(\) \{\n(.*?)^\}", ish.INSTALL.read_text(), re.S | re.M).group(1)
        self.assertRegex(body, r"--proto '=https,file'")
        self.assertRegex(body, r"--proto-redir '=https'")

    def test_the_first_download_uses_a_file_release_folder_with_the_limits(self):
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        curls = self.w.calls("curl")
        self.assertTrue(curls and all("--proto =https,file" in c and "--proto-redir =https" in c for c in curls), curls)


class RealTreeMore(unittest.TestCase):
    """The update must proceed on a checkout that was used the normal way: setup, a release engine or a source build, a
    server start and a request. Every path below is made by that use.
    Only a file of yours in a folder that git tracks stops the update."""

    TRACKED = ("server", "tools", "tools/crucible_cli", "engine", "ref", "tests", "deploy", "ops")

    def used_tree(self, w):
        def put(rel, text="x\n"):
            path = w.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        # ./crucible setup: .venv, the work folder (logs, choices, record of the state folder), the pack
        put(".venv/lib/python3.14/site-packages/numpy/__init__.py")
        put(".venv/pyvenv.cfg")
        for name in ("config.json", "serve.sh", "setup-state.json", "state-dir.json", "venv.log", "download.log", "build.log",
                     "server.log", "server.pid", "check.json", "update-pip-before.txt"):
            put(f".crucible/{name}")
        put(f"models/{PACK}/.cache/huggingface/download/shard.metadata")
        # make (source build) and install.sh (release engine): everything is under build/
        for name in ("crucible", "crucible.build-info", "crucible.d", "release.json", "split/cache/part0.o", "split/parts/p0.hip"):
            put(f"build/engine/{name}")
        put("build/tests/some_test")
        # a server start and a request: bytecode next to the sources, the engine's prompt file in the working folder
        for d in self.TRACKED:
            put(f"{d}/__pycache__/mod.cpython-314.pyc")
        put("server/tests/__pycache__/t.cpython-314.pyc")
        put("tools/crucible_cli/__pycache__/cli.cpython-314.pyc")
        put("tools/rival/.pytest_cache/v/cache/lastfailed")
        put(".pytest_cache/README.md")
        put("prompt.ids")
        put(".coverage")
        put("dist/release/SHA256SUMS")
        put("scratch/notes.txt")

    def test_a_used_checkout_updates_and_keeps_what_setup_made(self):
        w = World(self, tracked_dirs=self.TRACKED)
        self.used_tree(w)
        (w.home / ".config/crucible").mkdir(parents=True)
        (w.home / ".config/crucible/api-key").write_text("crucible-secret\n")
        (w.home / ".cache/crucible").mkdir(parents=True, exist_ok=True)
        (w.home / ".cache/crucible/update-check.json").write_text("{}")
        host = w.host()
        code = cli.main(["update", "--yes"], host=host, repo=w.dir)
        out = "\n".join(host.lines)
        self.assertEqual(code, 0, out)
        self.assertEqual((w.dir / "VERSION").read_text(), "0.2.0\n")
        for rel in (".venv/pyvenv.cfg", ".crucible/server.log", ".crucible/state-dir.json",
                    "prompt.ids", ".coverage", "scratch/notes.txt", "dist/release/SHA256SUMS", f"models/{PACK}/weights.safetensors"):
            self.assertTrue((w.dir / rel).exists(), rel)
        self.assertEqual(w.leftovers(), [])

    def test_the_developer_fixtures_of_ref_do_not_stop_the_update(self):
        # ref/fixtures is in .gitignore; ref/make_fixtures.py writes it (about 64 MB). It is regenerated, never yours.
        w = World(self, tracked_dirs=self.TRACKED)
        (w.dir / "ref/fixtures").mkdir()
        (w.dir / "ref/fixtures/layer0_qkv.npz").write_bytes(b"x")
        host = w.host()
        self.assertEqual(cli.main(["update", "--yes"], host=host, repo=w.dir), 0, "\n".join(host.lines))

    def test_a_file_of_yours_in_a_tracked_folder_still_stops_the_update(self):
        for rel in ("/".join(("docs", "my-notes.md")), "ref/my-fixtures/a.npz", "tools/crucible_cli/local.py", "server/__pycache__x.txt"):
            with self.subTest(rel):
                w = World(self, tracked_dirs=self.TRACKED)
                self.used_tree(w)
                path = w.dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("MINE\n")
                host = w.host()
                code = cli.main(["update", "--yes"], host=host, repo=w.dir)
                self.assertEqual(code, 1, "\n".join(host.lines))
                self.assertIn(Path(rel).name, "\n".join(host.lines))
                self.assertEqual((w.dir / "VERSION").read_text(), "0.1.0\n")


class GitignoreMore(unittest.TestCase):
    """Each entry of .gitignore is a path that the tools make. A new entry in a tracked folder needs a decision: add it to
    CACHE_NAMES or GENERATED_PATHS in update.py (the update skips it), or it stops the update."""

    TOP_LEVEL = {"build/", ".venv/", "/models/", "/prompt.ids", ".coverage", "scratch/", "/.crucible/", "dist/"}   # not in a tracked folder
    CACHES = {"__pycache__/", "*.pyc", ".pytest_cache/"}

    def test_every_ignored_path_in_a_tracked_folder_is_known_to_the_update(self):
        root = Path(__file__).resolve().parents[1]
        entries = {ln.strip() for ln in (root / ".gitignore").read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        for nested in (root / "tools" / "rival" / ".gitignore",):   # absent in some exports
            if nested.is_file():
                entries |= {ln.strip() for ln in nested.read_text().splitlines() if ln.strip()}
        for entry in sorted(entries - self.TOP_LEVEL):
            if entry in self.CACHES:
                self.assertTrue(entry.rstrip("/") in update.CACHE_NAMES or entry == "*.pyc", entry)
            else:
                self.assertTrue(any(entry.startswith(g) for g in update.GENERATED_PATHS), f"{entry} is new: decide in update.py")


class PackageNamesMore(unittest.TestCase):
    """No tool that the installer checks may be "not known" on a system that the installer knows (a clean ubuntu:24.04 has no curl)."""

    FAMILIES = {"arch": "pacman", "debian": "apt-get", "fedora": "dnf"}

    def pkg_for(self, family, name):
        text = ish.INSTALL.read_text()
        func = re.search(r"^pkg_for\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)
        r = subprocess.run(["bash", "-c", func + f'\npkg_for "$1" "$2"', "x", family, name], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def checked_tools(self):
        text = ish.INSTALL.read_text()
        body = re.search(r"^check_tools\(\) \{\n(.*?)^\}", text, re.S | re.M).group(1)
        tools = re.search(r"for tool in ([a-z0-9 ]+); do\n\s+command -v", body).group(1).split()
        self.assertTrue({"git", "curl", "python3", "gzip", "sha256sum"} <= set(tools), tools)
        return tools

    def test_every_checked_tool_has_a_package_on_every_known_family(self):
        for family in self.FAMILIES:
            for tool in self.checked_tools():
                with self.subTest(family=family, tool=tool):
                    self.assertNotEqual(self.pkg_for(family, f"tool-{tool}"), "", f"no package for {tool} on {family}")

    def test_every_tool_name_in_the_package_table_is_a_tool_that_is_checked_or_used(self):
        text = ish.INSTALL.read_text()
        func = re.search(r"^pkg_for\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)
        named = set(re.findall(r"tool-([a-z0-9]+)", func))
        self.assertTrue(set(self.checked_tools()) <= named, set(self.checked_tools()) - named)

    def test_a_missing_curl_goes_through_one_sudo_question(self):
        w = ish.Env(self)
        bin_ = w.t / "hostbin"
        bin_.mkdir()
        for tool in ish.HOST_TOOLS + ["sha256sum", "ldd", "gzip"]:
            path = shutil.which(tool)
            if path and tool != "curl":
                (bin_ / tool).symlink_to(path)
        (w.shims / "curl").unlink()
        (w.shims / "sudo").write_text(
            f'#!/bin/sh\nprintf "sudo %s\\n" "$*" >> "{w.log}"\ncase "$*" in\n  *" curl"*) ln -sf {shutil.which("curl")} {bin_}/curl ;;\nesac\nexit 0\n')
        r = w.install("--yes", answers="y\n", PATH=f"{w.shims}:{bin_}", SHIM_REAL_PATH=f"{bin_}:/usr/bin")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(out.count("(y/N)"), 1, out)
        self.assertEqual(w.calls("sudo"), ["sudo apt-get update", "sudo apt-get install -y curl"])


if __name__ == "__main__":
    unittest.main()
