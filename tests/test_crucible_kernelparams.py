"""Kernel parameters: setup finds the two required parameters and the boot method, prints the exact change, and applies it
with sudo only after the typed word "yes". It never restarts the machine. Every test here uses fake files in a temporary
folder and a fake runner. Nothing runs sudo, nothing touches /etc or /boot of the test machine."""
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crucible_cli_fakes import FakePrompt, FakeRunner, make_repo, wire

from crucible_cli import cli, doctor, kernelparams as kp, runner  # noqa: E402

TOKENS = ["amdgpu.gttsize=126976", "ttm.pages_limit=32505856"]
LINE = " ".join(TOKENS)
SMALL = 60_000_000_000
GRUB = ('# a comment\nGRUB_DEFAULT=0\nGRUB_TIMEOUT=5\nGRUB_CMDLINE_LINUX_DEFAULT="quiet splash"\n'
        'GRUB_CMDLINE_LINUX=""\nGRUB_DISABLE_RECOVERY=true\n')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)

    def machine(self, cmdline="root=/dev/nvme0n1p2 rw quiet splash", **kw):
        kw.setdefault("tools", ("make", "g++", "git", "sudo", "grub-mkconfig"))
        host = wire(self.t, cmdline=cmdline, gtt=SMALL, **kw)
        host.lines = []
        return host

    def put(self, host, rel, text, mode=0o644):
        """Write a fake system file. REL is under etc, boot or efi of the fake machine."""
        root = {"etc": host.etc, "boot": host.boot_dirs[0], "efi": host.boot_dirs[1]}[rel.split("/", 1)[0]]
        p = root / rel.split("/", 1)[1]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        p.chmod(mode)
        return p

    def grub(self, host, text=GRUB):
        self.put(host, "boot/grub/grub.cfg", "# generated\n")
        return self.put(host, "etc/default/grub", text)


class ParamsTest(unittest.TestCase):
    def test_the_required_tokens_are_the_ones_of_the_doctor(self):
        self.assertEqual(" ".join(kp.REQUIRED), doctor.GTT_PARAMS)
        self.assertEqual(kp.REQUIRED, TOKENS)

    def test_missing(self):
        self.assertEqual(kp.missing("root=/x rw"), TOKENS)
        self.assertEqual(kp.missing("root=/x amdgpu.gttsize=126976"), [TOKENS[1]])
        self.assertEqual(kp.missing("root=/x " + LINE), [])
        self.assertEqual(kp.missing("amdgpu.gttsize=65536 ttm.pages_limit=32505856"), [TOKENS[0]])
        self.assertEqual(kp.missing("xamdgpu.gttsize=126976 " + TOKENS[1]), [TOKENS[0]])

    def test_set_params_appends_and_keeps_the_rest_in_order(self):
        self.assertEqual(kp.set_params("quiet splash", TOKENS), "quiet splash " + LINE)
        self.assertEqual(kp.set_params("", TOKENS), LINE)

    def test_set_params_replaces_a_wrong_value_and_never_repeats_a_key(self):
        got = kp.set_params("a amdgpu.gttsize=1 b amdgpu.gttsize=2 c", TOKENS[:1])
        self.assertEqual(got, "a amdgpu.gttsize=126976 b c")
        again = kp.set_params(got, TOKENS)
        self.assertEqual(kp.set_params(again, TOKENS), again)
        self.assertEqual(again.split().count(TOKENS[0]), 1)

    def test_core_drops_the_words_that_the_loader_adds(self):
        self.assertEqual(kp.core("initrd=\\initramfs.img BOOT_IMAGE=/vmlinuz root=/x rw".split()), ["root=/x", "rw"])


class GrubTest(Base):
    def test_grub_is_found_and_the_line_is_changed_only_there(self):
        host = self.machine(cmdline="BOOT_IMAGE=/vmlinuz-linux root=/dev/x rw quiet splash")
        f = self.grub(host)
        plan = kp.plan(host)
        self.assertEqual((plan.method, plan.file, plan.can_apply), ("grub", f, True))
        want = GRUB.replace('"quiet splash"', f'"quiet splash {LINE}"')
        self.assertEqual(plan.new_text, want)
        self.assertEqual(plan.old_text, GRUB)
        self.assertEqual(plan.regen, ["grub-mkconfig", "-o", str(host.boot_dirs[0] / "grub/grub.cfg")])

    def test_single_quotes_no_quotes_and_a_wrong_value(self):
        for old, new in (("GRUB_CMDLINE_LINUX_DEFAULT='quiet'", f"GRUB_CMDLINE_LINUX_DEFAULT='quiet {LINE}'"),
                         ("GRUB_CMDLINE_LINUX_DEFAULT=quiet", f'GRUB_CMDLINE_LINUX_DEFAULT="quiet {LINE}"'),
                         ('GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gttsize=1 quiet"', f'GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gttsize=126976 quiet ttm.pages_limit=32505856"'),
                         ('GRUB_CMDLINE_LINUX_DEFAULT=""', f'GRUB_CMDLINE_LINUX_DEFAULT="{LINE}"')):
            host = self.machine(cmdline="BOOT_IMAGE=/v rw")
            self.grub(host, "A=1\n" + old + "\nB=2\n")
            self.assertEqual(kp.plan(host).new_text, "A=1\n" + new + "\nB=2\n", old)

    def test_a_line_with_shell_syntax_is_printed_not_changed(self):
        for bad in ('GRUB_CMDLINE_LINUX_DEFAULT="quiet $EXTRA"', 'GRUB_CMDLINE_LINUX_DEFAULT="a `id`"',
                    'GRUB_CMDLINE_LINUX_DEFAULT="a \\" b"'):
            host = self.machine(cmdline="BOOT_IMAGE=/v rw")
            self.grub(host, bad + "\n")
            plan = kp.plan(host)
            self.assertFalse(plan.can_apply, bad)
            self.assertIn("grub", plan.reason.lower())

    def test_two_lines_or_a_commented_line_are_not_edited(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw")
        self.grub(host, 'GRUB_CMDLINE_LINUX_DEFAULT="a"\nGRUB_CMDLINE_LINUX_DEFAULT="b"\n')
        self.assertFalse(kp.plan(host).can_apply)
        host = self.machine(cmdline="BOOT_IMAGE=/v rw")
        self.grub(host, '#GRUB_CMDLINE_LINUX_DEFAULT="a"\n')
        self.assertFalse(kp.plan(host).can_apply)

    def test_debian_style_regen(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw", tools=("sudo", "update-grub"))
        self.grub(host)
        self.assertEqual(kp.plan(host).regen, ["update-grub"])

    def test_fedora_style_regen(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw", tools=("sudo", "grub2-mkconfig"))
        self.put(host, "boot/grub2/grub.cfg", "x\n")
        self.put(host, "etc/default/grub", GRUB)
        self.assertEqual(kp.plan(host).regen, ["grub2-mkconfig", "-o", str(host.boot_dirs[0] / "grub2/grub.cfg")])

    def test_no_grub_tool_means_print_only(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw", tools=("sudo",))
        self.grub(host)
        plan = kp.plan(host)
        self.assertEqual(plan.method, "grub")
        self.assertFalse(plan.can_apply)
        self.assertIn("grub-mkconfig", plan.reason)

    def test_a_grub_file_without_a_grub_cfg_and_without_boot_image_is_not_grub(self):
        host = self.machine()
        self.put(host, "etc/default/grub", GRUB)
        self.assertEqual(kp.plan(host).method, "")


class SystemdBootTest(Base):
    ENTRY = "title Arch Linux\nlinux /vmlinuz-linux\ninitrd /initramfs-linux.img\noptions root=/dev/nvme0n1p2 rw quiet\n"

    def host(self, **kw):
        return self.machine(cmdline="initrd=\\initramfs-linux.img root=/dev/nvme0n1p2 rw quiet", **kw)

    def test_the_entry_that_matches_the_running_line_is_changed(self):
        host = self.host()
        f = self.put(host, "boot/loader/entries/arch.conf", self.ENTRY)
        self.put(host, "boot/loader/entries/old.conf", self.ENTRY.replace("quiet", "debug"))
        plan = kp.plan(host)
        self.assertEqual((plan.method, plan.file, plan.can_apply, plan.regen), ("systemd-boot", f, True, None))
        self.assertEqual(plan.new_text, self.ENTRY.replace("rw quiet", f"rw quiet {LINE}"))

    def test_the_entry_on_the_efi_folder_is_found_too(self):
        host = self.host()
        f = self.put(host, "efi/loader/entries/arch.conf", self.ENTRY)
        self.assertEqual(kp.plan(host).file, f)

    def test_two_matching_entries_mean_print_only_and_both_are_named(self):
        host = self.host()
        a = self.put(host, "boot/loader/entries/a.conf", self.ENTRY)
        b = self.put(host, "boot/loader/entries/b.conf", self.ENTRY)
        plan = kp.plan(host)
        self.assertFalse(plan.can_apply)
        text = "\n".join(kp.describe(plan, host))
        self.assertIn(str(a), text)
        self.assertIn(str(b), text)

    def test_no_entry_matches_the_running_line(self):
        host = self.host()
        self.put(host, "boot/loader/entries/a.conf", self.ENTRY.replace("quiet", "debug"))
        self.assertEqual(kp.plan(host).method, "")

    def test_an_entry_with_two_options_lines_is_not_edited(self):
        host = self.host()
        self.put(host, "boot/loader/entries/a.conf", self.ENTRY + "options extra\n")
        self.assertFalse(kp.plan(host).can_apply)


class UkiTest(Base):
    def host(self, **kw):
        return self.machine(cmdline="root=/dev/nvme0n1p2 rw quiet", **kw)

    def test_the_cmdline_file_that_matches_is_changed_and_the_image_is_built_again(self):
        host = self.host(tools=("sudo", "mkinitcpio"))
        f = self.put(host, "etc/kernel/cmdline", "root=/dev/nvme0n1p2 rw quiet\n")
        plan = kp.plan(host)
        self.assertEqual((plan.method, plan.file, plan.can_apply), ("uki", f, True))
        self.assertEqual(plan.new_text, f"root=/dev/nvme0n1p2 rw quiet {LINE}\n")
        self.assertEqual(plan.regen, ["mkinitcpio", "-P"])

    def test_without_mkinitcpio_it_is_print_only_with_the_reason(self):
        host = self.host(tools=("sudo",))
        self.put(host, "etc/kernel/cmdline", "root=/dev/nvme0n1p2 rw quiet\n")
        plan = kp.plan(host)
        self.assertEqual(plan.method, "uki")
        self.assertFalse(plan.can_apply)
        self.assertIn("mkinitcpio", plan.reason)

    def test_a_cmdline_file_that_differs_from_the_running_line_is_not_used(self):
        host = self.host(tools=("sudo", "mkinitcpio"))
        self.put(host, "etc/kernel/cmdline", "root=/dev/other rw\n")
        self.assertEqual(kp.plan(host).method, "")

    def test_the_text_warns_about_the_recovery_key(self):
        host = self.host(tools=("sudo", "mkinitcpio"))
        self.put(host, "etc/kernel/cmdline", "root=/dev/nvme0n1p2 rw quiet\n")
        text = "\n".join(kp.describe(kp.plan(host), host))
        self.assertIn("recovery key", text)


class UnknownTest(Base):
    def test_two_methods_that_match_mean_print_only_and_name_each(self):
        host = self.machine(cmdline="root=/dev/nvme0n1p2 rw quiet", tools=("sudo", "mkinitcpio"))
        self.put(host, "etc/kernel/cmdline", "root=/dev/nvme0n1p2 rw quiet\n")
        self.put(host, "boot/loader/entries/a.conf", "options root=/dev/nvme0n1p2 rw quiet\n")
        plan = kp.plan(host)
        self.assertFalse(plan.can_apply)
        text = "\n".join(kp.describe(plan, host))
        self.assertIn("etc/kernel/cmdline", text)
        self.assertIn("a.conf", text)

    def test_nothing_found_prints_the_line_and_every_way(self):
        host = self.machine()
        plan = kp.plan(host)
        self.assertEqual((plan.method, plan.can_apply), ("", False))
        text = "\n".join(kp.describe(plan, host))
        for word in (LINE, "GRUB_CMDLINE_LINUX_DEFAULT", "systemd-boot", "/etc/kernel/cmdline", "cat /proc/cmdline"):
            self.assertIn(word, text)

    def test_nothing_to_change_gives_no_plan(self):
        host = self.machine(cmdline="root=/x " + LINE)
        self.assertIsNone(kp.plan(host))

    def test_no_sudo_means_print_only(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw", tools=("grub-mkconfig",))
        self.grub(host)
        plan = kp.plan(host)
        self.assertFalse(plan.can_apply)
        self.assertIn("sudo", plan.reason)

    def test_an_unreadable_boot_folder_is_said_and_never_a_guess(self):
        host = self.machine(cmdline="root=/x rw")
        host.boot_dirs[0].chmod(0)
        self.addCleanup(host.boot_dirs[0].chmod, 0o755)
        if os.access(host.boot_dirs[0], os.R_OK):
            self.skipTest("this user reads any folder")
        plan = kp.plan(host)
        self.assertFalse(plan.can_apply)
        self.assertIn("read", "\n".join(kp.describe(plan, host)))


class DescribeTest(Base):
    def test_the_text_has_the_change_the_commands_and_the_restart_note(self):
        host = self.machine(cmdline="BOOT_IMAGE=/v rw")
        f = self.grub(host)
        text = "\n".join(kp.describe(kp.plan(host), host))
        for word in (LINE, str(f), '- GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"',
                     f'+ GRUB_CMDLINE_LINUX_DEFAULT="quiet splash {LINE}"', "sudo cp -p", ".crucible-bak-",
                     "sudo install -m 644", "sudo grub-mkconfig -o", "never restarts"):
            self.assertIn(word, text)


class ApplyTest(Base):
    def setUp(self):
        super().setUp()
        self.work = self.t / "work/.crucible"
        self.work.mkdir(parents=True)
        self.host = self.machine(cmdline="BOOT_IMAGE=/v rw")
        self.file = self.grub(self.host)
        self.plan = kp.plan(self.host)
        self.said = []

    def apply(self):
        return kp.apply(self.host, self.plan, self.work, self.said.append)

    def test_a_good_run_backs_up_writes_and_builds_the_menu_in_this_order(self):
        os.chmod(self.file, 0o640)
        self.apply()
        calls = self.host.runner.root_calls
        self.assertEqual([c[0] for c in calls], ["cp", "install", "grub-mkconfig"])
        backup = Path(calls[0][3])
        self.assertTrue(backup.name.startswith("grub.crucible-bak-"))
        self.assertEqual(backup.parent, self.file.parent)
        self.assertEqual(backup.read_text(), GRUB)
        self.assertEqual(self.file.read_text(), self.plan.new_text)
        self.assertEqual(calls[1][1:3], ["-m", "640"])
        self.assertEqual(stat.S_IMODE(self.file.stat().st_mode), 0o640)
        self.assertEqual(calls[2], self.plan.regen)
        self.assertTrue(any("restart" in s.lower() for s in self.said))
        self.assertEqual(os.listdir(self.work), [])   # the temp file is gone

    def test_the_temp_file_is_the_registered_source_and_is_in_the_work_folder(self):
        self.apply()
        src = self.host.runner.root_sources[1]
        self.assertEqual(len(src), 1)
        self.assertEqual(Path(src[0]).parent, self.work)
        self.assertEqual(self.host.runner.root_calls[1][3], src[0])

    def test_nothing_restarts_the_machine(self):
        self.apply()
        flat = [w for c in self.host.runner.root_calls + self.host.runner.calls for w in c]
        for word in ("reboot", "shutdown", "systemctl", "poweroff", "halt"):
            self.assertNotIn(word, flat)

    def test_a_file_that_changed_since_the_plan_is_not_touched(self):
        self.file.write_text(GRUB + "# changed\n")
        with self.assertRaises(kp.Failed):
            self.apply()
        self.assertEqual(self.host.runner.root_calls, [])
        self.assertEqual(self.file.read_text(), GRUB + "# changed\n")

    def test_a_failed_backup_stops_before_the_write(self):
        self.host.runner.root_fail = {"cp"}
        with self.assertRaises(kp.Failed):
            self.apply()
        self.assertEqual([c[0] for c in self.host.runner.root_calls], ["cp"])
        self.assertEqual(self.file.read_text(), GRUB)

    def test_a_failed_menu_build_restores_the_backup_and_says_so(self):
        self.host.runner.root_fail = {"grub-mkconfig"}
        with self.assertRaises(kp.Failed) as cm:
            self.apply()
        self.assertEqual(self.file.read_text(), GRUB)
        self.assertEqual([c[0] for c in self.host.runner.root_calls], ["cp", "install", "grub-mkconfig", "cp"])
        self.assertIn("restored", str(cm.exception).lower())

    def test_a_failed_restore_is_told_with_the_manual_step(self):
        class Flaky(FakeRunner):
            def run_root(self, argv, sources=()):
                res = super().run_root(argv, sources)
                if argv[0] == "cp" and len(self.root_calls) > 1:   # the restore, not the backup
                    self.root_fail.add("cp")
                    return type(res)(1, "no space")
                return res
        self.host.runner = Flaky()
        self.host.runner.root_fail = {"grub-mkconfig"}
        with self.assertRaises(kp.Failed) as cm:
            self.apply()
        self.assertIn("restore failed", str(cm.exception).lower())
        self.assertIn("crucible-bak", str(cm.exception))

    def test_a_backup_name_that_exists_is_not_overwritten(self):
        stamp = kp.stamp(self.host)
        (self.file.parent / f"grub.crucible-bak-{stamp}").write_text("old backup")
        with self.assertRaises(kp.Failed):
            self.apply()
        self.assertEqual(self.host.runner.root_calls, [])

    def test_a_print_only_plan_cannot_be_applied(self):
        shutil.rmtree(self.t / "machine")
        host = self.machine()
        plan = kp.plan(host)
        with self.assertRaises(kp.Failed):
            kp.apply(host, plan, self.work, self.said.append)
        self.assertEqual(host.runner.root_calls, [])

    def test_a_systemd_boot_entry_needs_no_menu_build(self):
        shutil.rmtree(self.t / "machine")
        host = self.machine(cmdline="initrd=\\i.img root=/x rw")
        self.put(host, "boot/loader/entries/a.conf", "title x\noptions root=/x rw\n")
        plan = kp.plan(host)
        kp.apply(host, plan, self.work, self.said.append)
        self.assertEqual([c[0] for c in host.runner.root_calls], ["cp", "install"])
        self.assertEqual(plan.file.read_text(), f"title x\noptions root=/x rw {LINE}\n")


class RealCommandsTest(unittest.TestCase):
    """The commands of a plan with the real paths pass the allowlist of the real runner. Nothing runs."""

    def plan(self, method, file, regen):
        return kp.Plan(tokens=TOKENS, method=method, file=Path(file), old_text="a\n", new_text="b\n", regen=regen,
                       can_apply=True)

    def test_each_plan_gives_commands_that_the_runner_accepts(self):
        for plan in (self.plan("grub", "/etc/default/grub", ["grub-mkconfig", "-o", "/boot/grub/grub.cfg"]),
                     self.plan("grub", "/etc/default/grub", ["update-grub"]),
                     self.plan("systemd-boot", "/boot/loader/entries/a.conf", None),
                     self.plan("systemd-boot", "/efi/loader/entries/a.conf", None),
                     self.plan("uki", "/etc/kernel/cmdline", ["mkinitcpio", "-P"])):
            for argv in plan.commands("20261111-000000", "/srv/u/repo/.crucible/kernel-param.new"):
                self.assertIsNone(runner.root_problem(argv, ["/srv/u/repo/.crucible/kernel-param.new"]), argv)

    def test_the_runner_refuses_everything_else(self):
        src = ["/srv/u/.crucible/new"]
        bad = [["rm", "-rf", "/etc/x"], ["sh", "-c", "id"], ["/usr/bin/cp", "-p", "/etc/a", "/etc/b"],
               ["cp", "-p", "/etc/shadow", "/srv/u/shadow"], ["cp", "-p", "/srv/u/x", "/etc/b"],
               ["install", "-m", "644", "/srv/u/other", "/etc/b"], ["install", "-m", "4755", src[0], "/etc/b"],
               ["install", "-m", "644", src[0], "/srv/u/b"], ["cp", "-p", "/etc/../root/x", "/etc/b"],
               ["cp", "-p", "etc/a", "/etc/b"], ["cp", "-r", "/etc/a", "/etc/b"], ["mkinitcpio", "-P", "-c", "/tmp/x"],
               ["grub-mkconfig", "-o", "/srv/u/grub.cfg"], ["cp"], [], ["install", "-m", "644", src[0], "/etc/b", "/etc/c"],
               ["sudo", "true"], ["cp", "-p", "/etc/a", "/etcx/b"]]
        for argv in bad:
            self.assertIsNotNone(runner.root_problem(argv, src), argv)
            with mock.patch("subprocess.run") as sp, mock.patch("subprocess.Popen") as po:
                with self.assertRaises(runner.Forbidden, msg=argv):
                    runner.Runner().run_root(argv, sources=src)
                sp.assert_not_called()
                po.assert_not_called()

    def test_a_good_command_goes_to_sudo_after_two_dashes_with_the_terminal_input(self):
        with mock.patch("subprocess.run") as sp:
            sp.return_value = mock.Mock(returncode=0, stdout="")
            res = runner.Runner().run_root(["cp", "-p", "/etc/default/grub", "/etc/default/grub.bak"])
        argv = sp.call_args.args[0]
        self.assertEqual(argv, ["sudo", "--", "cp", "-p", "/etc/default/grub", "/etc/default/grub.bak"])
        self.assertIsNone(sp.call_args.kwargs.get("stdin"))   # inherited, so sudo can ask on the terminal
        self.assertEqual(res.rc, 0)

    def test_the_old_entries_still_refuse_sudo(self):
        for name in ("sudo", "su", "doas", "pkexec"):
            with self.assertRaises(runner.Forbidden):
                runner.Runner().run([name, "true"])
            with self.assertRaises(runner.Forbidden):
                runner.Runner().spawn([name, "true"])


class SetupBase(Base):
    def setUp(self):
        super().setUp()
        self.repo = make_repo(self.t)

    def run_setup(self, argv=(), answers=(), interactive=False, prep=None, **machine):
        machine.setdefault("cmdline", "BOOT_IMAGE=/v rw quiet")
        # the first question of an interactive run is "Download ... and build now?"; the kernel question comes after it
        host = wire(self.t, gtt=SMALL, prompt=FakePrompt(["y", *answers]) if interactive else None, **machine)
        host.is_tty = interactive
        if prep:
            prep(host)
        base = ["--state-dir", str(self.t / "home/states")]
        if interactive:
            base += ["--pack", "4.05", "--models-dir", str(self.repo / "models"), "--port", "8790", "--service", "no"]
        else:
            base += ["--yes"]
        code = cli.main(["setup", *base, *argv], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def grub_machine(self, host):
        self.put(host, "boot/grub/grub.cfg", "x\n")
        self.put(host, "etc/default/grub", GRUB)
        self.grub_file = host.etc / "default/grub"


class SetupTest(SetupBase):
    def test_interactive_yes_applies_the_change_and_says_to_restart(self):
        code, host, out = self.run_setup(interactive=True, answers=["yes"], prep=self.grub_machine, tools=("make", "g++", "git", "sudo", "grub-mkconfig"))
        self.assertEqual(code, 1, out)
        self.assertEqual([c[0] for c in host.runner.root_calls], ["cp", "install", "grub-mkconfig"])
        self.assertIn(LINE, self.grub_file.read_text())
        self.assertIn("estart", out)
        self.assertEqual(host.runner.spawned, [])
        self.assertTrue(any("yes" in q for q in host.prompt.asked), host.prompt.asked)

    def test_every_other_answer_changes_nothing_and_prints_the_change(self):
        for answer in ("", "y", "no", "n", "YES please", "ja"):
            code, host, out = self.run_setup(interactive=True, answers=[answer], prep=self.grub_machine,
                                             tools=("make", "g++", "git", "sudo", "grub-mkconfig"))
            self.assertEqual(host.runner.root_calls, [], answer)
            self.assertEqual(self.grub_file.read_text(), GRUB, answer)
            self.assertIn(f'+ GRUB_CMDLINE_LINUX_DEFAULT="quiet splash {LINE}"', out)

    def test_the_word_yes_is_taken_with_spaces_and_capitals(self):
        code, host, out = self.run_setup(interactive=True, answers=["  Yes "], prep=self.grub_machine,
                                         tools=("make", "g++", "git", "sudo", "grub-mkconfig"))
        self.assertEqual(len(host.runner.root_calls), 3)

    def test_the_yes_flag_never_applies_and_never_asks(self):
        code, host, out = self.run_setup(prep=self.grub_machine, tools=("make", "g++", "git", "sudo", "grub-mkconfig"))
        self.assertEqual(host.runner.root_calls, [])
        self.assertEqual(self.grub_file.read_text(), GRUB)
        self.assertIn(LINE, out)
        self.assertIn("sudo grub-mkconfig", out)

    def test_a_dry_run_applies_nothing(self):
        code, host, out = self.run_setup(["--dry-run"], prep=self.grub_machine)
        self.assertEqual(host.runner.root_calls, [])

    def test_an_unknown_boot_method_prints_only_and_does_not_ask(self):
        code, host, out = self.run_setup(interactive=True, answers=["yes"])
        self.assertEqual(host.runner.root_calls, [])
        self.assertFalse(any("yes" in q for q in host.prompt.asked), host.prompt.asked)
        self.assertIn("GRUB_CMDLINE_LINUX_DEFAULT", out)

    def test_parameters_that_are_on_the_line_already_mean_restart_and_no_plan(self):
        code, host, out = self.run_setup(interactive=True, answers=["yes"], prep=self.grub_machine,
                                         cmdline="BOOT_IMAGE=/v rw " + LINE)
        self.assertEqual(host.runner.root_calls, [])
        self.assertNotIn("sudo cp", out)

    def test_a_machine_with_enough_gtt_gets_no_plan(self):
        host = wire(self.t, gtt=133_000_000_000, prompt=FakePrompt(["y", "yes"]))
        host.is_tty = True
        self.grub_machine(host)
        host.lines = []
        cli.main(["setup", "--state-dir", str(self.t / "home/states"), "--pack", "4.05", "--models-dir", str(self.repo / "models"),
                  "--port", "8790", "--service", "no"], host=host, repo=self.repo)
        self.assertEqual(host.runner.root_calls, [])
        self.assertFalse(any("yes" in q for q in host.prompt.asked))

    def test_the_doctor_never_applies(self):
        host = wire(self.t, gtt=SMALL)
        self.grub_machine(host)
        cli.main(["doctor"], host=host, repo=self.repo)
        self.assertEqual(host.runner.root_calls, [])
        self.assertEqual(self.grub_file.read_text(), GRUB)

    def test_no_option_applies_a_kernel_change(self):
        import argparse
        words = []
        for action in cli.parser()._actions + [a for sub in cli.parser()._subparsers._group_actions for p in sub.choices.values() for a in p._actions]:
            words += action.option_strings
        for w in words:
            self.assertFalse(any(k in w for k in ("kernel", "boot", "cmdline", "grub")), w)

    def test_a_failed_apply_is_told_and_the_run_ends_with_the_usual_failure(self):
        def prep(host):
            self.grub_machine(host)
            host.runner.root_fail = {"grub-mkconfig"}
        code, host, out = self.run_setup(interactive=True, answers=["yes"], prep=prep, tools=("make", "g++", "git", "sudo", "grub-mkconfig"))
        self.assertEqual(code, 1)
        self.assertEqual(self.grub_file.read_text(), GRUB)
        self.assertIn("restored", out.lower())


class TextClaimsTest(unittest.TestCase):
    def test_the_texts_no_longer_say_that_the_tool_never_changes_the_boot_settings(self):
        for text in (doctor.KERNEL_FIX, cli.DESCRIPTION):
            self.assertNotIn("never changes the boot loader", text)
            self.assertNotIn("never uses sudo", text)
        self.assertIn("yes", doctor.KERNEL_FIX)


if __name__ == "__main__":
    unittest.main()
