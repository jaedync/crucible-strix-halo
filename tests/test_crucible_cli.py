"""Tests of ./crucible setup and ./crucible doctor. No GPU, no network, no sudo, no real /proc or /sys."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from crucible_cli_fakes import (FakePrompt, FakeRunner, FakeHttp, NoPrompt, ROOT, make_repo, wire)  # noqa: E402

from crucible_cli import cli, doctor, files, launch, measured, packs  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

GTT_LINE = "amdgpu.gttsize=126976 ttm.pages_limit=32505856"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)

    def settings(self, host, **kw):
        return settings_mod.build(self.repo, host, **kw)

    def checks(self, **machine):
        host = wire(self.t, **machine)
        results = doctor.run_checks(host, self.settings(host))
        return {r.id: r for r in results}

    def setup(self, argv, host=None, **machine):
        host = host or wire(self.t, **machine)
        code = cli.main(["setup", *argv], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)


class DoctorTest(Base):
    def test_healthy_machine_has_no_failure(self):
        res = self.checks()
        self.assertEqual([r.id for r in res.values() if r.status == "fail"], [])
        self.assertEqual(res["gtt"].status, "pass")
        self.assertEqual(res["rocm-version"].status, "pass")

    def test_every_result_has_a_status_and_a_fix_when_it_fails(self):
        res = self.checks(gtt=60_000_000_000, version="6.4.0", py=(3, 9, 0))
        for r in res.values():
            self.assertIn(r.status, {"pass", "fail", "warn", "skip"})
            if r.status == "fail":
                self.assertTrue(r.fix, r.id)

    def test_old_python_fails_with_the_fix(self):
        r = self.checks(py=(3, 11, 4))["python"]
        self.assertEqual(r.status, "fail")
        self.assertIn("3.12", r.fix)

    def test_not_linux_fails(self):
        self.assertEqual(self.checks(platform="darwin")["os"].status, "fail")

    def test_no_rocm_fails_and_names_each_distribution(self):
        res = self.checks(rocm=False, hipcc=False)
        self.assertEqual(res["rocm-version"].status, "fail")
        self.assertEqual(res["hipcc"].status, "fail")
        for word in ("Arch", "Ubuntu", "Fedora"):
            self.assertIn(word, res["hipcc"].fix)
        self.assertIn("7.2.4", res["hipcc"].fix)

    def test_other_rocm_version_is_a_warning_not_a_failure(self):
        r = self.checks(version="7.1.0")["rocm-version"]
        self.assertEqual(r.status, "warn")
        self.assertIn("7.2.4", r.detail)

    def test_wrong_gpu_architecture_fails(self):
        host = wire(self.t)
        host.runner.gfx = "gfx1100"
        res = {r.id: r for r in doctor.run_checks(host, self.settings(host))}
        self.assertEqual(res["gpu-arch"].status, "fail")
        self.assertIn("gfx1151", res["gpu-arch"].detail)

    def test_small_gtt_prints_the_exact_kernel_line_and_why(self):
        r = self.checks(gtt=60_000_000_000)["gtt"]
        self.assertEqual(r.status, "fail")
        self.assertIn(GTT_LINE, r.fix)
        self.assertIn("never changes", r.fix)  # the tool does not edit the boot loader
        self.assertIn("memory", r.detail)

    def test_gtt_fix_covers_each_boot_method_and_does_not_apply_any(self):
        r = self.checks(gtt=60_000_000_000)["gtt"]
        for word in ("/boot/loader/entries/", "GRUB_CMDLINE_LINUX_DEFAULT", "/etc/kernel/cmdline", "cat /proc/cmdline"):
            self.assertIn(word, r.fix)
        self.assertEqual(r.blocks, ("start",))

    def test_gpu_memory_in_use_by_another_program_fails(self):
        r = self.checks(gtt_used=90_000_000_000)["gpu-free"]
        self.assertEqual(r.status, "fail")
        self.assertIn("another program", r.fix)

    def test_low_total_memory_fails(self):
        self.assertEqual(self.checks(mem_total_kb=32_000_000)["memory"].status, "fail")

    def test_missing_build_tools_fail_one_by_one(self):
        res = self.checks(tools=("git",))
        self.assertEqual(res["tool-make"].status, "fail")
        self.assertEqual(res["tool-cxx"].status, "warn")  # the engine builds with hipcc
        self.assertEqual(res["tool-git"].status, "pass")

    def test_gpu_device_access_fails_without_group_and_prints_text_only_fix(self):
        res = self.checks(deny=("dev/kfd",), groups=("users",))
        self.assertEqual(res["gpu-access"].status, "fail")
        self.assertIn("render", res["gpu-access"].fix)

    def test_small_disk_fails_for_the_pack(self):
        r = self.checks(free=50_000_000_000)["disk-models"]
        self.assertEqual(r.status, "fail")
        self.assertIn("GB", r.detail)

    def test_3_05_needs_less_disk_than_4_05(self):
        host = wire(self.t, free=100_000_000_000)
        r405 = {x.id: x for x in doctor.run_checks(host, self.settings(host, pack="4.05"))}["disk-models"]
        r305 = {x.id: x for x in doctor.run_checks(host, self.settings(host, pack="3.05"))}["disk-models"]
        self.assertEqual((r405.status, r305.status), ("fail", "pass"))

    def test_busy_port_fails(self):
        r = self.checks(busy_ports=(8790,))["port"]
        self.assertEqual(r.status, "fail")
        self.assertIn("--port", r.fix)

    def test_state_dir_outside_home_fails(self):
        host = wire(self.t)
        s = self.settings(host, state_dir=Path("/var/tmp/crucible-states"))
        r = {x.id: x for x in doctor.run_checks(host, s)}["state-dir"]
        self.assertEqual(r.status, "fail")
        self.assertIn("home", r.detail)

    def test_pack_not_downloaded_is_a_skip_and_incomplete_pack_is_a_failure(self):
        host = wire(self.t)
        s = self.settings(host)
        self.assertEqual({x.id: x for x in doctor.run_checks(host, s)}["pack"].status, "skip")
        s.ckpt.mkdir(parents=True)
        (s.ckpt / "config.json").write_text("{}")
        r = {x.id: x for x in doctor.run_checks(host, s)}["pack"]
        self.assertEqual(r.status, "fail")
        self.assertIn("ngram_embedding.safetensors", r.detail)

    def test_pack_with_a_missing_shard_fails_and_a_partial_file_fails(self):
        host = wire(self.t)
        s = self.settings(host)
        s.ckpt.mkdir(parents=True)
        for n in ("config.json", "tokenizer.json", "ngram_embedding.safetensors"):
            (s.ckpt / n).write_text("x")
        (s.ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-1.safetensors"}}))
        r = {x.id: x for x in doctor.run_checks(host, s)}["pack"]
        self.assertEqual(r.status, "fail")
        self.assertIn("m-1.safetensors", r.detail)
        (s.ckpt / "m-1.safetensors").write_text("x")
        (s.ckpt / ".cache/huggingface/download").mkdir(parents=True)
        (s.ckpt / ".cache/huggingface/download/m-2.safetensors.incomplete").write_text("x")
        r = {x.id: x for x in doctor.run_checks(host, s)}["pack"]
        self.assertEqual(r.status, "fail")
        self.assertIn("incomplete", r.detail)

    def test_text_report_one_line_per_check_with_fix_lines(self):
        host = wire(self.t, gtt=60_000_000_000)
        code = cli.main(["doctor"], host=host, repo=self.repo)
        text = "\n".join(host.lines)
        self.assertEqual(code, 1)
        self.assertRegex(text, r"(?m)^FAIL +gtt")
        self.assertRegex(text, r"(?m)^PASS +python")
        self.assertIn("fix:", text)

    def test_doctor_exit_zero_on_a_healthy_machine_and_json_works(self):
        host = wire(self.t)
        self.assertEqual(cli.main(["doctor", "--json"], host=host, repo=self.repo), 0)
        data = json.loads("\n".join(host.lines))
        self.assertTrue(all({"id", "status", "detail", "fix"} <= set(x) for x in data["checks"]))

    def test_doctor_never_runs_a_command_that_needs_root(self):
        host = wire(self.t, gtt=1)
        cli.main(["doctor"], host=host, repo=self.repo)
        for argv in host.runner.calls:
            self.assertNotIn(Path(argv[0]).name, files.FORBIDDEN_COMMANDS)


class SetupTest(Base):
    def test_yes_runs_the_whole_chain_without_a_prompt(self):
        code, host, out = self.setup(["--yes", "--state-dir", str(self.t / "home/states")])
        self.assertEqual(code, 0, out)
        names = [Path(c[0]).name for c in host.runner.calls]
        for expect in ("python", "hf", "make"):
            self.assertIn(expect, names)
        self.assertEqual(len(host.runner.spawned), 1)
        self.assertEqual(len(host.http.posts), 1)
        self.assertIn("http://127.0.0.1:8790/v1", out)
        self.assertIn("model", out)

    def test_without_yes_and_without_a_terminal_setup_refuses(self):
        code, host, out = self.setup([])
        self.assertEqual(code, 2)
        self.assertIn("--yes", out)
        self.assertEqual(host.runner.spawned, [])

    def test_interactive_run_asks_and_takes_the_defaults(self):
        host = wire(self.t, prompt=FakePrompt())
        host.is_tty = True
        code, host, out = self.setup([], host=host)
        self.assertEqual(code, 0, out)
        self.assertTrue(host.prompt.asked)

    def test_interactive_answer_selects_the_3_05_pack(self):
        host = wire(self.t, prompt=FakePrompt(["3.05"]))
        host.is_tty = True
        code, host, out = self.setup([], host=host)
        self.assertEqual(code, 0, out)
        self.assertIn("3.05bpw_h5_ng5", " ".join(host.runner.spawned[0]["argv"]))

    def test_default_pack_is_4_05_with_the_daily_layout(self):
        code, host, out = self.setup(["--yes"])
        spawn = host.runner.spawned[0]
        argv = spawn["argv"]
        self.assertIn("4.05bpw_h6_ng6", " ".join(argv))
        self.assertEqual(spawn["env"]["CRUCIBLE_PLE_HOST"], "1")
        self.assertEqual(spawn["env"]["CRUCIBLE_PF_LAUNCH_ROWS"], "16384+128")   # README words: tests/test_service_config.py
        self.assertEqual(argv[argv.index("--streams") + 1], "4")
        self.assertEqual(argv[argv.index("--ctx") + 1], "262144")
        self.assertEqual(argv[argv.index("--conversation-cache-mib") + 1], "9700")
        self.assertEqual(argv[argv.index("--host-reserve-mib") + 1], "11264")
        self.assertEqual(argv[argv.index("--state-disk-gib") + 1], "200")
        self.assertIn("--state-dir", argv)
        for flag in ("--vision", "--clamp-max-tokens"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--default-thinking") + 1], "on")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(spawn["cwd"], str(self.repo))

    def test_pack_3_05_gets_the_same_launch_rows_as_the_4_05_pack(self):
        code, host, out = self.setup(["--yes", "--pack", "3.05"])
        spawn = host.runner.spawned[0]
        self.assertEqual(spawn["env"]["CRUCIBLE_PF_LAUNCH_ROWS"], "16384+128")
        self.assertEqual(spawn["env"]["CRUCIBLE_PLE_HOST"], "1")
        hf = [c for c in host.runner.calls if Path(c[0]).name == "hf"][0]
        self.assertEqual(hf[hf.index("--revision") + 1], "3.05bpw_h5_ng5")
        self.assertEqual(hf[2], "turboderp/Qwen3.8-Flash-Next-exl3")

    def test_pack_option_rejects_other_values(self):
        with self.assertRaises(SystemExit) as cm:
            cli.main(["setup", "--yes", "--pack", "4.18"], host=wire(self.t), repo=self.repo)
        self.assertEqual(cm.exception.code, 2)

    def test_state_dir_must_be_inside_the_home_folder(self):
        code, host, out = self.setup(["--yes", "--state-dir", "/var/tmp/crucible-states"])
        self.assertEqual(code, 2)
        self.assertIn("home", out)
        self.assertEqual(host.runner.calls, [])

    def test_no_state_dir_leaves_out_the_disk_tier(self):
        code, host, out = self.setup(["--yes", "--no-state-dir"])
        argv = host.runner.spawned[0]["argv"]
        self.assertNotIn("--state-dir", argv)
        self.assertNotIn("--state-disk-gib", argv)

    def test_state_folder_is_made_with_mode_0700(self):
        sd = self.t / "home/crucible-states"
        self.setup(["--yes", "--state-dir", str(sd)])
        self.assertEqual(sd.stat().st_mode & 0o777, 0o700)

    def test_config_and_serve_script_are_written_in_the_work_folder(self):
        self.setup(["--yes"])
        cfg = json.loads((self.repo / ".crucible/config.json").read_text())
        self.assertEqual(cfg["pack"], "4.05")
        serve = (self.repo / ".crucible/serve.sh").read_text()
        self.assertIn("CRUCIBLE_PLE_HOST=1", serve)
        self.assertIn("-m server", serve)
        self.assertTrue(os.access(self.repo / ".crucible/serve.sh", os.X_OK))

    def test_venv_installs_the_requirements_and_the_download_client(self):
        code, host, out = self.setup(["--yes"])
        pips = [c for c in host.runner.calls if c[1:3] == ["-m", "pip"]]
        self.assertIn(str(self.repo / "requirements-cpu.txt"), sum(pips, []))
        self.assertIn("huggingface_hub", sum(pips, []))

    def test_make_drift_error_retries_with_split_check_off(self):
        host = wire(self.t)
        host.runner.make_output = "error: argument facts drift\n"
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 0, out)
        makes = [c for c in host.runner.calls if Path(c[0]).name == "make"]
        self.assertEqual(len(makes), 2)
        self.assertIn("SPLIT=0", makes[1])

    def test_other_build_error_stops_with_the_log_path(self):
        host = wire(self.t, runner=FakeRunner(fail=("make",)))
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 1)
        self.assertIn("build.log", out)
        self.assertEqual(host.runner.spawned, [])

    def test_resume_after_a_stopped_download_skips_the_done_steps(self):
        host = wire(self.t, runner=FakeRunner(fail=("download",)))
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 1)
        self.assertIn("download", out)
        state = json.loads((self.repo / ".crucible/setup-state.json").read_text())
        self.assertEqual(state["steps"]["venv"]["status"], "done")
        self.assertNotEqual(state["steps"]["download"]["status"], "done")
        host.runner.calls.clear()
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 0, out)
        self.assertFalse([c for c in host.runner.calls if c[1:3] == ["-m", "venv"]])
        self.assertTrue([c for c in host.runner.calls if Path(c[0]).name == "hf"])
        self.assertIn("Resuming", out)

    def test_second_run_with_everything_done_only_checks_and_starts(self):
        self.setup(["--yes", "--smoke"])
        host = wire(self.t)
        host.runner.calls.clear()
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 0, out)
        self.assertFalse([c for c in host.runner.calls if Path(c[0]).name == "hf"])
        self.assertFalse([c for c in host.runner.calls if c[1:3] == ["-m", "venv"]])

    def test_small_gtt_downloads_and_builds_but_does_not_start(self):
        code, host, out = self.setup(["--yes"], gtt=60_000_000_000)
        self.assertEqual(code, 1)
        self.assertIn(GTT_LINE, out)
        self.assertTrue([c for c in host.runner.calls if Path(c[0]).name == "hf"])
        self.assertTrue([c for c in host.runner.calls if Path(c[0]).name == "make"])
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("never changes", out)

    def test_missing_rocm_blocks_the_build_but_not_the_download(self):
        code, host, out = self.setup(["--yes"], rocm=False, hipcc=False)
        self.assertEqual(code, 1)
        self.assertTrue([c for c in host.runner.calls if Path(c[0]).name == "hf"])
        self.assertFalse([c for c in host.runner.calls if Path(c[0]).name == "make"])
        self.assertIn("7.2.4", out)

    def test_small_disk_blocks_the_download(self):
        code, host, out = self.setup(["--yes"], free=20_000_000_000)
        self.assertEqual(code, 1)
        self.assertFalse([c for c in host.runner.calls if Path(c[0]).name == "hf"])

    def test_busy_gpu_stops_before_the_start(self):
        code, host, out = self.setup(["--yes"], gtt_used=90_000_000_000)
        self.assertEqual(code, 1)
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("another program", out)

    def test_dry_run_prints_the_plan_and_runs_nothing(self):
        code, host, out = self.setup(["--yes", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(host.runner.spawned, [])
        self.assertEqual([c for c in host.runner.calls if Path(c[0]).name in ("hf", "make")], [])
        self.assertFalse((self.repo / ".venv").exists())
        for step in ("venv", "download", "build", "start", "test"):
            self.assertIn(step, out)

    def test_smoke_stops_the_server_after_the_test_request(self):
        code, host, out = self.setup(["--yes", "--smoke"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(host.runner.terminated), 1)
        self.assertEqual(host.runner.live, set())
        self.assertIn("first token", out)

    def test_server_that_dies_at_start_is_reported_with_the_log_tail(self):
        host = wire(self.t)
        host.runner.server_dies = True
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 1)
        self.assertIn("server.log", out)

    def test_timeline_records_each_step(self):
        self.setup(["--yes"])
        state = json.loads((self.repo / ".crucible/setup-state.json").read_text())
        for step in ("check", "venv", "download", "build", "config", "start", "test"):
            self.assertIn("seconds", state["steps"][step], step)

    def test_start_does_not_start_a_second_server(self):
        self.setup(["--yes"])
        host = wire(self.t)
        host.runner.live = {40001}
        code = cli.main(["start"], host=host, repo=self.repo)
        self.assertEqual(code, 0)
        self.assertEqual(host.runner.spawned, [])

    def test_stop_ends_only_the_recorded_process(self):
        self.setup(["--yes"])
        host = wire(self.t)
        host.runner.live = {40001}
        self.assertEqual(cli.main(["stop"], host=host, repo=self.repo), 0)
        self.assertEqual(host.runner.terminated, [40001])


class SafetyTest(Base):
    def test_no_command_in_a_full_run_uses_root_tools(self):
        for argv in (["--yes"], ["--yes", "--pack", "3.05"], ["--yes", "--smoke"]):
            code, host, out = self.setup(argv)
            for call in host.runner.calls:
                self.assertNotIn(Path(call[0]).name, files.FORBIDDEN_COMMANDS)

    def test_the_real_runner_refuses_sudo(self):
        from crucible_cli import runner
        for name in ("sudo", "/usr/bin/sudo", "doas", "pkexec", "su"):
            with self.assertRaises(runner.Forbidden):
                runner.Runner().run([name, "true"])
            with self.assertRaises(runner.Forbidden):
                runner.Runner().spawn([name, "true"])

    def test_the_real_runner_refuses_a_shell_string_and_a_shell(self):
        from crucible_cli import runner
        with self.assertRaises(runner.Forbidden):
            runner.Runner().run("make")
        with self.assertRaises(runner.Forbidden):
            runner.Runner().run(["sh", "-c", "sudo true"])

    def test_the_sandbox_refuses_boot_files_and_paths_outside_its_roots(self):
        box = files.Sandbox([self.t / "a", self.t / "b"])
        box.write_text(self.t / "a/x.txt", "ok")
        for bad in ("/boot/loader/entries/arch.conf", "/etc/default/grub", "/etc/kernel/cmdline",
                    str(self.t / "c/x.txt"), str(self.t / "a/../c/x.txt")):
            with self.assertRaises(files.OutsideRoots, msg=bad):
                box.write_text(Path(bad), "x")

    def test_the_sandbox_refuses_a_symlink_that_leaves_its_roots(self):
        (self.t / "a").mkdir()
        (self.t / "out").mkdir()
        (self.t / "a/link").symlink_to(self.t / "out")
        box = files.Sandbox([self.t / "a"])
        with self.assertRaises(files.OutsideRoots):
            box.write_text(self.t / "a/link/x.txt", "x")

    def test_a_full_run_writes_only_inside_the_repo_and_the_state_dir(self):
        host = wire(self.t)
        before = {p for p in self.t.rglob("*")}
        machine = self.t / "machine"
        sd = self.t / "home/states"
        self.setup(["--yes", "--state-dir", str(sd)], host=host)
        new = {p for p in self.t.rglob("*")} - before
        for p in new:
            if machine in p.parents:
                continue
            ok = any(root == p or root in p.parents for root in (self.repo, sd))
            self.assertTrue(ok, f"wrote outside the repo and the state folder: {p}")

    def test_source_never_calls_sudo_or_names_a_boot_file_for_writing(self):
        src = ROOT / "tools/crucible_cli"
        for path in list(src.glob("*.py")) + [ROOT / "crucible"]:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if path.name != "runner.py":
                        self.assertNotIn(name, {"system", "popen", "Popen", "call", "check_call", "check_output",
                                                "run_command", "execv", "execvp", "spawnl"}, f"{path.name}:{node.lineno}")
                    recv = getattr(f, "value", None) if isinstance(f, ast.Attribute) else None
                    via_box = (getattr(recv, "id", "") == "box") or (getattr(recv, "attr", "") == "box")
                    # box = a files.Sandbox. keyfile.py makes the API key file with O_EXCL and O_NOFOLLOW (a Sandbox
                    # cannot do that); steps.step_config calls box.check on the path first.
                    if path.name not in ("files.py", "keyfile.py") and not via_box:
                        self.assertNotIn(name, {"write_text", "write_bytes", "unlink", "rmtree", "rename",
                                                "chmod", "symlink_to", "copyfile", "copy2", "move", "truncate"},
                                         f"{path.name}:{node.lineno} writes outside the sandbox helper")
                        if name == "replace" and isinstance(f, ast.Attribute) and getattr(f.value, "id", "") == "os":
                            self.fail(f"{path.name}:{node.lineno} os.replace outside the sandbox helper")
                        if name == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                            self.assertNotRegex(str(node.args[1].value), "[wax+]", f"{path.name}:{node.lineno}")

    def test_source_uses_only_the_standard_library(self):
        std = set(sys.stdlib_module_names)
        local = {p.stem for p in (ROOT / "tools/crucible_cli").glob("*.py")} | {"crucible_cli"}
        for path in list((ROOT / "tools/crucible_cli").glob("*.py")) + [ROOT / "crucible"]:
            for node in ast.walk(ast.parse(path.read_text())):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    mods = [node.module]
                for mod in mods:
                    top = mod.split(".")[0]
                    self.assertTrue(top in std or top in local, f"{path.name} imports {mod}")

    def test_the_launcher_runs_help_and_is_executable(self):
        launcher = ROOT / "crucible"
        self.assertTrue(os.access(launcher, os.X_OK))
        proc = subprocess.run([sys.executable, str(launcher), "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for word in ("setup", "doctor", "start", "stop"):
            self.assertIn(word, proc.stdout)

    def test_launcher_doctor_runs_against_the_real_host_without_a_traceback(self):
        proc = subprocess.run([sys.executable, str(ROOT / "crucible"), "doctor"], capture_output=True, text=True,
                              timeout=120, env={**os.environ, "HOME": str(self.t)})
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertRegex(proc.stdout, r"(?m)^(PASS|FAIL|WARN|SKIP) +python")


class OwnHeadHookTest(Base):
    """Part 5 hook: the server command uses the user's own draft head only when its record exists and sits outside the repo."""

    def _settings(self):
        host = wire(self.t)
        s = self.settings(host)
        return s

    def test_default_has_no_sidecar_variable_and_the_dir_is_outside_the_repo(self):
        s = self._settings()
        self.assertNotIn("CRUCIBLE_MTP_SIDECAR", launch.server_command(s, "py")[0])
        self.assertEqual(s.own_head_dir, self.t / "home/.local/share/crucible/own-head")

    def test_xdg_data_home_moves_the_dir(self):
        host = wire(self.t)
        host.env = {**host.env, "XDG_DATA_HOME": str(self.t / "xdg")}
        self.assertEqual(self.settings(host).own_head_dir, self.t / "xdg/crucible/own-head")

    def _install(self, s, name="head-abc123def456.safetensors", **extra):
        (s.own_head_dir / "heads").mkdir(parents=True, exist_ok=True)
        side = s.own_head_dir / "heads" / name
        side.write_text("x")
        (s.own_head_dir / "head.json").write_text(json.dumps({"active": name, "previous": [], **extra}))
        return side

    def test_active_head_sets_the_sidecar(self):
        s = self._settings()
        side = self._install(s)
        self.assertEqual(launch.server_command(s, "py")[0]["CRUCIBLE_MTP_SIDECAR"], str(side))

    def test_null_missing_odd_or_git_tree_heads_are_ignored(self):
        s = self._settings()
        s.own_head_dir.mkdir(parents=True)
        rec = s.own_head_dir / "head.json"
        self.assertEqual(launch.own_head_env(s), {})   # no record: the parent variable is left alone
        for text in (json.dumps({"active": None, "previous": []}), json.dumps({"active": "head-1.safetensors"}),
                     json.dumps({"active": "../x"}), json.dumps({"active": "/etc/passwd"}), "not json",
                     json.dumps({"sidecar": "/tmp/x"})):
            rec.write_text(text)
            self.assertIsNone(launch.server_command(s, "py")[0]["CRUCIBLE_MTP_SIDECAR"], text)   # None = unset
        s.own_head_dir = self.repo / "own-head"   # a folder inside a git tree
        (self.repo / ".git").mkdir(exist_ok=True)
        self._install(s)
        self.assertIsNone(launch.server_command(s, "py")[0]["CRUCIBLE_MTP_SIDECAR"])

    def test_stock_record_removes_an_inherited_variable_and_no_record_keeps_it(self):
        s = self._settings()
        parent = {"PATH": "/bin", "CRUCIBLE_MTP_SIDECAR": "/profile/head"}
        self.assertEqual(launch.process_env(parent, launch.server_command(s, "py")[0])["CRUCIBLE_MTP_SIDECAR"], "/profile/head")
        s.own_head_dir.mkdir(parents=True)
        (s.own_head_dir / "head.json").write_text(json.dumps({"active": None, "previous": []}))
        self.assertNotIn("CRUCIBLE_MTP_SIDECAR", launch.process_env(parent, launch.server_command(s, "py")[0]))

    def test_serve_script_reads_the_head_at_each_start(self):
        host = wire(self.t)
        self.setup(["--yes"], host=host)
        script = self.repo / ".crucible/serve.sh"
        text = script.read_text()
        self.assertNotIn("CRUCIBLE_MTP_SIDECAR=/", text)
        self.assertIn("head_word", text)
        s = self.settings(host)
        head_part = text.split("exec env")[0].split("\n", 3)[3]   # the lines after "cd"
        head_part = head_part.replace(str(self.repo / ".venv/bin/python"), sys.executable).replace(str(self.repo / "tools"), str(ROOT / "tools"))

        def run(parent):
            out = subprocess.run(["sh", "-c", head_part + 'echo "${CRUCIBLE_MTP_SIDECAR-unset}"'], capture_output=True,
                                 text=True, env={"PATH": os.environ["PATH"], **parent})
            return out.stdout.strip(), out.stderr

        profile = {"CRUCIBLE_MTP_SIDECAR": "/profile/head"}
        self.assertEqual(run(profile)[0], "/profile/head")            # no record: a profile value stays
        side = self._install(s)                                      # installed after setup
        self.assertEqual(run(profile), (str(side), ""))
        (s.own_head_dir / "head.json").write_text(json.dumps({"active": None, "previous": [str(side.name)]}))
        self.assertEqual(run(profile)[0], "unset")                   # explicit stock removes it

    def test_yes_never_turns_on_capture_or_training(self):
        code, host, out = self.setup(["--yes"])
        argv = host.runner.spawned[0]["argv"]
        self.assertNotIn("--draft-capture", argv)
        self.assertFalse((self.t / "home/.local/share/crucible").exists())


class RealServicesTest(Base):
    """The real Runner and Http against local stand-ins: a tiny HTTP server and short child processes."""

    def test_http_get_and_streaming_chat_against_a_local_server(self):
        import http.server
        import threading
        from crucible_cli.http import Http

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in ({"reasoning_content": "hm"}, {"content": "Hel"}, {"content": "lo"}):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": piece}]}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        self.assertEqual(Http().get(base + "/health")[0], 200)
        self.assertEqual(Http().get(base + "/nope")[0], 404)
        res = Http().chat_stream(base + "/v1/chat/completions", {"stream": True})
        self.assertTrue(res["ok"])
        self.assertEqual((res["text"], res["tokens"]), ("Hello", 3))
        self.assertIsNotNone(res["first_token_s"])
        self.assertEqual(Http().get("http://127.0.0.1:9/health")[0], 0)

    def test_runner_runs_a_program_logs_its_output_and_returns_the_code(self):
        from crucible_cli import runner
        box = files.Sandbox([self.t / "w"])
        with box.open_log(self.t / "w/x.log") as log:
            res = runner.Runner().run([sys.executable, "-c", "print('a'); import sys; print('b', file=sys.stderr); sys.exit(3)"], log=log)
        self.assertEqual(res.rc, 3)
        self.assertEqual(sorted(res.out.split()), ["a", "b"])
        self.assertEqual(sorted((self.t / "w/x.log").read_text().split()), ["a", "b"])
        self.assertEqual(runner.Runner().run(["/no/such/program"]).rc, 127)

    def test_runner_stops_only_a_crucible_server_process(self):
        from crucible_cli import runner
        r = runner.Runner()
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(other.kill)
        self.assertFalse(r.alive(other.pid))          # not a server: never signalled
        self.assertFalse(r.terminate(other.pid, wait_s=1))
        self.assertIsNone(other.poll())
        srv = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "-m", "server"])
        self.addCleanup(srv.kill)
        for _ in range(50):  # the command line shows up a moment after the fork
            if r.alive(srv.pid):
                break
            time.sleep(0.1)
        self.assertTrue(r.alive(srv.pid))
        self.assertTrue(r.terminate(srv.pid, wait_s=10))
        srv.wait(timeout=10)
        self.assertFalse(r.alive(srv.pid))


class RealServerFlagsTest(Base):
    def test_the_real_server_accepts_every_flag_that_setup_writes(self):
        """The server parses its options and checks the state folder before it loads a model.
        A missing checkpoint then stops it. A bad flag would stop it earlier, with exit code 2."""
        home = self.t / "home"
        home.mkdir(exist_ok=True)
        for pack in ("4.05", "3.05"):
            s = settings_mod.Settings(repo=ROOT, pack=pack, models_dir=self.t / "none", state_dir=home / "states")
            env, argv = launch.server_command(s, sys.executable)
            proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120,
                                  env={**launch.process_env(os.environ, env), "HOME": str(home), "CRUCIBLE_NO_UPDATE_CHECK": "1"})
            self.assertNotEqual(proc.returncode, 2, proc.stderr[-400:])
            self.assertNotIn("unrecognized arguments", proc.stderr)
            self.assertNotIn("must be inside", proc.stderr)


class PackTest(unittest.TestCase):
    def test_pack_table_matches_the_public_readme(self):
        self.assertEqual(packs.PACKS["4.05"].revision, "4.05bpw_h6_ng6")
        self.assertEqual(packs.PACKS["3.05"].revision, "3.05bpw_h5_ng5")
        self.assertEqual(packs.DEFAULT, "4.05")
        text = (ROOT / "README.md").read_text()
        for p in packs.PACKS.values():
            self.assertIn(p.revision, text)
        self.assertIn(packs.REPO, text)

    def test_launch_flags_match_the_quickstart_guide_words(self):
        env, argv = launch.server_command(settings_mod.Settings(
            repo=Path("/r"), pack="4.05", models_dir=Path("/m"), state_dir=Path("/h/s")), "py")
        self.assertEqual(env, dict(measured.ENV))   # the pinned words are in tests/test_service_config.py
        self.assertEqual(argv[:3], ["py", "-m", "server"])


if __name__ == "__main__":
    unittest.main()
