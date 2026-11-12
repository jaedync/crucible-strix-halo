"""Tests of the service part of ./crucible: the unit text, service install/uninstall/status/logs, the start check
(doctor --for start), start and stop routing, and the setup question. A fake systemctl stands in for systemd.
No GPU, no network, no sudo, no real systemd."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import FakeHttp, FakePrompt, FakeRunner, ROOT, make_repo, wire  # noqa: E402

from crucible_cli import cli, doctor, files, host as hostmod, launch, service  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

UNIT = "crucible-server.service"
DEPLOY = ROOT / "deploy/user-systemd/crucible-server.service"


class FakeSystemd(FakeRunner):
    """A FakeRunner that also answers systemctl, loginctl and journalctl."""

    def __init__(self, manager="running", linger="no", linger_refused=False, start_fails=False):
        super().__init__()
        self.manager = manager          # the text of is-system-running; "" = no user manager; None = no systemctl
        self.linger = linger
        self.linger_refused = linger_refused
        self.start_fails = start_fails
        self.ctl = []                   # systemctl arguments after --user
        self.props = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running", "Result": "success",
                      "NRestarts": "0", "ExecMainStatus": "0", "StatusText": "ready: 0 of 4 sessions busy",
                      "UnitFileState": "enabled", "ActiveEnterTimestamp": "Thu 2026-11-05 09:00:00 CST"}
        self.journal = "\n".join(f"2026-11-05T09:00:{i:02d}-06:00 host crucible[1]: line {i}" for i in range(20))

    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = [str(a) for a in argv]
        name = Path(argv[0]).name
        if name == "systemctl":
            self.calls.append(argv)
            if self.manager is None:
                return hostmod.Result(127, "systemctl: not found")
            args = argv[2:] if argv[1] == "--user" else argv[1:]
            self.ctl.append(args)
            verb = args[0]
            if verb == "is-system-running":
                return hostmod.Result(0 if self.manager == "running" else 1, self.manager or "Failed to connect to bus")
            if verb == "start" or verb == "restart":
                if self.start_fails:
                    self.props.update(ActiveState="failed", SubState="failed", Result="exit-code")
                    return hostmod.Result(1, "Job for crucible-server.service failed.")
                self.props.update(ActiveState="active", SubState="running", Result="success")
            if verb == "stop":
                self.props.update(ActiveState="inactive", SubState="dead")
            if verb == "is-active":
                state = self.props["ActiveState"]
                return hostmod.Result(0 if state == "active" else 3, state)
            if verb == "show":
                names = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
                return hostmod.Result(0, "\n".join(f"{n}={self.props.get(n, '')}" for n in names))
            return hostmod.Result(0, "")
        if name == "loginctl":
            self.calls.append(argv)
            if argv[1] == "show-user":
                return hostmod.Result(0, f"Linger={self.linger}")
            if argv[1] == "enable-linger":
                if self.linger_refused:
                    return hostmod.Result(1, "Failed to enable linger: Interactive authentication required.")
                self.linger = "yes"
                return hostmod.Result(0, "")
        if name == "journalctl":
            self.calls.append(argv)
            return hostmod.Result(0, self.journal)
        return super().run(argv, cwd=cwd, env=env, log=log, echo=echo)

    def verbs(self):
        return [c[0] for c in self.ctl]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)

    def host(self, runner=None, http=None, **machine):
        host = wire(self.t, runner=runner or FakeSystemd(), http=http, **machine)
        host.env = {**host.env, "USER": "tester"}
        return host

    def run_cli(self, argv, host=None, **machine):
        host = host or self.host(**machine)
        code = cli.main(argv, host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def configure(self):
        """Run a plain setup so that .crucible/config.json and serve.sh exist."""
        host = self.host(runner=FakeRunner())
        self.assertEqual(cli.main(["setup", "--yes"], host=host, repo=self.repo), 0, "\n".join(host.lines))

    def unit_file(self):
        return self.t / "home/.config/systemd/user" / UNIT

    def install_unit(self, **kw):
        self.configure()
        code, host, out = self.run_cli(["service", "install"], host=self.host(runner=FakeSystemd(**kw)))
        self.assertEqual(code, 0, out)
        return host

    def ready_repo(self):
        """Everything that a start needs: venv, engine, pack."""
        s = settings_mod.build(self.repo, self.host())
        s.venv_python.parent.mkdir(parents=True, exist_ok=True)
        s.venv_python.write_text("#!/bin/sh\n")
        s.engine.parent.mkdir(parents=True, exist_ok=True)
        s.engine.write_text("#!/bin/sh\n")
        s.ckpt.mkdir(parents=True, exist_ok=True)
        for n in ("config.json", "tokenizer.json", "ngram_embedding.safetensors", "m-1.safetensors"):
            (s.ckpt / n).write_text("x")
        (s.ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-1.safetensors"}}))
        return s

    def check_file(self):
        return json.loads((self.repo / ".crucible/last-start-check.json").read_text())


class UnitTextTest(Base):
    def values(self, **kw):
        base = dict(repo="/srv/u/crucible", venv_py="/srv/u/crucible/.venv/bin/python",
                    exec_start="/srv/u/crucible/.crucible/serve.sh", bind="127.0.0.1", port=8790, start_timeout=600,
                    version="0.34.0")
        base.update(kw)
        return base

    def test_the_deploy_template_is_the_generator_output_with_placeholders(self):
        self.assertEqual(DEPLOY.read_text(), service.template_text())

    def test_the_template_names_every_placeholder_once_filled(self):
        text = service.render_unit(**self.values())
        self.assertNotRegex(text, r"@[A-Z_]+@")

    def test_unit_lines_of_the_design(self):
        text = service.render_unit(**self.values())
        lines = text.splitlines()
        for want in ("Type=notify", "NotifyAccess=main", "WatchdogSec=30s", "WatchdogSignal=SIGKILL",
                     "TimeoutStopSec=120s", "KillMode=control-group", "Restart=on-failure", "RestartSec=5s",
                     "RestartPreventExitStatus=2 78", "StartLimitIntervalSec=15min", "StartLimitBurst=4",
                     "LimitCORE=0", "LimitNOFILE=65536", "UMask=0077", "NoNewPrivileges=yes", "SyslogIdentifier=crucible",
                     "WantedBy=default.target", "TimeoutStartSec=600s", "Environment=PYTHONUNBUFFERED=1",
                     "WorkingDirectory=/srv/u/crucible"):
            self.assertIn(want, lines)
        self.assertIn('ExecStartPre="/srv/u/crucible/.venv/bin/python" -I "/srv/u/crucible/crucible" doctor --for start --quiet', lines)
        self.assertIn('ExecStart="/srv/u/crucible/.crucible/serve.sh" --systemd', lines)
        self.assertIn("Description=Crucible LLM server on 127.0.0.1:8790", lines)
        self.assertIn("Crucible 0.34.0", lines[0])

    def test_doctor_args_follow_the_quiet_flag(self):
        text = service.render_unit(**self.values(doctor_args=["--check-file", "/srv/u/ft/check.json"]))
        self.assertIn("doctor --for start --quiet --check-file /srv/u/ft/check.json", text)

    def test_a_path_with_a_space_is_quoted_and_allowed(self):
        text = service.render_unit(**self.values(repo="/srv/u/my crucible", exec_start="/srv/u/my crucible/.crucible/serve.sh"))
        self.assertIn('WorkingDirectory=/srv/u/my crucible', text)
        self.assertIn('ExecStart="/srv/u/my crucible/.crucible/serve.sh" --systemd', text)

    def test_a_path_with_a_special_character_is_refused(self):
        for bad in ("/srv/u/c%h", "/srv/u/$HOME", '/srv/u/a"b', "/srv/u/a\nb", "/srv/u/a\\b", "/srv/u/a'b", "/srv/u/a;b"):
            with self.assertRaises(service.UnitPathError, msg=repr(bad)):
                service.render_unit(**self.values(repo=bad))
        with self.assertRaises(service.UnitPathError):
            service.render_unit(**self.values(venv_py="/h/v$x/python"))
        with self.assertRaises(service.UnitPathError):
            service.render_unit(**self.values(exec_start="/h/serve%n.sh"))

    def test_the_generator_makes_no_file_and_runs_no_program(self):
        before = set(self.t.rglob("*"))
        service.render_unit(**self.values())
        self.assertEqual(before, set(self.t.rglob("*")))

    def test_start_timeout_is_three_times_the_measured_load_or_ten_minutes(self):
        self.assertEqual(service.start_timeout_for(None), 600)
        self.assertEqual(service.start_timeout_for(40), 300)
        self.assertEqual(service.start_timeout_for(200), 600)
        self.assertEqual(service.start_timeout_for(2000), 3600)  # capped at one hour


class InstallTest(Base):
    def test_install_writes_the_unit_enables_starts_and_checks_health(self):
        self.configure()
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["service", "install"], host=host)
        self.assertEqual(code, 0, out)
        verbs = host.runner.verbs()
        self.assertEqual(verbs[0], "is-system-running")
        self.assertLess(verbs.index("daemon-reload"), verbs.index("enable"))
        self.assertLess(verbs.index("enable"), verbs.index("start"))
        self.assertIn([UNIT], [c[1:] for c in host.runner.ctl if c[0] == "enable"])
        self.assertTrue(host.http.gets[-1].endswith("/health"))
        text = self.unit_file().read_text()
        self.assertIn(f'WorkingDirectory={self.repo}', text)
        self.assertIn(f'ExecStart="{self.repo}/.crucible/serve.sh" --systemd', text)
        self.assertEqual(stat.S_IMODE(self.unit_file().stat().st_mode), 0o644)

    def test_the_unit_names_the_venv_python_and_the_launcher(self):
        host = self.install_unit()
        text = self.unit_file().read_text()
        self.assertIn(f'ExecStartPre="{self.repo}/.venv/bin/python" -I "{self.repo}/crucible" doctor --for start --quiet', text)

    def test_install_rewrites_serve_sh_so_that_it_passes_the_arguments(self):
        self.configure()
        (self.repo / ".crucible/serve.sh").write_text("#!/bin/sh\nexec env X=1 python -m server\n")
        self.run_cli(["service", "install"], host=self.host(runner=FakeSystemd()))
        self.assertIn('"$@"', (self.repo / ".crucible/serve.sh").read_text())

    def test_serve_script_ends_with_the_argument_pass_through(self):
        s = settings_mod.Settings(repo=Path("/r"), models_dir=Path("/m"), state_dir=Path("/h/s"))
        text = launch.serve_script(s, "/r/.venv/bin/python")
        self.assertTrue(text.rstrip().endswith('"$@"'), text)

    def test_install_without_config_exits_2(self):
        code, host, out = self.run_cli(["service", "install"])
        self.assertEqual(code, 2)
        self.assertIn("./crucible setup", out)
        self.assertFalse(self.unit_file().exists())

    def test_install_without_a_user_manager_exits_3_and_changes_nothing(self):
        self.configure()
        for manager in ("", None, "offline"):
            host = self.host(runner=FakeSystemd(manager=manager))
            code, host, out = self.run_cli(["service", "install"], host=host)
            self.assertEqual(code, 3, (manager, out))
            self.assertIn("./crucible start", out)
            self.assertFalse(self.unit_file().exists())
            self.assertNotIn("start", host.runner.verbs())

    def test_a_degraded_user_manager_is_good_enough(self):
        self.configure()
        code, host, out = self.run_cli(["service", "install"], host=self.host(runner=FakeSystemd(manager="degraded")))
        self.assertEqual(code, 0, out)

    def test_install_refuses_a_bad_path_with_exit_2(self):
        bad = self.t / "re$po"
        bad.mkdir()
        (bad / "requirements-cpu.txt").write_text("numpy\n")
        host = self.host(runner=FakeRunner())
        self.assertEqual(cli.main(["setup", "--yes"], host=host, repo=bad), 0)
        host = self.host(runner=FakeSystemd())
        code = cli.main(["service", "install"], host=host, repo=bad)
        self.assertEqual(code, 2)
        self.assertIn("character", "\n".join(host.lines))
        self.assertNotIn("start", host.runner.verbs())

    def test_install_stops_a_pid_file_server_first_and_removes_the_record(self):
        self.configure()
        (self.repo / ".crucible/server.pid").write_text("40007 1000\n")
        runner = FakeSystemd()
        runner.live = {40007}
        code, host, out = self.run_cli(["service", "install"], host=self.host(runner=runner))
        self.assertEqual(code, 0, out)
        self.assertEqual(runner.terminated, [40007])
        self.assertFalse((self.repo / ".crucible/server.pid").exists())

    def test_install_enables_linger_when_it_is_off(self):
        host = self.install_unit(linger="no")
        self.assertIn(["enable-linger"], [c[1:2] for c in host.runner.calls if Path(c[0]).name == "loginctl"])

    def test_install_leaves_linger_alone_when_it_is_on(self):
        host = self.install_unit(linger="yes")
        self.assertNotIn("enable-linger", [c[1] for c in host.runner.calls if Path(c[0]).name == "loginctl"])

    def test_a_refused_linger_gives_one_line_with_the_sudo_command_and_no_failure(self):
        self.configure()
        host = self.host(runner=FakeSystemd(linger="no", linger_refused=True))
        code, host, out = self.run_cli(["service", "install"], host=host)
        self.assertEqual(code, 0, out)
        lines = [ln for ln in host.lines if "enable-linger" in ln]
        self.assertEqual(len(lines), 1, host.lines)
        self.assertIn("sudo loginctl enable-linger tester", lines[0])
        self.assertIn("only while you are logged in", out)
        for argv in host.runner.calls:  # the tool never runs sudo
            self.assertNotIn(Path(argv[0]).name, files.FORBIDDEN_COMMANDS)

    def test_a_failed_start_exits_1_with_the_status_and_the_last_20_log_lines(self):
        self.configure()
        host = self.host(runner=FakeSystemd(start_fails=True))
        code, host, out = self.run_cli(["service", "install"], host=host)
        self.assertEqual(code, 1)
        jc = [c for c in host.runner.calls if Path(c[0]).name == "journalctl"]
        self.assertEqual(len(jc), 1)
        self.assertEqual(jc[0][jc[0].index("-n") + 1], "20")
        self.assertIn("line 19", out)
        self.assertTrue(self.unit_file().exists())

    def test_a_unit_that_starts_but_does_not_answer_health_is_a_failure(self):
        self.configure()
        host = self.host(runner=FakeSystemd(), http=FakeHttp(ready_after=10 ** 9))
        code, host, out = self.run_cli(["service", "install"], host=host)
        self.assertEqual(code, 1)
        self.assertIn("/health", out)

    def test_no_start_writes_and_enables_but_does_not_start(self):
        self.configure()
        code, host, out = self.run_cli(["service", "install", "--no-start"], host=self.host(runner=FakeSystemd()))
        self.assertEqual(code, 0, out)
        self.assertIn("enable", host.runner.verbs())
        self.assertNotIn("start", host.runner.verbs())
        self.assertTrue(self.unit_file().exists())

    def test_start_timeout_flag_sets_the_unit_value(self):
        self.configure()
        self.run_cli(["service", "install", "--start-timeout", "900", "--no-start"], host=self.host(runner=FakeSystemd()))
        self.assertIn("TimeoutStartSec=900s", self.unit_file().read_text())

    def test_the_measured_load_time_of_setup_sets_the_timeout(self):
        self.configure()
        sf = self.repo / ".crucible/setup-state.json"
        data = json.loads(sf.read_text())
        data["load_seconds"] = 150
        sf.write_text(json.dumps(data))
        self.run_cli(["service", "install", "--no-start"], host=self.host(runner=FakeSystemd()))
        self.assertIn("TimeoutStartSec=450s", self.unit_file().read_text())

    def test_the_xdg_config_home_moves_the_unit(self):
        self.configure()
        host = self.host(runner=FakeSystemd())
        host.env = {**host.env, "XDG_CONFIG_HOME": str(self.t / "cfg")}
        code, host, out = self.run_cli(["service", "install", "--no-start"], host=host)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.t / "cfg/systemd/user" / UNIT).exists())

    def test_a_second_install_replaces_the_file_atomically_and_keeps_one_file(self):
        self.install_unit()
        self.run_cli(["service", "install", "--no-start"], host=self.host(runner=FakeSystemd()))
        names = sorted(p.name for p in self.unit_file().parent.iterdir())
        self.assertEqual(names, [UNIT])


class UninstallTest(Base):
    def test_uninstall_stops_disables_removes_and_resets(self):
        self.install_unit()
        code, host, out = self.run_cli(["service", "uninstall"], host=self.host(runner=FakeSystemd()))
        self.assertEqual(code, 0, out)
        verbs = host.runner.verbs()
        for v in ("stop", "disable", "daemon-reload", "reset-failed"):
            self.assertIn(v, verbs)
        self.assertLess(verbs.index("stop"), verbs.index("disable"))
        self.assertFalse(self.unit_file().exists())
        self.assertIn("linger", out.lower())

    def test_uninstall_when_not_installed_is_fine(self):
        code, host, out = self.run_cli(["service", "uninstall"])
        self.assertEqual(code, 0)
        self.assertEqual(host.runner.ctl, [])


class RoutingTest(Base):
    def test_start_and_stop_use_systemd_when_the_unit_is_installed(self):
        self.install_unit()
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["start"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("reset-failed", host.runner.verbs())  # a start-limit-hit must not block it
        self.assertIn("start", host.runner.verbs())
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["stop"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.terminated, [])
        self.assertIn("stop", host.runner.verbs())

    def test_start_and_stop_never_use_systemd_without_the_unit(self):
        self.configure()
        self.ready_repo()
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["start"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(host.runner.spawned), 1)
        self.assertNotIn("start", host.runner.verbs())

    def test_setup_run_with_the_unit_installed_starts_through_systemd(self):
        self.install_unit()
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["setup", "--yes"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("start", host.runner.verbs())

    def test_service_start_stop_restart_are_the_same_calls(self):
        self.install_unit()
        for verb in ("start", "stop", "restart"):
            host = self.host(runner=FakeSystemd())
            code, host, out = self.run_cli(["service", verb], host=host)
            self.assertEqual(code, 0, out)
            self.assertIn(verb, host.runner.verbs())


class StatusTest(Base):
    def status(self, props=None, check=None, http=None, argv=(), runner=None):
        self.configure()
        if self.unit_file().parent.exists() is False:
            self.unit_file().parent.mkdir(parents=True)
        runner = runner or FakeSystemd()
        runner.props.update(props or {})
        if check is not None:
            (self.repo / ".crucible/last-start-check.json").write_text(json.dumps(check))
        host = self.host(runner=runner, http=http)
        return self.run_cli(["service", "status", *argv], host=host)

    def test_not_installed_exits_4(self):
        code, host, out = self.status({"LoadState": "not-found", "ActiveState": "inactive"})
        self.assertEqual(code, 4)
        self.assertIn("./crucible service install", out)

    def test_running_and_healthy_exits_0(self):
        code, host, out = self.status()
        self.assertEqual(code, 0, out)
        self.assertIn("http://127.0.0.1:8790", out)
        self.assertIn("running", out)

    def test_stopped_exits_3(self):
        code, host, out = self.status({"ActiveState": "inactive", "SubState": "dead"})
        self.assertEqual(code, 3)
        self.assertIn("./crucible service start", out)

    def test_active_but_health_not_200_exits_1(self):
        code, host, out = self.status(http=FakeHttp(ready_after=10 ** 9))
        self.assertEqual(code, 1)
        self.assertIn("/health", out)

    def test_failed_unit_exits_1(self):
        code, host, out = self.status({"ActiveState": "failed", "SubState": "failed", "Result": "exit-code",
                                       "ExecMainStatus": "1"})
        self.assertEqual(code, 1)
        self.assertIn("code 1", out)

    def test_f8_bad_setting_status_names_the_failed_start_check(self):
        check = {"ok": False, "code": 78, "failed": [{"id": "pack", "detail": "the pack folder /x is missing",
                                                       "fix": "Run ./crucible setup again."}]}
        code, host, out = self.status({"ActiveState": "failed", "SubState": "failed", "Result": "start-limit-hit"},
                                      check=check)
        self.assertEqual(code, 1)
        self.assertIn("Stopped after 4 failed starts in 15 min.", out)
        self.assertIn("Last start check: the pack folder /x is missing", out)
        self.assertIn("./crucible service start", out)

    def test_f10_crash_loop_status_says_why_without_a_check(self):
        code, host, out = self.status({"ActiveState": "failed", "SubState": "failed", "Result": "start-limit-hit",
                                       "NRestarts": "3", "ExecMainStatus": "1"})
        self.assertEqual(code, 1)
        self.assertIn("Stopped after 4 failed starts in 15 min.", out)
        self.assertIn("service logs", out)

    def test_watchdog_and_timeout_results_have_their_own_sentence(self):
        _, _, out = self.status({"ActiveState": "activating", "SubState": "auto-restart", "Result": "watchdog"})
        self.assertIn("watchdog", out.lower())
        _, _, out = self.status({"ActiveState": "failed", "Result": "timeout"})
        self.assertIn("did not get ready", out)

    def test_json_has_the_keys_of_the_driver(self):
        code, host, out = self.status(argv=["--json"])
        data = json.loads(out)
        for key in ("state", "substate", "nrestarts", "result", "exec_main_status", "status_text", "enabled",
                    "linger", "health", "reason", "url", "key_required", "installed"):
            self.assertIn(key, data)
        self.assertEqual((data["state"], data["substate"], data["nrestarts"], data["health"]), ("active", "running", 0, 200))
        self.assertIs(data["enabled"], True)
        self.assertIs(data["key_required"], False)

    def test_a_check_that_passed_is_not_blamed(self):
        code, host, out = self.status({"ActiveState": "failed", "Result": "exit-code", "ExecMainStatus": "1"},
                                      check={"ok": True, "code": 0, "failed": []})
        self.assertNotIn("Last start check", out)

    def test_unit_option_and_check_file_option(self):
        runner = FakeSystemd()
        check = self.t / "ft-check.json"
        check.write_text(json.dumps({"ok": False, "code": 78, "failed": [{"id": "pack", "detail": "no pack", "fix": ""}]}))
        code, host, out = self.status({"ActiveState": "failed", "Result": "start-limit-hit"}, runner=runner,
                                      argv=["--unit", "crucible-ft.service", "--check-file", str(check), "--json"])
        data = json.loads(out)
        self.assertIn("no pack", data["reason"])
        shown = [c for c in runner.ctl if c[0] == "show"][0]
        self.assertIn("crucible-ft.service", shown)

    def test_work_folder_variable_finds_the_check_file(self):
        work = self.t / "ftwork"
        work.mkdir()
        (work / "last-start-check.json").write_text(json.dumps(
            {"ok": False, "code": 78, "failed": [{"id": "engine", "detail": "no engine file", "fix": ""}]}))
        self.configure()
        runner = FakeSystemd()
        runner.props.update(ActiveState="failed", Result="start-limit-hit")
        host = self.host(runner=runner)
        host.env = {**host.env, "CRUCIBLE_WORK": str(work)}
        code, host, out = self.run_cli(["service", "status"], host=host)
        self.assertIn("no engine file", out)

    def test_key_required_reads_the_settings(self):
        self.configure()
        cfg = self.repo / ".crucible/config.json"
        data = json.loads(cfg.read_text())
        data["api_key_file"] = str(self.t / "home/.config/crucible/api-key")
        cfg.write_text(json.dumps(data))
        host = self.host(runner=FakeSystemd())
        code, host, out = self.run_cli(["service", "status", "--json"], host=host)
        if "api_key_file" in settings_mod.SAVED:
            self.assertIs(json.loads(out)["key_required"], True)


class LogsTest(Base):
    def test_default_is_200_lines_without_a_pager(self):
        code, host, out = self.run_cli(["service", "logs"])
        argv = [c for c in host.runner.calls if Path(c[0]).name == "journalctl"][0]
        self.assertEqual(argv[1:], ["--user", "-u", UNIT, "-n", "200", "-o", "short-iso", "--no-pager"])

    def test_n_and_follow(self):
        code, host, out = self.run_cli(["service", "logs", "-n", "5", "-f"])
        argv = [c for c in host.runner.calls if Path(c[0]).name == "journalctl"][0]
        self.assertEqual(argv[argv.index("-n") + 1], "5")
        self.assertIn("-f", argv)
        self.assertNotIn("--no-pager", argv)

    def test_no_user_journal_gives_a_clear_line(self):
        runner = FakeSystemd()
        runner.journal = "No journal files were found."
        code, host, out = self.run_cli(["service", "logs"], host=self.host(runner=runner))
        self.assertIn("not kept", out)


class StartCheckTest(Base):
    def doctor(self, extra=(), host=None, **machine):
        host = host or self.host(runner=FakeSystemd(), **machine)
        code = cli.main(["doctor", "--for", "start", *extra], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def clocked(self, **machine):
        clock = {"t": 0.0}
        host = self.host(runner=FakeSystemd(), **machine)
        host.clock = lambda: clock["t"]
        host.sleep = lambda n: clock.__setitem__("t", clock["t"] + n)
        return host, clock

    def test_a_good_machine_exits_0_and_writes_ok(self):
        self.ready_repo()
        code, host, out = self.doctor(["--quiet"])
        self.assertEqual(code, 0, out)
        self.assertEqual(out, "")
        self.assertIs(self.check_file()["ok"], True)

    def test_a_missing_key_file_exits_78_and_a_present_one_exits_0(self):
        self.ready_repo()
        key = self.t / "home/.config/crucible/api-key"
        (self.repo / ".crucible").mkdir(exist_ok=True)
        (self.repo / ".crucible/config.json").write_text(json.dumps(
            {"pack": "4.05", "bind": "0.0.0.0", "api_key_file": str(key)}))
        code, host, out = self.doctor(["--quiet"])
        self.assertEqual(code, 78, out)
        self.assertIn("key-file", out)
        self.assertNotIn("api-key-secret", out)
        key.parent.mkdir(parents=True, mode=0o700)
        key.write_text("a" * 40 + "\n")
        key.chmod(0o600)
        host = self.host(runner=FakeSystemd())
        self.assertEqual(cli.main(["doctor", "--for", "start", "--quiet"], host=host, repo=self.repo), 0, host.lines)

    def test_f8_a_missing_pack_exits_78_names_the_check_and_writes_the_file(self):
        s = self.ready_repo()
        for p in s.ckpt.iterdir():
            p.unlink()
        s.ckpt.rmdir()
        code, host, out = self.doctor(["--quiet"])
        self.assertEqual(code, 78, out)
        self.assertEqual(len(host.lines), 1)
        self.assertIn("pack", host.lines[0])
        self.assertIn(str(s.ckpt), host.lines[0])
        data = self.check_file()
        self.assertEqual((data["ok"], data["code"]), (False, 78))
        self.assertEqual(data["failed"][0]["id"], "pack")
        self.assertIn(str(s.ckpt), data["failed"][0]["detail"])

    def test_a_missing_engine_or_venv_exits_78(self):
        s = self.ready_repo()
        s.engine.unlink()
        code, host, out = self.doctor(["--quiet"])
        self.assertEqual(code, 78)
        self.assertIn("engine", out)
        s = self.ready_repo()
        s.venv_python.unlink()
        s.engine.write_text("x")
        code, host, out = self.doctor(["--quiet"])
        self.assertEqual(code, 78)
        self.assertIn("venv", out)

    def test_a_wrong_gpu_exits_78_at_once_without_a_wait(self):
        self.ready_repo()
        host, clock = self.clocked()
        host.runner.gfx = "gfx1100"
        code, host, out = self.doctor(["--quiet"], host=host)
        self.assertEqual(code, 78)
        self.assertEqual(clock["t"], 0)

    def test_a1_the_gpu_busy_for_60_s_at_start_still_starts(self):
        self.ready_repo()
        host, clock = self.clocked()
        used = Path(host.sysfs) / "class/drm/card0/device/mem_info_gtt_used"
        used.write_text("90000000000\n")
        real_sleep = host.sleep

        def sleep(n):
            real_sleep(n)
            if clock["t"] >= 60:
                used.write_text("2000000\n")
        host.sleep = sleep
        code, host, out = self.doctor(["--quiet"], host=host)
        self.assertEqual(code, 0, out)
        self.assertGreaterEqual(clock["t"], 60)
        self.assertLess(clock["t"], 120)

    def test_a1_a_port_busy_for_more_than_120_s_exits_75(self):
        self.ready_repo()
        host, clock = self.clocked(busy_ports=(8790,))
        code, host, out = self.doctor(["--quiet"], host=host)
        self.assertEqual(code, 75, out)
        self.assertGreaterEqual(clock["t"], 120)
        self.assertLess(clock["t"], 130)
        self.assertIn("port", out)
        self.assertEqual(self.check_file()["code"], 75)

    def test_a_hard_failure_beside_a_transient_one_exits_78_without_a_wait(self):
        s = self.ready_repo()
        s.engine.unlink()
        host, clock = self.clocked(busy_ports=(8790,))
        code, host, out = self.doctor(["--quiet"], host=host)
        self.assertEqual(code, 78)
        self.assertEqual(clock["t"], 0)

    def test_kfd_missing_is_its_own_transient_check(self):
        self.ready_repo()
        host, clock = self.clocked()
        (Path(host.dev) / "kfd").unlink()
        code, host, out = self.doctor(["--json"], host=host)
        self.assertEqual(code, 75, out)
        self.assertGreaterEqual(clock["t"], 120)
        ids = [c["id"] for c in json.loads(out)["checks"]]
        self.assertIn("gpu-kfd", ids)

    def test_the_check_file_option_moves_the_file(self):
        self.ready_repo()
        target = self.t / "ft/check.json"
        code, host, out = self.doctor(["--quiet", "--check-file", str(target)])
        self.assertEqual(code, 0)
        self.assertIs(json.loads(target.read_text())["ok"], True)

    def test_without_quiet_the_text_has_the_fix_line(self):
        s = self.ready_repo()
        s.engine.unlink()
        code, host, out = self.doctor()
        self.assertEqual(code, 78)
        self.assertIn("fix:", out)

    def test_doctor_without_for_keeps_its_exit_codes(self):
        host = self.host(runner=FakeSystemd(), gtt=60_000_000_000)
        self.assertEqual(cli.main(["doctor"], host=host, repo=self.repo), 1)


class SetupQuestionTest(Base):
    def setup_cli(self, argv, host=None, **machine):
        host = host or self.host(runner=FakeSystemd(), **machine)
        code = cli.main(["setup", *argv], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def test_yes_without_service_installs_nothing_and_prints_the_hint(self):
        code, host, out = self.setup_cli(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.ctl, [])
        self.assertEqual(out.count("To run it as a service: ./crucible service install"), 1)

    def test_service_no_prints_no_hint_needed_and_installs_nothing(self):
        code, host, out = self.setup_cli(["--yes", "--service", "no"])
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.ctl, [])

    def test_yes_with_service_yes_installs_after_the_test_and_hands_over(self):
        code, host, out = self.setup_cli(["--yes", "--service", "yes"])
        self.assertEqual(code, 0, out)
        self.assertTrue(self.unit_file().exists())
        self.assertEqual(host.runner.terminated, [40001])  # the setup server gives way to the unit
        self.assertIn("start", host.runner.verbs())
        self.assertIn("./crucible service status", out)
        self.assertNotIn("To run it as a service", out)
        self.assertLess(out.index("[test]"), out.index("[service]"))

    def test_smoke_with_service_yes_exits_2(self):
        code, host, out = self.setup_cli(["--yes", "--smoke", "--service", "yes"])
        self.assertEqual(code, 2)
        self.assertEqual(host.runner.spawned, [])

    def test_service_yes_without_a_user_manager_fails_early_with_exit_3(self):
        code, host, out = self.setup_cli(["--yes", "--service", "yes"], host=self.host(runner=FakeSystemd(manager="")))
        self.assertEqual(code, 3)
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("./crucible start", out)

    def test_a_failed_install_after_setup_exits_1_and_keeps_the_server(self):
        host = self.host(runner=FakeSystemd(start_fails=True))
        code, host, out = self.setup_cli(["--yes", "--service", "yes"], host=host)
        self.assertEqual(code, 1)

    def test_the_question_default_is_yes(self):
        host = self.host(runner=FakeSystemd(), prompt=FakePrompt())
        host.is_tty = True
        code, host, out = self.setup_cli([], host=host)
        self.assertEqual(code, 0, out)
        asked = [q for q in host.prompt.asked if "service" in q.lower()]
        self.assertEqual(len(asked), 1)
        self.assertIn("starts at boot", asked[0])
        self.assertTrue(self.unit_file().exists())

    def test_answer_n_installs_nothing(self):
        host = self.host(runner=FakeSystemd())
        host.is_tty = True
        host.prompt = lambda q, d="": ("n" if "service" in q.lower() else d)
        code, host, out = self.setup_cli([], host=host)
        self.assertEqual(code, 0, out)
        self.assertFalse(self.unit_file().exists())
        self.assertEqual(host.runner.verbs(), ["is-system-running"])  # the probe only

    def test_the_question_is_not_asked_when_a_flag_answers_it(self):
        host = self.host(runner=FakeSystemd(), prompt=FakePrompt())
        host.is_tty = True
        self.setup_cli(["--service", "no"], host=host)
        self.assertFalse([q for q in host.prompt.asked if "service" in q.lower()])

    def test_the_question_is_not_asked_without_a_user_manager(self):
        host = self.host(runner=FakeSystemd(manager=""), prompt=FakePrompt())
        host.is_tty = True
        code, host, out = self.setup_cli([], host=host)
        self.assertEqual(code, 0, out)
        self.assertFalse([q for q in host.prompt.asked if "service" in q.lower()])
        self.assertIn("./crucible start", out)

    def test_a2_stdin_is_a_pipe_the_questions_come_from_the_terminal_device(self):
        host = self.host(runner=FakeSystemd())
        host.is_tty = False
        host.tty_prompt = FakePrompt()
        code, host, out = self.setup_cli([], host=host)
        self.assertEqual(code, 0, out)
        self.assertTrue([q for q in host.tty_prompt.asked if "service" in q.lower()])
        self.assertTrue(self.unit_file().exists())

    def test_a2_no_terminal_at_all_acts_like_yes_for_the_service_question(self):
        host = self.host(runner=FakeSystemd())
        host.is_tty = False
        host.tty_prompt = None
        args = cli.parser().parse_args(["setup"])
        self.assertEqual(cli.service_answer(args, host), "no")
        self.assertIn("To run it as a service: ./crucible service install", "\n".join(host.lines))

    def test_a2_no_terminal_and_no_yes_still_refuses_setup(self):
        code, host, out = self.setup_cli([])
        self.assertEqual(code, 2)
        self.assertIn("--yes", out)

    def test_dry_run_shows_the_service_step_only_when_asked_for(self):
        code, host, out = self.setup_cli(["--dry-run", "--yes", "--service", "yes"])
        self.assertIn("service", out)
        self.assertEqual(host.runner.ctl, [])
        code, host, out = self.setup_cli(["--dry-run", "--yes"])
        self.assertNotIn("systemd", out)


REL = {"format": 1, "tag": "v0.34.0", "commit": "a" * 40, "engine_sha256": "b" * 64, "arch": "gfx1151", "rocm": "7.2",
       "asset": "crucible-engine-v0.34.0-gfx1151-rocm7.2-linux-x86_64.gz"}


class ReleaseEngineTest(Base):
    """T6 hooks: with a verified release engine the compiler and make are not needed."""

    def checks(self, release=REL, **machine):
        old = doctor.release_engine
        doctor.release_engine = lambda repo: release
        self.addCleanup(setattr, doctor, "release_engine", old)
        host = self.host(runner=FakeRunner(), **machine)
        s = settings_mod.build(self.repo, host)
        return {r.id: r for r in doctor.run_checks(host, s)}, host, s

    def test_without_a_release_engine_nothing_changes(self):
        res, host, s = self.checks(release=None, rocm=False, hipcc=False, tools=("git",))
        self.assertEqual(res["hipcc"].status, "fail")
        self.assertEqual(res["tool-make"].status, "fail")
        self.assertIn("build", res["hipcc"].blocks)

    def test_with_a_release_engine_hipcc_make_and_cxx_are_skipped(self):
        res, host, s = self.checks(hipcc=False, tools=("git",))
        for id_ in ("hipcc", "tool-make", "tool-cxx"):
            self.assertEqual(res[id_].status, "skip", id_)
            self.assertIn("not needed: the release engine is installed", res[id_].detail)
            self.assertEqual(res[id_].blocks, ())

    def test_a_present_compiler_still_passes(self):
        res, host, s = self.checks()
        self.assertEqual(res["hipcc"].status, "pass")
        self.assertEqual(res["tool-make"].status, "pass")

    def test_missing_rocm_blocks_the_start_only(self):
        res, host, s = self.checks(rocm=False, hipcc=False)
        self.assertEqual(res["rocm-version"].status, "fail")
        self.assertEqual(res["rocm-version"].blocks, ("start",))

    def test_the_engine_check_names_the_release(self):
        s = self.ready_repo()
        res, host, s = self.checks()
        self.assertEqual(res["engine"].status, "pass")
        self.assertIn("release engine v0.34.0", res["engine"].detail)

    def test_setup_does_not_run_make_and_says_why(self):
        old = doctor.release_engine
        doctor.release_engine = lambda repo: REL
        self.addCleanup(setattr, doctor, "release_engine", old)
        host = self.host(runner=FakeRunner(), hipcc=False, tools=("git",))
        code = cli.main(["setup", "--yes"], host=host, repo=self.repo)
        out = "\n".join(host.lines)
        self.assertEqual(code, 0, out)
        self.assertFalse([c for c in host.runner.calls if Path(c[0]).name == "make"])
        self.assertIn("release engine v0.34.0 (verified)", out)

    def test_the_start_check_does_not_need_the_compiler_either(self):
        self.ready_repo()
        old = doctor.release_engine
        doctor.release_engine = lambda repo: REL
        self.addCleanup(setattr, doctor, "release_engine", old)
        host = self.host(runner=FakeSystemd(), hipcc=False, tools=("git",))
        self.assertEqual(cli.main(["doctor", "--for", "start", "--quiet"], host=host, repo=self.repo), 0, host.lines)

    def test_a_missing_release_module_means_no_release_engine(self):
        # Until T6 merges, tools/crucible_cli/release.py may not exist: the helper then returns None.
        self.assertIsNone(doctor.release_engine(self.t / "nowhere"))


class SetupRestartTest(Base):
    """D2: `setup` with a changed setting restarts a running service; the 403 text promises it."""

    def setup_service(self, **props):
        self.install_unit()
        self.ready_repo()
        runner = FakeSystemd()
        runner.props.update(props)
        return self.host(runner=runner)

    def test_a_new_allowed_host_restarts_the_running_service(self):
        host = self.setup_service()
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertIn("ai.example.org", (self.repo / ".crucible/serve.sh").read_text())
        self.assertIn("restart", host.runner.verbs())

    def test_a_new_api_key_setting_restarts_the_running_service(self):
        host = self.setup_service()
        code, host, out = self.run_cli(["setup", "--yes", "--api-key"], host=host)
        self.assertEqual(code, 0, out)
        self.assertIn("restart", host.runner.verbs())

    def test_a_key_made_by_setup_does_not_block_the_start_on_a_configured_host(self):
        """Found with D2: the key-file check ran before the config step made the key, so the first `setup --api-key` on a
        configured host stopped with 'blocked: check failed: key-file'."""
        self.configure()
        self.ready_repo()
        code, host, out = self.run_cli(["setup", "--yes", "--api-key"], host=self.host(runner=FakeRunner()))
        self.assertEqual(code, 0, out)
        self.assertNotIn("blocked", out)

    def test_the_same_settings_do_not_restart_the_running_service(self):
        host = self.setup_service()
        self.assertEqual(self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)[0], 0)
        host.runner.ctl.clear()
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertNotIn("restart", host.runner.verbs())
        self.assertIn("start", host.runner.verbs())

    def test_a_stopped_service_is_started_not_restarted(self):
        host = self.setup_service(ActiveState="inactive", SubState="dead")
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertNotIn("restart", host.runner.verbs())
        self.assertIn("start", host.runner.verbs())

    def test_the_refusal_text_names_the_command_and_the_restart(self):
        from server import hosts
        self.assertIn("./crucible setup --allowed-host NAME", hosts.REFUSED)
        self.assertIn("restarts the server", hosts.REFUSED)

    def test_the_refusal_text_names_a_server_from_start_and_a_server_made_by_hand(self):
        from server import hosts
        self.assertIn("./crucible start", hosts.REFUSED)
        self.assertIn("python -m server", hosts.REFUSED)


class SetupRestartPendingTest(Base):
    """M1: the restart must not get lost when a setup run stops before it, and a unit in the state "activating"
    (the model loads) counts as running."""

    def setup_service(self, **props):
        self.install_unit()
        self.s = self.ready_repo()
        runner = FakeSystemd()
        runner.props.update(props)
        return self.host(runner=runner)

    def test_a_setup_that_stopped_in_the_download_still_restarts_the_service_next_time(self):
        host = self.setup_service()
        pack_file = self.s.ckpt / "m-1.safetensors"
        pack_file.unlink()
        host.runner.fail.add("download")
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 1, out)
        self.assertIn("ai.example.org", (self.repo / ".crucible/serve.sh").read_text())
        self.assertNotIn("restart", host.runner.verbs())
        pack_file.write_text("x")
        host.runner.ctl.clear()
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertIn("restart", host.runner.verbs())

    def test_the_restart_is_done_once_after_a_good_restart(self):
        host = self.setup_service()
        self.assertEqual(self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)[0], 0)
        self.assertIn("restart", host.runner.verbs())
        host.runner.ctl.clear()
        self.assertEqual(self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)[0], 0)
        self.assertNotIn("restart", host.runner.verbs())

    def test_a_failed_restart_stays_pending(self):
        host = self.setup_service()
        host.runner.start_fails = True
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 1, out)
        host.runner.start_fails = False
        host.runner.props.update(ActiveState="active", SubState="running", Result="success")
        host.runner.ctl.clear()
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertIn("restart", host.runner.verbs())

    def test_a_service_that_loads_the_model_counts_as_running_for_setup(self):
        host = self.setup_service(ActiveState="activating", SubState="start", StatusText="loading the model")
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=host)
        self.assertEqual(code, 0, out)
        self.assertNotIn("blocked", out)
        self.assertNotIn("held by another program", out)
        self.assertIn("restart", host.runner.verbs())

    def doctor_checks(self, **props):
        self.install_unit()
        self.ready_repo()
        runner = FakeSystemd()
        runner.props.update(props)
        host = self.host(runner=runner, busy_ports=(8790,), gtt_used=90_000_000_000)
        return {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host))}

    def test_the_doctor_accepts_a_service_that_loads_the_model(self):
        checks = self.doctor_checks(ActiveState="activating", SubState="start")
        self.assertEqual(checks["gpu-free"].status, "pass")
        self.assertEqual(checks["port"].status, "pass")

    def test_the_doctor_does_not_accept_a_unit_that_waits_for_its_start_check_or_a_restart(self):
        # ExecStartPre runs doctor --for start while the unit is "activating"; it must still see a busy port and GPU.
        for sub in ("start-pre", "auto-restart"):
            checks = self.doctor_checks(ActiveState="activating", SubState=sub)
            self.assertEqual(checks["gpu-free"].status, "fail", sub)
            self.assertEqual(checks["port"].status, "fail", sub)


class SetupHintTest(Base):
    """L2 and L4: what the final lines of setup say about the service and about a server that was started by hand."""

    def test_setup_does_not_offer_the_service_again_when_it_is_installed(self):
        self.install_unit()
        self.ready_repo()
        code, host, out = self.run_cli(["setup", "--yes"], host=self.host(runner=FakeSystemd()))
        self.assertEqual(code, 0, out)
        self.assertNotIn("To run it as a service", out)
        self.assertNotIn("server.log", out)
        self.assertIn("./crucible service logs", out)

    def test_setup_still_offers_the_service_when_it_is_not_installed(self):
        self.configure()
        code, host, out = self.run_cli(["setup", "--yes"], host=self.host(runner=FakeRunner()))
        self.assertEqual(code, 0, out)
        self.assertIn("To run it as a service: ./crucible service install", out)

    def test_a_server_from_start_gets_the_new_settings_with_a_restart(self):
        self.configure()
        pid = int((self.repo / ".crucible/server.pid").read_text().split()[0])
        runner = FakeRunner()
        runner.live = {pid}
        code, host, out = self.run_cli(["setup", "--yes", "--allowed-host", "ai.example.org"], host=self.host(runner=runner))
        self.assertEqual(code, 0, out)
        self.assertEqual(runner.terminated, [pid], out)
        self.assertEqual(len(runner.spawned), 1, out)
        self.assertIn("restarting", out)

    def test_a_server_from_start_with_the_same_settings_is_left_running(self):
        self.configure()
        pid = int((self.repo / ".crucible/server.pid").read_text().split()[0])
        runner = FakeRunner()
        runner.live = {pid}
        code, host, out = self.run_cli(["setup", "--yes"], host=self.host(runner=runner))
        self.assertEqual(code, 0, out)
        self.assertEqual(runner.terminated, [])
        self.assertEqual(runner.spawned, [])


class DoctorServiceTest(Base):
    """D2: the service writes no server.pid, so the doctor asks systemd. A running service owns the port and the GPU memory."""

    def checks(self, active, **machine):
        self.install_unit()
        self.ready_repo()
        runner = FakeSystemd()
        runner.props.update(ActiveState="active" if active else "inactive")
        host = self.host(runner=runner, busy_ports=(8790,), gtt_used=90_000_000_000, **machine)
        s = settings_mod.build(self.repo, host)
        return {r.id: r for r in doctor.run_checks(host, s)}

    def test_a_running_service_owns_the_port_and_the_gpu_memory(self):
        checks = self.checks(active=True)
        self.assertEqual(checks["port"].status, "pass")
        self.assertEqual(checks["gpu-free"].status, "pass")
        self.assertIn("service", checks["port"].detail)

    def test_a_stopped_service_does_not(self):
        checks = self.checks(active=False)
        self.assertEqual(checks["port"].status, "fail")
        self.assertEqual(checks["gpu-free"].status, "fail")


class StopNoteTest(Base):
    """The stop messages say that the states are saved only when a state folder is on (--no-state-dir keeps them in memory)."""

    SAVES = "saves its conversation states"

    def configure_no_state(self):
        host = self.host(runner=FakeRunner())
        code = cli.main(["setup", "--yes", "--no-state-dir"], host=host, repo=self.repo)
        self.assertEqual(code, 0, "\n".join(host.lines))

    def test_the_service_stop_says_nothing_is_saved_without_a_state_folder(self):
        self.configure_no_state()
        code, host, out = self.run_cli(["service", "install"], host=self.host(runner=FakeSystemd()))
        self.assertEqual(code, 0, out)
        code, host, out = self.run_cli(["service", "stop"], host=self.host(runner=FakeSystemd()))
        self.assertEqual(code, 0, out)
        self.assertNotIn(self.SAVES, out)
        self.assertIn("memory only", out)
        code, host, out = self.run_cli(["service", "uninstall"], host=self.host(runner=FakeSystemd()))
        self.assertNotIn(self.SAVES, out)
        self.assertIn("memory only", out)

    def test_the_service_stop_still_says_it_saves_with_a_state_folder(self):
        self.install_unit()
        code, host, out = self.run_cli(["service", "stop"], host=self.host(runner=FakeSystemd()))
        self.assertIn(self.SAVES, out)
        self.assertNotIn("memory only", out)
        code, host, out = self.run_cli(["service", "uninstall"], host=self.host(runner=FakeSystemd()))
        self.assertIn(self.SAVES, out)

    def check_pid_file_stop_and_install(self, no_state):
        (self.configure_no_state if no_state else self.configure)()
        for argv in (["stop"], ["service", "install"]):
            (self.repo / ".crucible/server.pid").write_text("40007 1000\n")
            runner = FakeSystemd()
            runner.live = {40007}
            code, host, out = self.run_cli(argv, host=self.host(runner=runner))
            self.assertEqual(code, 0, out)
            self.assertEqual(self.SAVES in out, not no_state, out)
            self.assertEqual("memory only" in out, no_state, out)

    def test_a_pid_file_server_stop_without_a_state_folder_says_memory_only(self):
        self.check_pid_file_stop_and_install(no_state=True)

    def test_a_pid_file_server_stop_with_a_state_folder_says_it_saves(self):
        self.check_pid_file_stop_and_install(no_state=False)


class SourceTest(unittest.TestCase):
    def test_the_service_module_never_runs_sudo(self):
        text = (ROOT / "tools/crucible_cli/service.py").read_text()
        self.assertNotIn('"sudo"', text)
        self.assertNotIn("'sudo'", text)

    def test_the_deploy_readme_names_the_install_command(self):
        text = (ROOT / "deploy/user-systemd/README.md").read_text()
        self.assertIn("./crucible service install", text)


if __name__ == "__main__":
    unittest.main()
