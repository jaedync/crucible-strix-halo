"""The night fault driver (tools/service_faulttest.py) against a simulated machine (tests/service_sim.py).
Good behavior passes every step. Each broken behavior fails exactly the step that must catch it."""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import ROOT  # noqa: F401  (puts tools/ on sys.path)
from service_sim import Behavior, Sim

import service_faulttest as ft

IDS = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11"]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)

    def config(self, **kw):
        return ft.Config(repo=self.t / "repo", python="/usr/bin/python3", ckpt=self.t / "ckpt", engine=self.t / "engine",
                         state_dir=self.t / "states", work=self.t / "work", report=self.t / "report.json", **kw)

    def drive(self, **behavior):
        sim = Sim(Behavior(**behavior))
        cfg = self.config()
        report = ft.Driver(cfg, sim).run()
        return sim, report

    def statuses(self, report):
        return {s["id"]: s["status"] for s in report["steps"]}


class HappyRun(Base):
    def test_every_step_passes_and_the_unit_is_gone_at_the_end(self):
        sim, report = self.drive()
        self.assertEqual(list(self.statuses(report)), IDS)
        self.assertEqual(set(self.statuses(report).values()), {"pass"}, report["steps"])
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(sim.removed)
        self.assertIsNone(sim.unit_text)
        self.assertFalse(sim.guard)

    def test_the_report_file_is_json_with_a_fact_per_step(self):
        sim, report = self.drive()
        data = json.loads((self.t / "report.json").read_text())
        self.assertEqual(data["verdict"], "pass")
        facts = {s["id"]: s["facts"] for s in data["steps"]}
        self.assertAlmostEqual(facts["N1"]["load_seconds"], 4.0, delta=1.5)
        self.assertEqual(facts["N2"]["largest_gap_s"], 3.2)
        self.assertGreater(facts["N2"]["prompt_tokens"], 60000)
        self.assertLessEqual(facts["N5"]["found_seconds"], 90)
        self.assertEqual(facts["N8"]["lost_states"], 0)

    def test_the_report_holds_no_home_path(self):
        sim, report = self.drive()
        text = (self.t / "report.json").read_text()
        self.assertNotIn(str(Path.home()), text)

    def test_the_guard_timer_starts_before_the_unit_and_stops_after_it(self):
        sim, report = self.drive()
        kinds = [c[0] if c[0] != "run" else "run:" + c[1] for c in sim.calls]
        self.assertTrue(any(k.startswith("run:") and "systemd-run" in k for k in kinds))
        first_start = next(i for i, c in enumerate(sim.calls) if c[0] == "start")
        guard = next(i for i, c in enumerate(sim.calls) if c[0] == "run" and "--on-active=80min" in c)
        self.assertLess(guard, first_start)
        self.assertTrue(any(c[0] == "stop" and c[-1].startswith("crucible-ft-guard") for c in sim.calls))

    def test_a_guard_timer_left_by_a_dead_run_is_cleared_and_the_run_goes_on(self):
        sim, report = self.drive(old_guard=True)
        self.assertEqual(report["verdict"], "pass", report["steps"])
        run = next(i for i, c in enumerate(sim.calls) if c[0] == "run" and "--on-active=80min" in c)
        cleared = [i for i, c in enumerate(sim.calls) if c[0] == "stop" and c[-1] == "crucible-ft-guard.timer"]
        self.assertTrue(cleared and cleared[0] < run, "the old guard must be stopped before the new one starts")
        self.assertFalse(sim.guard)

    def test_the_unit_is_a_runtime_unit(self):
        sim, report = self.drive()
        self.assertTrue(any(c[0] == "enable" and "--runtime" in c for c in sim.calls))

    def test_n4_resends_a_conversation_and_sees_a_restore(self):
        sim, report = self.drive()
        facts = {s["id"]: s["facts"] for s in report["steps"]}
        self.assertGreater(facts["N4"]["restored_tokens"], 0)


class Broken(Base):
    def only_fails(self, expected, **behavior):
        sim, report = self.drive(**behavior)
        got = {k for k, v in self.statuses(report).items() if v == "fail"}
        self.assertEqual(got, set(expected), report["steps"])
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(sim.removed)  # a failed step never skips the uninstall
        return report

    def test_a_late_watchdog_fails_n5_and_n6(self):
        self.only_fails({"N5", "N6"}, found=120.0)

    def test_a_slow_heal_after_an_engine_kill_fails_n3(self):
        self.only_fails({"N3"}, heal_extra=30.0)

    def test_lost_states_fail_n4_and_n8(self):
        self.only_fails({"N4", "N8"}, lose_states=True)

    def test_a_sigkill_in_the_stop_fails_n8(self):
        self.only_fails({"N8"}, stop_sigkill=True)

    def test_a_large_engine_gap_fails_n2(self):
        report = self.only_fails({"N2"}, gap=30.0)
        self.assertIn("25", " ".join(s["notes"] for s in report["steps"] if s["id"] == "N2"))

    def test_a_missing_gap_line_fails_a_long_prefill(self):
        self.only_fails({"N2"}, gap_line=False)

    def test_a_bad_setting_that_never_fails_fails_n10(self):
        self.only_fails({"N10"}, bad_setting_hangs=True)

    def test_a_leftover_process_fails_n11(self):
        self.only_fails({"N11"}, leftover=True)

    def test_gpu_memory_that_does_not_return_fails_n11(self):
        self.only_fails({"N11"}, mem_back=False)

    def test_a_server_that_exits_on_a_full_disk_fails_n9(self):
        self.only_fails({"N9"}, full_disk_crash=True)

    def test_a_locked_memory_limit_that_differs_from_a_login_shell_fails_n1(self):
        self.only_fails({"N1"}, locked_same=False)

    def test_an_open_files_limit_below_the_unit_value_fails_n1(self):
        self.only_fails({"N1"}, nofile=1024)

    def test_an_open_files_limit_below_the_login_shell_is_fine(self):
        sim, report = self.drive(nofile=65536)  # a login shell has 524288 in the sim
        self.assertEqual(self.statuses(report)["N1"], "pass")

    def test_the_unit_text_sets_the_open_files_limit_of_the_rule(self):
        self.assertIn(f"LimitNOFILE={ft.NOFILE_MIN}\n", ft.render_unit(self.config()))

    def test_a_core_dump_fails_n5_and_n6(self):
        self.only_fails({"N5", "N6"}, core_dump=True)

    def test_blocked_user_namespaces_skip_n9(self):
        sim, report = self.drive(userns=False)
        self.assertEqual(self.statuses(report)["N9"], "skip")
        self.assertEqual(report["verdict"], "pass")
        self.assertIn("namespace", [s for s in report["steps"] if s["id"] == "N9"][0]["notes"])

    def test_n10_names_the_failed_check(self):
        sim, report = self.drive(status_json=False)
        n10 = [s for s in report["steps"] if s["id"] == "N10"][0]
        self.assertEqual(n10["status"], "pass")
        self.assertIn("pack", n10["facts"]["reason"])


class Control(Base):
    def test_an_error_in_a_step_is_a_fail_and_the_next_steps_still_run(self):
        sim = Sim()
        orig = sim.tokenize_len
        calls = []

        def boom(text):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("boom")
            return orig(text)
        sim.tokenize_len = boom
        report = ft.Driver(self.config(), sim).run()
        st = self.statuses(report)
        self.assertIn("fail", st.values())
        self.assertEqual(list(st), IDS)
        self.assertTrue(sim.removed)

    def test_a_stop_signal_to_the_driver_still_uninstalls(self):
        sim = Sim()
        orig = sim.systemctl

        starts = []

        def hit(*args):
            starts.append(args[0] == "start")
            if args[0] == "start" and starts.count(True) == 2:
                raise KeyboardInterrupt
            return orig(*args)
        sim.systemctl = hit
        with self.assertRaises(KeyboardInterrupt):
            ft.Driver(self.config(), sim).run()
        self.assertTrue(sim.removed)
        self.assertFalse(sim.guard)

    def test_steps_after_the_time_limit_are_skipped(self):
        sim = Sim()
        cfg = self.config()
        cfg.max_minutes = 1
        report = ft.Driver(cfg, sim).run()
        st = self.statuses(report)
        self.assertEqual(st["N1"], "pass")
        self.assertIn("skip", st.values())
        self.assertTrue(sim.removed)
        self.assertEqual(report["verdict"], "fail")  # an incomplete night is not a pass

    def test_only_selects_steps_and_n1_always_runs_first(self):
        sim = Sim()
        cfg = self.config()
        cfg.only = ["N3"]
        report = ft.Driver(cfg, sim).run()
        self.assertEqual(list(self.statuses(report)), ["N1", "N3", "N11"])


class UnitText(Base):
    def test_the_unit_follows_the_design(self):
        text = ft.render_unit(self.config())
        for line in ("Type=notify", "NotifyAccess=main", "WatchdogSec=30s", "WatchdogSignal=SIGKILL", "KillMode=control-group",
                     "TimeoutStopSec=120s", "Restart=on-failure", "RestartPreventExitStatus=2 78", "LimitCORE=0",
                     "StartLimitBurst=4", "StartLimitIntervalSec=15min", "UMask=0077"):
            self.assertIn(line, text)
        self.assertRegex(text, r"ExecStart=\".*ft-serve\.sh\" --systemd --watchdog-stall-seconds 50")
        self.assertRegex(text, r"ExecStartPre=.*doctor --for start --quiet .*--port 8791")
        self.assertNotRegex(text, r"[%]")  # no systemd specifier

    def test_a_scaled_run_scales_the_watchdog(self):
        cfg = self.config()
        cfg.watchdog_sec, cfg.stall_sec = 3, 10
        text = ft.render_unit(cfg)
        self.assertIn("WatchdogSec=3s", text)
        self.assertIn("--watchdog-stall-seconds 10", text)

    def test_a_bad_path_is_refused(self):
        cfg = self.config()
        cfg.state_dir = self.t / "a$b"
        with self.assertRaises(ValueError):
            ft.render_unit(cfg)

    def test_the_serve_script_runs_the_server_on_the_ft_port_with_the_ft_state_folder(self):
        text = ft.serve_script(self.config())
        self.assertIn("--port 8791", text)
        self.assertIn("--state-dir", text)
        self.assertIn('"$@"', text)
        self.assertNotIn("8790", text)


class Drift(Base):
    """The unit of the fault test must stay the unit of ./crucible service install (T4: service.render_unit)."""

    DIFFERS = ("# Written", "Description=", "Documentation=", "ExecStartPre=", "ExecStart=", "WatchdogSec=", "SyslogIdentifier=")

    def test_the_fault_unit_equals_the_install_unit_except_for_the_known_lines(self):
        try:
            from crucible_cli import service
            render = service.render_unit
        except (ImportError, AttributeError):
            self.skipTest("service.render_unit is not in this tree yet (T4)")
        cfg = self.config(pack="4.05", port=8791)
        mine = ft.render_unit(cfg)
        theirs = render(repo=cfg.repo, venv_py=cfg.python, exec_start=cfg.serve, bind=cfg.bind, port=cfg.port,
                        start_timeout=cfg.start_timeout, version="x")
        keep = lambda text: [ln for ln in text.splitlines() if ln and not ln.startswith(self.DIFFERS)]
        self.assertEqual(keep(mine), keep(theirs))
        pre = lambda text: next(ln for ln in text.splitlines() if ln.startswith("ExecStartPre="))
        self.assertTrue(pre(mine).startswith(pre(theirs)), (pre(mine), pre(theirs)))
        run = lambda text: next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))
        self.assertTrue(run(mine).startswith(run(theirs)), (run(mine), run(theirs)))


class Restore(Base):
    def test_a_disk_restore_is_read_from_the_server_turn_line(self):
        sim = Sim()
        conv = ft.Conv("k", "w1 w2", 20000)
        sim.chat = lambda c, max_tokens=8, timeout=60: dict(ok=True, status=200, prompt_tokens=c.tokens, cached_tokens=0)
        line = 'x crucible-ft[1]: crucible_turn {"request_id": 3, "prompt_tokens": 20064, "cached_tokens": null, "state_disk": "ready", "state_disk_tokens": 20000}'
        sim.journal = lambda since: "other line\n" + line + "\n"
        _, got = ft.Driver(self.config(), sim).resend(conv)
        self.assertEqual(got, 20000)

    def test_a_turn_line_of_another_prompt_does_not_count(self):
        sim = Sim()
        sim.chat = lambda c, max_tokens=8, timeout=60: dict(ok=True, status=200, prompt_tokens=c.tokens, cached_tokens=0)
        sim.journal = lambda since: 'crucible_turn {"prompt_tokens": 5, "state_disk_tokens": 5}\n'
        _, got = ft.Driver(self.config(), sim).resend(ft.Conv("k", "w1", 20000))
        self.assertEqual(got, 0)


class Preflight(Base):
    def test_a_real_run_names_a_missing_engine_and_pack(self):
        cfg = self.config()
        problems = ft.preflight(cfg, manager_ok=True)
        self.assertTrue(any("engine" in p for p in problems), problems)
        self.assertTrue(any("pack folder" in p for p in problems), problems)

    def test_no_user_manager_is_named(self):
        cfg = self.config()
        cfg.fake = True
        self.assertTrue(any("user manager" in p for p in ft.preflight(cfg, manager_ok=False)))
        self.assertEqual(ft.preflight(cfg, manager_ok=True), [])


class Cli(Base):
    def test_a_real_run_needs_the_real_flag(self):
        code = ft.main(["--repo", str(self.t)], machine=Sim())
        self.assertEqual(code, 2)

    def test_help_lists_the_steps(self):
        out = subprocess.run([sys.executable, str(ROOT / "tools/service_faulttest.py"), "--help"], capture_output=True,
                             text=True, timeout=30)
        self.assertEqual(out.returncode, 0)
        for i in IDS:
            self.assertIn(i, out.stdout)
        self.assertNotIn("\u2014", out.stdout)

    def test_the_plan_prints_and_changes_nothing(self):
        sim = Sim()
        code = ft.main(["--real", "--plan", "--repo", str(self.t)], machine=sim)
        self.assertEqual(code, 0)
        self.assertEqual(sim.calls, [])

    def test_exit_code_is_one_for_a_failed_night(self):
        sim = Sim(Behavior(found=200.0))
        code = ft.main(["--fake", "--repo", str(self.t), "--work", str(self.t / "w"), "--report", str(self.t / "r.json")],
                       machine=sim)
        self.assertEqual(code, 1)


class Snapshot(Base):
    def test_compare_flags_a_leftover_ft_unit_and_a_changed_linger(self):
        before = dict(units=["crucible-server.service loaded active"], timers=["x.timer"], linger="yes", runtime_units=[])
        after_ok = dict(before)
        self.assertEqual(ft.compare_snapshots(before, after_ok)["verdict"], "pass")
        after_bad = dict(before, units=before["units"] + ["crucible-ft.service loaded active"], linger="no")
        res = ft.compare_snapshots(before, after_bad)
        self.assertEqual(res["verdict"], "fail")
        self.assertTrue(any("crucible-ft" in p for p in res["problems"]))
        self.assertTrue(any("linger" in p for p in res["problems"]))

    def test_compare_lists_every_difference(self):
        before = dict(units=["a", "b"], timers=["t1"], linger="yes", runtime_units=[])
        after = dict(units=["a", "c"], timers=[], linger="yes", runtime_units=[])
        res = ft.compare_snapshots(before, after)
        self.assertEqual(res["units_removed"], ["b"])
        self.assertEqual(res["units_added"], ["c"])
        self.assertEqual(res["timers_removed"], ["t1"])


if __name__ == "__main__":
    unittest.main()
