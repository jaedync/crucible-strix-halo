"""Tests of the setup-extras options of ./crucible setup: the 16-bit n-gram table (--ngram16) and the abliterated mode
(--abliterated). No GPU, no network."""
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from crucible_cli_fakes import FakePrompt, FakeRunner, ROOT, make_repo, wire  # noqa: E402

from crucible_cli import cli, doctor, launch, options  # noqa: E402
from crucible_cli.commands import command_text  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

NG_BYTES = 102_400_496_832
NG_SHA = "71205584be130551594a3e1f43949f47c576a3cc65486f1e51f4e7ba45fa42be"


class NgramRunner(FakeRunner):
    """Runs tools/ngram16/fetch.py as a fake: a sparse file of the right size and its sha256 record."""

    def __init__(self, *a, ngram_rc=0, **kw):
        super().__init__(*a, **kw)
        self.ngram_rc = ngram_rc
        self.ngram_calls = []

    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = [str(a) for a in argv]
        if len(argv) > 1 and argv[1].endswith("tools/ngram16/fetch.py"):
            self.calls.append(argv)
            self.ngram_calls.append(argv)
            if self.ngram_rc:
                from crucible_cli.host import Result
                return Result(self.ngram_rc, "error: not enough free disk space")
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            f = out / "ngram_embedding_bf16.safetensors"
            with open(f, "wb") as fh:
                fh.truncate(NG_BYTES)   # sparse: no real disk use
            (out / "ngram_embedding_bf16.safetensors.sha256").write_text(f"{NG_SHA}  {f.name}\n")
            from crucible_cli.host import Result
            return Result(0, "checked")
        return super().run(argv, cwd=cwd, env=env, log=log, echo=echo)


class KeyedPrompt(FakePrompt):
    def __init__(self, keyed):
        super().__init__()
        self.keyed = keyed

    def __call__(self, question, default=""):
        self.asked.append(question)
        return next((a for k, a in self.keyed.items() if k in question), default)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)

    def setup(self, argv, host=None, **machine):
        host = host or wire(self.t, runner=NgramRunner(), **machine)
        code = cli.main(["setup", *argv], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def config(self):
        return json.loads((self.repo / ".crucible/config.json").read_text())


class Ngram16Test(Base):
    def test_default_is_off_and_the_engine_gets_no_word(self):
        code, host, out = self.setup(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.ngram_calls, [])
        self.assertNotIn("CRUCIBLE_PLE_BF16", host.runner.spawned[0]["env"])
        self.assertFalse(self.config().get("ngram16"))

    def test_yes_builds_the_table_and_the_start_and_the_service_get_the_word(self):
        code, host, out = self.setup(["--yes", "--ngram16", "yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(host.runner.ngram_calls), 1)
        call = host.runner.ngram_calls[0]
        self.assertIn("--delete-source", call)
        want = str(self.repo / "models/ngram16/ngram_embedding_bf16.safetensors")
        self.assertEqual(host.runner.spawned[0]["env"]["CRUCIBLE_PLE_BF16"], want)
        self.assertEqual(host.runner.spawned[0]["env"]["CRUCIBLE_PLE_HOST"], "1")
        self.assertTrue(self.config()["ngram16"])
        serve = (self.repo / ".crucible/serve.sh").read_text()
        self.assertIn(f"CRUCIBLE_PLE_BF16={want}", serve)

    def test_a_second_run_keeps_the_saved_choice_and_does_not_build_again(self):
        self.assertEqual(self.setup(["--yes", "--ngram16", "yes"])[0], 0)
        host = wire(self.t, runner=NgramRunner())
        code, host, out = self.setup(["--yes"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.ngram_calls, [])
        self.assertIn("CRUCIBLE_PLE_BF16", host.runner.spawned[0]["env"])

    def test_no_turns_it_off_and_keeps_the_file(self):
        self.assertEqual(self.setup(["--yes", "--ngram16", "yes"])[0], 0)
        host = wire(self.t, runner=NgramRunner())
        code, host, out = self.setup(["--yes", "--ngram16", "no"], host=host)
        self.assertEqual(code, 0, out)
        self.assertNotIn("CRUCIBLE_PLE_BF16", host.runner.spawned[0]["env"])
        self.assertTrue((self.repo / "models/ngram16/ngram_embedding_bf16.safetensors").exists())

    def test_disk_check_comes_first_and_counts_the_pack(self):
        # 250 GB free: the pack (110) fits, the pack and the peak of the table build (207) do not
        code, host, out = self.setup(["--yes", "--ngram16", "yes"], free=250_000_000_000)
        self.assertEqual(code, 1)
        self.assertEqual(host.runner.ngram_calls, [])
        self.assertIn("disk-ngram16", out)
        self.assertIn("--ngram16 no", out)

    def test_a_failed_machine_check_stops_the_table_download_too(self):
        code, host, out = self.setup(["--yes", "--ngram16", "yes"], gtt=60_000_000_000)
        self.assertEqual(code, 1)
        self.assertEqual(host.runner.ngram_calls, [])

    def test_a_failed_build_is_reported_and_blocks_the_start(self):
        host = wire(self.t, runner=NgramRunner(ngram_rc=2))
        code, host, out = self.setup(["--yes", "--ngram16", "yes"], host=host)
        self.assertEqual(code, 1)
        self.assertEqual(host.runner.spawned, [])
        self.assertIn("ngram16.log", out)

    def test_the_question_shows_the_measured_rows_and_enter_means_no(self):
        prompt = KeyedPrompt({})
        host = wire(self.t, runner=NgramRunner(), prompt=prompt)
        host.is_tty = True
        code, host, out = self.setup(["--service", "no"], host=host)
        self.assertEqual(code, 0, out)
        self.assertTrue(any("16-bit n-gram table" in q for q in prompt.asked), prompt.asked)
        self.assertEqual(host.runner.ngram_calls, [])
        for row in ("5.7% lower", "1.3% slower", "2.2% faster", "104.3 GB", "102.4 GB"):
            self.assertIn(row, out)

    def test_the_question_answer_yes_builds(self):
        prompt = KeyedPrompt({"16-bit n-gram table": "y"})
        host = wire(self.t, runner=NgramRunner(), prompt=prompt)
        host.is_tty = True
        code, host, out = self.setup(["--service", "no"], host=host)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(host.runner.ngram_calls), 1)

    def test_the_option_text_holds_only_rows_of_the_guide(self):
        guide = (ROOT / "tools/ngram16/fetch.py").read_text()
        self.assertIn(str(NG_BYTES), guide.replace("_", ""))
        for row in options.NGRAM16_ROWS:
            self.assertRegex(row, r"\d")

    def test_doctor_fails_the_start_when_the_saved_table_is_missing(self):
        self.assertEqual(self.setup(["--yes", "--ngram16", "yes"])[0], 0)
        (self.repo / "models/ngram16/ngram_embedding_bf16.safetensors").unlink()
        host = wire(self.t, runner=NgramRunner())
        s = settings_mod.build(self.repo, host)
        fails = {f.id for f in doctor.start_failures(host, s)}
        self.assertIn("ngram16", fails)

    def test_a_running_server_is_not_stopped_for_the_build(self):
        self.assertEqual(self.setup(["--yes"])[0], 0)
        host = wire(self.t, runner=NgramRunner())
        pid = int((self.repo / ".crucible/server.pid").read_text().split()[0])
        host.runner.live.add(pid)
        code, host, out = self.setup(["--yes", "--ngram16", "yes"], host=host)
        self.assertEqual(code, 1)
        self.assertEqual(host.runner.ngram_calls, [])
        self.assertEqual(host.runner.terminated, [])
        self.assertIn(command_text(self.repo, "stop"), out)  # the full launcher path (r373 L2)


SIDE_305 = b"fake heretic sidecar 3.05"
SIDE_405 = b"fake heretic sidecar 4.05 header"


LICENSE_TEXT = b"Qwen Community License 1.0\n(test copy)\n"


class AblRunner(NgramRunner):
    """hf download of the sidecar writes a fake file; tools/ablation_pack.py writes the fake 4.05 copy."""

    def __init__(self, *a, source=SIDE_305, license=LICENSE_TEXT, **kw):
        super().__init__(*a, **kw)
        self.source = source
        self.license = license
        self.abl_calls = []

    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = [str(a) for a in argv]
        if Path(argv[0]).name == "hf" and options.ABL_REPO in argv:
            self.calls.append(argv)
            self.abl_calls.append(argv)
            out = Path(argv[argv.index("--local-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / options.ABL_FILE).write_bytes(self.source)
            if "LICENSE" in argv:
                (out / "LICENSE").write_bytes(self.license)
            from crucible_cli.host import Result
            return Result(0, "")
        if len(argv) > 1 and argv[1].endswith("tools/ablation_pack.py"):
            self.calls.append(argv)
            self.abl_calls.append(argv)
            out = Path(argv[4])
            out.mkdir(parents=True, exist_ok=True)
            (out / options.ABL_405_FILE).write_bytes(SIDE_405)
            from crucible_cli.host import Result
            return Result(0, "SIDECAR ok")
        return super().run(argv, cwd=cwd, env=env, log=log, echo=echo)


class AblLicensePinTest(unittest.TestCase):
    def test_the_pin_of_the_license_is_the_published_file(self):
        # sha256 of LICENSE at andrevp/...-heretic@da2b870, byte-equal to the LICENSE of Qwen and of turboderp (Nov 12)
        self.assertEqual(options.ABL_LICENSE_SHA256, "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08")


class AbliteratedTest(Base):
    def setUp(self):
        super().setUp()
        import hashlib
        from unittest import mock
        pins = {"source": hashlib.sha256(SIDE_305).hexdigest(), "4.05": hashlib.sha256(SIDE_405).hexdigest()}
        patcher = mock.patch.object(options, "ABL_SHA256", {"3.05": pins["source"], "4.05": pins["4.05"]})
        patcher.start()
        self.addCleanup(patcher.stop)
        lic = mock.patch.object(options, "ABL_LICENSE_SHA256", hashlib.sha256(LICENSE_TEXT).hexdigest())
        lic.start()
        self.addCleanup(lic.stop)

    def run_setup(self, argv, runner=None, **machine):
        host = wire(self.t, runner=runner or AblRunner(), **machine)
        return self.setup(argv, host=host)

    def test_default_is_no_and_the_engine_gets_no_word(self):
        code, host, out = self.run_setup(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.abl_calls, [])
        env = host.runner.spawned[0]["env"]
        self.assertNotIn("CRUCIBLE_ABLATE", env)
        self.assertNotIn("CRUCIBLE_ABLATE_MODE", env)

    def test_yes_on_the_4_05_pack_downloads_the_pinned_file_repackages_it_and_starts_in_request_mode(self):
        code, host, out = self.run_setup(["--yes", "--abliterated", "yes"])
        self.assertEqual(code, 0, out)
        dl, pack = host.runner.abl_calls
        self.assertEqual(dl[dl.index("--revision") + 1], options.ABL_REVISION)
        self.assertIn(options.ABL_FILE, dl)
        self.assertIn("repackage", pack)
        self.assertEqual(pack[pack.index("--pack") + 1], "405")
        env = host.runner.spawned[0]["env"]
        self.assertTrue(env["CRUCIBLE_ABLATE"].endswith(options.ABL_405_FILE))
        self.assertEqual(env["CRUCIBLE_ABLATE_MODE"], "request")
        self.assertFalse(Path(env["CRUCIBLE_ABLATE"]).is_relative_to(self.repo))   # the tool refuses the repo
        serve = (self.repo / ".crucible/serve.sh").read_text()
        self.assertIn("CRUCIBLE_ABLATE_MODE=request", serve)
        # how a client selects the mode
        self.assertIn("-abliterated", out)
        self.assertIn("crucible_ablate", out)

    def test_yes_on_the_3_05_pack_uses_the_download_as_it_is(self):
        code, host, out = self.run_setup(["--yes", "--abliterated", "yes", "--pack", "3.05"])
        self.assertEqual(code, 0, out)
        self.assertEqual(len(host.runner.abl_calls), 1)
        self.assertTrue(host.runner.spawned[0]["env"]["CRUCIBLE_ABLATE"].endswith(options.ABL_FILE))

    def test_a_download_with_another_sha256_is_refused_and_removed(self):
        code, host, out = self.run_setup(["--yes", "--abliterated", "yes"], runner=AblRunner(source=b"something else"))
        self.assertEqual(code, 1)
        self.assertIn("sha256", out)
        self.assertEqual(host.runner.spawned, [])
        self.assertEqual(list(Path(self.t / "home/.local/share/crucible/ablation").rglob(options.ABL_FILE)), [])

    def test_the_license_text_comes_with_the_sidecar_and_sits_next_to_each_copy(self):
        """Qwen Community License 1.0, condition 1: the notice goes with every copy (license read of Nov 12)."""
        code, host, out = self.run_setup(["--yes", "--abliterated", "yes"])
        self.assertEqual(code, 0, out)
        dl = host.runner.abl_calls[0]
        self.assertIn("LICENSE", dl)
        root = self.t / "home/.local/share/crucible/ablation"
        self.assertEqual((root / options.ABL_SOURCE_FOLDER / "LICENSE").read_bytes(), LICENSE_TEXT)
        self.assertEqual((root / options.ABL_405_FOLDER / "LICENSE").read_bytes(), LICENSE_TEXT)

    def test_a_license_text_with_another_sha256_is_refused(self):
        code, host, out = self.run_setup(["--yes", "--abliterated", "yes"], runner=AblRunner(license=b"other terms"))
        self.assertEqual(code, 1, out)
        self.assertIn("LICENSE", out)
        self.assertEqual(host.runner.spawned, [])


    def test_a_second_run_does_not_download_again(self):
        self.assertEqual(self.run_setup(["--yes", "--abliterated", "yes"])[0], 0)
        code, host, out = self.run_setup(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(host.runner.abl_calls, [])
        self.assertEqual(host.runner.spawned[0]["env"]["CRUCIBLE_ABLATE_MODE"], "request")

    def test_no_turns_it_off(self):
        self.assertEqual(self.run_setup(["--yes", "--abliterated", "yes"])[0], 0)
        code, host, out = self.run_setup(["--yes", "--abliterated", "no"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("CRUCIBLE_ABLATE", host.runner.spawned[0]["env"])

    def test_the_question_shows_the_cost_table_and_the_responsibility_note_and_enter_means_no(self):
        prompt = KeyedPrompt({})
        host = wire(self.t, runner=AblRunner(), prompt=prompt)
        host.is_tty = True
        code, host, out = self.setup(["--service", "no", "--ngram16", "no"], host=host)
        self.assertEqual(code, 0, out)
        self.assertTrue(any("abliterated mode" in q.lower() for q in prompt.asked), prompt.asked)
        self.assertEqual(host.runner.abl_calls, [])
        for row in ("0.0206", "0.0248", "94.87%", "2.7% slower", "Responsibility", "Qwen Community License"):
            self.assertIn(row, out)

    def test_the_cost_rows_are_the_rows_of_the_guide(self):
        guide = (ROOT / "docs/ABLITERATED.md").read_text()
        for row in options.ABL_ROWS:
            for number in __import__("re").findall(r"\d[\d.,]*%?", row):
                self.assertIn(number.rstrip(".,"), guide, row)

    def test_doctor_fails_the_start_when_the_saved_sidecar_is_missing(self):
        self.assertEqual(self.run_setup(["--yes", "--abliterated", "yes"])[0], 0)
        for f in Path(self.t / "home/.local/share/crucible/ablation").rglob("*.safetensors"):
            f.unlink()
        host = wire(self.t, runner=AblRunner())
        s = settings_mod.build(self.repo, host)
        self.assertIn("abliterated", {f.id for f in doctor.start_failures(host, s)})


class StateWriteLimitTest(Base):
    """Item 11: the SSD wear guard of the state folder is a setup option with one line, and status shows the ledger."""

    def test_setup_prints_the_default_limit_and_its_yearly_size(self):
        code, host, out = self.setup(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertIn("state writes: at most 300 GiB a day (118 TB a year); compare with the TBW of your SSD", out)
        argv = host.runner.spawned[0]["argv"]
        self.assertNotIn("--state-disk-write-gib-per-day", argv)   # the default stays the engine's own default

    def test_the_option_goes_to_the_server_and_is_saved(self):
        code, host, out = self.setup(["--yes", "--state-write-gib-day", "100"])
        self.assertEqual(code, 0, out)
        argv = host.runner.spawned[0]["argv"]
        self.assertEqual(argv[argv.index("--state-disk-write-gib-per-day") + 1], "100")
        self.assertIn("at most 100 GiB a day (39 TB a year)", out)
        self.assertEqual(self.config()["state_write_gib_day"], 100)
        self.assertIn("--state-disk-write-gib-per-day 100", (self.repo / ".crucible/serve.sh").read_text())

    def test_zero_makes_the_folder_read_only_and_bad_values_are_refused(self):
        code, host, out = self.setup(["--yes", "--state-write-gib-day", "0"])
        self.assertEqual(code, 0, out)
        self.assertIn("read only", out)
        for bad in ("-1", "65537", "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    cli.main(["setup", "--yes", "--state-write-gib-day", bad], host=wire(self.t), repo=self.repo)

    def test_no_line_and_no_flag_without_a_state_folder(self):
        code, host, out = self.setup(["--yes", "--no-state-dir", "--state-write-gib-day", "50"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("state writes:", out)
        self.assertNotIn("--state-disk-write-gib-per-day", host.runner.spawned[0]["argv"])

    def status(self, host=None):
        host = host or wire(self.t, runner=NgramRunner())
        code = cli.main(["status"], host=host, repo=self.repo)
        return code, "\n".join(host.lines)

    def test_status_shows_the_bytes_written_today_and_the_limit(self):
        import time
        self.assertEqual(self.setup(["--yes", "--state-dir", str(self.t / "home/states")])[0], 0)
        today = time.strftime("%Y-%m-%d", time.localtime(1000.0))
        (self.t / "home/states/write-ledger").write_text(f"{today} {12_900_000_000}\n")
        code, out = self.status()
        self.assertEqual(code, 0, out)
        self.assertIn("state writes today: 12.9 GB of 300 GiB (322.1 GB)", out)

    def test_status_counts_an_old_day_as_zero_and_a_missing_ledger_as_none(self):
        self.assertEqual(self.setup(["--yes", "--state-dir", str(self.t / "home/states")])[0], 0)
        code, out = self.status()
        self.assertIn("state writes today: 0.0 GB of 300 GiB", out)
        (self.t / "home/states/write-ledger").write_text("2001-01-01 999999999999\n")
        code, out = self.status()
        self.assertIn("state writes today: 0.0 GB of 300 GiB", out)

    def test_status_without_a_setup_says_so(self):
        code, out = self.status()
        self.assertEqual(code, 1)
        self.assertIn(command_text(self.repo, "setup"), out)


class DiskMeasureTest(Base):
    """Item 8: setup and doctor measure the state folder's disk one time (O_DIRECT write and read, deleted after)."""

    def host_with(self, write_mbps, read_mbps, calls=None, **machine):
        host = wire(self.t, runner=NgramRunner(), **machine)

        def fake(path, nbytes):
            if calls is not None:
                calls.append((str(path), nbytes))
            return {"write_mbps": write_mbps, "read_mbps": read_mbps, "bytes": nbytes}
        host.disk_measure = fake
        return host

    def doctor(self, host, *argv):
        code = cli.main(["doctor", "--state-dir", str(self.t / "home/states"), *argv], host=host, repo=self.repo)
        return code, "\n".join(host.lines)

    def test_a_fast_ssd_passes_and_names_the_restore_time(self):
        calls = []
        code, out = self.doctor(self.host_with(3000, 5000, calls))
        self.assertIn("write 3000 MB/s, read 5000 MB/s", out)
        self.assertIn("100K-token state (2.8 GB) in about 0.6 s", out)
        self.assertEqual(calls[0][1], 2 * 1024 ** 3)
        self.assertNotIn("WARN  disk-speed", out)

    def test_a_slow_disk_warns_with_the_fix(self):
        code, out = self.doctor(self.host_with(300, 450))
        self.assertIn("WARN  disk-speed", out)
        self.assertIn("about 6.2 s", out)
        self.assertIn("--no-state-dir", out)
        self.assertEqual(code, 0)   # a warning never blocks

    def test_one_time_only_and_the_flag_skips_it(self):
        calls = []
        host = self.host_with(3000, 5000, calls)
        self.doctor(host)
        self.doctor(self.host_with(3000, 5000, calls))
        self.assertEqual(len(calls), 1)   # the saved result is used the second time
        calls.clear()
        (self.repo / ".crucible/disk-measure.json").unlink()
        code, out = self.doctor(self.host_with(3000, 5000, calls), "--no-disk-check")
        self.assertEqual(calls, [])
        self.assertIn("SKIP  disk-speed", out)

    def test_setup_measures_and_doctor_for_start_never_does(self):
        calls = []
        code, host, out = self.setup(["--yes"], host=self.host_with(3000, 5000, calls))
        self.assertEqual(code, 0, out)
        self.assertEqual(len(calls), 1)
        (self.repo / ".crucible/disk-measure.json").unlink()
        host = self.host_with(3000, 5000, calls)
        s = settings_mod.build(self.repo, host)
        doctor.for_start(host, s, quiet=True)
        self.assertEqual(len(calls), 1)

    def test_no_measure_while_the_server_runs_or_without_space_or_without_a_folder(self):
        calls = []
        self.assertEqual(self.setup(["--yes"], host=self.host_with(3000, 5000, calls))[0], 0)
        (self.repo / ".crucible/disk-measure.json").unlink()
        host = self.host_with(3000, 5000, calls)
        host.runner.live.add(int((self.repo / ".crucible/server.pid").read_text().split()[0]))
        code, out = self.doctor(host)
        self.assertIn("SKIP  disk-speed", out)
        code, out = self.doctor(self.host_with(3000, 5000, calls, free=1_000_000_000))
        self.assertIn("SKIP  disk-speed", out)
        host = self.host_with(3000, 5000, calls)
        cli.main(["doctor", "--no-state-dir"], host=host, repo=self.repo)
        self.assertEqual(len(calls), 1)

    def test_the_measure_file_is_never_left_in_the_state_folder(self):
        from crucible_cli import diskmeasure
        folder = self.t / "home/states"
        folder.mkdir(parents=True)
        res = diskmeasure.measure(folder, 8 * 1024 * 1024)
        self.assertEqual(sorted(os.listdir(folder)), [])
        self.assertEqual(sorted(os.listdir(folder.parent)), ["states"])
        self.assertTrue(res is None or (res["write_mbps"] > 0 and res["read_mbps"] > 0))

    def test_the_measure_keeps_a_file_that_it_did_not_make(self):
        # review repro: a file at the old fixed names (PID name next to the folder, plain name inside it) stays
        from crucible_cli import diskmeasure
        folder = self.t / "home/states"
        folder.mkdir(parents=True)
        old = [folder.parent / f"{diskmeasure.NAME}-{os.getpid()}", folder / diskmeasure.NAME]
        for p in old:
            p.write_text("USER DATA\n")
        diskmeasure.measure(folder, 4096, 4096)
        for p in old:
            self.assertEqual(p.read_text(), "USER DATA\n")
        self.assertEqual(sorted(os.listdir(folder.parent)), sorted(["states", old[0].name]))

    def test_a_name_collision_picks_a_new_name_and_keeps_the_old_file(self):
        from crucible_cli import diskmeasure
        folder = self.t / "home/states"
        folder.mkdir(parents=True)
        taken = folder.parent / f"{diskmeasure.NAME}-taken"
        taken.write_text("USER DATA\n")
        names = iter([taken.name, f"{diskmeasure.NAME}-fresh"])
        with mock.patch.object(diskmeasure, "_name", lambda: next(names)):
            diskmeasure.measure(folder, 4096, 4096)
        self.assertEqual(taken.read_text(), "USER DATA\n")
        self.assertEqual(sorted(os.listdir(folder.parent)), sorted(["states", taken.name]))

    def test_when_every_name_is_taken_the_measure_gives_none_and_deletes_nothing(self):
        from crucible_cli import diskmeasure
        folder = self.t / "home/states"
        folder.mkdir(parents=True)
        taken = folder.parent / f"{diskmeasure.NAME}-taken"
        taken.write_text("USER DATA\n")
        with mock.patch.object(diskmeasure, "_name", lambda: taken.name):
            self.assertIsNone(diskmeasure.measure(folder, 4096, 4096))
        self.assertEqual(taken.read_text(), "USER DATA\n")

    def test_a_file_system_without_o_direct_gives_none_and_removes_the_file_that_the_measure_made(self):
        from crucible_cli import diskmeasure
        folder = self.t / "home/states"
        folder.mkdir(parents=True)
        with mock.patch.object(diskmeasure, "_time", side_effect=OSError(22, "Invalid argument")):
            self.assertIsNone(diskmeasure.measure(folder, 4096, 4096))
        self.assertEqual(sorted(os.listdir(folder.parent)), ["states"])

    def test_the_measure_names_differ_between_calls(self):
        from crucible_cli import diskmeasure
        self.assertNotEqual(diskmeasure._name(), diskmeasure._name())


if __name__ == "__main__":
    unittest.main()
