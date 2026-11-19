"""The memory class of the machine (tools/crucible_cli/machine.py): a 128 GB machine keeps every measured default, a 96 GB
machine gets the 3.05 pack with a layout that fits, and a smaller machine gets a clear refusal. Fake MemTotal values only.

The MemTotal values are those of the test machine (131,156,752 kB on a 128 GiB machine: 3,060,976 kB go to the firmware and the
0.5 GiB VRAM carve-out) with the same loss on a 96 GiB and a 64 GiB machine."""
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import make_repo, wire
from crucible_cli import doctor, kernelparams, launch, machine, measured, settings as settings_mod

MEM_128 = 131_156_752
MEM_96 = 97_602_320
MEM_64 = 64_047_888
GTT_96 = "amdgpu.gttsize=94208 ttm.pages_limit=24117248"   # 92 GiB: MemTotal 93 GiB less 1 GiB, as 124 of 125 on the test machine

# The server command of a 128 GB machine before machine.py: the measured flags (README table) and the same order.
ARGV_128_TAIL = ["--streams", "4", "--ctx", "262144", "--prefill", "4096", "--conversation-cache-mib", "9700",
                 "--host-reserve-mib", "11264"]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name)
        self.repo = make_repo(self.t)

    def tearDown(self):
        self._tmp.cleanup()

    def settings(self, mem, **kw):
        host = wire(self.t, mem_total_kb=mem)
        return host, settings_mod.build(self.repo, host, **kw)

    def checks(self, mem, gtt=133_000_000_000, **kw):
        host = wire(self.t, mem_total_kb=mem, gtt=gtt)
        return {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host, **kw))}


class FitTest(Base):
    def test_a_128_gb_machine_keeps_every_measured_value(self):
        for mem in (MEM_128, 130_000_000, 120 * 1024 * 1024, None):
            with self.subTest(mem=mem):
                f = machine.fit(mem)
                self.assertIs(f, machine.FULL)
        f = machine.FULL
        self.assertEqual((f.pack, f.streams, f.ctx), ("4.05", measured.STREAMS, measured.CTX))
        self.assertEqual(f.cache_mib, measured.SERVER_FLAGS["--conversation-cache-mib"])
        self.assertEqual(f.gtt_params, doctor.GTT_PARAMS)
        self.assertEqual(f.gtt_min_bytes, doctor.GTT_MIN_BYTES)
        self.assertEqual(f.layouts, measured.LAYOUTS)
        self.assertTrue(f.supported)

    def test_a_96_gb_machine_gets_the_3_05_pack_and_two_long_sessions(self):
        f = machine.fit(MEM_96)
        self.assertTrue(f.supported)
        self.assertEqual(f.name, "96 GB")
        self.assertEqual((f.pack, f.packs), ("3.05", ("3.05",)))
        self.assertEqual((f.streams, f.ctx, f.cache_mib), (2, 262144, "4096"))
        self.assertEqual(f.gtt_params, GTT_96)
        self.assertEqual(f.gtt_min_bytes, 89 * 2 ** 30)
        self.assertEqual(f.layouts["default"], (2, 262144, 0, 0))
        self.assertEqual(f.layouts["long"], (1, 524288, 0, 0))
        self.assertNotIn("two-long", f.layouts)

    def test_the_class_edges(self):
        self.assertEqual(machine.fit(machine.FULL_MIN_KB - 1).name, machine.fit(machine.MID_MIN_KB).name)
        self.assertTrue(machine.fit(machine.MID_MIN_KB).supported)
        self.assertFalse(machine.fit(machine.MID_MIN_KB - 1).supported)

    def test_a_64_gb_machine_is_not_supported_and_says_why(self):
        f = machine.fit(MEM_64)
        self.assertFalse(f.supported)
        self.assertEqual(f.packs, ())
        for word in ("96 GB", "3.05 pack", "9 GiB", "61 GiB"):
            self.assertIn(word, machine.refusal(MEM_64))

    def test_the_gtt_values_follow_the_memory(self):
        self.assertEqual(machine.gtt_params(MEM_96), GTT_96)
        self.assertEqual(machine.gtt_params(MEM_64), "amdgpu.gttsize=61440 ttm.pages_limit=15728640")


class SettingsTest(Base):
    def test_defaults_follow_the_machine(self):
        _, s = self.settings(MEM_96)
        self.assertEqual((s.pack, s.streams, s.ctx, s.short_streams), ("3.05", 2, 262144, 0))
        _, s = self.settings(MEM_128)
        self.assertEqual((s.pack, s.streams, s.ctx), ("4.05", 4, 262144))

    def test_the_user_choice_wins(self):
        _, s = self.settings(MEM_96, pack="4.05", streams=3)
        self.assertEqual((s.pack, s.streams), ("4.05", 3))

    def test_layouts_of_a_96_gb_machine(self):
        _, s = self.settings(MEM_96, layout="long")
        self.assertEqual((s.streams, s.ctx, s.short_streams, s.short_ctx), (1, 524288, 0, 0))
        with self.assertRaises(ValueError) as e:
            self.settings(MEM_96, layout="two-long")
        self.assertIn("96 GB", str(e.exception))

    def test_layouts_of_a_128_gb_machine_do_not_change(self):
        for name, want in measured.LAYOUTS.items():
            _, s = self.settings(MEM_128, layout=name)
            self.assertEqual((s.streams, s.ctx, s.short_streams, s.short_ctx), want)


class CommandTest(Base):
    def test_the_128_gb_command_is_the_measured_one(self):
        _, s = self.settings(MEM_128)
        env, argv = launch.server_command(s, "/py")
        i = argv.index("--streams")
        self.assertEqual(argv[i:i + len(ARGV_128_TAIL)], ARGV_128_TAIL)
        self.assertEqual(env, {**measured.ENV, **launch.own_head_env(s)})
        # A Settings object without a machine (older callers, tests) gives the same command.
        bare = settings_mod.Settings(repo=self.repo)
        self.assertEqual(launch.server_command(bare, "/py")[1][i:i + len(ARGV_128_TAIL)], ARGV_128_TAIL)

    def test_the_96_gb_command(self):
        _, s = self.settings(MEM_96)
        _, argv = launch.server_command(s, "/py")
        i = argv.index("--streams")
        self.assertEqual(argv[i:i + 10], ["--streams", "2", "--ctx", "262144", "--prefill", "4096",
                                          "--conversation-cache-mib", "4096", "--host-reserve-mib", "11264"])
        self.assertIn("Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5", argv[3])
        self.assertIn("--vision", argv)


class DoctorTest(Base):
    def test_128_gb_lines_do_not_change(self):
        r = self.checks(MEM_128)
        self.assertEqual((r["memory"].status, r["memory"].title), ("pass", "System memory 120 GiB or more"))
        self.assertEqual(r["gtt"].status, "pass")

    def test_96_gb_passes_with_the_3_05_pack(self):
        r = self.checks(MEM_96, gtt=94208 * 2 ** 20)
        self.assertEqual(r["memory"].status, "pass", r["memory"].fix)
        self.assertIn("3.05 pack", r["memory"].detail)
        self.assertIn("2 sessions of 262,144 tokens", r["memory"].detail)
        self.assertEqual(r["gtt"].status, "pass", r["gtt"].detail)

    def test_96_gb_refuses_the_4_05_pack(self):
        r = self.checks(MEM_96, pack="4.05")["memory"]
        self.assertEqual((r.status, r.blocks), ("fail", ("start",)))
        self.assertIn("--pack 3.05", r.fix)

    def test_96_gb_with_a_small_gtt_gets_its_own_kernel_line(self):
        r = self.checks(MEM_96, gtt=48 * 2 ** 30)["gtt"]
        self.assertEqual(r.status, "fail")
        self.assertIn(GTT_96, r.fix)
        self.assertNotIn(doctor.GTT_PARAMS, r.fix)

    def test_64_gb_fails_with_the_reason(self):
        r = self.checks(MEM_64)["memory"]
        self.assertEqual((r.status, r.blocks), ("fail", ("start",)))
        self.assertIn("96 GB", r.fix)

    def checks_with(self, meminfo=None, vram=None, mem=MEM_96, gtt=94208 * 2 ** 20):
        """The doctor lines of a fake machine whose /proc/meminfo text is MEMINFO, or whose VRAM (UMA frame buffer) is VRAM."""
        host = wire(self.t, mem_total_kb=mem, gtt=gtt)
        if meminfo is not None:
            (host.proc / "meminfo").write_text(meminfo)
        if vram is not None:
            (Path(host.sysfs) / "class/drm/card0/device/mem_info_vram_total").write_text(f"{vram}\n")
        return {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host))}

    def test_no_memtotal_gives_the_unknown_fail_not_a_crash(self):
        # machine.fit(None) is the 128 GB class, so the 96 GB branch formatted None.
        for text in ("MemAvailable:   100000000 kB\n", ""):
            with self.subTest(meminfo=text):
                r = self.checks_with(meminfo=text)["memory"]
                self.assertEqual((r.status, r.blocks), ("fail", ("start",)))
                self.assertIn("unknown", r.detail)
                self.assertTrue(r.fix)

    def test_96_gb_pass_line_names_the_start_time_fit(self):
        # At the bottom of the class the start-time fit may start fewer sessions or a smaller cache.
        r = self.checks_with()["memory"]
        self.assertEqual(r.status, "pass")
        self.assertIn("a smaller conversation cache or fewer sessions", r.detail)
        self.assertIn("shorter --ctx", r.detail)

    def test_96_gb_pass_line_gives_the_uma_advice_when_the_vram_is_large(self):
        # A 128 GB machine with a large UMA frame buffer looks like a 96 GB machine.
        r = self.checks_with(vram=16 * 2 ** 30)["memory"]
        self.assertEqual(r.status, "pass")
        self.assertIn("UMA frame buffer", r.detail)
        self.assertIn("512 MB", r.detail)
        self.assertIn("16 GiB", r.detail)
        small = self.checks_with(vram=512 * 2 ** 20)["memory"]
        self.assertNotIn("UMA frame buffer", small.detail)
        full = self.checks_with(vram=16 * 2 ** 30, mem=MEM_128, gtt=133_000_000_000)["memory"]
        self.assertNotIn("UMA frame buffer", full.detail)


class SetupTextTest(Base):
    """The setup question, the pack loop and the --layout help on a 96 GB and a 64 GB machine."""

    def ask_pack(self, mem, answers):
        from crucible_cli import cli
        host = wire(self.t, mem_total_kb=mem)
        args = cli.parser().parse_args(["setup", "--models-dir", str(self.repo / "models"), "--no-state-dir",
                                        "--port", "8790", "--bind", "127.0.0.1", "--ngram16", "no", "--abliterated", "no",
                                        "--no-api-key"])
        it = iter(answers)
        seen = []

        def ask(question, default):
            seen.append(question)
            if question.startswith(("Pack", "Type")):
                return next(it)
            return default
        host.prompt = ask
        cli._ask_settings(args, host, self.repo)
        return args.pack, [q for q in seen if q.startswith(("Pack", "Type"))]

    def test_the_96_gb_pack_loop_does_not_take_4_05(self):
        pack, asked = self.ask_pack(MEM_96, ["4.05", "3.05"])
        self.assertEqual(pack, "3.05")
        self.assertEqual(len(asked), 2)
        self.assertNotIn("4.05 or 3.05", asked[1])

    def test_the_128_gb_pack_question_does_not_change(self):
        # The question of the setup before the memory classes, with the 4.18 pack.
        pack, asked = self.ask_pack(MEM_128, ["4.05"])
        self.assertEqual((pack, asked), ("4.05", ["Pack: 4.05 (best quality, 107 GB), 4.18 (the same quality, slower, "
                                                  "122 GB) or 3.05 (smaller, 85 GB)"]))

    def test_the_128_gb_class_takes_the_4_18_pack_and_the_96_gb_class_does_not(self):
        self.assertIn("4.18", machine.fit(MEM_128).packs)
        pack, asked = self.ask_pack(MEM_128, ["4.18"])
        self.assertEqual((pack, len(asked)), ("4.18", 1))
        self.assertNotIn("4.18", machine.fit(MEM_96).packs)
        pack, asked = self.ask_pack(MEM_96, ["4.18", "3.05"])
        self.assertEqual((pack, len(asked)), ("3.05", 2))

    def test_the_64_gb_question_does_not_say_that_3_05_fits(self):
        pack, asked = self.ask_pack(MEM_64, ["3.05"])
        self.assertNotIn("no room for the 4.05 pack", asked[0])
        self.assertIn("doctor", asked[0])
        self.assertIn("96 GB", asked[0])

    def test_the_layout_help_names_the_machine_class(self):
        from crucible_cli import cli
        text = " ".join(cli.parser()._subparsers._group_actions[0].choices["setup"].format_help().split())
        self.assertIn("On a 128 GB machine: default (4 x 262,144)", text)
        self.assertIn("On a 96 GB machine: default (2 x 262,144), long (1 x 524,288)", text)


class KernelParamsTest(Base):
    def test_the_plan_asks_for_the_values_of_this_machine(self):
        for mem, want in ((MEM_128, doctor.GTT_PARAMS), (MEM_96, GTT_96)):
            with self.subTest(mem=mem):
                host = wire(self.t, mem_total_kb=mem, cmdline="root=/dev/nvme0n1p2 rw")
                p = kernelparams.plan(host)
                self.assertEqual(p.tokens, want.split())
        host = wire(self.t, mem_total_kb=MEM_96, cmdline=f"root=/dev/nvme0n1p2 rw {GTT_96}")
        self.assertIsNone(kernelparams.plan(host))


if __name__ == "__main__":
    unittest.main()
