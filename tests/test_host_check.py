"""The GTT host rule as one pure function (tools/crucible_cli/hostcheck.py) and its test vectors
(tests/data/host_check/gtt_vectors.json). The vectors are written by hand, not made by the function. A port of the rule to
another language (a container host check) reads the same file. The doctor uses the function: for each vector that the
doctor can see (no pack rule, not a container), its gtt line has the status and text of the vector. Fake machines only."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import ROOT, make_repo, wire
from crucible_cli import doctor, hostcheck, settings as settings_mod

VECTORS_FILE = ROOT / "tests/data/host_check/gtt_vectors.json"
VECTORS = json.loads(VECTORS_FILE.read_text())["vectors"]
INPUT_KEYS = {"gtt_total_bytes", "memtotal_kb", "pack", "container", "params_on_cmdline"}
OUTPUT_KEYS = {"status", "code", "reason", "params"}
CMDLINE_PARAMS = "root=/dev/nvme0n1p2 rw amdgpu.gttsize=126976 ttm.pages_limit=32505856"
CMDLINE_PLAIN = "root=/dev/nvme0n1p2 rw"


class VectorFile(unittest.TestCase):
    def test_every_vector_has_the_documented_keys(self):
        for v in VECTORS:
            with self.subTest(v["name"]):
                self.assertEqual(set(v["input"]), INPUT_KEYS)
                self.assertEqual(set(v["output"]), OUTPUT_KEYS)
                self.assertIn(v["output"]["status"], hostcheck.STATUSES)
                self.assertIn(v["output"]["code"], hostcheck.CODES)

    def test_names_are_unique(self):
        names = [v["name"] for v in VECTORS]
        self.assertEqual(len(names), len(set(names)))

    def test_the_file_covers_each_status_and_code(self):
        self.assertEqual({v["output"]["status"] for v in VECTORS}, set(hostcheck.STATUSES))
        self.assertEqual({v["output"]["code"] for v in VECTORS}, set(hostcheck.CODES))


class Rule(unittest.TestCase):
    def test_each_vector(self):
        for v in VECTORS:
            with self.subTest(v["name"]):
                got = hostcheck.gtt_verdict(**v["input"])
                self.assertEqual(got.as_dict(), v["output"])

    def test_the_input_list_is_not_changed(self):
        totals = [0, None, 133143986176]
        hostcheck.gtt_verdict(totals, 131156752)
        self.assertEqual(totals, [0, None, 133143986176])

    def test_an_unknown_pack_is_an_error(self):
        with self.assertRaises(ValueError):
            hostcheck.gtt_verdict([133143986176], 131156752, pack="9.99")


class DoctorUsesTheRule(unittest.TestCase):
    """The gtt line of ./crucible doctor for each vector with no pack and no container."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.t = Path(self._tmp.name)
        self.repo = make_repo(self.t)

    def doctor_gtt(self, inp):
        mem = inp["memtotal_kb"]
        cmdline = CMDLINE_PARAMS if inp["params_on_cmdline"] else CMDLINE_PLAIN
        host = wire(self.t, mem_total_kb=mem if mem is not None else 131156752, cmdline=cmdline)
        if mem is None:
            (host.proc / "meminfo").write_text("MemAvailable:   100000000 kB\n")
        drm = Path(host.sysfs) / "class/drm"
        shutil.rmtree(drm)
        for n, value in enumerate(inp["gtt_total_bytes"]):
            dev = drm / f"card{n}/device"
            dev.mkdir(parents=True)
            (dev / "mem_info_gtt_used").write_text("2000000\n")
            if value is not None:
                (dev / "mem_info_gtt_total").write_text(f"{value}\n")
        drm.mkdir(parents=True, exist_ok=True)
        return {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host))}["gtt"]

    def test_doctor_vectors(self):
        seen = 0
        for v in VECTORS:
            inp, out = v["input"], v["output"]
            if inp["pack"] is not None or inp["container"]:
                continue
            seen += 1
            with self.subTest(v["name"]):
                r = self.doctor_gtt(inp)
                self.assertEqual((r.status, r.detail), (out["status"], out["reason"]))
                if out["code"] == "gtt-low":
                    self.assertEqual(r.fix, doctor.kernel_fix(out["params"]))
                    self.assertEqual(r.blocks, ("start",))
                elif out["code"] == "no-gtt":
                    # The doctor keeps the fix text that it had: the 128 GB class line (doctor.KERNEL_FIX).
                    self.assertTrue(r.fix.endswith(doctor.KERNEL_FIX))
                    self.assertIn("lsmod | grep amdgpu", r.fix)
                    self.assertEqual(r.blocks, ("start",))
                else:
                    self.assertEqual((r.fix, r.blocks), ("", ()))
        self.assertGreaterEqual(seen, 12)


if __name__ == "__main__":
    unittest.main()
