"""The HIP runtime check: the latest ROCm (10.x) and 7.2.4 pass with no warning and no message that asks for an older
version. No GPU, no network, no sudo. Synthetic machines only."""
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import make_repo, wire

from crucible_cli import doctor, rocm  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

HIP_10_1 = "libamdhip64.so.7.16.26385"
HIP_7_2_4 = "libamdhip64.so.7.2.53211"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)

    def checks(self, **machine):
        host = wire(self.t, **machine)
        return {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host))}


class ClassifyTest(unittest.TestCase):
    def test_ten_and_later_are_current(self):
        for v in ("10.1", "10.1.0", "10.2.3", "11.0", "12"):
            self.assertEqual(rocm.classify(v), "current", v)

    def test_7_2_4_is_tested(self):
        for v in ("7.2.4", "7.2.4-1"):
            self.assertEqual(rocm.classify(v), "tested", v)

    def test_others_are_other(self):
        for v in ("7.1.0", "7.2.3", "6.4.0", "7.9"):
            self.assertEqual(rocm.classify(v), "other", v)

    def test_junk_is_other(self):
        for v in ("", "abc", None):
            self.assertEqual(rocm.classify(v), "other", repr(v))

    def test_hip_library_names_map_to_a_rocm_version(self):
        self.assertEqual(rocm.version_from_lib(HIP_10_1), "10.1")
        self.assertEqual(rocm.version_from_lib(HIP_7_2_4), "7.2.4")
        self.assertIsNone(rocm.version_from_lib("libamdhip64.so.7.99.1"))
        self.assertIsNone(rocm.version_from_lib("libamdhip64.so"))


class DetectTest(Base):
    def test_version_file_wins(self):
        host = wire(self.t, version="10.1.0", libs=(HIP_7_2_4,))
        rt = rocm.detect(host)
        self.assertEqual((rt.version, rt.state, rt.source), ("10.1.0", "current", "system"))

    def test_library_name_is_the_second_source(self):
        host = wire(self.t, rocm=False, libs=(HIP_10_1, "libamdhip64.so.7"), hipcc=False)
        rt = rocm.detect(host)
        self.assertEqual((rt.version, rt.state), ("10.1", "current"))
        self.assertEqual(rt.lib, Path(host.rocm) / "lib")

    def test_unknown_library_version_is_unknown_not_none(self):
        host = wire(self.t, rocm=False, libs=("libamdhip64.so.7.99.1",), hipcc=False)
        rt = rocm.detect(host)
        self.assertEqual(rt.state, "unknown")

    def test_old_soname_is_not_usable(self):
        host = wire(self.t, rocm=False, libs=("libamdhip64.so.6.4.60400",), hipcc=False)
        self.assertEqual(rocm.detect(host).state, "none")

    def test_nothing_is_none(self):
        host = wire(self.t, rocm=False, hipcc=False)
        rt = rocm.detect(host)
        self.assertEqual((rt.state, rt.version, rt.source), ("none", None, "none"))

    def test_library_dir_from_the_environment_counts(self):
        other = self.t / "elsewhere"
        other.mkdir()
        (other / HIP_10_1).write_text("")
        host = wire(self.t, rocm=False, hipcc=False, env={"LD_LIBRARY_PATH": f"/nonexistent:{other}"})
        rt = rocm.detect(host)
        self.assertEqual((rt.state, rt.lib, rt.source), ("current", other, "env"))

    def test_private_runtime_is_the_last_choice(self):
        work = self.t / "repo/.crucible"
        lib = work / "rocm-runtime/lib"
        lib.mkdir(parents=True)
        (lib / "libamdhip64.so.7").write_text("")
        (work / "rocm-runtime/runtime.json").write_text('{"rocm": "10.1", "hip": "7.16.26385"}')
        host = wire(self.t, rocm=False, hipcc=False)
        rt = rocm.detect(host, work=work)
        self.assertEqual((rt.source, rt.version, rt.state, rt.lib), ("private", "10.1", "current", lib))
        # a system runtime wins over the private one
        host2 = wire(self.t, version="7.2.4")
        self.assertEqual(rocm.detect(host2, work=work).source, "system")


class DoctorRocmTest(Base):
    def test_ten_one_passes_with_no_warning(self):
        res = self.checks(version="10.1.0")
        self.assertEqual(res["rocm-version"].status, "pass")
        self.assertIn("10.1", res["rocm-version"].detail)
        self.assertEqual([r.id for r in res.values() if r.status in ("fail", "warn")], [])

    def test_later_major_passes_with_no_warning(self):
        self.assertEqual(self.checks(version="11.2.0")["rocm-version"].status, "pass")

    def test_7_2_4_passes_with_no_warning(self):
        r = self.checks(version="7.2.4")["rocm-version"]
        self.assertEqual(r.status, "pass")
        self.assertNotIn("only", r.detail)

    def test_no_message_asks_for_an_older_version(self):
        for v in ("10.1.0", "7.2.4"):
            for r in self.checks(version=v).values():
                self.assertNotIn("only 7.2.4", r.detail + r.fix)
                self.assertNotIn("Install ROCm 7.2.4", r.detail + r.fix)

    def test_other_version_warns_and_names_both_tested_versions(self):
        r = self.checks(version="7.1.0")["rocm-version"]
        self.assertEqual(r.status, "warn")
        self.assertIn("10.1", r.detail)
        self.assertIn("7.2.4", r.detail)

    def test_version_from_the_library_name_passes(self):
        res = self.checks(rocm=False, libs=(HIP_10_1,))
        self.assertEqual(res["rocm-version"].status, "pass")
        self.assertIn("10.1", res["rocm-version"].detail)

    def test_no_runtime_fails_and_recommends_the_current_rocm(self):
        res = self.checks(rocm=False, hipcc=False)
        r = res["rocm-version"]
        self.assertEqual(r.status, "fail")
        self.assertIn("current ROCm", r.fix)
        self.assertIn("10.1", r.fix)
        self.assertNotIn("Install ROCm 7.2.4", r.fix)
        for word in ("Arch", "Ubuntu", "Fedora"):
            self.assertIn(word, res["hipcc"].fix)

    def test_title_does_not_name_one_version(self):
        self.assertNotIn("7.2.4", self.checks()["rocm-version"].title)

    def test_missing_rocminfo_falls_back_to_the_kfd_topology(self):
        r = self.checks(rocminfo=False, kfd_gfx=110501)["gpu-arch"]
        self.assertEqual(r.status, "pass")
        self.assertIn("gfx1151", r.detail)

    def test_kfd_topology_with_another_gpu_fails(self):
        r = self.checks(rocminfo=False, kfd_gfx=110000)["gpu-arch"]
        self.assertEqual(r.status, "fail")
        self.assertIn("gfx1100", r.detail)

    def test_missing_rocminfo_and_no_topology_is_a_warning_not_a_failure(self):
        r = self.checks(rocminfo=False)["gpu-arch"]
        self.assertEqual(r.status, "warn")


if __name__ == "__main__":
    unittest.main()
