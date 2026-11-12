"""The private ROCm runtime: the fallback for a machine that has no HIP runtime. Setup offers a pinned download into the
install folder. No root, no change to system packages. The download is fake here: a synthetic wheel (a zip file) that a
fake HTTP client serves with byte ranges. No network, no GPU."""
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from crucible_cli_fakes import FakeHttp, FakePrompt, FakeRunner, make_repo, wire

import make_runtime_pin  # noqa: E402  (tools/ is on the path through crucible_cli_fakes)
from crucible_cli import cli, doctor, launch, rocm, runtime  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

URL = "https://example.invalid/whl/rocm_sdk_core-10.1.0-py3-none-linux_x86_64.whl"
HIP = "_rocm_sdk_core/lib/libamdhip64.so.7"
FILES = {
    HIP: b"HIPLIB" * 9000,
    "_rocm_sdk_core/lib/libhsa-runtime64.so.1": b"HSA" * 4000,
    "_rocm_sdk_core/lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1": b"ZZ" * 3000,
    "_rocm_sdk_core/lib/rocm_sysdeps/share/libdrm/amdgpu.ids": b"1002,\tdevice\n" * 50,   # stored, not deflated
    "_rocm_sdk_core/share/doc/rocr/LICENSE.md": b"The license text of rocr.\n",
    "_rocm_sdk_core/lib/libnot-wanted.so": b"NO" * 100,
}
WANT = [(n, n.replace("_rocm_sdk_core/", "", 1)) for n in list(FILES)[:4]]
LICENSES = [("_rocm_sdk_core/share/doc/rocr/LICENSE.md", "licenses/rocr-LICENSE.md")]
REL = {"format": 1, "tag": "v0.34.0", "commit": "a" * 40, "engine_sha256": "b" * 64, "arch": "gfx1151", "rocm": "7.2",
       "asset": "crucible-engine-v0.34.0-gfx1151-rocm7.2-linux-x86_64.gz"}


def make_wheel():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in FILES.items():
            kind = zipfile.ZIP_STORED if name.endswith(".ids") else zipfile.ZIP_DEFLATED
            z.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data, compress_type=kind)
    return buf.getvalue()


WHEEL = make_wheel()


def make_pin():
    return make_runtime_pin.build_pin(make_runtime_pin.BytesSource(WHEEL), URL, rocm="10.1", hip="7.16.26385",
                                      want=WANT, licenses=LICENSES)


class FakeDownload(FakeHttp):
    """Serves a wheel by byte ranges. fail: member data offsets that answer HTTP 404. cut: answer fewer bytes."""

    def __init__(self, data=WHEEL, status=206, cut=0, corrupt=False, **kw):
        super().__init__(**kw)
        self.data, self.status, self.cut, self.corrupt = data, status, cut, corrupt
        self.downloads = []

    def download(self, url, out, start=None, end=None, timeout=600):
        self.downloads.append((url, start, end))
        if self.status != 206:
            return self.status, 0
        body = self.data[start:end + 1]
        if self.cut:
            body = body[:-self.cut]
        if self.corrupt:
            body = bytes([body[0] ^ 1]) + body[1:]
        out.write(body)
        return 206, len(body)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)
        self.pin = make_pin()
        self.pin_file = self.t / "pin.json"
        self.pin_file.write_text(json.dumps(self.pin))
        old = runtime.PIN_FILE
        runtime.PIN_FILE = self.pin_file
        self.addCleanup(setattr, runtime, "PIN_FILE", old)

    def machine(self, http=None, **kw):
        host = wire(self.t, http=http or FakeDownload(), **kw)
        return host, settings_mod.build(self.repo, host)

    def tree(self, root):
        return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


class PinBuilderTest(Base):
    def test_every_member_has_a_place_a_size_a_hash_and_a_range(self):
        by = {m["to"]: m for m in self.pin["members"]}
        self.assertEqual(sorted(by), sorted(t for _, t in WANT))
        m = by["lib/libamdhip64.so.7"]
        self.assertEqual(m["sha256"], hashlib.sha256(FILES[HIP]).hexdigest())
        self.assertEqual(m["size"], len(FILES[HIP]))
        self.assertNotEqual(WHEEL[m["offset"]:m["offset"] + 2], b"PK")   # the data, not a zip header
        self.assertEqual(m["method"], 8)
        self.assertEqual(by["lib/rocm_sysdeps/share/libdrm/amdgpu.ids"]["method"], 0)

    def test_the_pin_names_the_runtime_and_the_source(self):
        self.assertEqual((self.pin["rocm"], self.pin["hip"], self.pin["url"]), ("10.1", "7.16.26385", URL))
        self.assertEqual(self.pin["licenses"][0]["to"], "licenses/rocr-LICENSE.md")

    def test_a_member_that_is_not_in_the_wheel_is_an_error(self):
        with self.assertRaises(make_runtime_pin.PinError):
            make_runtime_pin.build_pin(make_runtime_pin.BytesSource(WHEEL), URL, "10.1", "7.16.26385",
                                       want=[("_rocm_sdk_core/lib/nothing.so", "lib/nothing.so")], licenses=[])

    def test_the_command_answers_help(self):
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-I", str(Path(make_runtime_pin.__file__)), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--wheel", r.stdout)


class ShippedPinTest(unittest.TestCase):
    def test_the_pin_in_the_repo_is_complete_and_safe(self):
        pin = runtime.load_pin(runtime.PIN_FILE.parent / "runtime-pin.json")
        self.assertEqual(pin["rocm"], "10.1")
        self.assertEqual(pin["hip"], "7.16.26385")
        self.assertTrue(pin["url"].startswith("https://stable.repo.amd.com/"))
        tos = [m["to"] for m in pin["members"]]
        self.assertIn("lib/libamdhip64.so.7", tos)
        self.assertTrue(pin["licenses"])
        for m in pin["members"] + pin["licenses"]:
            self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse(m["to"].startswith("/") or ".." in m["to"].split("/"), m["to"])

    def test_the_pin_hip_version_is_the_one_that_the_detector_knows(self):
        pin = runtime.load_pin(runtime.PIN_FILE.parent / "runtime-pin.json")
        self.assertEqual(rocm.HIP_TO_ROCM[pin["hip"]], pin["rocm"])


class InstallTest(Base):
    def test_the_files_have_the_pinned_content_in_the_install_folder(self):
        host, s = self.machine()
        runtime.install(host, s, self.pin)
        base = s.work / "rocm-runtime"
        self.assertEqual((base / "lib/libamdhip64.so.7").read_bytes(), FILES[HIP])
        self.assertEqual((base / "lib/rocm_sysdeps/share/libdrm/amdgpu.ids").read_bytes(), FILES[WANT[3][0]])
        self.assertNotIn("libnot-wanted.so", " ".join(self.tree(base)))
        self.assertEqual((base / "licenses/rocr-LICENSE.md").read_bytes(), FILES[LICENSES[0][0]])

    def test_the_record_names_the_runtime(self):
        host, s = self.machine()
        runtime.install(host, s, self.pin)
        doc = json.loads((s.work / "rocm-runtime/runtime.json").read_text())
        self.assertEqual((doc["rocm"], doc["hip"], doc["url"]), ("10.1", "7.16.26385", URL))
        self.assertRegex(doc["pin_sha256"], r"^[0-9a-f]{64}$")

    def test_the_detector_finds_it_as_a_current_private_runtime(self):
        host, s = self.machine(rocm=False, hipcc=False)
        runtime.install(host, s, self.pin)
        rt = rocm.detect(host, s.work)
        self.assertEqual((rt.version, rt.state, rt.source), ("10.1", "current", "private"))
        self.assertEqual(rt.lib, s.work / "rocm-runtime/lib")

    def test_a_wrong_hash_stops_the_install_and_leaves_nothing(self):
        host, s = self.machine()
        pin = json.loads(json.dumps(self.pin))
        pin["members"][1]["sha256"] = "0" * 64
        with self.assertRaises(runtime.RuntimeFailed) as cm:
            runtime.install(host, s, pin)
        self.assertIn(pin["members"][1]["to"], str(cm.exception))
        self.assertIn("sha256", str(cm.exception))
        self.assertEqual([n for n in os.listdir(s.work) if n.startswith("rocm-runtime")] if s.work.exists() else [], [])

    def test_a_corrupt_download_is_refused(self):
        host, s = self.machine(http=FakeDownload(corrupt=True))
        with self.assertRaises(runtime.RuntimeFailed):
            runtime.install(host, s, self.pin)
        self.assertFalse((s.work / "rocm-runtime").exists())

    def test_a_short_download_is_refused(self):
        host, s = self.machine(http=FakeDownload(cut=5))
        with self.assertRaises(runtime.RuntimeFailed) as cm:
            runtime.install(host, s, self.pin)
        self.assertIn("bytes", str(cm.exception))

    def test_an_http_error_is_reported_with_the_code(self):
        host, s = self.machine(http=FakeDownload(status=404))
        with self.assertRaises(runtime.RuntimeFailed) as cm:
            runtime.install(host, s, self.pin)
        self.assertIn("404", str(cm.exception))
        self.assertFalse((s.work / "rocm-runtime").exists())

    def test_a_server_that_ignores_the_range_is_refused(self):
        host, s = self.machine(http=FakeDownload(status=200))
        with self.assertRaises(runtime.RuntimeFailed):
            runtime.install(host, s, self.pin)

    def test_a_member_that_grows_past_its_size_is_refused(self):
        host, s = self.machine()
        pin = json.loads(json.dumps(self.pin))
        pin["members"][0]["size"] -= 1
        with self.assertRaises(runtime.RuntimeFailed):
            runtime.install(host, s, pin)

    def test_a_pin_with_a_path_that_leaves_the_folder_is_refused_before_any_download(self):
        for bad in ("../evil.so", "/etc/evil.so", "lib/../../evil.so", "lib//x", ""):
            host, s = self.machine()
            pin = json.loads(json.dumps(self.pin))
            pin["members"][0]["to"] = bad
            with self.assertRaises(runtime.RuntimeFailed, msg=bad):
                runtime.install(host, s, pin)
            self.assertEqual(host.http.downloads, [], bad)

    def test_a_pin_url_must_be_https(self):
        host, s = self.machine()
        pin = dict(self.pin, url="http://example.invalid/x.whl")
        with self.assertRaises(runtime.RuntimeFailed):
            runtime.install(host, s, pin)

    def test_a_second_run_downloads_nothing(self):
        host, s = self.machine()
        runtime.install(host, s, self.pin)
        n = len(host.http.downloads)
        note = runtime.install(host, s, self.pin)
        self.assertEqual(len(host.http.downloads), n)
        self.assertIn("already", note)

    def test_a_damaged_file_is_downloaded_again(self):
        host, s = self.machine()
        runtime.install(host, s, self.pin)
        (s.work / "rocm-runtime/lib/libamdhip64.so.7").write_bytes(b"short")
        runtime.install(host, s, self.pin)
        self.assertEqual((s.work / "rocm-runtime/lib/libamdhip64.so.7").read_bytes(), FILES[HIP])

    def test_an_older_private_runtime_is_replaced_whole(self):
        host, s = self.machine()
        old = s.work / "rocm-runtime/lib"
        old.mkdir(parents=True)
        (old / "stale.so").write_text("old")
        runtime.install(host, s, self.pin)
        self.assertFalse((old / "stale.so").exists())
        self.assertTrue((old / "libamdhip64.so.7").exists())
        self.assertEqual([n for n in os.listdir(s.work) if n != "rocm-runtime"], [])

    def test_a_failed_install_keeps_the_older_runtime(self):
        host, s = self.machine(http=FakeDownload(status=500))
        old = s.work / "rocm-runtime/lib"
        old.mkdir(parents=True)
        (old / "libamdhip64.so.7").write_text("old")
        with self.assertRaises(runtime.RuntimeFailed):
            runtime.install(host, s, self.pin)
        self.assertEqual((old / "libamdhip64.so.7").read_text(), "old")

    def test_not_enough_disk_stops_before_the_download(self):
        host, s = self.machine(free=1000)
        with self.assertRaises(runtime.RuntimeFailed) as cm:
            runtime.install(host, s, self.pin)
        self.assertIn("disk", str(cm.exception))
        self.assertEqual(host.http.downloads, [])

    def test_a_mirror_url_in_the_environment_replaces_the_source(self):
        host, s = self.machine(env={"CRUCIBLE_RUNTIME_URL": "https://mirror.invalid/w.whl"})
        runtime.install(host, s, self.pin)
        self.assertEqual({u for u, _, _ in host.http.downloads}, {"https://mirror.invalid/w.whl"})

    def test_no_command_runs_and_nothing_outside_the_work_folder_changes(self):
        host, s = self.machine()
        before = self.tree(self.t / "home")
        runtime.install(host, s, self.pin)
        self.assertEqual(host.runner.calls, [])
        self.assertEqual(self.tree(self.t / "home"), before)
        self.assertEqual(os.listdir(s.work), ["rocm-runtime"])

    def test_the_download_asks_for_the_data_ranges_of_the_pin_only(self):
        host, s = self.machine()
        runtime.install(host, s, self.pin)
        want = {(m["offset"], m["offset"] + m["csize"] - 1) for m in self.pin["members"] + self.pin["licenses"]}
        self.assertEqual({(a, b) for _, a, b in host.http.downloads}, want)

    def test_the_text_for_the_question_names_size_source_and_place(self):
        text = runtime.offer_text(self.pin, Path("/x/.crucible"))
        for word in ("10.1", "example.invalid", "/x/.crucible/rocm-runtime", "no root"):
            self.assertIn(word, text)
        self.assertIn("system", text)


class RealHttpTest(unittest.TestCase):
    """The real Http.download against a local server that answers byte ranges (loopback only)."""

    def serve(self, ranges=True):
        import http.server
        import threading
        data = bytes(range(256)) * 100

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/gone":
                    self.send_response(404)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                if rng and ranges:
                    a, b = (int(x) for x in rng.split("=")[1].split("-"))
                    body = data[a:b + 1]
                    self.send_response(206)
                else:
                    body = data
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}", data

    def test_a_range_comes_back_as_206_with_the_bytes(self):
        from crucible_cli.http import Http
        import io
        base, data = self.serve()
        out = io.BytesIO()
        self.assertEqual(Http().download(base + "/f", out, start=10, end=19), (206, 10))
        self.assertEqual(out.getvalue(), data[10:20])

    def test_a_server_without_ranges_answers_200_and_an_error_gives_its_code(self):
        from crucible_cli.http import Http
        import io
        base, _ = self.serve(ranges=False)
        self.assertEqual(Http().download(base + "/f", io.BytesIO(), start=0, end=9)[0], 200)
        self.assertEqual(Http().download(base + "/gone", io.BytesIO(), start=0, end=9), (404, 0))

    def test_no_server_gives_zero(self):
        from crucible_cli.http import Http
        import io
        self.assertEqual(Http().download("http://127.0.0.1:9/x", io.BytesIO(), start=0, end=1), (0, 0))


class SettingsTest(Base):
    def test_a_private_runtime_sets_the_library_folder(self):
        host, s = self.machine(rocm=False, hipcc=False)
        self.assertIsNone(s.runtime_lib)
        runtime.install(host, s, self.pin)
        self.assertEqual(settings_mod.build(self.repo, host).runtime_lib, s.work / "rocm-runtime/lib")

    def test_a_system_runtime_wins_over_a_private_one(self):
        host, s = self.machine(version="10.1.0")
        runtime.install(host, s, self.pin)
        self.assertIsNone(settings_mod.build(self.repo, host).runtime_lib)

    def test_the_serve_script_and_the_command_put_the_folder_first_in_the_library_path(self):
        host, s = self.machine(rocm=False, hipcc=False)
        runtime.install(host, s, self.pin)
        s2 = settings_mod.build(self.repo, host)
        script = launch.serve_script(s2, s2.venv_python)
        lib = str(s.work / "rocm-runtime/lib")
        self.assertIn(f"LD_LIBRARY_PATH={lib}", script)
        env, _ = launch.server_command(s2, s2.venv_python)
        self.assertEqual(env["LD_LIBRARY_PATH"], lib)

    def test_without_a_private_runtime_the_library_path_is_left_alone(self):
        host, s = self.machine(version="7.2.4")
        script = launch.serve_script(s, s.venv_python)
        self.assertNotIn("LD_LIBRARY_PATH", script)
        self.assertNotIn("LD_LIBRARY_PATH", launch.server_command(s, s.venv_python)[0])


class SetupBase(Base):
    def setUp(self):
        super().setUp()
        self.old = doctor.release_engine
        doctor.release_engine = lambda repo: REL
        self.addCleanup(setattr, doctor, "release_engine", self.old)

    def run_setup(self, argv, interactive=False, answers=(), **machine):
        machine.setdefault("rocm", False)
        machine.setdefault("hipcc", False)
        machine.setdefault("rocminfo", False)
        machine.setdefault("kfd_gfx", 110501)
        host = wire(self.t, http=FakeDownload(), prompt=FakePrompt(answers) if interactive else None, **machine)
        host.is_tty = interactive
        base = ["--state-dir", str(self.t / "home/states")]
        if interactive:
            base += ["--pack", "4.05", "--models-dir", str(self.repo / "models"), "--port", "8790", "--service", "no"]
        code = cli.main(["setup", *base, *argv], host=host, repo=self.repo)
        return code, host, "\n".join(host.lines)

    def downloaded(self, host):
        return bool(host.http.downloads)


class SetupOfferTest(SetupBase):
    def test_interactive_default_is_yes_and_setup_goes_on_to_a_running_server(self):
        code, host, out = self.run_setup([], interactive=True)
        self.assertEqual(code, 0, out)
        self.assertTrue(self.downloaded(host))
        self.assertTrue(any("private ROCm runtime" in q for q in host.prompt.asked), host.prompt.asked)
        self.assertEqual(host.runner.spawned[0]["env"]["LD_LIBRARY_PATH"], str(self.repo / ".crucible/rocm-runtime/lib"))
        self.assertNotIn("FAIL", out)

    def test_the_answer_no_downloads_nothing_and_prints_the_two_ways_out(self):
        code, host, out = self.run_setup([], interactive=True, answers=["n"] * 1)
        self.assertEqual(code, 1, out)
        self.assertFalse(self.downloaded(host))
        self.assertIn("--private-runtime yes", out)
        self.assertIn("10.1", out)

    def test_yes_without_the_flag_does_not_download_and_says_how(self):
        code, host, out = self.run_setup(["--yes"])
        self.assertEqual(code, 1, out)
        self.assertFalse(self.downloaded(host))
        self.assertIn("--private-runtime yes", out)

    def test_yes_with_the_flag_downloads_and_does_not_ask(self):
        code, host, out = self.run_setup(["--yes", "--private-runtime", "yes"])
        self.assertEqual(code, 0, out)
        self.assertTrue(self.downloaded(host))
        self.assertTrue((self.repo / ".crucible/rocm-runtime/lib/libamdhip64.so.7").exists())

    def test_the_flag_no_skips_the_offer(self):
        code, host, out = self.run_setup(["--yes", "--private-runtime", "no"])
        self.assertFalse(self.downloaded(host))
        self.assertNotIn("--private-runtime yes", out)

    def test_a_system_runtime_is_never_replaced(self):
        for version in ("10.1.0", "7.2.4"):
            code, host, out = self.run_setup(["--yes", "--private-runtime", "yes"], rocm=True, version=version)
            self.assertFalse(self.downloaded(host), version)
            self.assertFalse((self.repo / ".crucible/rocm-runtime").exists(), version)
            self.assertNotIn("older ROCm", out)
            self.assertNotIn("older version", out)

    def test_a_system_runtime_means_no_question(self):
        code, host, out = self.run_setup([], interactive=True, rocm=True, version="10.1.0")
        self.assertFalse(any("runtime" in q.lower() for q in host.prompt.asked), host.prompt.asked)

    def test_a_source_build_gets_no_offer(self):
        doctor.release_engine = lambda repo: None
        code, host, out = self.run_setup(["--yes", "--private-runtime", "yes"])
        self.assertFalse(self.downloaded(host))
        self.assertIn("build", out)

    def test_dry_run_downloads_nothing_and_names_the_step(self):
        code, host, out = self.run_setup(["--yes", "--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertFalse(self.downloaded(host))
        self.assertIn("private ROCm runtime", out)

    def test_a_failed_download_prints_the_error_and_the_usual_fix(self):
        host = wire(self.t, http=FakeDownload(status=503), rocm=False, hipcc=False, rocminfo=False, kfd_gfx=110501)
        code = cli.main(["setup", "--yes", "--private-runtime", "yes", "--state-dir", str(self.t / "home/states")],
                        host=host, repo=self.repo)
        out = "\n".join(host.lines)
        self.assertEqual(code, 1, out)
        self.assertIn("503", out)
        self.assertIn("Install the current ROCm", out)
        self.assertFalse((self.repo / ".crucible/rocm-runtime").exists())

    def test_setup_run_twice_downloads_once(self):
        self.run_setup(["--yes", "--private-runtime", "yes"])
        code, host, out = self.run_setup(["--yes", "--private-runtime", "yes"])
        self.assertFalse(self.downloaded(host))
        self.assertEqual(code, 0, out)

    def test_the_doctor_counts_the_private_runtime_as_a_pass(self):
        host, s = self.machine(rocm=False, hipcc=False, rocminfo=False, kfd_gfx=110501)
        runtime.install(host, s, self.pin)
        rows = {r.id: r for r in doctor.run_checks(host, settings_mod.build(self.repo, host))}
        self.assertEqual(rows["rocm-version"].status, "pass")
        self.assertIn("private", rows["rocm-version"].detail)


if __name__ == "__main__":
    unittest.main()
