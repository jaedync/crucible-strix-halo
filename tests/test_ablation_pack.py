import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import shutil
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "ablation_pack", pathlib.Path(__file__).resolve().parents[1] / "tools/ablation_pack.py")
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)

HIDDEN = PACK.HIDDEN


def unit_direction(seed=0.37):
    raw = [math.sin(seed * (i + 1)) + 0.01 * (i % 7) for i in range(HIDDEN)]
    norm = math.sqrt(sum(v * v for v in raw))
    values = [struct.unpack("<f", struct.pack("<f", v / norm))[0] for v in raw]
    return values


def direction_bytes(values):
    return struct.pack(f"<{HIDDEN}f", *values)


class RepackageTests(unittest.TestCase):
    def setUp(self):
        self.original_sha = PACK.HERETIC_SHA256
        # The reviewed Heretic sidecar shape: 36 sites, 26 MoE and 10 attention.
        sites = [(layer, "abl_mlp") for layer in range(17, 43)]
        sites += [(layer, "abl_attn") for layer in range(30, 40)]
        a = unit_direction(0.71)
        b = PACK.scaled(a, 2.95)
        tensors = []
        for layer, site in sites:
            tensors.append((PACK.site_key(layer, site, "a"), a))
            tensors.append((PACK.site_key(layer, site, "b"), b))
        self.source = PACK.safetensors_bytes(tensors, {"tool": "test"})
        PACK.HERETIC_SHA256 = hashlib.sha256(self.source).hexdigest()

    def tearDown(self):
        PACK.HERETIC_SHA256 = self.original_sha

    def test_values_are_unchanged_and_the_pack_moves(self):
        data, sites = PACK.build_heretic_repackage(self.source, "405")
        _, source_tensors = PACK.parse_safetensors(self.source)
        meta, tensors = PACK.parse_safetensors(data)
        self.assertEqual(sites, 36)
        self.assertEqual(tensors, source_tensors)
        self.assertEqual(meta["mode"], "heretic-repackage")
        self.assertEqual(meta["pack"], "4.05bpw_h6_ng6")
        self.assertEqual(meta["site_count"], "36")
        self.assertEqual(meta["sites"], "abl_attn,abl_mlp")
        self.assertEqual(meta["original_sha256"], PACK.HERETIC_SHA256)
        self.assertEqual(meta["tensors"], "values unchanged from the source sidecar")
        self.assertNotIn("lambda", meta)

    def test_refuses_another_source(self):
        other = PACK.safetensors_bytes([(PACK.site_key(17, "abl_mlp", "a"), unit_direction(0.11)),
                                        (PACK.site_key(17, "abl_mlp", "b"), unit_direction(0.12))],
                                       {"tool": "other"})
        with self.assertRaises(PACK.SidecarError) as caught:
            PACK.build_heretic_repackage(other, "405")
        self.assertIn("not the reviewed Heretic sidecar", str(caught.exception))

    def test_refuses_a_source_without_its_b_vectors(self):
        a = unit_direction(0.71)
        tensors = [(PACK.site_key(17, "abl_mlp", "a"), a)]
        source = PACK.safetensors_bytes(tensors, {"tool": "test"})
        PACK.HERETIC_SHA256 = hashlib.sha256(source).hexdigest()
        with self.assertRaises(PACK.SidecarError) as caught:
            PACK.build_heretic_repackage(source, "405")
        self.assertIn("missing a or b", str(caught.exception))


class OutputPathTests(unittest.TestCase):
    def test_refuses_a_path_inside_the_repository(self):
        with self.assertRaises(PACK.SidecarError):
            PACK._refuse_in_repo(PACK.REPO_ROOT / "results" / "sidecars")
        PACK._refuse_in_repo(pathlib.Path(tempfile.gettempdir()) / "abl-b1")


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA = "05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f"
PACK_405_SHA = "1bb0d4132a5798d2a7da7820b65af1e5d37e39a0cf5726f1986be924f2678baa"


class PinTests(unittest.TestCase):
    def test_the_engine_pins_the_source_and_the_405_copy(self):
        header = (ROOT / "engine/layers/ablation.h").read_text()
        self.assertIn(f'"{SOURCE_SHA}"', header)
        self.assertIn(f'"{PACK_405_SHA}"', header)
        self.assertEqual(PACK.HERETIC_SHA256, SOURCE_SHA)
        self.assertEqual(PACK.HERETIC_OUTPUT_SHA256["405"], PACK_405_SHA)

    def test_this_tool_only_repackages_the_reviewed_sidecar(self):
        self.assertFalse(hasattr(PACK, "read_direction"))
        self.assertFalse(hasattr(PACK, "build_direction_sidecar"))


class RealSourceTests(unittest.TestCase):
    """Needs the published file: set CRUCIBLE_HERETIC_SOURCE to its path."""

    @unittest.skipUnless(os.environ.get("CRUCIBLE_HERETIC_SOURCE"),
                         "set CRUCIBLE_HERETIC_SOURCE to abliteration.safetensors")
    def test_the_405_copy_has_the_pinned_sha256(self):
        source = pathlib.Path(os.environ["CRUCIBLE_HERETIC_SOURCE"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                code = PACK.main(["repackage", str(source), tmp, "--pack", "405"])
            written = pathlib.Path(tmp) / "heretic-andrevp-4.05bpw_h6_ng6.safetensors"
            self.assertEqual(code, 0)
            self.assertEqual(hashlib.sha256(written.read_bytes()).hexdigest(), PACK_405_SHA)
        self.assertIn(f"sha256={PACK_405_SHA}", report.getvalue())
        self.assertIn("sites=36", report.getvalue())

    def test_a_wrong_output_sha_writes_nothing(self):
        # A source that passes the source check but gives another output sha.
        a = unit_direction(0.71)
        tensors = [(PACK.site_key(17, "abl_mlp", "a"), a),
                   (PACK.site_key(17, "abl_mlp", "b"), PACK.scaled(a, 2.0))]
        data = PACK.safetensors_bytes(tensors, {"tool": "test"})
        saved = PACK.HERETIC_SHA256
        PACK.HERETIC_SHA256 = hashlib.sha256(data).hexdigest()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = pathlib.Path(tmp) / "src.safetensors"
                source.write_bytes(data)
                out = pathlib.Path(tempfile.gettempdir()) / "abl-pack-test-out"
                shutil.rmtree(out, ignore_errors=True)
                with contextlib.redirect_stderr(io.StringIO()) as err, \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = PACK.main(["repackage", str(source), str(out), "--pack", "405"])
                self.assertEqual(code, 1)
                self.assertIn("nothing written", err.getvalue())
                self.assertFalse(out.exists())
        finally:
            PACK.HERETIC_SHA256 = saved


if __name__ == "__main__":
    unittest.main()
