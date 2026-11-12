"""./crucible setup --layout: the named session layouts (default, long, two-long) of the server. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import make_repo, wire  # noqa: E402

from crucible_cli import cli, launch, measured  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402


def flag_values(argv):
    return {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--")}


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)
        self.host = wire(self.t)

    def build(self, **kw):
        return settings_mod.build(self.repo, self.host, **kw)

    def flags(self, s):
        return launch.server_command(s, "/venv/bin/python")[1]

    def test_the_three_names_and_their_flags(self):
        self.assertEqual(list(measured.LAYOUTS), ["default", "long", "two-long"])
        self.assertEqual(measured.LAYOUTS["default"], (4, 262144, 0, 0))
        self.assertEqual(measured.LAYOUTS["long"], (3, 524288, 2, 262144))
        self.assertEqual(measured.LAYOUTS["two-long"], (2, 524288, 0, 0))

    def test_default_layout_keeps_the_measured_command(self):
        s = self.build()
        self.assertEqual(s.layout, "default")
        argv = self.flags(s)
        f = flag_values(argv)
        self.assertEqual((f["--streams"], f["--ctx"]), ("4", "262144"))
        self.assertNotIn("--short-streams", argv)
        self.assertIn("--state-dir", argv)

    def test_long_layout_gives_one_long_and_two_short_sessions_and_no_state_folder(self):
        s = self.build(layout="long")
        argv = self.flags(s)
        f = flag_values(argv)
        self.assertEqual((f["--streams"], f["--ctx"], f["--short-streams"], f["--short-ctx"]),
                         ("3", "524288", "2", "262144"))
        self.assertIsNone(s.state_dir)  # the server refuses --state-dir with YaRN (first version)
        self.assertNotIn("--state-dir", argv)

    def test_two_long_layout(self):
        s = self.build(layout="two-long")
        argv = self.flags(s)
        f = flag_values(argv)
        self.assertEqual((f["--streams"], f["--ctx"]), ("2", "524288"))
        self.assertNotIn("--short-streams", argv)
        self.assertNotIn("--state-dir", argv)

    def test_the_layout_is_saved_and_default_goes_back(self):
        s = self.build(layout="long")
        s.work.mkdir(parents=True, exist_ok=True)
        (s.work / "config.json").write_text(json.dumps(s.to_json()))
        again = self.build()
        self.assertEqual((again.layout, again.streams, again.ctx), ("long", 3, 524288))
        self.assertIsNone(again.state_dir)
        back = self.build(layout="default")
        self.assertEqual((back.layout, back.streams, back.ctx), ("default", 4, 262144))
        self.assertIsNotNone(back.state_dir)

    def test_a_context_above_262144_keeps_the_cache_in_memory(self):
        s = self.build(streams=1, ctx=524288)
        self.assertIsNone(s.state_dir)
        self.assertNotIn("--state-dir", self.flags(s))

    def test_the_cli_takes_the_names_and_refuses_streams_or_ctx_with_a_named_layout(self):
        ap = cli.parser()
        self.assertEqual(ap.parse_args(["setup", "--layout", "two-long"]).layout, "two-long")
        with self.assertRaises(SystemExit):
            ap.parse_args(["setup", "--layout", "huge"])
        args = ap.parse_args(["setup", "--layout", "long", "--ctx", "262144"])
        with self.assertRaises(ValueError):
            settings_mod.build(self.repo, self.host, layout=args.layout, ctx=args.ctx)


if __name__ == "__main__":
    unittest.main()
