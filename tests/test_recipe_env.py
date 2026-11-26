"""tools/recipe_env.py prints the server arguments and the environment of tools/crucible_cli/measured.py (the measured
run: the README table and docs/DOCKER.md) for a container recipe, at the working tree or at a git ref (a release tag).
A recipe copies the printed lines, never the values by hand. No network; a temporary git repository for the ref tests."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import ROOT
from crucible_cli import measured

SCRIPT = ROOT / "tools/recipe_env.py"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t" + chr(64) + "example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t" + chr(64) + "example.invalid", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

OLD_MEASURED = '''"""An older table."""
ENV = {
    "CRUCIBLE_PLE_HOST": "1",
    "CRUCIBLE_DRAFT_HEAD": "q2",
}
SERVER_FLAGS = {
    "--streams": "4",
    "--ctx": "262144",
}
STREAMS = int(SERVER_FLAGS["--streams"])
'''


def run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True,
                          timeout=60, env={**os.environ, **GIT_ENV})


class WorkingTree(unittest.TestCase):
    def test_json_equals_measured(self):
        r = run("--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout)
        self.assertEqual(got["args"], [list(kv) for kv in measured.SERVER_FLAGS.items()])
        self.assertEqual(got["env"], [list(kv) for kv in measured.ENV.items()])
        self.assertEqual(got["ref"], None)

    def test_the_two_words_that_the_probe_lacked_are_printed(self):
        env = dict(json.loads(run("--format", "json").stdout)["env"])
        self.assertEqual((env["CRUCIBLE_EXACT_RESUME"], env["CRUCIBLE_TWIN"]), ("1", "1"))

    def test_text_lines_equal_measured(self):
        r = run()
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [line for line in r.stdout.splitlines() if not line.startswith("#")]
        want = [f"arg {k} {v}" for k, v in measured.SERVER_FLAGS.items()] + \
               [f"env {k}={v}" for k, v in measured.ENV.items()]
        self.assertEqual(lines, want)

    def test_cpp_holds_each_pair_in_order(self):
        r = run("--format", "cpp")
        self.assertEqual(r.returncode, 0, r.stderr)
        pairs = [line.strip() for line in r.stdout.splitlines() if line.strip().startswith("{\"")]
        want = [f'{{"{k}", "{v}"}},' for k, v in measured.SERVER_FLAGS.items()] + \
               [f'{{"{k}", "{v}"}},' for k, v in measured.ENV.items()]
        self.assertEqual(pairs, want)
        self.assertIn("kCrucibleArgs", r.stdout)
        self.assertIn("kCrucibleEnv", r.stdout)


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class AtARef(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        (self.repo / "tools/crucible_cli").mkdir(parents=True)
        self.git("init", "-q")
        self.write(OLD_MEASURED)
        self.git("add", ".")
        self.git("commit", "-q", "-m", "old")
        self.git("tag", "v0.0.1")
        self.write((ROOT / "tools/crucible_cli/measured.py").read_text())
        self.git("commit", "-q", "-am", "new")

    def git(self, *args):
        subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false", *args],
                       cwd=self.repo, check=True, capture_output=True, timeout=60, env={**os.environ, **GIT_ENV})

    def write(self, text):
        (self.repo / "tools/crucible_cli/measured.py").write_text(text)

    def test_a_tag_gives_the_table_of_that_tag(self):
        r = run("--repo", str(self.repo), "--ref", "v0.0.1", "--format", "json")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout)
        self.assertEqual(got["ref"], "v0.0.1")
        self.assertEqual(got["env"], [["CRUCIBLE_PLE_HOST", "1"], ["CRUCIBLE_DRAFT_HEAD", "q2"]])
        self.assertEqual(got["args"], [["--streams", "4"], ["--ctx", "262144"]])

    def test_the_text_names_the_ref(self):
        r = run("--repo", str(self.repo), "--ref", "v0.0.1")
        self.assertEqual(r.stdout.splitlines()[0], "# tools/crucible_cli/measured.py at v0.0.1")

    def test_the_tip_equals_measured(self):
        got = json.loads(run("--repo", str(self.repo), "--ref", "HEAD", "--format", "json").stdout)
        self.assertEqual(got["env"], [list(kv) for kv in measured.ENV.items()])

    def test_an_unknown_ref_fails_with_its_name(self):
        r = run("--repo", str(self.repo), "--ref", "v9.9.9")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("v9.9.9", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_a_table_without_env_fails(self):
        self.write('SERVER_FLAGS = {"--ctx": "1"}\n')
        self.git("commit", "-q", "-am", "no env")
        r = run("--repo", str(self.repo), "--ref", "HEAD")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ENV", r.stderr)

    def test_a_value_that_is_not_a_string_fails(self):
        self.write('ENV = {"A": 1}\nSERVER_FLAGS = {"--ctx": "1"}\n')
        self.git("commit", "-q", "-am", "int")
        r = run("--repo", str(self.repo), "--ref", "HEAD")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("string", r.stderr)


if __name__ == "__main__":
    unittest.main()
