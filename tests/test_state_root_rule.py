"""The state folder rule, host copy (tools/crucible_cli/state_root.py), against the shared table
tests/data/state_root_rule.tsv. tests/state_root_test.cpp runs the same table against the engine copy
(engine/runtime/state_root.h), so the two give the same verdict for the same path (a state folder outside
the home folder, Nov 12)."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from crucible_cli import state_root  # noqa: E402

TABLE = Path(__file__).resolve().parent / "data" / "state_root_rule.tsv"
# An identity folder name, built from one group so that the leak guard does not see a 50-character data run.
IDENT = "-".join(["0123456789abcdef"] * 3)


def rows():
    out = []
    for line in TABLE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, setup, path, uid, gid, code = line.split("\t")
        out.append((name, setup, path, uid, gid, code))
    return out


def sticky_root_tmp():
    """True when /tmp is a root folder with the sticky bit (root shows as the overflow user in a user namespace)."""
    try:
        st = os.lstat("/tmp")
    except OSError:
        return False
    return st.st_uid in state_root.trusted_owners(os.geteuid()) and st.st_uid != os.geteuid() and bool(st.st_mode & 0o1000)


class Case:
    def __init__(self, base):
        self.t = Path(tempfile.mkdtemp(dir=base))
        os.chmod(self.t, 0o700)
        self.h = self.t / "home"
        self.h.mkdir(mode=0o700)
        self.s = None

    def sub(self, text):
        if "S" in text.split("/")[0:1] and self.s is None:
            self.s = Path(tempfile.mkdtemp(dir="/tmp", prefix="state-root-"))
        out = []
        for i, part in enumerate(text.split("/")):
            if i == 0 and part == "T":
                part = str(self.t)
            elif i == 0 and part == "H":
                part = str(self.h)
            elif i == 0 and part == "S":
                part = str(self.s)
            elif part == "ID":
                part = IDENT
            out.append(part)
        return "/".join(out)

    def build(self, setup):
        if setup == "-":
            return
        for step in setup.split(";"):
            kind, *args = step.split(" ")
            if kind == "dir":
                p = Path(self.sub(args[0]))
                p.mkdir()
                os.chmod(p, int(args[1], 8))
            elif kind == "file":
                Path(self.sub(args[0])).write_text("x")
            elif kind == "link":
                Path(self.sub(args[0])).symlink_to(self.sub(args[1]))
            else:
                raise ValueError(step)

    def cleanup(self):
        for p in (self.t, self.s):
            if p is None:
                continue
            for root, dirs, _ in os.walk(p):
                for d in dirs:
                    q = os.path.join(root, d)
                    if not os.path.islink(q):
                        os.chmod(q, 0o700)
            shutil.rmtree(p, ignore_errors=True)


class StateRootTableTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, True)
        base_verdict = state_root.state_root_verdict(os.path.join(self.base, "probe"))
        # A gate that cannot run the rows fails: a skip would let the run pass with no row checked.
        self.assertTrue(base_verdict.ok, f"the temp folder fails the rule itself, so no row can run: {base_verdict.message}")

    def run_row(self, row):
        name, setup, path, uid, gid, code = row
        case = Case(self.base)
        self.addCleanup(case.cleanup)
        case.build(setup)
        real_uid = os.geteuid()
        group = os.lstat(case.t).st_gid
        return state_root.state_root_verdict(case.sub(path), home=str(case.h),
                                             uid=real_uid if uid == "self" else real_uid + 1,
                                             gid=group if gid == "self" else group + 1)

    def test_every_row(self):
        table = rows()
        self.assertGreater(len(table), 40)
        ran = 0
        for row in table:
            with self.subTest(row=row[0]):
                if row[2].startswith("S") and not sticky_root_tmp():
                    continue
                ran += 1
                verdict = self.run_row(row)
                self.assertEqual(verdict.code, row[5], f"{row[0]}: {verdict.message}")
                self.assertEqual(verdict.message == "", verdict.ok, row[0])
        self.assertGreater(ran, 40)

    def test_the_table_names_each_code(self):
        codes = {r[5] for r in rows()}
        for code in ("ok", "relative", "system", "home", "link", "not-dir", "parent-owner", "parent-open", "owner",
                     "shared", "foreign", "unreadable", "contents-unreadable"):
            self.assertIn(code, codes)

    def test_the_real_path_keeps_the_last_name_and_resolves_the_parents(self):
        case = Case(self.base)
        self.addCleanup(case.cleanup)
        (case.t / "fast").mkdir()
        (case.h / "fast").symlink_to(case.t / "fast")
        verdict = state_root.state_root_verdict(case.h / "fast" / "states")
        self.assertEqual(verdict.path, str(case.t / "fast" / "states"))
        self.assertFalse(verdict.exists)

    def test_messages_name_the_folder_and_the_reason(self):
        case = Case(self.base)
        self.addCleanup(case.cleanup)
        (case.t / "shared").mkdir()
        os.chmod(case.t / "shared", 0o777)
        verdict = state_root.state_root_verdict(case.t / "shared" / "states")
        self.assertEqual(verdict.message, f"other users can write {case.t / 'shared'} (mode 0777, no sticky bit), so they "
                                          f"could move or replace {case.t / 'shared' / 'states'}.")
        self.assertEqual(state_root.state_root_verdict("/etc/x").message, "/etc/x is a system folder.")
        # A parent link to a folder that is not there (a second disk that is not mounted): refused, as the engine does.
        (case.h / "fast").symlink_to(case.t / "gone")
        verdict = state_root.state_root_verdict(case.h / "fast" / "states")
        self.assertEqual(verdict.message, f"{case.h / 'fast'} is a link to a folder that is not there "
                                          f"({case.t / 'gone'}).")


if __name__ == "__main__":
    unittest.main()
