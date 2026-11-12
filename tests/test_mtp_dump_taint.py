"""Leak guard: the native mtp_dump propagates the taint itself (bench/mtp_taint.h, used by bench/mtp_dump.hip).

CPU only. The stand-in tests/mtp_taint_tool.cpp calls the same header as mtp_dump. Its output must equal what
`python -m training.mtp_tune.taint tag-dump` writes, and the Python tagger must accept it (idempotent).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.leak_canary import CANARY, CANARY_BYTES  # noqa: E402
from training.mtp_tune import taint  # noqa: E402


def pub(rows):
    """A public list: the public line with the digest of the rows, then the rows."""
    return taint.public_line(rows.encode()) + "\n" + rows



@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    exe = tmp_path_factory.mktemp("bin") / "mtp_taint_tool"
    cxx = os.environ.get("CXX") or "g++"
    subprocess.run([cxx, "-std=c++20", "-O1", str(ROOT / "tests" / "mtp_taint_tool.cpp"), "-o", str(exe)], check=True)
    return exe


def run(tool, list_text, tmp_path, *extra, env=None):
    lst = tmp_path / "in.list"
    lst.write_text(list_text)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CRUCIBLE_")}
    clean.update(env or {})
    p = subprocess.run([str(tool), str(lst), str(out), *extra], capture_output=True, text=True, env=clean)
    return p, lst, out


BODY = "s1 5,6,7\ns2 8,9\n"


@pytest.mark.parametrize("header", ["#" + CANARY + "\n", "", "#nospace\n", "\n"])
def test_tainted_or_unknown_list_tags_every_output(tool, tmp_path, header):
    p, lst, out = run(tool, header + BODY, tmp_path)
    assert p.returncode == 0, p.stderr
    for name, n in (("s1", 3), ("s2", 2)):
        stk = (out / f"{name}.stk").read_bytes()
        assert stk.endswith(CANARY_BYTES) and len(stk) == n * 8 + len(CANARY_BYTES)
        first, rest = (out / f"{name}.ids").read_text().split("\n", 1)
        assert first == "#" + CANARY
        assert rest.strip() == ",".join(map(str, [5, 6, 7] if n == 3 else [8, 9]))
    assert not (out / "PROVENANCE.json").exists()
    assert taint.dir_tainted(str(out)) and not taint.dir_is_clean(str(out))


def test_native_output_equals_the_python_tagger_and_it_is_idempotent(tool, tmp_path):
    p, lst, out = run(tool, "#" + CANARY + "\n" + BODY, tmp_path)
    assert p.returncode == 0
    before = {f.name: f.read_bytes() for f in out.iterdir()}
    # the Python tagger accepts what is already tagged and changes nothing
    assert taint.tag_dump(str(lst), str(out)) == 0
    assert {f.name: f.read_bytes() for f in out.iterdir()} == before
    # and equals what the tagger writes to an untagged dump
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in out.iterdir():
        data = f.read_bytes()
        if f.suffix == ".stk":
            data = data[: -len(CANARY_BYTES)]
        elif f.suffix == ".ids":
            data = data.split(b"\n", 1)[1]
        (raw / f.name).write_bytes(data)
    assert taint.tag_dump(str(lst), str(raw)) == 4
    assert {f.name: f.read_bytes() for f in raw.iterdir()} == before


def test_public_list_gets_the_marker_the_tagger_writes(tool, tmp_path):
    p, lst, out = run(tool, pub(BODY), tmp_path)
    assert p.returncode == 0, p.stderr
    assert not (out / "s1.stk").read_bytes().endswith(CANARY_BYTES)
    assert (out / "s1.ids").read_text() == "5,6,7\n"
    assert taint.dir_is_clean(str(out))
    native = (out / "PROVENANCE.json").read_bytes()
    # the Python tagger gives the same bytes and changes nothing else
    assert taint.tag_dump(str(lst), str(out)) == 0
    assert (out / "PROVENANCE.json").read_bytes() == native
    twin = tmp_path / "twin"
    shutil.copytree(out, twin)
    (twin / "PROVENANCE.json").unlink()
    taint.mark_dir(str(twin), "public")
    assert (twin / "PROVENANCE.json").read_bytes() == native


def test_a_public_run_in_a_tainted_folder_stays_tainted(tool, tmp_path):
    p, lst, out = run(tool, "#" + CANARY + "\n" + BODY, tmp_path)
    assert p.returncode == 0
    (tmp_path / "in2.list").write_text(pub("s3 1,2\n"))
    q = subprocess.run([str(tool), str(tmp_path / "in2.list"), str(out)], capture_output=True)
    assert q.returncode == 1 and not taint.dir_is_clean(str(out))      # H8: the folder holds files of another list


def test_check_output_of_a_tainted_list_is_refused(tool, tmp_path):
    p, lst, out = run(tool, "#" + CANARY + "\n" + BODY, tmp_path, "--check", "1")
    assert p.returncode == 2 and not list(out.iterdir())
    sub = tmp_path / "x"
    sub.mkdir()
    p, lst, out = run(tool, BODY, sub, "--check", "1")
    assert p.returncode == 2 and not list(out.iterdir())


@pytest.mark.parametrize("name", ["CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR", "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY",
                                  "CRUCIBLE_ROUTE_DUMP", "CRUCIBLE_PROFILE", "CRUCIBLE_PF_OUT"])
def test_diagnostic_switch_stops_the_start_before_any_write(tool, tmp_path, name):
    p, lst, out = run(tool, "#" + CANARY + "\n" + BODY, tmp_path, env={name: str(tmp_path / "cap")})
    assert p.returncode == 2 and name in p.stderr and not list(out.iterdir())


def test_registry_switches_are_the_same_set_as_the_python_launcher():
    from training.mtp_tune.launch import diagnostic_switches
    text = (ROOT / "bench" / "mtp_taint.h").read_text()
    header = (ROOT / "engine" / "switches.h").read_text()
    registry = set(re.findall(r'\{"(CRUCIBLE_\w+)",\s*K::kDiagnostic', header))
    assert registry <= diagnostic_switches()
    assert "kDiagnostic" in text and "kSwitches" in text      # the C++ list is read from the registry, not copied


def test_mtp_dump_uses_the_header_before_it_touches_anything():
    src = (ROOT / "bench" / "mtp_dump.hip").read_text()
    main = src[src.index("int main("):]
    assert '#include "mtp_taint.h"' in src
    first_use = main.index("mtp_taint::DiagnosticSwitchInEnvironment")
    assert first_use < main.index("SetRuntimeMemoryDefaults") and first_use < main.index("std::ifstream")
    for call in ("ListProvenance", "CheckOutputAllowed", "WriteStk", "WriteIds", "FinishFolder"):
        assert "mtp_taint::" + call in main, call
    # no raw writer of .stk or .ids is left
    assert 'WriteAll(out + "/" + name + ".stk"' not in src and 'std::ofstream f(out + "/" + name + ".ids")' not in src


def test_the_source_holds_no_canary_literal():
    text = (ROOT / "bench" / "mtp_taint.h").read_text()
    assert CANARY not in text and CANARY.split("-")[-1] not in text


def test_h7_a_merged_list_is_tainted_in_the_native_tool_too(tool, tmp_path):
    merged = pub("a 1,2,3\n#" + CANARY + "\nb 7,8,9\n")
    p, lst, out = run(tool, merged, tmp_path)
    assert p.returncode == 0
    assert (out / "a.stk").read_bytes().endswith(CANARY_BYTES) and not (out / "PROVENANCE.json").exists()
    assert not taint.dir_is_clean(str(out)) and taint.list_public(str(lst)) is False


def test_h8_a_public_run_in_a_folder_with_foreign_files_writes_no_marker(tool, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "b.stk").write_bytes(b"\0" * 64)            # an earlier untagged dump of another list
    (out / "b.ids").write_text("7,8,9\n")
    p, lst, out = run(tool, pub("a 1,2,3\n"), tmp_path)
    assert p.returncode == 1 and not (out / "PROVENANCE.json").exists()
    assert not taint.dir_is_clean(str(out))
