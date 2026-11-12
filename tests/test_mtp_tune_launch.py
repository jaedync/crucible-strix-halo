"""WP part 1: children that touch private data run with an allow-listed environment, umask 077, no core file, and a
private work folder; error messages hide paths and file content. Synthetic data only."""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.mtp_tune import launch, taint  # noqa: E402

FAKE = """#!/bin/bash
# A stand-in for mtp_dump: it writes to the capture switches of the engine when they are set, like the engine does.
out="$1"
env | sort > "$out/env.txt"
ulimit -c > "$out/core.txt"
pwd > "$out/cwd.txt"
for var in CRUCIBLE_QSA_CAPTURE_DIR CRUCIBLE_ROUTE_DUMP; do
  target="${!var:-}"
  if [ -n "$target" ]; then echo leaked > "$target/leaked-by-$var"; fi
done
echo ok > "$out/s0.stk"
"""


def fake(tmp_path):
    path = tmp_path / "fake_dump"
    path.write_text(FAKE)
    path.chmod(0o755)
    return path


def git_dir(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    return repo


def test_an_inherited_capture_switch_does_not_reach_the_child(tmp_path, monkeypatch):
    repo = git_dir(tmp_path)
    monkeypatch.setenv("CRUCIBLE_QSA_CAPTURE_DIR", str(repo))
    monkeypatch.setenv("CRUCIBLE_ROUTE_DUMP", str(repo / "r.log"))
    monkeypatch.setenv("SOME_API_TOKEN", "x")
    out, work = tmp_path / "out", tmp_path / "work"
    rc = launch.run_private([str(fake(tmp_path)), str(out)], out_dir=out, work_dir=work)
    assert rc == 0
    assert sorted(p.name for p in repo.iterdir()) == [".git"]            # nothing written into the work tree
    env = (out / "env.txt").read_text()
    assert "CRUCIBLE_" not in env and "SOME_API_TOKEN" not in env
    assert "PATH=" in env and (out / "core.txt").read_text().strip() == "0"
    assert (out / "cwd.txt").read_text().strip() == str(work)


def test_the_files_of_the_child_are_private_and_the_work_folder_too(tmp_path):
    out, work = tmp_path / "out", tmp_path / "work"
    launch.run_private([str(fake(tmp_path)), str(out)], out_dir=out, work_dir=work)
    assert stat.S_IMODE((out / "s0.stk").stat().st_mode) == 0o600
    assert stat.S_IMODE(work.stat().st_mode) == 0o700


def test_an_output_or_work_folder_inside_a_git_tree_is_refused(tmp_path):
    repo = git_dir(tmp_path)
    with pytest.raises(ValueError, match="outside every git work tree"):
        launch.run_private([str(fake(tmp_path)), "x"], out_dir=repo / "o", work_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="outside every git work tree"):
        launch.run_private([str(fake(tmp_path)), "x"], out_dir=tmp_path / "o", work_dir=repo / "w")


def test_extra_variables_must_be_named_and_never_a_capture_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_SOMETHING", "1")
    monkeypatch.setenv("CRUCIBLE_ROUTE_DUMP", "/x")
    assert "CRUCIBLE_SOMETHING" not in launch.private_env()
    assert launch.private_env(extra=("CRUCIBLE_SOMETHING",))["CRUCIBLE_SOMETHING"] == "1"
    with pytest.raises(ValueError, match="diagnostic"):
        launch.private_env(extra=("CRUCIBLE_ROUTE_DUMP",))


def test_the_dump_scripts_call_the_launcher_not_the_binary_directly():
    for name in ("dump.sh", "postgen.sh", "stage1.sh"):
        text = (ROOT / "training/mtp_tune" / name).read_text()
        assert "mtp_tune.launch run" in text, name
        for line in text.splitlines():
            if "build/bench/mtp_dump" in line:
                assert "launch run" in line or line.lstrip().startswith("#"), (name, line)


def test_redact_hides_directories_and_keeps_a_short_name():
    text = taint.redact("cannot read /home/someone/private/data/x.ids: bad")
    assert "someone" not in text and "private" not in text and "x.ids" in text
    assert taint.redact("no path here") == "no path here"


def test_a_malformed_file_error_has_no_content_and_no_path(tmp_path):
    bad = tmp_path / "secretdir" / "t.ids"
    bad.parent.mkdir()
    bad.write_text("hello private words,3\n")
    with pytest.raises(ValueError) as caught:
        taint.read_ids(bad)
    message = str(caught.value)
    assert "hello" not in message and "private" not in message and "secretdir" not in message
    lst = tmp_path / "secretdir" / "t.list"
    lst.write_text("name 1,two,3\n")
    with pytest.raises(ValueError) as caught:
        list(taint.read_list(lst))
    assert "two" not in str(caught.value) and "secretdir" not in str(caught.value)


def test_the_training_code_redacts_paths_in_messages():
    for name, needle in (("data.py", "taint.redact"), ("train.py", "taint.redact"), ("calib.py", "taint.redact")):
        assert needle in (ROOT / "training/mtp_tune" / name).read_text(), name
