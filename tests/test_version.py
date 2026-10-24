"""VERSION file, server/version.py, /v1/models and /v1/crucible/version."""
import json
import re
import subprocess
from pathlib import Path

from server import version as v
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, invoke

ROOT = Path(__file__).resolve().parents[1]


def test_version_file_is_plain_numbers_without_v():
    text = (ROOT / "VERSION").read_text()
    assert re.fullmatch(r"0\.\d+\.\d+\n?", text), text


def test_read_version_reads_the_file(tmp_path):
    (tmp_path / "VERSION").write_text("0.22.2\n")
    assert v.read_version(tmp_path) == "0.22.2"


def test_read_version_without_a_valid_file_is_unknown(tmp_path):
    assert v.read_version(tmp_path) == "unknown"
    (tmp_path / "VERSION").write_text("v0.22.2 beta\n")
    assert v.read_version(tmp_path) == "unknown"


def test_git_commit_of_a_checkout(tmp_path):
    env = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               PATH="/usr/bin:/bin", HOME=str(tmp_path))
    for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "x"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, env=env)
    sha = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, env=env,
                         capture_output=True, text=True).stdout.strip()
    assert sha.startswith(v.git_commit(tmp_path))
    assert 7 <= len(v.git_commit(tmp_path)) <= 12


def test_git_commit_is_none_without_a_checkout_of_its_own(tmp_path):
    # A source tarball inside another repository must not report the outer repository's commit.
    assert v.git_commit(tmp_path) is None


def test_models_list_reports_the_version():
    response = invoke(ScriptBackend([]), PiecesTokenizer([]), method="GET", path="/v1/models")
    body = json.loads(response.body)
    assert response.statuses == [200]
    assert body["crucible_version"] == v.read_version()
    assert body["data"][0]["crucible_version"] == v.read_version()
    assert body["data"][0]["id"] == "crucible"


def test_version_endpoint():
    response = invoke(ScriptBackend([]), PiecesTokenizer([]), method="GET", path="/v1/crucible/version")
    body = json.loads(response.body)
    assert response.statuses == [200]
    assert body["object"] == "crucible.version"
    assert body["version"] == v.read_version()
    assert body["commit"] is None or re.fullmatch(r"[0-9a-f]{7,12}", body["commit"])


def test_version_endpoint_checks_the_host_like_the_others():
    # Any Host passes by default. With a strict list (CRUCIBLE_ALLOWED_HOSTS) this route is checked too.
    from server.hosts import parse_allowed_hosts
    response = invoke(ScriptBackend([]), PiecesTokenizer([]), method="GET", path="/v1/crucible/version",
                      headers=[("Host", "evil.example")],
                      service_options=dict(allowed_hosts=parse_allowed_hosts("localhost,127.0.0.1")))
    assert response.statuses == [403]


def test_main_has_a_version_flag(capsys):
    import pytest
    from server.__main__ import main
    import sys
    argv, sys.argv = sys.argv, ["server", "--version"]
    try:
        with pytest.raises(SystemExit) as stop:
            main()
    finally:
        sys.argv = argv
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("crucible " + v.read_version())


def test_engine_version_header_matches_the_version_file():
    header = (ROOT / "engine/cli/version.h").read_text()
    default = re.search(r'#define ENGINE_BUILD_VERSION "([^"]+)"', header)
    assert default and default.group(1) == (ROOT / "VERSION").read_text().strip()


def test_the_engine_build_gets_no_version_flag():
    # A compiler flag reaches the device compile and the part cache keys (tools/devsplit.py).
    assert "ENGINE_BUILD_VERSION" not in (ROOT / "Makefile").read_text()
