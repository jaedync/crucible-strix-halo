"""VERSION file, server/version.py, /v1/models and /v1/crucible/version."""
import json
import re
import subprocess

import pytest
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


def test_the_version_files_and_the_installer_pin_name_one_release():
    # A release ships VERSION, the engine --version text, the installer pin and the Docker tag from one number.
    number = (ROOT / "VERSION").read_text().strip()
    header = (ROOT / "engine/cli/version.h").read_text()
    assert f'#define ENGINE_BUILD_VERSION "{number}"' in header
    assert f'PIN_TAG="v{number}"' in (ROOT / "install.sh").read_text()
    assert f"ARG CRUCIBLE_VERSION=v{number}" in (ROOT / "Dockerfile").read_text()
GIT_ENV = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               PATH="/usr/bin:/bin", GIT_CONFIG_NOSYSTEM="1")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
                          env=GIT_ENV | dict(HOME=str(root))).stdout.strip()


def checkout(tmp_path, version="0.22.2"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "VERSION").write_text(version + "\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "VERSION")
    git(tmp_path, "commit", "-q", "-m", "x")
    return tmp_path


def test_a_tree_without_git_is_the_release_of_its_version_file(tmp_path):
    # The release tarball and the Docker image (it removes .git) print their tag, as before.
    (tmp_path / "VERSION").write_text("0.22.2\n")
    assert v.build_label(tmp_path) == "v0.22.2"


@pytest.mark.parametrize("annotated", [False, True])
def test_a_checkout_of_the_release_tag_is_the_release(tmp_path, annotated):
    root = checkout(tmp_path)
    git(root, "tag", *(["-a", "-m", "v0.22.2"] if annotated else []), "v0.22.2")
    assert v.build_label(root) == "v0.22.2"


def test_a_checkout_that_is_not_the_release_tag_is_a_development_build(tmp_path):
    root = checkout(tmp_path)
    assert v.build_label(root) == f"development build {v.git_commit(root)}, based on v0.22.2"
    git(root, "tag", "v0.22.2")
    git(root, "commit", "-q", "--allow-empty", "-m", "after the release")
    assert v.build_label(root) == f"development build {v.git_commit(root)}, based on v0.22.2"


def test_a_tag_of_another_version_does_not_make_a_release(tmp_path):
    root = checkout(tmp_path)
    git(root, "tag", "v0.22.1")
    assert v.build_label(root).startswith("development build ")


def test_the_startup_line_of_a_release_and_of_a_development_build(tmp_path):
    (tmp_path / "VERSION").write_text("0.22.2\n")
    assert v.startup_line(tmp_path) == "Crucible v0.22.2"
    root = checkout(tmp_path / "c")
    git(root, "tag", "v0.22.2")
    assert v.startup_line(root) == f"Crucible v0.22.2 ({v.git_commit(root)})"
    git(root, "commit", "-q", "--allow-empty", "-m", "after the release")
    assert v.startup_line(root) == f"Crucible development build {v.git_commit(root)}, based on v0.22.2"


def test_the_version_file_is_not_older_than_the_newest_release_tag():
    # Drift guard: a release bumps VERSION in the public tree. Trunk must take the same bump, or a development
    # build says that it is based on an old release. Tags that are not in this checkout cannot be checked.
    if v.git_commit(ROOT) is None:
        pytest.skip("not a git checkout")
    tags = subprocess.run(["git", "-C", str(ROOT), "tag", "--list", "v*"], capture_output=True,
                          text=True).stdout.split()
    releases = [tuple(map(int, t[1:].split("."))) for t in tags if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    if not releases:
        pytest.skip("no release tags in this checkout")
    if subprocess.run(["git", "-C", str(ROOT), "tag", "--points-at", "HEAD", "--list", "v*"],
                      capture_output=True, text=True).stdout.strip():
        pytest.skip("a release checkout")
    newest = max(releases)
    assert tuple(map(int, v.read_version().split("."))) >= newest, \
        f"VERSION is {v.read_version()}, the newest release is v{'.'.join(map(str, newest))}"
