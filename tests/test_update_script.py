"""tools/update.sh against a temporary git repository with fake release tags and a fake make."""
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")

# The fake make writes the binary from VERSION, or fails, as the `behavior` file of the checked-out release says.
# KILL_MODE stops the update script itself (SIGKILL, as a stopped unit or a memory kill does) at one point of the build.
FAKE_MAKE = """#!/bin/sh
case "$(cat behavior)" in
  build-fail) rm -f build/engine/crucible; echo "fake compiler error" >&2; exit 1 ;;
esac
mkdir -p build/engine
case "${KILL_MODE:-}" in
  before-build) kill -9 $PPID; exit 137 ;;
  partial-final) echo "half of a binary" > build/engine/crucible; kill -9 $PPID; exit 137 ;;
  no-engine) rm -f build/engine/crucible; kill -9 $PPID; exit 137 ;;
  partial-tmp) echo "half of a binary" > build/engine/crucible.tmpa851efc; kill -9 $PPID; exit 137 ;;
esac
mkdir -p build/engine
echo "binary $(cat VERSION)" > build/engine/crucible
"""
FAKE_SELFTEST = """import os, sys
behavior = open("behavior").read().strip()
if sys.argv[1] == "cpu":
    print("cpu self-test", sys.argv[2])
    sys.exit(1 if behavior == "test-fail" else 0)
open("build/gpu-called", "w").write(sys.argv[2])
sys.exit({"gpu-fail": 1, "gpu-skip": 3}.get(behavior, 0))
"""
FAKE_SYSTEMCTL = """#!/bin/sh
echo "$@" >> "$SYSTEMCTL_LOG"
"""
# KILL_CP stops the update script when it copies the engine: after the marker, before the copy lands.
FAKE_CP = """#!/bin/sh
[ -n "${KILL_CP:-}" ] && { kill -9 $PPID; exit 137; }
exec /bin/cp "$@"
"""


def git(cwd, *args):
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=os.environ | GIT_ENV)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def release(origin, tag, behavior="ok", extra_script=""):
    """One commit in the origin work tree with a VERSION and a tag."""
    (origin / "VERSION").write_text(tag[1:] + "\n")
    (origin / "behavior").write_text(behavior + "\n")
    script = (ROOT / "tools/update.sh").read_text()
    # A newer release may carry a different update.sh: the running script must survive its own replacement.
    (origin / "tools/update.sh").write_text(script + extra_script)
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", tag)
    git(origin, "tag", tag)


class Setup:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.origin = tmp_path / "origin"
        (self.origin / "tools/update_selftest").mkdir(parents=True)
        git(self.origin, "init", "-q", "-b", "main")
        (self.origin / ".gitignore").write_text("build/\n")
        (self.origin / "tools/update_selftest/selftest.py").write_text(FAKE_SELFTEST)
        self.make = tmp_path / "fake-make"
        self.make.write_text(FAKE_MAKE)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        (self.bin / "systemctl").write_text(FAKE_SYSTEMCTL)
        (self.bin / "cp").write_text(FAKE_CP)
        for tool in (self.make, self.bin / "systemctl", self.bin / "cp"):
            tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        self.systemctl_log = tmp_path / "systemctl.log"
        self.work = tmp_path / "work"

    def clone(self, tag):
        """The user's checkout at `tag`, with that release built."""
        git(self.tmp, "clone", "-q", str(self.origin), str(self.work))
        git(self.work, "checkout", "-q", "--detach", tag)
        (self.work / "build/engine").mkdir(parents=True)
        (self.work / "build/engine/crucible").write_text("binary " + tag[1:] + "\n")

    def run(self, *args, **env):
        environment = os.environ | GIT_ENV | dict(CRUCIBLE_MAKE=str(self.make), PYTHON="python3",
                                                  PATH=f"{self.bin}:{os.environ['PATH']}",
                                                  SYSTEMCTL_LOG=str(self.systemctl_log)) | env
        return subprocess.run(["bash", str(self.work / "tools/update.sh"), *args], cwd=self.tmp, env=environment,
                              capture_output=True, text=True, timeout=60)

    def head(self):
        return git(self.work, "rev-parse", "HEAD")

    def tag_of_head(self):
        return git(self.work, "describe", "--tags", "--exact-match")

    def binary(self):
        path = self.work / "build/engine/crucible"
        return path.read_text().strip() if path.exists() else None


@pytest.fixture
def repo(tmp_path):
    setup = Setup(tmp_path)
    release(setup.origin, "v0.22.0")
    release(setup.origin, "v0.22.1")
    return setup


def test_updates_to_the_newest_tag_and_keeps_the_previous_binary(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert repo.tag_of_head() == "v0.22.2"
    assert repo.binary() == "binary 0.22.2"
    assert (repo.work / "build/engine/crucible.prev").read_text().strip() == "binary 0.22.1"
    assert (repo.work / "build/update-prev-tag").read_text().strip() == "v0.22.1"
    assert "cpu self-test 0.22.2" in done.stdout
    assert "make rollback" in done.stdout


def test_newest_means_numerically_newest(repo):
    for tag in ("v0.22.2", "v0.22.9", "v0.22.10"):
        release(repo.origin, tag)
    repo.clone("v0.22.0")
    assert repo.run().returncode == 0
    assert repo.tag_of_head() == "v0.22.10"


def test_up_to_date_changes_nothing(repo):
    repo.clone("v0.22.1")
    done = repo.run()
    assert done.returncode == 0
    assert "up to date" in done.stdout
    assert not (repo.work / "build/update-prev-tag").exists()


def test_a_checkout_between_releases_on_the_release_line_may_update(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.0")
    git(repo.work, "checkout", "-q", "--detach", "v0.22.1~0")
    assert repo.run().returncode == 0
    assert repo.tag_of_head() == "v0.22.2"


def test_local_changes_are_refused_and_kept(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    (repo.work / "VERSION").write_text("my edit\n")
    before = repo.head()
    done = repo.run()
    assert done.returncode == 1
    assert "local changes" in done.stderr
    assert (repo.work / "VERSION").read_text() == "my edit\n"
    assert repo.head() == before and repo.binary() == "binary 0.22.1"


def test_staged_changes_are_refused(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    (repo.work / "VERSION").write_text("staged\n")
    git(repo.work, "add", "VERSION")
    assert repo.run().returncode == 1
    assert (repo.work / "VERSION").read_text() == "staged\n"


def test_a_head_off_the_release_line_is_refused(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    (repo.work / "mine.txt").write_text("my work\n")
    git(repo.work, "add", "mine.txt")
    git(repo.work, "commit", "-q", "-m", "mine")
    before = repo.head()
    done = repo.run()
    assert done.returncode == 1
    assert "release line" in done.stderr
    assert repo.head() == before and (repo.work / "mine.txt").exists()


def test_ahead_of_the_newest_release_is_not_an_error(repo):
    repo.clone("v0.22.1")
    (repo.work / "mine.txt").write_text("x\n")
    git(repo.work, "add", "mine.txt")
    git(repo.work, "commit", "-q", "-m", "after the release")
    done = repo.run()
    assert done.returncode == 0 and "ahead" in done.stdout


def test_a_failing_build_rolls_back_the_source_and_the_binary(repo):
    release(repo.origin, "v0.22.2", behavior="build-fail")
    repo.clone("v0.22.1")
    done = repo.run()
    assert done.returncode == 1
    assert "build of v0.22.2 failed" in done.stderr
    assert repo.tag_of_head() == "v0.22.1"
    assert repo.binary() == "binary 0.22.1"
    assert not (repo.work / "build/update-prev-tag").exists()
    assert "restoring v0.22.1" in done.stdout


def test_a_failing_self_test_rolls_back(repo):
    release(repo.origin, "v0.22.2", behavior="test-fail")
    repo.clone("v0.22.1")
    done = repo.run()
    assert done.returncode == 1
    assert "did not pass the self-test" in done.stderr
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"


def test_a_failing_build_without_an_earlier_binary_leaves_none(repo):
    release(repo.origin, "v0.22.2", behavior="build-fail")
    repo.clone("v0.22.1")
    (repo.work / "build/engine/crucible").unlink()
    assert repo.run().returncode == 1
    assert repo.binary() is None


def test_the_gpu_part_runs_only_with_a_model_path(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    done = repo.run()
    assert "GPU self-test skipped" in done.stdout
    assert not (repo.work / "build/gpu-called").exists()
    repo.run("--rollback")
    (repo.work / "build/update-hold").unlink()  # the rollback holds v0.22.2
    done = repo.run(CKPT="/models/x")
    assert done.returncode == 0, done.stderr
    assert (repo.work / "build/gpu-called").read_text() == "/models/x"


def test_a_failing_gpu_self_test_rolls_back(repo):
    release(repo.origin, "v0.22.2", behavior="gpu-fail")
    repo.clone("v0.22.1")
    done = repo.run(CKPT="/models/x")
    assert done.returncode == 1
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"


def test_a_pack_without_expected_ids_is_a_skip_not_a_failure(repo):
    release(repo.origin, "v0.22.2", behavior="gpu-skip")
    repo.clone("v0.22.1")
    done = repo.run(CKPT="/models/x")
    assert done.returncode == 0
    assert repo.tag_of_head() == "v0.22.2"


def test_rollback_restores_the_binary_and_the_tag(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 0
    done = repo.run("--rollback")
    assert done.returncode == 0, done.stderr
    assert repo.tag_of_head() == "v0.22.1"
    assert repo.binary() == "binary 0.22.1"
    assert not (repo.work / "build/update-prev-tag").exists()
    assert not (repo.work / "build/engine/crucible.prev").exists()


def test_rollback_holds_the_release_it_left_until_a_newer_one_exists(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 0
    assert repo.run("--rollback").returncode == 0
    assert (repo.work / "build/update-hold").read_text().strip() == "v0.22.2"
    done = repo.run()  # the daily timer
    assert done.returncode == 0 and "on hold" in done.stdout, done.stdout
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert repo.run().returncode == 0 and repo.tag_of_head() == "v0.22.1"  # and again the next day
    release(repo.origin, "v0.22.3")
    assert repo.run().returncode == 0
    assert repo.tag_of_head() == "v0.22.3"
    assert not (repo.work / "build/update-hold").exists()  # a newer release lifts the hold


def test_deleting_the_hold_file_installs_the_held_release(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 0 and repo.run("--rollback").returncode == 0
    (repo.work / "build/update-hold").unlink()
    assert repo.run().returncode == 0
    assert repo.tag_of_head() == "v0.22.2"


def test_a_bad_hold_file_is_ignored(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    (repo.work / "build").mkdir(exist_ok=True)
    (repo.work / "build/update-hold").write_text("not a tag; rm -rf /\n")
    assert repo.run().returncode == 0
    assert repo.tag_of_head() == "v0.22.2"
    assert not (repo.work / "build/update-hold").exists()


def test_a_failed_rollback_writes_no_hold(repo):
    repo.clone("v0.22.1")
    assert repo.run("--rollback").returncode == 1
    assert not (repo.work / "build/update-hold").exists()


def test_a_clone_under_a_symlinked_folder_updates(repo, tmp_path):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    link = tmp_path / "link"
    link.symlink_to(repo.work)
    environment = os.environ | GIT_ENV | dict(CRUCIBLE_MAKE=str(repo.make), PYTHON="python3",
                                              PATH=f"{repo.bin}:{os.environ['PATH']}")
    done = subprocess.run(["bash", str(link / "tools/update.sh")], cwd=tmp_path, env=environment, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert repo.tag_of_head() == "v0.22.2"


def test_rollback_without_an_update_says_so(repo):
    repo.clone("v0.22.1")
    done = repo.run("--rollback")
    assert done.returncode == 1 and "no earlier update" in done.stderr


def test_rollback_keeps_local_changes(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 0
    (repo.work / "VERSION").write_text("my edit\n")
    done = repo.run("--rollback")
    assert done.returncode == 1 and "local changes" in done.stderr
    assert (repo.work / "VERSION").read_text() == "my edit\n"
    assert repo.binary() == "binary 0.22.2"


def test_a_new_update_script_in_the_release_does_not_break_the_running_one(repo):
    release(repo.origin, "v0.22.2", extra_script="\necho 'a line that the old script never reads' >&2\nfalse\n")
    repo.clone("v0.22.1")
    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert "never reads" not in done.stderr


def test_unreachable_remote_changes_nothing(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    done = repo.run(CRUCIBLE_UPDATE_REMOTE="nowhere")
    assert done.returncode == 1 and "Could not fetch" in done.stderr
    assert repo.tag_of_head() == "v0.22.1"


def test_restart_unit_only_after_a_real_update(repo):
    repo.clone("v0.22.1")
    assert repo.run(CRUCIBLE_RESTART_UNIT="crucible-server.service").returncode == 0
    assert not repo.systemctl_log.exists()  # up to date: no restart
    release(repo.origin, "v0.22.2")
    assert repo.run(CRUCIBLE_RESTART_UNIT="crucible-server.service").returncode == 0
    assert repo.systemctl_log.read_text().strip() == "--user try-restart -- crucible-server.service"


def test_no_restart_after_a_failed_update(repo):
    release(repo.origin, "v0.22.2", behavior="build-fail")
    repo.clone("v0.22.1")
    assert repo.run(CRUCIBLE_RESTART_UNIT="crucible-server.service").returncode == 1
    assert not repo.systemctl_log.exists()


@pytest.mark.parametrize("unit", ["a b.service", "x;rm.service", "no-suffix", "--now.timer"])
def test_a_bad_unit_name_is_not_passed_to_systemctl(repo, unit):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    done = repo.run(CRUCIBLE_RESTART_UNIT=unit)
    assert done.returncode == 0 and not repo.systemctl_log.exists()
    assert "No restart" in done.stderr


def test_unknown_argument_prints_usage(repo):
    repo.clone("v0.22.1")
    done = repo.run("--wat")
    assert done.returncode == 2 and "usage" in done.stderr


def test_makefile_targets_call_the_script():
    text = (ROOT / "Makefile").read_text()
    assert "update:" in text and "tools/update.sh" in text
    assert "rollback:" in text and "--rollback" in text
    # `bash` first: an export or a copy that lost the mode 755 must still update.
    assert "bash tools/update.sh\n" in text and "bash tools/update.sh --rollback" in text
    assert os.access(ROOT / "tools/update.sh", os.X_OK)


# ---- tools/update_selftest/selftest.py ----


def load_selftest():
    spec = importlib.util.spec_from_file_location("selftest", ROOT / "tools/update_selftest/selftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_engine(tmp_path, version="0.22.2", ids="5 6 7"):
    engine = tmp_path / "crucible"
    engine.write_text(f"""#!/bin/sh
if [ "$1" = "--version" ]; then echo "crucible {version}"; exit 0; fi
echo "ids: {ids}"
echo "prefill 4 tok"
""")
    engine.chmod(0o755)
    return engine


def selftest_root(tmp_path, version="0.22.2"):
    root = tmp_path / "root"
    (root / "engine/cli").mkdir(parents=True)
    (root / "VERSION").write_text(version + "\n")
    (root / "engine/cli/version.h").write_text(f'#define ENGINE_BUILD_VERSION "{version}"\n')
    return root


def test_selftest_cpu_checks_version_and_binary(tmp_path):
    selftest = load_selftest()
    root = selftest_root(tmp_path)
    selftest.ROOT = ROOT  # the server modules come from this tree
    assert selftest.cpu("0.22.2", root, fake_engine(tmp_path)) == 0
    assert selftest.cpu("0.22.3", root, fake_engine(tmp_path)) == 1  # the tag and the file disagree
    assert selftest.cpu("0.22.2", root, fake_engine(tmp_path, version="0.22.1")) == 1  # an old binary


def test_selftest_gpu_compares_tokens_and_skips_unknown_packs(tmp_path, monkeypatch, capsys):
    selftest = load_selftest()
    monkeypatch.setattr(selftest, "EXPECTED_DIR", tmp_path)
    ckpt = tmp_path / "Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
    ckpt.mkdir()
    engine = fake_engine(tmp_path)
    assert selftest.gpu(ckpt, engine) == selftest.SKIPPED
    assert "no expected token IDs for the pack '3.05bpw_h5_ng5'" in capsys.readouterr().out
    (tmp_path / "3.05bpw_h5_ng5.json").write_text(json.dumps(dict(prompt_ids=[1], n=3, expected=[5, 6, 7])))
    assert selftest.gpu(ckpt, engine) == 0
    (tmp_path / "3.05bpw_h5_ng5.json").write_text(json.dumps(dict(prompt_ids=[1], n=3, expected=[5, 6, 8])))
    assert selftest.gpu(ckpt, engine) == 1
    assert "differ" in capsys.readouterr().out


def test_selftest_gpu_reports_an_engine_failure_and_a_bad_file(tmp_path, monkeypatch):
    selftest = load_selftest()
    monkeypatch.setattr(selftest, "EXPECTED_DIR", tmp_path)
    ckpt = tmp_path / "pack"
    ckpt.mkdir()
    (tmp_path / "pack.json").write_text("{not json")
    assert selftest.gpu(ckpt, fake_engine(tmp_path)) == 1
    (tmp_path / "pack.json").write_text(json.dumps(dict(prompt_ids=[1], n=3, expected=[5, 6, 7])))
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\nexit 9\n")
    broken.chmod(0o755)
    assert selftest.gpu(ckpt, broken) == 1


def test_selftest_pack_name_cannot_leave_the_directory(tmp_path):
    selftest = load_selftest()
    for name in (".hidden", "a b", "x;y"):
        folder = tmp_path / name
        folder.mkdir()
        assert selftest.expected_path(folder) is None


def test_selftest_record_round_trips(tmp_path, monkeypatch):
    selftest = load_selftest()
    monkeypatch.setattr(selftest, "EXPECTED_DIR", tmp_path)
    ckpt = tmp_path / "Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6"
    ckpt.mkdir()
    engine = fake_engine(tmp_path)
    assert selftest.record(ckpt, engine) == 0
    assert json.loads((tmp_path / "4.05bpw_h6_ng6.json").read_text())["expected"] == [5, 6, 7]
    assert selftest.gpu(ckpt, engine) == 0


# ---- deploy/user-systemd: opt-in templates ----

UNITS = ROOT / "deploy/user-systemd"


def test_units_are_not_enabled_by_themselves():
    service = (UNITS / "crucible-update.service").read_text()
    assert "[Install]" not in service  # a oneshot started by its timer, never enabled alone
    timer = (UNITS / "crucible-update.timer").read_text()
    assert "WantedBy=timers.target" in timer and "OnCalendar=" in timer and "Persistent=true" in timer


def test_the_update_service_runs_make_update_in_the_checkout_and_names_the_restart_unit():
    service = (UNITS / "crucible-update.service").read_text()
    assert "ExecStart=/usr/bin/make update" in service
    assert "WorkingDirectory=@CHECKOUT@" in service
    assert "CRUCIBLE_RESTART_UNIT=crucible-server.service" in service
    assert "CKPT" not in service  # the timer never loads the model


def test_the_update_service_cannot_take_the_server_down_with_it():
    service = (UNITS / "crucible-update.service").read_text()
    # The build runs next to a loaded model in shared memory: the kernel kills the build first, and a cap stops it
    # before the system runs out.
    assert "OOMScoreAdjust=1000" in service
    assert re.search(r"^MemoryMax=\d+G$", service, re.M)
    # With the default policy (stop), systemd ends the whole unit when the kernel kills the build, and the update
    # script never restores the previous release. `continue` lets the script see the failed build.
    assert re.search(r"^OOMPolicy=continue$", service, re.M)


def test_no_repository_file_installs_or_enables_the_units():
    # Nothing but the README text may name `systemctl --user enable`.
    for path in [ROOT / "Makefile", ROOT / "tools/update.sh", *ROOT.glob("tools/*.sh")]:
        text = path.read_text()
        assert "enable" not in text.replace("disable", ""), path


def test_the_server_unit_template_names_its_placeholders():
    text = (UNITS / "crucible-server.service").read_text()
    assert "@CHECKOUT@" in text and "@MODEL@" in text and "-m server" in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze is not installed")
def test_units_parse_with_systemd_analyze(tmp_path):
    for name in ("crucible-update.service", "crucible-update.timer"):
        (tmp_path / name).write_text((UNITS / name).read_text().replace("@CHECKOUT@", str(tmp_path)))
    done = subprocess.run(["systemd-analyze", "--user", "verify", *map(str, tmp_path.glob("crucible-update.*"))],
                          capture_output=True, text=True)
    assert "Unknown" not in done.stderr and "Invalid" not in done.stderr, done.stderr


# --- an update that a kill stopped half way: the next update or rollback finishes the restore first ---

OFFLINE = dict(CRUCIBLE_UPDATE_REMOTE="/nonexistent-remote")  # the fetch fails: only the restore at the start runs


def stopped_update(repo, mode="before-build", **env):
    """v0.22.1 is built; v0.22.2 is out; the update is killed. Returns the work folder state checks use."""
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    done = repo.run(KILL_MODE=mode, **env)
    assert done.returncode == -9, done.stderr
    assert repo.tag_of_head() == "v0.22.2" or mode == "cp"
    return done


def test_the_marker_is_gone_after_a_good_update_and_after_a_failed_build(repo):
    release(repo.origin, "v0.22.2", "build-fail")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 1
    assert not (repo.work / "build/update-running").exists()
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    release(repo.origin, "v0.22.3")
    assert repo.run().returncode == 0
    assert not (repo.work / "build/update-running").exists()


def test_a_kill_after_the_checkout_and_before_the_build_is_healed_by_the_next_run(repo):
    stopped_update(repo)
    marker = (repo.work / "build/update-running").read_text()
    assert marker == "v0.22.1\n1\n"
    assert repo.binary() == "binary 0.22.1"  # the checkout is new, the binary is the old one: do not start a server now
    healed = repo.run(**OFFLINE)
    assert "stopped before it finished" in healed.stdout
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert not (repo.work / "build/update-running").exists()
    # A normal run after it updates as usual.
    done = repo.run()
    assert done.returncode == 0, done.stderr
    assert repo.tag_of_head() == "v0.22.2" and repo.binary() == "binary 0.22.2"
    assert (repo.work / "build/engine/crucible.prev").read_text().strip() == "binary 0.22.1"
    assert not (repo.work / "build/update-running").exists()


def test_a_normal_update_heals_and_goes_on_in_the_same_run(repo):
    stopped_update(repo)
    done = repo.run()
    assert "stopped before it finished" in done.stdout and done.returncode == 0
    assert repo.tag_of_head() == "v0.22.2" and repo.binary() == "binary 0.22.2"


def test_a_kill_during_the_build_leaves_a_binary_that_the_next_run_replaces(repo):
    # The worst case: a build that writes the final path. The linker (lld) writes a temporary file and renames it, so
    # this should not happen; the restore covers it if it does.
    stopped_update(repo, "partial-final")
    assert repo.binary() == "half of a binary"
    repo.run(**OFFLINE)
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert not (repo.work / "build/update-running").exists()
    assert not (repo.work / "build/engine/crucible.prev").exists()
    done = repo.run()
    assert done.returncode == 0 and repo.binary() == "binary 0.22.2"


def test_a_kill_during_the_link_leaves_the_engine_whole_and_the_restore_removes_the_linker_temporary_file(repo):
    stopped_update(repo, "partial-tmp")
    assert repo.binary() == "binary 0.22.1"
    assert (repo.work / "build/engine/crucible.tmpa851efc").exists()
    repo.run(**OFFLINE)
    assert not list((repo.work / "build/engine").glob("crucible.tmp*"))
    assert repo.binary() == "binary 0.22.1" and repo.tag_of_head() == "v0.22.1"
    assert repo.run().returncode == 0 and repo.binary() == "binary 0.22.2"


def test_each_kill_leaves_a_state_that_the_next_run_heals_so_a_repeated_kill_cannot_loop(repo):
    stopped_update(repo, "partial-final")
    for _ in range(3):
        again = repo.run(KILL_MODE="partial-final")
        assert again.returncode == -9
        assert "stopped before it finished" in again.stdout
    repo.run(**OFFLINE)
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert repo.run().returncode == 0 and repo.binary() == "binary 0.22.2"


def test_a_kill_after_the_marker_and_before_the_copy_keeps_the_engine_and_the_old_copy_is_not_used(repo):
    # An update that passed leaves crucible.prev (release 0.22.0). The next update must not move that copy over the
    # engine of 0.22.1 when it is killed before its own copy lands.
    repo.clone("v0.22.0")
    assert repo.run().returncode == 0 and repo.tag_of_head() == "v0.22.1"
    assert (repo.work / "build/engine/crucible.prev").read_text().strip() == "binary 0.22.0"
    release(repo.origin, "v0.22.2")
    assert repo.run(KILL_CP="1").returncode == -9
    assert repo.tag_of_head() == "v0.22.1"  # killed before the checkout
    assert (repo.work / "build/update-running").exists()
    repo.run(**OFFLINE)
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert repo.run().returncode == 0 and repo.binary() == "binary 0.22.2"


def test_a_rollback_after_a_kill_heals_first_and_then_reports_that_nothing_is_left_to_undo(repo):
    stopped_update(repo, "partial-final")
    done = repo.run("--rollback")
    assert "stopped before it finished" in done.stdout
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert "no earlier update" in done.stderr and done.returncode == 1


def test_the_restore_can_run_again_after_a_kill_inside_it(repo):
    repo.clone("v0.22.1")
    work = repo.work / "build"
    # The state after the engine move and before the marker went: the copy is gone, the engine is the previous one.
    (work / "update-running").write_text("v0.22.1\n1\n")
    repo.run(**OFFLINE)
    assert repo.binary() == "binary 0.22.1" and not (work / "update-running").exists()
    # The same state of an update that began with no binary: the engine goes.
    (work / "update-running").write_text("v0.22.1\n0\n")
    repo.run(**OFFLINE)
    assert repo.binary() is None


def test_a_rollback_that_a_kill_stopped_is_finished_and_the_release_stays_on_hold(repo):
    release(repo.origin, "v0.22.2")
    repo.clone("v0.22.1")
    assert repo.run().returncode == 0
    # The kill came after the hold and the marker were written, before the restore.
    (repo.work / "build/update-hold").write_text("v0.22.2\n")
    (repo.work / "build/update-running").write_text("v0.22.1\n1\n")
    healed = repo.run()
    assert "stopped before it finished" in healed.stdout
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert "on hold" in healed.stdout  # the hold stops the next update from installing v0.22.2 again


def test_a_marker_that_is_not_valid_is_removed(repo):
    repo.clone("v0.22.1")
    (repo.work / "build/update-running").write_text("not a ref; rm -rf /\n1\n")
    done = repo.run()
    assert "is not valid" in done.stderr
    assert not (repo.work / "build/update-running").exists()
    assert repo.binary() == "binary 0.22.1" and repo.tag_of_head() == "v0.22.1"


def test_local_changes_stop_the_heal_and_keep_the_marker(repo):
    stopped_update(repo)
    (repo.work / "VERSION").write_text("my edit\n")
    done = repo.run(**OFFLINE)
    assert done.returncode == 1 and "local changes" in done.stderr
    assert (repo.work / "build/update-running").exists()
    assert (repo.work / "VERSION").read_text() == "my edit\n"



def test_a_kill_after_the_linker_removed_the_old_binary_restores_the_previous_one(repo):
    stopped_update(repo, "no-engine")
    assert repo.binary() is None
    repo.run(**OFFLINE)
    assert repo.tag_of_head() == "v0.22.1" and repo.binary() == "binary 0.22.1"
    assert repo.run().returncode == 0 and repo.binary() == "binary 0.22.2"
