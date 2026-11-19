"""E3: ./crucible status and ./crucible start show why the last server stopped, from .crucible/server.log. On Nov 14 the
engine died, the terminal showed nothing, and the user saw only a busy port. No GPU, no network."""
from crucible_cli_fakes import FakeHttp, FakeRunner, make_repo, wire
from crucible_cli import cli, logcause

DIED = ("Crucible is ready at http://127.0.0.1:8790/v1 (content special tokens: text)\n"
        "HIP error out of memory at engine.hip:646\n"
        "The engine stopped (exit code 1). The server exits with code 3 and frees port 8790. "
        "The engine lines above give the cause. Check the server log before you start it again.\n")
KILLED = ("Crucible is ready at http://127.0.0.1:8790/v1 (content special tokens: text)\n"
          "The engine stopped (killed by signal 9, SIGKILL; the kernel out-of-memory killer sends this signal). "
          "The server exits with code 3 and frees port 8790.\n")


def test_the_cause_comes_from_the_lines_after_the_last_ready_line(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(DIED + "Crucible is ready at http://127.0.0.1:8790/v1 (content special tokens: text)\n")
    assert logcause.last_stop(log) == []          # the last run started again after the death
    log.write_text(DIED)
    assert logcause.last_stop(log) == ["HIP error out of memory at engine.hip:646", DIED.splitlines()[2]]
    assert logcause.last_stop(tmp_path / "missing.log") == []


def configured(tmp_path, log_text, ready_after=10 ** 6):
    host = wire(tmp_path, runner=FakeRunner(), http=FakeHttp(ready_after=ready_after))
    repo = make_repo(tmp_path)
    (repo / ".crucible").mkdir()
    (repo / ".crucible/config.json").write_text("{}")
    (repo / ".crucible/server.log").write_text(log_text)
    return host, repo


def test_status_names_the_cause_of_the_last_stop(tmp_path):
    host, repo = configured(tmp_path, KILLED)
    cli.main(["status"], host=host, repo=repo)
    out = "\n".join(host.lines)
    assert "server: not running" in out
    assert "last stop" in out and "signal 9, SIGKILL" in out


def test_start_names_the_cause_before_it_starts_again(tmp_path):
    host, repo = configured(tmp_path, DIED, ready_after=0)
    from crucible_cli import settings, steps
    steps.Setup(host, settings.build(repo, host)).step_start([])
    out = "\n".join(host.lines)
    assert "The last server stopped" in out and "HIP error out of memory" in out
    assert out.index("The last server stopped") < out.index("Server process")
