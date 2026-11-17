from pathlib import Path

from crucible_cli_fakes import wire as make_machine, make_repo
from crucible_cli import cli, settings, steps


def test_root_stop_is_refused_without_signals(tmp_path):
    host = make_machine(tmp_path)
    host.uid = 0
    repo = make_repo(tmp_path)
    lines = []
    host.out = lines.append
    assert cli.main(['stop'], host=host, repo=repo) == 2
    assert lines == [f'Run {repo}/crucible without sudo. The server runs as your user.']
    assert host.runner.terminated == []


def test_unrecorded_server_is_named_and_not_stopped(tmp_path):
    host = make_machine(tmp_path)
    repo = make_repo(tmp_path)
    proc = host.proc / '4120'
    proc.mkdir()
    (proc / 'cmdline').write_bytes(b'python\0-m\0server\0model\0')
    (proc / 'comm').write_text('python\n')
    (proc / 'cwd').symlink_to(repo)
    lines = []
    host.out = lines.append
    assert steps.stop(host, settings.build(repo, host)) == 1
    assert lines == ['A Crucible server still runs: PID 4120 (python). This tool did not stop it. '
                     'Use the terminal, service or container that started it.']
    assert host.runner.terminated == []


def test_busy_health_with_no_visible_pid_is_not_nothing_to_stop(tmp_path):
    host = make_machine(tmp_path, busy_ports=(8790,))
    repo = make_repo(tmp_path)
    lines = []
    host.out = lines.append
    assert steps.stop(host, settings.build(repo, host)) == 1
    assert lines == ['A server still answers on port 8790. Its PID is not available to this user. '
                     'Use the terminal, service or container that started it.']
    assert host.runner.terminated == []
