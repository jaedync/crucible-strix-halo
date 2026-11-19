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


def _server_process(host, pid, cwd, port_inode=None):
    proc = host.proc / str(pid)
    (proc / 'fd').mkdir(parents=True)
    (proc / 'cmdline').write_bytes(b'python\0-m\0server\0model\0')
    (proc / 'comm').write_text('python\n')
    (proc / 'cwd').symlink_to(cwd)
    if port_inode is not None:
        (proc / 'fd' / '5').symlink_to(f'socket:[{port_inode}]')


def _listen(host, port, inode):
    (host.proc / 'net').mkdir(exist_ok=True)
    (host.proc / 'net' / 'tcp').write_text(
        '  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
        f'   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 {inode} 1 0\n')


def test_a_server_of_another_install_does_not_fail_this_stop(tmp_path):
    # Two installs (for example a second copy for Kiln): the stop of one must not fail on the server of the other.
    host = make_machine(tmp_path)
    repo = make_repo(tmp_path)
    other = tmp_path / 'other-install'
    (other / 'server').mkdir(parents=True)
    (other / 'crucible').write_text('')
    (other / 'server/api.py').write_text('')
    _server_process(host, 4121, other)
    lines = []
    host.out = lines.append
    assert steps.stop(host, settings.build(repo, host)) == 0, lines
    assert host.runner.terminated == []


def test_the_server_on_the_configured_port_is_named_with_its_pid(tmp_path):
    host = make_machine(tmp_path, busy_ports=(8790,))
    repo = make_repo(tmp_path)
    other = tmp_path / 'by-hand'
    other.mkdir()
    _server_process(host, 4122, other, port_inode=777)
    _listen(host, 8790, 777)
    lines = []
    host.out = lines.append
    assert steps.stop(host, settings.build(repo, host)) == 1
    assert len(lines) == 1 and 'port 8790' in lines[0] and 'PID 4122 (python)' in lines[0], lines
    assert host.runner.terminated == []
