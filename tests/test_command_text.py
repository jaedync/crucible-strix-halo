"""Printed launcher commands work outside the installation folder."""
import json
import shlex
from tools.crucible_cli import cli
from tools.crucible_cli.commands import command_text, output_text
from crucible_cli_fakes import make_machine


def test_command_path_is_absolute_and_shell_quoted(tmp_path):
    repo = tmp_path / 'install "with spaces"'
    assert command_text(repo, 'stop') == shlex.quote(str(repo / 'crucible')) + ' stop'


def test_output_replaces_each_command_and_preserves_json(tmp_path):
    repo = tmp_path / 'install "with spaces"'
    text = 'Stop: ./crucible stop. Start: ./crucible start'
    expected = text.replace('./crucible', command_text(repo))
    assert output_text(repo, text) == expected
    assert json.loads(output_text(repo, json.dumps({'fix': text}))) == {'fix': expected}


def test_main_output_names_install_path(tmp_path):
    host = make_machine(tmp_path)
    lines = []
    host.out = lines.append
    repo = tmp_path / 'another folder'
    assert cli.main(['start'], host=host, repo=repo) == 1
    assert lines == [f'No configuration yet. Run {command_text(repo, "setup")} first.']


def test_help_names_install_path(tmp_path):
    assert command_text(tmp_path, 'stop') in cli.parser(repo=tmp_path).format_help()


def test_json_output_keeps_its_layout(tmp_path):
    # doctor --json and service status --json print indented JSON; the launcher path must not fold it into one line.
    text = json.dumps({'checks': [{'fix': 'Run ./crucible setup'}], 'ok': False}, indent=2)
    out = output_text(tmp_path, text)
    assert out == text.replace('./crucible', command_text(tmp_path))
