"""Soft LAN key policy. Synthetic addresses and mocked startup need no GPU."""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from crucible_cli_fakes import make_repo, wire
from crucible_cli import cli, keyfile, launch, settings
from server import api_key
from server.hosts import host_allowed

LOCAL = ("127.0.0.1", "127.2.3.4", "::1", "10.0.0.1", "10.255.255.254",
         "172.16.0.1", "172.31.255.254", "192.168.1.2", "169.254.1.1",
         "fe80::1", "febf::1", "fc00::1", "fdff::1")
PUBLIC = ("8.8.8.8", "172.15.0.1", "172.32.0.1", "192.169.1.2", "2001:4860::1",
          "fec0::1", "0.0.0.0", "::", "::ffff:127.0.0.1", "192.0.2.1", "2001:db8::1", "224.0.0.1",
          # IPv4-mapped private addresses need a key too (r373 secrev L8): a mapped bind is not a private IPv4 bind.
          "::ffff:10.0.0.1", "::ffff:192.168.1.2")


@pytest.mark.parametrize("address", LOCAL + PUBLIC)
@pytest.mark.parametrize("has_key", (False, True))
def test_address_policy(address, has_key):
    refusal = api_key.bind_error(address, has_key, {})
    assert (refusal is None) == (has_key or address in LOCAL)
    assert keyfile.is_private_bind(address) == (address in LOCAL)
    if refusal:
        assert "--api-key-file" in refusal
        assert "private address" in refusal


@pytest.fixture
def setup(tmp_path):
    return make_repo(tmp_path), wire(tmp_path)


def test_lan_has_no_key_by_default_and_can_toggle_without_deleting_file(setup):
    repo, host = setup
    assert settings.build(repo, host, bind="192.168.1.2").api_key_file is None
    s = settings.build(repo, host, bind="192.168.1.2", api_key=True)
    keyfile.ensure_key(host, s.api_key_file)
    saved_path = s.api_key_file
    (repo / ".crucible").mkdir(exist_ok=True)
    import json
    (repo / ".crucible/config.json").write_text(json.dumps(s.to_json()))
    assert settings.build(repo, host).api_key_file == saved_path
    off = settings.build(repo, host, api_key=False)
    assert off.api_key_file is None
    assert saved_path.exists()
    assert "--allowed-host" in launch.server_command(off, "/python")[1]
    (repo / ".crucible/config.json").write_text(json.dumps(off.to_json()))
    assert settings.build(repo, host).api_key_file is None


def test_flags_are_mutually_exclusive_and_noninteractive(setup):
    repo, host = setup
    for flag, enabled in (("--api-key", True), ("--no-api-key", False)):
        args = cli.parser().parse_args(["setup", "--yes", "--bind", "10.0.0.1", flag])
        assert (cli._settings(args, host, repo).api_key_file is not None) == enabled
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["setup", "--api-key", "--no-api-key"])
    with pytest.raises(ValueError, match="private address"):
        settings.build(repo, host, bind="0.0.0.0", api_key=False)


def test_setup_asks_network_address_and_lan_key_with_no_defaults(setup):
    repo, host = setup
    args = cli.parser().parse_args(["setup", "--pack", "4.05", "--models-dir", str(repo / "models"),
                                    "--no-state-dir", "--port", "8790"])
    seen = []
    answers = iter(("yes", "192.168.1.2", ""))
    def ask(question, default):
        seen.append((question, default))
        return next(answers) or default
    host.prompt = ask
    cli._ask_settings(args, host, repo)
    assert seen == [("Will other machines use this server?", "no"),
                    ("Listen address", ""), ("Use an API key?", "no")]
    assert args.bind == "192.168.1.2"
    assert cli._settings(args, host, repo).api_key_file is None


def test_setup_network_no_keeps_loopback_and_asks_listen_address(setup):
    repo, host = setup
    args = cli.parser().parse_args(["setup", "--pack", "4.05", "--models-dir", str(repo / "models"),
                                    "--no-state-dir", "--port", "8790"])
    seen = []
    host.prompt = lambda question, default: seen.append((question, default)) or default
    cli._ask_settings(args, host, repo)
    assert seen == [("Will other machines use this server?", "no"), ("Listen address", "127.0.0.1")]
    assert args.bind == "127.0.0.1"


@pytest.mark.parametrize("has_key", (False, True))
@pytest.mark.parametrize("address, authority", (("192.168.1.2", "192.168.1.2"), ("fd00::1", "[fd00::1]")))
def test_mocked_hand_start_warns_once_and_keeps_host_check(has_key, address, authority, tmp_path):
    import server.__main__ as startup
    path = tmp_path / "key"
    path.write_text("synthetic-" + "a" * 40 + "\n")
    path.chmod(0o600)
    argv = ["server", "/model", "--host", address, "--no-update-check"]
    if has_key:
        argv += ["--api-key-file", str(path)]
    with patch.object(sys, "argv", argv), patch.object(startup, "ModelTokenizer"), \
            patch("server.batch_backend.BatchedEngineBackend"), patch.object(startup, "bind_server") as bind, \
            patch.object(startup.signal, "signal"), patch.object(startup, "Service") as service, \
            patch("sys.stderr", io.StringIO()) as err, patch.dict("os.environ", {}, clear=True):
        bind.return_value.serve_forever.side_effect = KeyboardInterrupt
        assert startup.main() == 0
    hosts = service.call_args.kwargs["allowed_hosts"]
    if not has_key:
        assert host_allowed(authority + ":8790", hosts)
        assert not host_allowed("attacker.example:8790", hosts)
        expected = (f"WARNING: no API key on {authority}:8790. Every device that can reach this server can send prompts "
                    "and fill its sessions. Add a key with: ./crucible setup --api-key")
        assert err.getvalue().splitlines().count(expected) == 1
    else:
        assert "WARNING: no API key" not in err.getvalue()


@pytest.mark.parametrize("flag", ("--api-key", "--no-api-key"))
def test_installer_forwards_the_script_flags(flag):
    import subprocess
    text = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
    function = text.split("# ---- arguments", 1)[1].split("parse_args()", 1)[1].split("# ---- the system", 1)[0]
    script = 'parse_args()' + function + '\nparse_args "$@"\nprintf "%s\\n" "${PASS[@]}"\n'
    result = subprocess.run(["bash", "-c", script, "installer", "--yes", "--bind", "10.0.0.1", flag],
                            capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.splitlines() == ["--yes", "--bind", "10.0.0.1", flag]


def test_setup_retries_an_empty_remote_address(setup):
    repo, host = setup
    args = cli.parser().parse_args(["setup", "--pack", "4.05", "--models-dir", str(repo / "models"),
                                    "--no-state-dir", "--port", "8790"])
    answers = iter(("maybe", "yes", "", "10.0.0.1", "maybe", "no"))
    host.prompt = lambda question, default: next(answers)
    cli._ask_settings(args, host, repo)
    assert args.bind == "10.0.0.1"
    assert args.api_key is False


def test_launch_host_policy_with_and_without_key(setup):
    repo, host = setup
    for enabled in (False, True):
        s = settings.build(repo, host, bind="192.168.1.2", api_key=enabled)
        names = launch.allowed_hosts(s)
        if not enabled:
            assert "192.168.1.2" in names
            assert "localhost" in names
        else:
            assert names is None
