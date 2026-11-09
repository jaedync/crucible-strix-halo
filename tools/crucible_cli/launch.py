"""The server command. The words are the ones of docs/QUICKSTART.md, step 6."""
import json
import os
import shlex
from pathlib import Path

from . import keyfile, measured

# The Host names that the server accepts on a loopback bind (DNS rebinding: a web page whose name points at 127.0.0.1
# sends its own name in the Host header, and a server that accepts any name answers it).
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def in_git_tree(path):
    """True when PATH or a parent folder holds a .git entry (a repo, a worktree or a submodule)."""
    p = Path(os.path.realpath(path))
    return any((q / ".git").exists() for q in (p, *p.parents))


def head_choice(own_head_dir):
    """What the own-head record says. Returns (kind, path):
    ("off", None)   no record: the variable CRUCIBLE_MTP_SIDECAR of the parent is left alone (a profile may set it);
    ("stock", None) the record exists and names no usable head: the stock head, so the variable is removed;
    ("head", PATH)  the record names a head file.
    Schema (part 5 design 2.1): <dir>/head.json = {"active": "head-<sha12>.safetensors" | null, "previous": [...]},
    the files in <dir>/heads/. A name with a path part, a missing file or a folder inside any git tree is not usable."""
    if own_head_dir is None:
        return "off", None
    base = Path(own_head_dir)
    try:
        text = (base / "head.json").read_text()
    except OSError:
        return "off", None
    try:
        active = json.loads(text).get("active")
    except (ValueError, AttributeError):
        return "stock", None
    if not isinstance(active, str) or not active or active != os.path.basename(active) or active.startswith("."):
        return "stock", None
    side = base / "heads" / active
    if not side.is_file() or in_git_tree(side):
        return "stock", None
    return "head", os.path.realpath(side)


def head_sidecar(own_head_dir):
    """The path of the head to serve, or None. For the serve script: prints the path, '-' (stock: unset) or nothing."""
    return head_choice(own_head_dir)[1]


def head_word(own_head_dir):
    kind, path = head_choice(own_head_dir)
    return path if kind == "head" else ("-" if kind == "stock" else "")


def own_head_env(s):
    """Hook for part 5 (train your own draft head). The value None means: remove the variable from the process."""
    kind, path = head_choice(s.own_head_dir)
    if kind == "head":
        return {"CRUCIBLE_MTP_SIDECAR": path}
    return {"CRUCIBLE_MTP_SIDECAR": None} if kind == "stock" else {}


def allowed_hosts(s):
    """The comma list for --allowed-host, or None for no list.

    A loopback bind gets the loopback names, the bind address and the names that the user added (a reverse proxy sends
    its own name). A bind that is not loopback needs the API key, which a web page cannot send, so it has no list unless
    the user added names; then the list also holds the bind address."""
    if not keyfile.is_loopback(s.bind) and not s.allowed_hosts:
        return None
    names = list(LOOPBACK_HOSTS)
    ip = keyfile.address(s.bind)
    if not ip.is_unspecified:
        names.append(f"[{ip}]" if ip.version == 6 else str(ip))
    names += list(s.allowed_hosts)
    return ",".join(dict.fromkeys(name.lower() for name in names))


def server_command(s, python):
    env = {**measured.ENV, **s.spec.extra_env, **own_head_env(s)}
    flags = measured.SERVER_FLAGS
    argv = [str(python), "-m", "server", str(s.ckpt), "--engine", str(s.engine), "--host", s.bind,
            "--port", str(s.port), "--streams", str(s.streams), "--ctx", str(s.ctx), "--prefill", flags["--prefill"],
            "--conversation-cache-mib", flags["--conversation-cache-mib"], "--host-reserve-mib", flags["--host-reserve-mib"]]
    if s.api_key_file is not None:
        argv += ["--api-key-file", str(s.api_key_file)]  # the path only: the key is never in an argument
    hosts = allowed_hosts(s)
    if hosts is not None:
        argv += ["--allowed-host", hosts]
    if s.state_dir is not None:
        argv += ["--state-dir", str(s.state_dir), "--state-disk-gib", str(s.state_disk_gib)]
    argv += ["--vision", "--clamp-max-tokens", "--default-thinking", "on", "--max-body-mib", "16"]
    return env, argv


def process_env(base, env):
    """The environment of the server: `base` plus `env`. A measured word (tools/crucible_cli/measured.py) is a default: a
    word that `base` already holds, also an empty one, stays as the user set it."""
    merged = dict(base)
    for key, value in env.items():
        if value is None:
            merged.pop(key, None)
        elif key not in measured.ENV or key not in base:
            merged[key] = value
    return merged


def serve_script(s, python):
    """The script resolves the own head each time it runs, so a head installed later is used at the next start."""
    env, argv = server_command(s, python)
    env.pop("CRUCIBLE_MTP_SIDECAR", None)  # resolved below, at each start
    # A measured word is a default: the script sets it only when the caller's environment does not hold it (also empty).
    defaults = "".join(f'[ -n "${{{k}+x}}" ] || {k}={shlex.quote(v)}\nexport {k}\n' for k, v in env.items() if k in measured.ENV)
    words = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if k not in measured.ENV)
    cmd = " ".join(shlex.quote(a) for a in argv)
    q = shlex.quote
    head = ""
    if s.own_head_dir is not None:
        look = ("import sys; sys.path.insert(0, sys.argv[1]); from crucible_cli.launch import head_word; "
                "print(head_word(sys.argv[2]))")
        head = (f"HEAD=$({q(os.fspath(python))} -I -c {q(look)} {q(os.fspath(Path(s.repo) / 'tools'))} {q(os.fspath(s.own_head_dir))} 2>/dev/null)\n"
                'case "$HEAD" in\n  "") ;;\n  -) unset CRUCIBLE_MTP_SIDECAR ;;\n  *) export CRUCIBLE_MTP_SIDECAR="$HEAD" ;;\nesac\n')
    return (f"#!/bin/sh\n# Written by ./crucible setup. The server command of this machine.\n"
            f"cd {q(os.fspath(s.repo))} || exit 1\n{defaults}{head}exec env {words} {cmd} \"$@\"\n")
