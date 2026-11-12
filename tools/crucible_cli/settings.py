"""One object for every choice of a run. Saved in .crucible/config.json so that start and doctor reuse it."""
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import keyfile, measured, packs, rocm

UNSET = object()
# The state folder is owned by the engine, which refuses a folder that holds other files. So the proof that setup made
# the folder (or took an empty one) is a record in the work folder, never a file inside the state folder.
STATE_MARKER = "state-dir.json"
STATE_IDENTITY = re.compile(r"^[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}$")
SAVED = ("pack", "models_dir", "state_dir", "port", "streams", "ctx", "state_disk_gib", "jobs", "bind", "api_key_file",
         "allowed_hosts", "layout", "short_streams", "short_ctx")

# One name for the Host list: a host name or an IPv4 address, or an IPv6 literal in brackets, then an optional port.
HOST_NAME = re.compile(r"(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|\[[0-9A-Fa-f:.]+\])(?::([0-9]{1,5}))?")


def host_name_ok(text):
    """True when TEXT is one name for the Host list (the server's --allowed-host). No list, no space, no scheme, no path."""
    match = HOST_NAME.fullmatch(text) if isinstance(text, str) else None
    return bool(match) and (match.group(1) is None or 0 < int(match.group(1)) <= 65535)


@dataclass
class Settings:
    repo: Path
    pack: str = packs.DEFAULT
    models_dir: Path = None
    state_dir: Path = None
    port: int = 8790
    bind: str = "127.0.0.1"
    api_key_file: Path = None  # the path only; the key is never in settings. None: no key (loopback only)
    streams: int = measured.STREAMS
    ctx: int = measured.CTX
    layout: str = "default"   # a name of measured.LAYOUTS; it sets streams, ctx, short_streams and short_ctx
    short_streams: int = 0
    short_ctx: int = 0
    state_disk_gib: int = 200
    jobs: int = 1
    allowed_hosts: tuple = ()  # more Host names for the server (a reverse proxy's name); the loopback names are always there
    own_head_dir: Path = None  # part 5: where the user's own draft head lives (outside the repo); None = off
    runtime_lib: Path = None   # the library folder of the private ROCm runtime; None when the machine has a HIP runtime of its own

    def __post_init__(self):
        self.repo = Path(self.repo)
        if self.models_dir is None:
            self.models_dir = self.repo / "models"
        self.allowed_hosts = tuple(self.allowed_hosts or ())
        if isinstance(self.api_key_file, str):
            self.api_key_file = Path(self.api_key_file)

    @property
    def spec(self):
        return packs.PACKS[self.pack]

    @property
    def ckpt(self):
        return Path(self.models_dir) / self.spec.folder

    @property
    def work(self):
        return self.repo / ".crucible"

    @property
    def venv(self):
        return self.repo / ".venv"

    @property
    def venv_python(self):
        return self.venv / "bin" / "python"

    @property
    def engine(self):
        return self.repo / "build" / "engine" / "crucible"

    @property
    def base_url(self):
        """Where this machine reaches the server. A wildcard bind (0.0.0.0, ::) is reached by loopback."""
        ip = keyfile.address(self.bind)
        if ip.is_unspecified:
            ip = ipaddress.ip_address("127.0.0.1" if ip.version == 4 else "::1")
        return f"http://{'[%s]' % ip if ip.version == 6 else ip}:{self.port}"

    def to_json(self):
        out = {}
        for key in SAVED:
            value = getattr(self, key)
            out[key] = str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        return out


def load_saved(repo):
    try:
        return json.loads((Path(repo) / ".crucible/config.json").read_text())
    except (OSError, ValueError):
        return {}


def build(repo, host, **kw):
    """Defaults, then the saved config, then the flags that the user gave (UNSET or None means not given)."""
    saved = load_saved(repo)
    values = {k: saved[k] for k in SAVED if k in saved}
    for key, value in kw.items():
        if value is not UNSET and value is not None:
            values[key] = value
    layout = kw.get("layout")
    if layout is not None:
        if layout not in measured.LAYOUTS:
            raise ValueError(f"--layout must be one of {', '.join(measured.LAYOUTS)}, not {layout!r}.")
        if layout != "default" and (kw.get("streams") is not None or kw.get("ctx") is not None):
            raise ValueError(f"--layout {layout} sets the sessions and the context; do not give --streams or --ctx with it.")
        streams = kw.get("streams")
        ctx = kw.get("ctx")
        values["streams"], values["ctx"], values["short_streams"], values["short_ctx"] = measured.LAYOUTS[layout]
        if streams is not None:
            values["streams"] = streams
        if ctx is not None:
            values["ctx"] = ctx
    elif kw.get("streams") is not None or kw.get("ctx") is not None:
        values["layout"], values["short_streams"], values["short_ctx"] = "default", 0, 0  # own sizes: no short class
    api_key = values.pop("api_key", False)
    state_dir = values.pop("state_dir", UNSET)
    if state_dir is None and kw.get("state_dir") in (None, UNSET) and int(saved.get("ctx") or 0) > measured.PLAIN_CTX:
        state_dir = UNSET  # the saved "no folder" came from a YaRN layout: a plain layout gets the default folder back
    no_state = kw.get("no_state_dir", False)
    s = Settings(repo=Path(repo), **{k: v for k, v in values.items() if k != "no_state_dir"})
    xdg = host.env.get("XDG_DATA_HOME")
    s.own_head_dir = (Path(xdg) if xdg else Path(host.home) / ".local/share") / "crucible/own-head"
    if isinstance(s.models_dir, str):
        s.models_dir = Path(s.models_dir)
    rt = rocm.detect(host, s.work)
    s.runtime_lib = rt.lib if rt.source == "private" else None   # a system runtime always wins
    # A bind address that is not loopback needs the key (the server refuses to start without it). --api-key asks for
    # the key on loopback too (a reverse proxy). A saved path stays when the bind goes back to loopback.
    if s.api_key_file is None and (api_key or not keyfile.is_loopback(s.bind)):
        s.api_key_file = keyfile.default_path(host)
    if no_state:
        s.state_dir = None
    elif state_dir is UNSET:
        s.state_dir = Path(host.home) / ".cache/crucible/states"
    else:
        s.state_dir = Path(state_dir).expanduser() if state_dir else None
    if s.ctx > measured.PLAIN_CTX:
        s.state_dir = None  # YaRN: the server refuses --state-dir (first version), so the cache stays in memory
    return s


def stop_note(s):
    """What a stop message says about the conversation states. They reach the disk only when a state folder is on."""
    if getattr(s, "state_dir", None) is None:
        return "The conversation cache is in memory only (no state folder), so it is lost."
    return "It saves its conversation states first; this can take a minute."


def state_folder_problem(path, owned=False):
    """None when PATH may be the state folder, else the sentence that says why not.

    It may be new, empty, or hold only what the engine writes there (write-ledger and the identity folders: the same rule
    that the engine applies at its start). A folder that setup made before (OWNED) is accepted as it is."""
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        return f"{path} is a link."
    if not path.exists():
        return None
    if not path.is_dir():
        return f"{path} is not a folder."
    if owned:
        return None
    try:
        names = os.listdir(path)
    except OSError as exc:
        return f"{path} cannot be read ({exc})."
    foreign = sorted(n for n in names if n != "write-ledger" and not (STATE_IDENTITY.match(n) and (path / n).is_dir()
                                                                       and not (path / n).is_symlink()))
    if foreign:
        return f"{path} holds files that Crucible did not make ({', '.join(foreign[:3])}{', ...' if len(foreign) > 3 else ''})."
    return None


def state_marker_path(repo):
    return Path(repo) / ".crucible" / STATE_MARKER


def state_owned(repo, state):
    """True when setup recorded STATE as its own state folder (the same real path, a plain file as the record)."""
    marker = state_marker_path(repo)
    try:
        if marker.is_symlink() or not marker.is_file():
            return False
        recorded = json.loads(marker.read_text()).get("state_dir")
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(recorded, str) and recorded == os.path.realpath(state)


def state_marker_text(state):
    return json.dumps({"state_dir": os.path.realpath(state), "made_by": "crucible setup"}) + "\n"
