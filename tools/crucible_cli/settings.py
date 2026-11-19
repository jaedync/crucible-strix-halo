"""One object for every choice of a run. Saved in .crucible/config.json so that start and doctor reuse it."""
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import keyfile, machine, measured, options, packs, rocm, state_root

UNSET = object()
# The state folder is owned by the engine, which refuses a folder that holds other files. So the proof that setup made
# the folder (or took an empty one) is a record in the work folder, never a file inside the state folder.
STATE_MARKER = "state-dir.json"
STATE_IDENTITY = state_root.IDENTITY
SAVED = ("pack", "models_dir", "state_dir", "port", "streams", "ctx", "state_disk_gib", "jobs", "bind", "api_key_file",
         "allowed_hosts", "layout", "short_streams", "short_ctx", "ngram16", "abliterated",
         "state_write_gib_day")

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
    api_key_file: Path = None  # the path only; the key is never in settings. None: no key (private addresses only)
    streams: int = measured.STREAMS
    ctx: int = measured.CTX
    layout: str = "default"   # a name of measured.LAYOUTS; it sets streams, ctx, short_streams and short_ctx
    short_streams: int = 0
    short_ctx: int = 0
    state_disk_gib: int = 200
    state_write_gib_day: int = 300   # the SSD wear guard of the state folder (engine/runtime/state_disk.h, default 300)
    jobs: int = 1
    allowed_hosts: tuple = ()  # more Host names for the server (a reverse proxy's name); the loopback names are always there
    own_head_dir: Path = None  # part 5: where the user's own draft head lives (outside the repo); None = off
    runtime_lib: Path = None   # the library folder of the private ROCm runtime; None when the machine has a HIP runtime of its own
    ngram16: bool = False      # --ngram16: the 16-bit n-gram table (CRUCIBLE_PLE_BF16), built in the models folder
    abliterated: bool = False  # --abliterated: CRUCIBLE_ABLATE with CRUCIBLE_ABLATE_MODE=request (stock stays the default)
    ablation_dir: Path = None  # where the sidecar lives: outside the repo (tools/ablation_pack.py refuses the repo)
    memory: object = None      # the machine.Fit of this machine (not saved); None: the measured 128 GB values

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
    def ngram16_dir(self):
        return Path(self.models_dir) / options.NGRAM16_FOLDER

    @property
    def ngram16_file(self):
        return self.ngram16_dir / options.NGRAM16_FILE

    @property
    def ablation_file(self):
        if self.pack == "4.05":
            return Path(self.ablation_dir) / options.ABL_405_FOLDER / options.ABL_405_FILE
        return Path(self.ablation_dir) / options.ABL_SOURCE_FOLDER / options.ABL_FILE

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


def ngram16_ready(s):
    """True when the 16-bit table is complete: the size and the sha256 record that the build wrote (it is not read again)."""
    f = s.ngram16_file
    try:
        if f.is_symlink() or f.stat().st_size != options.NGRAM16_BYTES:
            return False
        return Path(str(f) + ".sha256").read_text().split()[0] == options.NGRAM16_SHA256
    except (OSError, IndexError):
        return False


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ablation_ready(s, path=None, pack=None):
    """True when the sidecar of the pack (or PATH for PACK) is a plain file with the sha256 that the engine pins."""
    path = Path(path or s.ablation_file)
    try:
        return not path.is_symlink() and path.is_file() and file_sha256(path) == options.ABL_SHA256[pack or s.pack]
    except (OSError, KeyError):
        return False


ENGINE_WRITE_GIB_DAY = 300   # the engine's default of --state-disk-write-gib-per-day (engine/runtime/state_disk.h)


def write_limit_line(s):
    """One line about the SSD wear guard: the most that the state folder writes in a day and in a year."""
    n = s.state_write_gib_day
    if n == 0:
        return "state writes: none (--state-write-gib-day 0: the state folder is read only)"
    tb = n * 1024 ** 3 * 365 / 1e12
    return (f"state writes: at most {n} GiB a day ({tb:.0f} TB a year); compare with the TBW of your SSD "
            "(--state-write-gib-day N)")


def ledger_today(state_dir, now):
    """Bytes that the state folder wrote today (the engine's write-ledger: one line "YYYY-MM-DD BYTES", local day)."""
    import time
    try:
        day, count = (Path(state_dir) / "write-ledger").read_text().split()[:2]
        return int(count) if day == time.strftime("%Y-%m-%d", time.localtime(now)) else 0
    except (OSError, ValueError):
        return 0


def load_saved(repo):
    try:
        return json.loads((Path(repo) / ".crucible/config.json").read_text())
    except (OSError, ValueError):
        return {}


def build(repo, host, **kw):
    """Defaults, then the saved config, then the flags that the user gave (UNSET or None means not given)."""
    saved = load_saved(repo)
    fit = machine.of_host(host)
    # The defaults of this machine's memory class (on a 128 GB machine the measured ones, as the dataclass has them).
    values = {"pack": fit.pack, "streams": fit.streams, "ctx": fit.ctx}
    values.update({k: saved[k] for k in SAVED if k in saved})
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
        values["streams"], values["ctx"], values["short_streams"], values["short_ctx"] = machine.layout_choice(fit, layout)
        if streams is not None:
            values["streams"] = streams
        if ctx is not None:
            values["ctx"] = ctx
    elif kw.get("streams") is not None or kw.get("ctx") is not None:
        values["layout"], values["short_streams"], values["short_ctx"] = "default", 0, 0  # own sizes: no short class
    api_key = values.pop("api_key", None)
    state_dir = values.pop("state_dir", UNSET)
    if state_dir is None and kw.get("state_dir") in (None, UNSET) and int(saved.get("ctx") or 0) > measured.PLAIN_CTX:
        state_dir = UNSET  # the saved "no folder" came from a YaRN layout: a plain layout gets the default folder back
    no_state = kw.get("no_state_dir", False)
    s = Settings(repo=Path(repo), memory=fit, **{k: v for k, v in values.items() if k != "no_state_dir"})
    xdg = host.env.get("XDG_DATA_HOME")
    s.own_head_dir = (Path(xdg) if xdg else Path(host.home) / ".local/share") / "crucible/own-head"
    s.ablation_dir = (Path(xdg) if xdg else Path(host.home) / ".local/share") / "crucible/ablation"
    if isinstance(s.models_dir, str):
        s.models_dir = Path(s.models_dir)
    rt = rocm.detect(host, s.work)
    s.runtime_lib = rt.lib if rt.source == "private" else None   # a system runtime always wins
    # None preserves the saved choice. Public and wildcard binds still need a key.
    if api_key is False:
        if not keyfile.is_private_bind(s.bind):
            raise ValueError("Without an API key, listen on a private address. Set --api-key or change --bind.")
        s.api_key_file = None  # keep the file on disk, but stop using it
    elif s.api_key_file is None and (api_key is True or not keyfile.is_private_bind(s.bind)):
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


def state_dir_verdict(home, repo, path, look_inside=False):
    """The verdict of the engine rule on PATH as the state folder (tools/crucible_cli/state_root.py, the host copy of
    engine/runtime/state_root.h). A folder that setup recorded as its own is accepted with what it holds, unless
    LOOK_INSIDE (uninstall: remove only what the engine writes)."""
    owned = False if look_inside else state_owned(repo, path)
    return state_root.state_root_verdict(os.path.abspath(path), home=home, contents=not owned)


def state_dir_problem(home, repo, path, look_inside=False):
    """The one check of the state folder (cache folder item 2): setup, doctor and uninstall use it. None when PATH may be the
    state folder, else the sentence that says why not. The folder may be anywhere that only this user (or root) can change
    (Nov 12), but not a system folder or the home folder itself."""
    verdict = state_dir_verdict(home, repo, path, look_inside=look_inside)
    return None if verdict.ok else verdict.message


def state_folder_problem(path, owned=False):
    """The same rule without a home folder: None when PATH may be the state folder, else the sentence that says why not.
    A folder that setup made before (OWNED) is accepted with what it holds."""
    verdict = state_root.state_root_verdict(os.path.abspath(path), contents=not owned)
    return None if verdict.ok else verdict.message


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
