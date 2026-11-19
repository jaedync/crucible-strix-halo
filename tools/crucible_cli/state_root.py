"""Where the conversation state folder (--state-dir) can be: the host copy of the engine rule.

engine/runtime/state_root.h CheckStateRoot is the engine copy. Both give the same code for the same path: the shared table
tests/data/state_root_rule.tsv tests the two. docs/CONVERSATION_CACHE.md "Where the state folder can be" gives the reason
for each part. The rule does not need HOME: any folder that only this user (or root) can change is accepted.
"""
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

IDENTITY = re.compile(r"^[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}$")
LEDGER = "write-ledger"
# A system folder and every folder inside it.
SYSTEM_TREES = ("/etc", "/usr", "/proc", "/sys", "/dev", "/boot", "/efi", "/bin", "/sbin", "/lib", "/lib32", "/lib64",
                "/var/lib")
# Shared top folders: refused as the state folder itself, accepted as a parent.
SYSTEM_TOPS = ("/", "/run", "/tmp", "/var", "/var/tmp", "/home", "/mnt", "/media", "/opt", "/srv", "/root")


@dataclass(frozen=True)
class Verdict:
    code: str      # "ok" or the refusal word (the table codes)
    message: str   # "" when ok, else one sentence for the user
    path: str      # the real path (links in the parent folders resolved, the last name kept)
    exists: bool   # the folder is there now

    @property
    def ok(self):
        return self.code == "ok"


def is_system(real):
    return real in SYSTEM_TOPS or any(real == t or real.startswith(t + "/") for t in SYSTEM_TREES)


def _user_namespace():
    """True inside a user namespace (bwrap, a rootless container): root of the host shows there as the overflow user."""
    try:
        with open("/proc/self/uid_map") as f:
            return f.read().split() != ["0", "0", "4294967295"]
    except OSError:
        return False


def _overflow_uid():
    try:
        with open("/proc/sys/kernel/overflowuid") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 65534


def trusted_owners(uid):
    owners = {0, uid}
    if _user_namespace():
        owners.add(_overflow_uid())
    return owners


def _resolve_parent(parent):
    """PARENT with its links resolved up to the first name that stat() cannot reach (missing, a link to a missing folder,
    or not readable); the names from there are kept as they are, so the parent check sees them. The engine copy
    (state_root.h StateRootResolveParent) does the same steps. os.path.realpath alone follows a link to a missing folder."""
    parts = Path(parent).parts
    prefix, i = parts[0], 1
    while i < len(parts):
        try:
            os.stat(os.path.join(prefix, parts[i]))
        except OSError:
            break
        prefix = os.path.join(prefix, parts[i])
        i += 1
    try:
        real = os.path.realpath(prefix, strict=True)
    except OSError:
        return parent  # changed between the two calls: the parent check sees the change
    return os.path.join(real, *parts[i:])


def real_path(path):
    """The path with its parent links resolved and the last name kept, the way the engine reads it."""
    norm = os.path.normpath(path)
    if norm in ("/", "//"):
        return "/"
    parent, name = os.path.split(norm)
    return os.path.join(_resolve_parent(parent), name)


def _link_problem(link):
    """A link in a parent folder after the resolve: its target is not there (a disk that is not mounted), or it changed."""
    try:
        os.stat(link)
    except OSError as exc:
        try:
            target = os.readlink(link)
        except OSError:
            target = ""
        if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
            return "link", f"{link} is a link to a folder that is not there ({target})."
        return "link", f"{link} is a link that cannot be followed ({target}: {exc.strerror})."
    return "link", f"{link} is a link (it changed after the check)."


def _refuse(code, message, path, exists=False):
    return Verdict(code, message, path, exists)


def _parent_problem(real, uid, gid, owners):
    """The first parent folder that another user could change, as (code, message), or None. Missing parents are new."""
    parts = Path(real).parts[:-1]
    current = "/"
    for i, part in enumerate(parts):
        current = "/" if i == 0 else os.path.join(current, part)
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            return None
        except OSError as exc:
            return "unreadable", f"{current} cannot be read ({exc.strerror})."
        if stat.S_ISLNK(st.st_mode):
            return _link_problem(current)
        if not stat.S_ISDIR(st.st_mode):
            return "not-dir", f"{current} is not a folder."
        if st.st_uid not in owners:
            return "parent-owner", f"{current} belongs to another user, who could move or replace {real}."
        open_bits = st.st_mode & (0o002 | (0o020 if st.st_gid != gid else 0))
        if open_bits and not st.st_mode & stat.S_ISVTX:
            mode = f"{stat.S_IMODE(st.st_mode):04o}"
            return "parent-open", f"other users can write {current} (mode {mode}, no sticky bit), so they could move or replace {real}."
    return None


def _content_problem(real, uid):
    try:
        names = sorted(os.listdir(real))
    except OSError as exc:
        return "contents-unreadable", f"the list of {real} cannot be read ({exc.strerror})."
    foreign, other_owner = [], []
    for name in names:
        try:
            st = os.lstat(os.path.join(real, name))
        except OSError:
            foreign.append(name)
            continue
        if name != LEDGER and not (IDENTITY.match(name) and stat.S_ISDIR(st.st_mode)):
            foreign.append(name)
        elif st.st_uid != uid:
            other_owner.append(name)
    if foreign:
        more = ", ..." if len(foreign) > 3 else ""
        return "foreign", f"{real} holds files that Crucible did not make ({', '.join(foreign[:3])}{more})."
    if other_owner:
        return "foreign", f"{real} holds files of another user ({', '.join(other_owner[:3])})."
    return None


def state_root_verdict(path, home=None, uid=None, gid=None, contents=True):
    """The verdict on PATH as the state folder. HOME (optional) refuses the home folder itself. CONTENTS False skips the
    look inside (a folder that setup made before)."""
    uid = os.geteuid() if uid is None else uid
    gid = os.getegid() if gid is None else gid
    path = os.fspath(path)
    if not os.path.isabs(path):
        return _refuse("relative", f"{path} is not an absolute path.", path)
    real = real_path(path)
    if is_system(real):
        return _refuse("system", f"{real} is a system folder.", real)
    if home and real == os.path.realpath(os.path.abspath(os.fspath(home))):
        return _refuse("home", f"{real} is your home folder. Use a folder of its own, for example ~/.cache/crucible/states.", real)
    problem = _parent_problem(real, uid, gid, trusted_owners(uid))
    if problem:
        return _refuse(problem[0], problem[1], real)
    try:
        st = os.lstat(real)
    except FileNotFoundError:
        return Verdict("ok", "", real, False)
    except OSError as exc:
        return _refuse("unreadable", f"{real} cannot be read ({exc.strerror}).", real, True)
    if stat.S_ISLNK(st.st_mode):
        return _refuse("link", f"{real} is a link.", real, True)
    if not stat.S_ISDIR(st.st_mode):
        return _refuse("not-dir", f"{real} is not a folder.", real, True)
    if st.st_uid != uid:
        return _refuse("owner", f"{real} belongs to another user.", real, True)
    if st.st_mode & stat.S_ISVTX:
        return _refuse("shared", f"{real} is a shared folder (it has the sticky bit).", real, True)
    if contents:
        problem = _content_problem(real, uid)
        if problem:
            return _refuse(problem[0], problem[1], real, True)
    return Verdict("ok", "", real, True)
