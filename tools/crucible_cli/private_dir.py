"""A private folder of this user (mode 0700), made or opened one name at a time from "/" (review cache-folder-r4 L-A).

The host copy of engine/runtime/state_private_dir.h. Path.mkdir(parents=True) and os.chmod follow a link in any parent:
in a shared sticky folder another user can make a missing parent (the `x` of /tmp/x/states) a link after the check
(state_root.py) and before the mkdir, and the mode change then hits another folder of this user. Here each name opens
with O_NOFOLLOW | O_DIRECTORY from the fd of its parent, and each parent fd gets the parent rule (fstat), so a link at
any depth stops the call. Only the fd of a folder of this user gets the fchmod.
"""
import os
import stat

from . import state_root

_FLAGS = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC  # needs only the search bit of the parent (review M1)
_READ = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC


class Refused(Exception):
    """The one sentence for the user that says why the folder was not made or opened. MISSING: a name was not there."""

    def __init__(self, message, missing=False):
        super().__init__(message)
        self.missing = missing


def _open_problem(at, name, shown, path, exc):
    """Why NAME in the folder AT did not open as a folder."""
    try:
        st = os.stat(name, dir_fd=at, follow_symlinks=False)
    except OSError:
        st = None
    if st is not None and stat.S_ISLNK(st.st_mode):
        return Refused(f"{shown} is a link, and Crucible does not follow a link while it makes {path} (another user can "
                       "make one in a shared folder). Remove the link, or use a folder that only you can change.")
    if st is not None and not stat.S_ISDIR(st.st_mode):
        return Refused(f"{shown} is not a folder.")
    return Refused(f"{shown} cannot be opened ({exc.strerror}).")


def _open_or_make(at, name, shown, path, make):
    """The fd of the folder NAME in AT, and whether this call made it (mode 0700 before the umask)."""
    try:
        return os.open(name, _FLAGS, dir_fd=at), False
    except FileNotFoundError:
        if not make:
            raise Refused(f"{shown} is not there.", missing=True) from None
    except OSError as exc:
        raise _open_problem(at, name, shown, path, exc) from None
    made = True
    try:
        os.mkdir(name, 0o700, dir_fd=at)
    except FileExistsError:
        made = False  # another process made it first: the checks below see what it is
    except OSError as exc:
        raise Refused(f"cannot create {shown}: {exc.strerror}.") from None
    try:
        return os.open(name, _FLAGS, dir_fd=at), made
    except OSError as exc:
        raise _open_problem(at, name, shown, path, exc) from None


def _set_private(fd, shown):
    """Mode 0700 on the folder of this user that the O_PATH fd FD holds. fchmod refuses an O_PATH fd, so the change goes to
    a new fd of "." (it needs the read and search bits of the folder). Without them (for example mode 0300), chmod of
    /proc/self/fd/N changes the inode of FD: that name always resolves to the inode of the fd, so no other folder can take
    its place."""
    try:
        own = os.open(".", _READ, dir_fd=fd)
    except PermissionError:
        own = None
    except OSError as exc:
        raise Refused(f"{shown} cannot be opened ({exc.strerror}).") from None
    try:
        if own is None:
            os.chmod(f"/proc/self/fd/{fd}", 0o700)
        else:
            os.fchmod(own, 0o700)
    except OSError as exc:
        raise Refused(f"cannot set mode 0700 on {shown}: {exc.strerror}.") from None
    finally:
        if own is not None:
            os.close(own)


def _fstat(fd, shown):
    try:
        return os.fstat(fd)
    except OSError as exc:
        raise Refused(f"{shown} cannot be read ({exc.strerror}).") from None


def make_private_dir(path, make=True, uid=None, gid=None, owners=None):
    """Make (MAKE) or open the folder PATH (absolute, no "." or ".." names). Each missing name gets mode 0700. Each parent
    must pass state_root.parent_folder_problem. The last folder must belong to UID; its mode becomes 0700. A folder of
    another user is never changed. Raises Refused."""
    uid = os.geteuid() if uid is None else uid
    gid = os.getegid() if gid is None else gid
    owners = state_root.trusted_owners(uid) if owners is None else owners
    path = os.fspath(path)
    if not os.path.isabs(path):
        raise Refused(f"{path} is not an absolute path.")
    names = [name for name in path.split("/") if name]
    for name in names:
        if name in (".", ".."):
            raise Refused(f"{path} has a '{name}' name.")
    try:
        at = os.open("/", _FLAGS & ~os.O_NOFOLLOW)
    except OSError as exc:
        raise Refused(f"/ cannot be opened ({exc.strerror}).") from None
    try:
        shown = "/"
        for name in names:
            problem = state_root.parent_folder_problem(shown, path, _fstat(at, shown), owners, gid)
            if problem:
                raise Refused(problem[1])
            shown = os.path.join(shown, name)
            fd, made = _open_or_make(at, name, shown, path, make)
            os.close(at)
            at = fd
            # A folder that this call made gets 0700 exactly (the umask cuts the mkdir mode). It must be ours: another
            # user may have made the name between the mkdir and the open.
            if made:
                if _fstat(at, shown).st_uid != uid:
                    raise Refused(f"{shown} belongs to another user (it came while Crucible made it). Use a folder "
                                  "that only you can change.")
                _set_private(at, shown)
        st = _fstat(at, shown)
        if st.st_uid != uid:
            raise Refused(f"{path} belongs to another user (uid {st.st_uid}), so Crucible does not change it. Use a "
                          "folder of your own.")
        if stat.S_IMODE(st.st_mode) != 0o700:
            _set_private(at, path)
    finally:
        os.close(at)
