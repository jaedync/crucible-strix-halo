"""One measure of the disk of the state folder: a sequential O_DIRECT write and read of about 2 GiB, deleted after.

The file goes next to the state folder (its parent folder, on the same file system), never inside it: the engine refuses
to start when the state folder holds a file that it did not make. When the parent is on another file system, the file
goes inside the folder for the time of the measure only. The file has a new random name and is made with O_EXCL; the
measure deletes only the file that it made (same device and inode). Standard library only."""
import fcntl
import mmap
import os
import secrets
import time
from pathlib import Path

SIZE = 2 * 1024 ** 3
CHUNK = 8 * 1024 * 1024
NAME = ".crucible-disk-measure"
TRIES = 8


def _where(folder):
    """The folder for the measure file, or None."""
    folder = Path(folder)
    probe = folder
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    parent = folder.parent if folder.exists() else probe
    try:
        if folder.exists() and os.stat(parent).st_dev != os.stat(folder).st_dev:
            return folder
    except OSError:
        return None
    return parent


def _name():
    return f"{NAME}-{os.getpid()}-{secrets.token_hex(8)}"


def _create(where):
    """(path, fd) of a new file that this call made, or None. An existing file is never opened."""
    for _ in range(TRIES):
        path = where / _name()
        try:
            return path, os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except FileExistsError:
            continue
        except OSError:
            return None
    return None


def _remove(path, ident):
    """Delete path only when it is still the file that _create made."""
    try:
        st = os.lstat(path)
        if (st.st_dev, st.st_ino) == ident:
            os.unlink(path)
    except OSError:
        pass


def _time(fd, buf, count):
    """(write seconds, read seconds) of count writes of buf and one read of the file, both O_DIRECT."""
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_DIRECT)
    t0 = time.monotonic()
    for _ in range(count):
        os.write(fd, buf)
    os.fsync(fd)
    write_s = time.monotonic() - t0
    os.lseek(fd, 0, os.SEEK_SET)
    t0 = time.monotonic()
    while os.readv(fd, [buf]) > 0:
        pass
    return write_s, time.monotonic() - t0


def measure(folder, nbytes=SIZE, chunk=CHUNK):
    """{"write_mbps", "read_mbps", "bytes"}, or None when the file system does not take O_DIRECT (tmpfs, for example)."""
    where = _where(folder)
    if where is None:
        return None
    chunk = min(chunk, nbytes)
    buf = mmap.mmap(-1, chunk)          # page aligned, as O_DIRECT needs
    buf.write(os.urandom(chunk))        # not zeros: some drives compress
    count = max(1, nbytes // chunk)
    made = _create(where)
    if made is None:
        buf.close()
        return None
    path, fd = made
    ident = None
    try:
        st = os.fstat(fd)
        ident = (st.st_dev, st.st_ino)
        write_s, read_s = _time(fd, buf, count)
    except OSError:
        return None
    finally:
        os.close(fd)
        if ident is not None:
            _remove(path, ident)
        buf.close()
    total = count * chunk
    return {"write_mbps": round(total / 1e6 / max(write_s, 1e-6)), "read_mbps": round(total / 1e6 / max(read_s, 1e-6)),
            "bytes": total}
