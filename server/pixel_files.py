"""Private, short-lived pixel files: the engine reads image pixels from them, not from the JSON line.

A JSON line of base64 pixels costs the engine about 20 ms per MB to read, and every other batch session waits
for it. Each image goes to a mode-0600 file with a random name in a mode-0700
directory under /dev/shm (memory, not disk). The engine opens only the files of images that it encodes.
The backend removes a request's files when the engine has finished with the request, and close() removes
the directory. A new store removes the directories of dead servers of this user.

A container gets a 64 MB /dev/shm unless it sets --shm-size, and one 2560x1440 image is 11 MB of RGB. A request
whose bytes do not fit in /dev/shm (statvfs, less the bytes of writes in progress and SHM_MARGIN) goes to the
temp folder (tempfile.gettempdir(), so TMPDIR works). A fault during a write to /dev/shm (for example ENOSPC when
another process fills it) removes the partial files and writes the request to the temp folder once. The files
hold the same bytes in each folder, so the engine's tokens do not change, and the same clean-up rules apply.
"""
import errno
import os
import secrets
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from .api import ApiError

PREFIX = "crucible-vision-"
# Room in /dev/shm that pixel files never take: other users of the folder (ROCm, Python multiprocessing, other
# processes of the container) keep it.
SHM_MARGIN = 16 << 20
NO_ROOM = "The server has no room for the image pixels. Try again later."


def default_root():
    return Path("/dev/shm") if os.path.isdir("/dev/shm") else Path(tempfile.gettempdir())


class PixelFiles:
    """The files of one engine request. remove() is idempotent."""

    def __init__(self, paths):
        self.paths = tuple(paths)
        self._removed = False

    def remove(self):
        if self._removed:
            return
        self._removed = True
        for path in self.paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class PixelStore:
    def __init__(self, root=None, fallback=None):
        self.root = Path(root) if root is not None else default_root()
        self.fallback = Path(fallback) if fallback is not None else Path(tempfile.gettempdir())
        self.directories = {}  # folder -> this server's private directory in it
        self.pending = 0  # bytes that writes to self.root have taken and not yet finished
        self.logged = False
        self.lock = threading.Lock()

    def _ensure(self, root):
        with self.lock:
            directory = self.directories.get(root)
            if directory is None:
                _remove_stale(root)
                directory = tempfile.mkdtemp(prefix=f"{PREFIX}{os.getpid()}-", dir=root)
                os.chmod(directory, 0o700)
                self.directories[root] = directory
            return directory

    def _has_room(self, size):
        """Under self.lock. Unknown free space counts as room: a fault during the write still goes to the fallback."""
        try:
            stats = os.statvfs(self.root)
        except OSError:
            return True
        return size + self.pending + SHM_MARGIN <= stats.f_bavail * stats.f_frsize

    def _note_fallback(self, size, error=None):
        with self.lock:
            if self.logged:
                return
            self.logged = True
        # The fixed folder and the error name only: the temp folder's path can name a user.
        cause = "has no room for" if error is None or error.errno in (errno.ENOSPC, errno.EDQUOT) else \
            f"failed ({errno.errorcode.get(error.errno, 'OSError')}) to take"
        print(f"crucible: {self.root} {cause} the pixel files of a request ({size / 1e6:.1f} MB); they go to "
              "the temp folder. In a container, set --shm-size (docs/DOCKER.md).", file=sys.stderr, flush=True)

    def write(self, images):
        """Write each image's RGB bytes to a new private file. Returns PixelFiles in image order."""
        size = sum(memoryview(image.rgb).nbytes for image in images)
        if self.fallback == self.root:
            return self._write_last(self.root, images)
        with self.lock:
            fits = self._has_room(size)
            if fits:
                self.pending += size
        failure = None
        if fits:
            try:
                return self._write_in(self.root, images)
            except OSError as error:
                failure = error  # _write_in removed the partial files; the temp folder gets one try below
            finally:
                with self.lock:
                    self.pending -= size
        self._note_fallback(size, failure)
        return self._write_last(self.fallback, images)

    def _write_last(self, root, images):
        try:
            return self._write_in(root, images)
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise ApiError(NO_ROOM, status=503, code="pixel_space") from None
            raise

    def _write_in(self, root, images):
        directory = self._ensure(root)
        paths = []
        try:
            for image in images:
                path = os.path.join(directory, secrets.token_hex(16))
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
                paths.append(path)
                try:
                    view = memoryview(image.rgb)
                    while view:
                        view = view[os.write(fd, view):]
                finally:
                    os.close(fd)
        except BaseException:
            PixelFiles(paths).remove()
            raise
        return PixelFiles(paths)

    def close(self):
        with self.lock:
            for directory in self.directories.values():
                shutil.rmtree(directory, ignore_errors=True)
            self.directories = {}


def _remove_stale(root):
    """Remove pixel directories of this user whose server process no longer exists."""
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(PREFIX):
            continue
        pid = entry.name[len(PREFIX):].split("-", 1)[0]
        try:
            if not pid.isdigit() or entry.stat(follow_symlinks=False).st_uid != os.getuid():
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(entry.path, ignore_errors=True)
        except (OSError, ValueError):
            continue
