"""Image pixels go to the engine as private files, not in the JSON line."""
import errno
import os
import stat
import subprocess
import sys
import tempfile
import threading

import pytest

from server import pixel_files
from server.api import ApiError
from server.pixel_files import PixelStore
from server.vision import VisionImage, image_digest


def image(height=32, width=64, fill=7):
    rgb = bytes([fill]) * (height * width * 3)
    return VisionImage(height, width, rgb, image_digest(height, width, rgb))


def test_files_are_private_and_hold_the_pixels(tmp_path):
    store = PixelStore(root=tmp_path)
    one, two = image(fill=1), image(fill=2)
    files = store.write([one, two])
    assert len(files.paths) == 2 and len(set(files.paths)) == 2
    directory = os.path.dirname(files.paths[0])
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert os.path.basename(directory).startswith(f"crucible-vision-{os.getpid()}-")
    for path, item in zip(files.paths, (one, two)):
        assert os.path.isabs(path)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert open(path, "rb").read() == item.rgb
        # Names carry no image identity.
        assert item.digest[:16] not in path
    files.remove()
    assert not any(os.path.exists(path) for path in files.paths)
    files.remove()  # idempotent
    store.close()
    assert not os.path.exists(directory)


def test_close_removes_files_that_a_request_still_holds(tmp_path):
    store = PixelStore(root=tmp_path)
    files = store.write([image()])
    store.close()
    assert not os.path.exists(files.paths[0])
    files.remove()  # no error after close


def test_a_failed_write_leaves_no_file(tmp_path, monkeypatch):
    # An I/O fault in both folders: the request fails and leaves no file in either folder.
    shm, fallback = tmp_path / "shm", tmp_path / "tmp"
    shm.mkdir(), fallback.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback))
    store = PixelStore(root=shm)
    calls = []
    real = os.write

    def failing(fd, data):
        calls.append(fd)
        if len(calls) >= 2:
            raise OSError(errno.EIO, "Input/output error")
        return real(fd, data)
    monkeypatch.setattr(os, "write", failing)
    with pytest.raises(OSError):
        store.write([image(fill=1), image(fill=2)])
    monkeypatch.undo()
    assert files_under(shm) == [] and files_under(fallback) == []
    store.close()


def test_stale_directories_of_dead_servers_are_removed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = tmp_path / f"crucible-vision-{dead.pid}-abc"
    stale.mkdir(mode=0o700)
    (stale / "left").write_bytes(b"x")
    live = tmp_path / f"crucible-vision-{os.getppid()}-def"
    live.mkdir(mode=0o700)
    other = tmp_path / "unrelated"
    other.mkdir()
    store = PixelStore(root=tmp_path)
    store.write([image()]).remove()
    assert not stale.exists() and live.exists() and other.exists()
    store.close()


def test_the_default_root_is_memory_backed():
    store = PixelStore()
    assert str(store.root) in ("/dev/shm",) or not os.path.isdir("/dev/shm")


# A small /dev/shm (a container gets 64 MB unless it sets --shm-size). FakeShm makes a folder act as a tmpfs of a
# given size: statvfs reports its free space, and a write past the size is short, then fails with ENOSPC, as on tmpfs.
SIZE = 256 * 256 * 3


def files_under(root):
    return sorted(os.path.join(folder, name) for folder, _, names in os.walk(root) for name in names)


class FakeShm:
    def __init__(self, monkeypatch, sizes, reported=None):
        self.sizes = {str(root): size for root, size in sizes.items()}
        self.reported = reported or {}  # a statvfs value that differs from the real room (another process wrote)
        self.failures = []
        self.hook = None
        self.real_write, self.real_statvfs = os.write, os.statvfs
        monkeypatch.setattr(os, "write", self.write)
        monkeypatch.setattr(os, "statvfs", self.statvfs)

    def root_of(self, path):
        path = os.path.realpath(path)
        return next((root for root in self.sizes if path == root or path.startswith(root + os.sep)), None)

    def room(self, root):
        return self.sizes[root] - sum(os.path.getsize(path) for path in files_under(root))

    def statvfs(self, path):
        root = self.root_of(path)
        if root is None:
            return self.real_statvfs(path)
        free = self.reported.get(root, self.room(root)) // 4096
        return os.statvfs_result((4096, 4096, self.sizes[root] // 4096, free, free, 1000, 1000, 1000, 0, 255))

    def write(self, fd, data):
        root = self.root_of(os.readlink(f"/proc/self/fd/{fd}"))
        if root is None:
            return self.real_write(fd, data)
        if self.hook is not None:
            self.hook(root)
        room = self.room(root)
        if room <= 0:
            self.failures.append(root)
            raise OSError(errno.ENOSPC, "No space left on device")
        return self.real_write(fd, bytes(data[:room]))


@pytest.fixture
def folders(tmp_path, monkeypatch):
    shm, fallback = tmp_path / "shm", tmp_path / "tmp"
    shm.mkdir(), fallback.mkdir()
    # The fallback folder is the temp folder of the server (tempfile.gettempdir(), so TMPDIR works).
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback))
    return shm, fallback


def margin():
    return getattr(pixel_files, "SHM_MARGIN", 0)


def check_bytes(files, images):
    for path, item in zip(files.paths, images, strict=True):
        assert open(path, "rb").read() == item.rgb
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_a_request_that_does_not_fit_in_shm_goes_to_the_temp_folder(folders, monkeypatch):
    shm, fallback = folders
    fake = FakeShm(monkeypatch, {shm: margin() + SIZE + SIZE // 2})
    store = PixelStore(root=shm)
    images = [image(256, 256, fill=1), image(256, 256, fill=2)]
    files = store.write(images)
    # The engine reads the same bytes from the other folder, so the tokens do not change.
    check_bytes(files, images)
    assert all(fake.root_of(path) is None and path.startswith(str(fallback)) for path in files.paths)
    assert files_under(shm) == [] and fake.failures == []
    assert stat.S_IMODE(os.stat(os.path.dirname(files.paths[0])).st_mode) == 0o700
    # The clean-up rules do not change: the files leave after the request, the folders at close.
    files.remove()
    assert files_under(fallback) == []
    store.close()
    assert os.listdir(shm) == [] and os.listdir(fallback) == []


def test_a_request_that_fits_stays_in_shm(folders, monkeypatch):
    shm, fallback = folders
    FakeShm(monkeypatch, {shm: margin() + 2 * SIZE})
    store = PixelStore(root=shm)
    images = [image(256, 256, fill=1), image(256, 256, fill=2)]
    files = store.write(images)
    check_bytes(files, images)
    assert all(path.startswith(str(shm)) for path in files.paths)
    files.remove()
    store.close()
    assert os.listdir(fallback) == []


def test_enospc_during_the_write_removes_the_partial_file_and_writes_once_to_the_temp_folder(folders, monkeypatch):
    shm, fallback = folders
    # statvfs shows room, but another process fills /dev/shm before this write ends.
    fake = FakeShm(monkeypatch, {shm: SIZE + SIZE // 2}, reported={str(shm): 1 << 30})
    store = PixelStore(root=shm)
    images = [image(256, 256, fill=1), image(256, 256, fill=2)]
    files = store.write(images)
    check_bytes(files, images)
    assert fake.failures == [str(shm)]  # one fault, then the fallback folder; no second try in /dev/shm
    assert files_under(shm) == []  # the partial file and the first image left /dev/shm
    assert all(path.startswith(str(fallback)) for path in files.paths)
    files.remove()
    store.close()
    assert os.listdir(shm) == [] and os.listdir(fallback) == []


def test_two_requests_that_fit_alone_but_not_together(folders, monkeypatch):
    shm, fallback = folders
    fake = FakeShm(monkeypatch, {shm: margin() + SIZE + SIZE // 2})
    store = PixelStore(root=shm)
    one, two = image(256, 256, fill=1), image(256, 256, fill=2)
    first_writes, second_done = threading.Event(), threading.Event()
    first = threading.current_thread()

    def hold(root):
        # The first request starts its write, and the second request writes completely before the first continues.
        if threading.current_thread() is first and not first_writes.is_set():
            first_writes.set()
            second_done.wait(5)
    fake.hook = hold
    results = {}

    def second():
        first_writes.wait(5)
        try:
            results["second"] = store.write([two])
        except BaseException as error:  # noqa: BLE001 (the test reports the error)
            results["second"] = error
        finally:
            second_done.set()
    thread = threading.Thread(target=second)
    thread.start()
    try:
        results["first"] = store.write([one])
    except OSError as error:
        results["first"] = error
    thread.join(5)
    assert not isinstance(results["first"], BaseException), results["first"]
    assert not isinstance(results["second"], BaseException), results["second"]
    check_bytes(results["first"], [one])
    check_bytes(results["second"], [two])
    # The first request holds its room in /dev/shm, so the second goes to the temp folder without a fault.
    assert results["first"].paths[0].startswith(str(shm))
    assert results["second"].paths[0].startswith(str(fallback))
    assert fake.failures == []
    results["first"].remove(), results["second"].remove()
    store.close()


def test_the_fallback_writes_one_log_line(folders, monkeypatch, capsys):
    shm, _ = folders
    FakeShm(monkeypatch, {shm: margin() + SIZE // 2})
    store = PixelStore(root=shm)
    for fill in (1, 2, 3):
        store.write([image(256, 256, fill=fill)]).remove()
    lines = [line for line in capsys.readouterr().err.splitlines() if "pixel" in line]
    assert len(lines) == 1 and str(shm) in lines[0] and "--shm-size" in lines[0], lines
    store.close()


def test_no_room_in_both_folders_is_a_503_and_leaves_no_file(folders, monkeypatch):
    shm, fallback = folders
    FakeShm(monkeypatch, {shm: SIZE // 2, fallback: SIZE // 2}, reported={str(shm): 1 << 30, str(fallback): 1 << 30})
    store = PixelStore(root=shm)
    with pytest.raises(ApiError) as caught:
        store.write([image(256, 256)])
    assert caught.value.status == 503
    assert files_under(shm) == [] and files_under(fallback) == []
    store.close()


def test_without_dev_shm_a_full_temp_folder_is_tried_once(folders, monkeypatch):
    _, fallback = folders
    fake = FakeShm(monkeypatch, {fallback: SIZE // 2})
    store = PixelStore(root=fallback)
    with pytest.raises(ApiError):
        store.write([image(256, 256)])
    assert fake.failures == [str(fallback)] and files_under(fallback) == []
    store.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a mode-0500 folder")
def test_a_shm_folder_that_refuses_the_directory_uses_the_temp_folder(folders, capsys):
    shm, fallback = folders
    shm.chmod(0o500)
    try:
        store = PixelStore(root=shm)
        item = image()
        files = store.write([item])
        check_bytes(files, [item])
        assert files.paths[0].startswith(str(fallback))
        assert "EACCES" in capsys.readouterr().err
        files.remove()
        store.close()
    finally:
        shm.chmod(0o700)
