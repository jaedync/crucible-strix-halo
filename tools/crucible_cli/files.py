"""Every write of this tool goes through a Sandbox. A path outside its roots raises OutsideRoots.

The tool never edits the boot loader, the kernel command line or any system file. It never uses sudo.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

# Commands that the runner refuses: root tools and shells.
FORBIDDEN_COMMANDS = frozenset({"sudo", "su", "doas", "pkexec", "sh", "bash", "dash", "zsh", "fish", "csh", "ksh"})
SYSTEM_PREFIXES = ("/boot", "/efi", "/etc", "/usr", "/bin", "/sbin", "/lib", "/var/lib", "/sys", "/proc", "/dev")


class OutsideRoots(Exception):
    pass


def _real(path):
    return os.path.realpath(os.path.abspath(os.fspath(path)))


class Sandbox:
    def __init__(self, roots):
        self.roots = []
        for root in roots:
            real = _real(root)
            if real == "/" or any(real == p or real.startswith(p + "/") for p in SYSTEM_PREFIXES):
                raise ValueError(f"a system folder cannot be a write root: {root}")
            self.roots.append(real)

    def check(self, path):
        real = _real(path)
        if not any(real == r or real.startswith(r + "/") for r in self.roots):
            raise OutsideRoots(f"{path} is outside the folders that this tool may write")
        return Path(real)

    def mkdir(self, path, mode=0o777):
        real = self.check(path)
        real.mkdir(parents=True, exist_ok=True, mode=mode)
        if mode != 0o777:
            os.chmod(real, mode)

    def write_text(self, path, text, executable=False, mode=None):
        real = self.check(path)
        real.parent.mkdir(parents=True, exist_ok=True)
        tmp = real.with_name(real.name + ".tmp")
        tmp.write_text(text)
        if executable:
            os.chmod(tmp, 0o755)
        elif mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, real)

    def write_json(self, path, data):
        self.write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")

    def open_log(self, path):
        real = self.check(path)
        real.parent.mkdir(parents=True, exist_ok=True)
        return open(real, "ab")

    def remove(self, path):
        real = self.check(path)
        if real.exists():
            real.unlink()

    # Trees. A path here is checked as a name inside a root: its folder is resolved, the last name is not, so that a link
    # is removed or moved as a link and never followed out of the roots.
    def _name(self, path):
        path = os.path.abspath(os.fspath(path))
        real = os.path.join(_real(os.path.dirname(path)), os.path.basename(path))
        if not any(real == r or real.startswith(r + "/") for r in self.roots):
            raise OutsideRoots(f"{path} is outside the folders that this tool may write")
        return real

    def rename(self, src, dst):
        os.rename(self._name(src), self._name(dst))

    def replace(self, src, dst):
        os.replace(self._name(src), self._name(dst))

    def unlink(self, path, missing_ok=False):
        Path(self._name(path)).unlink(missing_ok=missing_ok)

    def rmtree(self, path, ignore_errors=False):
        """Remove a folder, a file or a link. A link is removed and not followed."""
        real = self._name(path)
        if os.path.islink(real) or os.path.isfile(real):
            os.unlink(real)
        elif os.path.isdir(real):
            if any(real == r for r in self.roots):
                raise OutsideRoots(f"{path} is a root of this tool and is never removed")
            shutil.rmtree(real, ignore_errors=ignore_errors)

    def rmdir(self, path):
        os.rmdir(self._name(path))

    def chmod(self, path, mode):
        os.chmod(self._name(path), mode)

    def copyfile(self, src, dst):
        shutil.copyfile(src, self._name(dst))

    def mkdtemp(self, prefix, dir):
        return Path(tempfile.mkdtemp(prefix=prefix, dir=self._name(dir)))

    def open_out(self, path, mode="wb"):
        return open(self._name(path), mode)
