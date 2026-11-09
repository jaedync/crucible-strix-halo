"""./crucible update and ./crucible uninstall.

`update` fetches new code and runs it, so every step is a check first and a change last.

  1. Name the release once. The tag comes from --tag or from the newest `vX.Y.Z` tag of the repository list. The commit of
     that tag is read at the same time and pinned: the clone and the release manifest must show the same commit.
  2. Download SHA256SUMS, the manifest and the engine. Check the sha256 of every file against SHA256SUMS. Unpack the engine
     with a size limit and check its sha256 against the manifest. The engine is never run. Nothing of the new release runs
     before every check passed, and a bad or partial file ends the update with no change.
  3. Clone the tag into a staging folder next to the install folder (same file system). A release engine goes in place with
     its release.json; a source build install runs make in the staging folder. The live tree is not touched.
  4. Only then stop the server. A journal file records what must be undone. The folders that are not in git (the work folder,
     the venv, the models, your own files) move to the staging tree, and one atomic exchange (renameat2 RENAME_EXCHANGE)
     puts the new tree in place. The old tree is then in the staging folder.
  5. The new code writes the service unit (`service install --no-start`), the service starts, /health answers, and one
     small request returns a token. Any failure puts the old tree, the old unit and the old files back and starts the old
     release (exit 4, or exit 5 when that fails too). A kill at any point leaves the journal; the next run puts the old
     release back first.

The tool never runs sudo, never touches the model files and never changes the boot settings.

Exit codes of update: 0 updated or up to date, 1 failed before any change, 2 usage, 4 rolled back, 5 rollback failed.
Exit codes of uninstall: 0 done or nothing removed, 1 the service did not stop (nothing removed), 2 usage or not an install.
"""
import ctypes
import errno
import fcntl
import gzip
import hashlib
import http.client
import json
import os
import re
import signal
import urllib.error
import urllib.request
from pathlib import Path

from . import packs, release, service, settings as settings_mod, steps
from .files import SYSTEM_PREFIXES, OutsideRoots, Sandbox

ARCH = "gfx1151"
ROCM_SERIES = "7.2"
REPO_URL_DEFAULT = "https://github.com/jaedync/crucible-strix-halo.git"
RELEASE_BASE_DEFAULT = "https://github.com/jaedync/crucible-strix-halo/releases/download"
TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+\Z")
HEX40 = re.compile(r"^[0-9a-f]{40}\Z")
HEX64 = re.compile(r"^[0-9a-f]{64}\Z")
VERSION_RE = re.compile(r"^([0-9]+(\.[0-9]+)*)?$")
LIB_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
MAX_SMALL_BYTES = 1 << 20        # SHA256SUMS and the manifest
MAX_DOWNLOAD_BYTES = 3 << 30     # the compressed engine
MAX_ENGINE_BYTES = 4 << 30       # the unpacked engine
MIN_FREE_BYTES = 2 << 30
FLUSH_SECONDS = 85               # the longest save of conversation states at a stop
KEEP_OUT = frozenset({".git", "build", "__pycache__"})   # top-level names that never move to the new tree
CACHE_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})   # made by tools; never yours
GENERATED_PATHS = ("ref/fixtures/",)   # made by ref/make_fixtures.py, listed in .gitignore, made again on demand; never yours
PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]*\])?==[A-Za-z0-9._+!-]+$")
WORK_PREFIX = ".{name}-update-"
JOURNAL = ".{name}.update.json"
HOLD = ".crucible/update-hold"
PIP_BEFORE = ".crucible/update-pip-before.txt"
SMOKE = {"model": "crucible", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8, "temperature": 0,
         "stream": True}

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_ROLLED_BACK, EXIT_ROLLBACK_FAILED = 0, 1, 2, 4, 5


class CheckFailed(Exception):
    """A check failed before any change. The text goes to the user."""


class Usage(Exception):
    pass


class RestoreFailed(Exception):
    pass


def crash_point(name):
    """A seam for the tests: they raise here to simulate a kill. It does nothing in a real run."""


def asset_stem(tag):
    return f"crucible-engine-{tag}-{ARCH}-rocm{ROCM_SERIES}-linux-x86_64"


def vkey(tag):
    return tuple(int(x) for x in tag[1:].split("."))


# release list and sums -----------------------------------------------------------------------------------------------
def parse_tags(text):
    """From `git ls-remote --tags`: {"tags": {tag: commit}, "newest": tag or None}. A peeled line (^{}) names the commit."""
    tags = {}
    for line in text.splitlines():
        sha, _, ref = line.partition("\t")
        if not HEX40.match(sha) or not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/"):]
        peeled = name.endswith("^{}")
        name = name[:-3] if peeled else name
        if not TAG_RE.match(name):
            continue
        if peeled or name not in tags:
            tags[name] = sha
    newest = max(tags, key=vkey) if tags else None
    return {"tags": tags, "newest": newest}


def sums_hash(text, name):
    """The sha256 that SHA256SUMS lists for NAME. Refuses a missing name, two different hashes and a bad hash."""
    found = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        file = parts[1][1:] if parts[1].startswith("*") else parts[1]
        if file == name:
            found.add(parts[0])
    if not found:
        raise CheckFailed(f"{name} is not listed in SHA256SUMS.")
    if len(found) > 1:
        raise CheckFailed(f"SHA256SUMS lists {name} with different hashes.")
    want = found.pop()
    if not HEX64.match(want):
        raise CheckFailed(f"SHA256SUMS has a bad line for {name}.")
    return want


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise urllib.error.URLError("a redirect to an address that is not https was refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsOnlyRedirect)


def fetch(url, dest, max_bytes, box, timeout=60):
    """Download URL to DEST through DEST.part. A short or too large file is an error and leaves no DEST."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    try:
        with _OPENER.open(url, timeout=timeout) as resp:
            expected = resp.headers.get("Content-Length")
            total = 0
            with box.open_out(part, "wb") as f:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    total += len(block)
                    if total > max_bytes:
                        raise OSError(f"the file is larger than {max_bytes} bytes")
                    f.write(block)
            if expected is not None and expected.isdigit() and int(expected) != total:
                raise OSError("the download stopped early")
    except (OSError, http.client.HTTPException) as exc:
        box.unlink(part, missing_ok=True)
        raise CheckFailed(f"The download of {url} failed ({exc}).") from None
    box.replace(part, dest)


# swap of two folders -------------------------------------------------------------------------------------------------
def exchange(a, b):
    """Swap two paths in one atomic step (renameat2 with RENAME_EXCHANGE). Raises OSError when the system cannot."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        fn = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, "renameat2 is not available") from None
    if fn(-100, os.fsencode(a), -100, os.fsencode(b), 2) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


NO_EXCHANGE = (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP)


def swap_tmp(b):
    b = Path(b)
    return b.with_name(b.name + ".swap")


def _box_for(*paths):
    return Sandbox([os.path.dirname(os.path.realpath(p)) for p in paths])


def swap_dirs(a, b, box=None):
    """Put B where A was and A where B was. One atomic exchange when the file system has it, else three renames (A, B, tmp)."""
    box = box or _box_for(a, b)
    try:
        exchange(a, b)
        return
    except OSError as exc:
        if exc.errno not in NO_EXCHANGE:
            raise
    tmp = swap_tmp(b)
    box.rename(a, tmp)
    crash_point("swap:1")
    box.rename(b, a)
    crash_point("swap:2")
    box.rename(tmp, b)


def heal_swap(a, b, box=None):
    """Finish or undo the three-rename swap that a kill stopped, so that A and B are folders again."""
    box = box or _box_for(a, b)
    tmp = swap_tmp(b)
    if not tmp.exists():
        return
    if not Path(a).exists():
        box.rename(tmp, a)     # killed after the first rename: the old tree goes back to A
    else:
        box.rename(tmp, b)     # killed after the second rename: the old tree goes to B


def _ident(path):
    st = os.stat(path)
    return [st.st_dev, st.st_ino]


# git -----------------------------------------------------------------------------------------------------------------
def top_level_names(text):
    return {n for n in text.split("\0") if n}


class Update:
    def __init__(self, host, repo, args, ask):
        self.host, self.args, self.ask = host, args, ask
        self.real = Path(os.path.realpath(repo))
        self.parent = self.real.parent
        self.name = self.real.name
        self.say = host.out
        self.journal_path = self.parent / JOURNAL.format(name=self.name)
        self.env = {**host.env, "GIT_TERMINAL_PROMPT": "0"}
        self.repo_url = host.env.get("CRUCIBLE_REPO_URL") or REPO_URL_DEFAULT
        self.sums_pin = host.env.get("CRUCIBLE_SUMS_SHA256") or ""
        self.base_override = host.env.get("CRUCIBLE_RELEASE_BASE") or ""
        self.s = None
        try:
            self.box = Sandbox([self.parent, service.unit_dir(host)])
        except ValueError:
            self.box = None

    # helpers ----------------------------------------------------------------------------------------------------
    def git(self, *argv, cwd=None):
        return self.host.runner.run(["git", *argv], cwd=str(cwd or self.real), env=self.env)

    def run_new(self, *argv, echo=False):
        """Run the ./crucible command of the folder that is now in place."""
        return self.host.runner.run([self.host.python_exe, str(self.real / "crucible"), *argv], cwd=str(self.real),
                                    env=dict(self.env), echo=echo)

    def read(self, rel):
        try:
            return (self.real / rel).read_text().strip()
        except OSError:
            return ""

    # entry ------------------------------------------------------------------------------------------------------
    def run(self):
        host = self.host
        if self.box is None:
            self.say(f"{self.real} is inside a system folder. Update does not change files there. Nothing was changed.")
            return EXIT_FAILED
        if not self.repo_url.startswith(("https://", "file://")):
            raise Usage("CRUCIBLE_REPO_URL must start with https:// or file://.")
        if self.base_override and not self.base_override.startswith(("https://", "file://")):
            raise Usage("CRUCIBLE_RELEASE_BASE must start with https:// or file://.")
        if not self.args.yes and not self.args.check and self.ask is None:
            self.say("Update asks one question, and no terminal is attached. Add --yes to update without the question.")
            return EXIT_USAGE
        try:
            lock = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            self.say(f"Cannot open {self.parent}: {exc}. Nothing was changed.")
            return EXIT_FAILED
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.say("Another update is running. Nothing was changed.")
                return EXIT_FAILED
            return self._locked()
        finally:
            os.close(lock)

    def _locked(self):
        try:
            code = self.recover()
            if code is not None:
                return code
            if not ((self.real / ".git").is_dir() and (self.real / "crucible").is_file()):
                self.say(f"{self.real} is not a git checkout of Crucible. Nothing was changed. "
                         "To get a normal install, run the installer again (see the README).")
                return EXIT_FAILED
            self.clean_stale()
            return self.update()
        except CheckFailed as exc:
            self.say(f"{exc} Nothing was changed.")
            return EXIT_FAILED

    # update -----------------------------------------------------------------------------------------------------
    def update(self):
        host, args, say = self.host, self.args, self.say
        self.s = settings_mod.build(self.real, host)
        if self.git("status", "--porcelain", "--untracked-files=no").out.strip():
            raise CheckFailed("You have local changes in tracked files. Commit them or undo them, then run this again.")
        head = self.git("rev-parse", "HEAD")
        if head.rc != 0 or not HEX40.match(head.out.strip()):
            raise CheckFailed("This folder has no commit to update from.")
        old_commit = head.out.strip()
        old_tag = self.current_tag()
        target, target_commit = self.resolve()
        crash_point("resolved")
        shown = old_tag or old_commit[:12]
        if old_commit == target_commit:
            say(f"Crucible is up to date ({shown}).")
            return EXIT_OK
        if not args.tag and self.read(HOLD) == target:
            say(f"Release {target} is on hold because it did not start on this machine, and the update went back. "
                f"To try it again, run: ./crucible update --tag {target}")
            return EXIT_OK
        if args.check:
            say(f"Update available: {shown} to {target}. Run ./crucible update to install it.")
            return EXIT_OK
        self.refuse_own_files()
        free = host.disk_free(self.parent)
        if free < MIN_FREE_BYTES:
            raise CheckFailed(f"There is not enough disk space next to the install folder ({free >> 20} MiB free, "
                              f"{MIN_FREE_BYTES >> 20} MiB needed).")
        if not os.access(self.parent, os.W_OK | os.X_OK):
            raise CheckFailed(f"Cannot write next to the install folder: {self.parent}. The new release is staged there.")
        release_mode = (not args.source_build) and release.engine_release(self.real) is not None
        work = self.box.mkdtemp(WORK_PREFIX.format(name=self.name), self.parent)
        self.box.chmod(work, 0o700)
        keep_work = False
        try:
            stage = work / "new"
            self.clone(stage, target, target_commit)
            verdict = self.relation(stage, old_commit, target)
            if verdict:
                say(verdict)
                return EXIT_OK
            engine = self.get_release(work, target, target_commit) if release_mode else None
            carry = self.carry_list(stage, target)
            if engine:
                self.place_engine(stage, engine, target)
            else:
                self.build(stage, work)
            pip_req = self.python_change(stage)
            crash_point("staged")
            if not args.yes:
                if not self.confirm(shown, target, release_mode, pip_req is not None):
                    say("Nothing was changed.")
                    return EXIT_OK
            self.refuse_own_files()
            keep_work = True
            return self.swap_in(work, stage, carry, target, shown, pip_req)
        finally:
            if not keep_work or not self.journal_path.exists():
                self.box.rmtree(work, ignore_errors=True)

    def own_files(self):
        """Paths of your files (not in git, ignored or not) inside the folders that git tracks. The swap deletes them with the
        old tree. Files at the top level are carried to the new tree and are not listed. Tool caches and GENERATED_PATHS are not listed."""
        tree = self.git("ls-tree", "-d", "--name-only", "-z", "HEAD").out
        folders = [n for n in tree.split("\0") if n and n not in KEEP_OUT]
        if not folders:
            return []
        res = self.git("--literal-pathspecs", "ls-files", "-z", "--others", "--", *folders)
        if res.rc != 0:
            raise CheckFailed("Could not list your own files in the install folder, so the update cannot tell what it would delete.")
        mine = []
        for path in res.out.split("\0"):
            parts = path.split("/")
            if path and not (CACHE_NAMES & set(parts)) and not path.endswith((".pyc", ".pyo")) and not path.startswith(GENERATED_PATHS):
                mine.append(path)
        return sorted(mine)

    def refuse_own_files(self):
        mine = self.own_files()
        if mine:
            shown = ", ".join(mine[:8]) + (f" and {len(mine) - 8} more" if len(mine) > 8 else "")
            raise CheckFailed(f"You have {len(mine)} file(s) of your own inside the folders that the update replaces ({shown}). "
                              "The update would delete them. Move them out of the install folder, or commit them, then run this again.")

    def current_tag(self):
        res = self.git("tag", "--points-at", "HEAD", "--list", "v[0-9]*")
        tags = [t for t in res.out.split() if TAG_RE.match(t)]
        return max(tags, key=vkey) if tags else ""

    def resolve(self):
        """The target tag and its commit, read once. Everything after this uses the pair."""
        res = self.git("ls-remote", "--tags", self.repo_url)
        if res.rc != 0:
            raise CheckFailed(f"Could not read the release list from {self.repo_url}. Check the network and try again.")
        found = parse_tags(res.out)
        tag = self.args.tag or found["newest"]
        if not tag or tag not in found["tags"]:
            raise CheckFailed(f"No release {tag or ''} was found in the release list.".replace("release  ", "release "))
        return tag, found["tags"][tag]

    # download and checks ----------------------------------------------------------------------------------------
    def release_base(self, tag):
        return (self.base_override or f"{RELEASE_BASE_DEFAULT}/{tag}").rstrip("/")

    def get_release(self, work, tag, pinned):
        """Download and check the release engine of TAG. Returns the unpacked engine path and the manifest."""
        say = self.say
        dl = work / "dl"
        dl.mkdir()
        stem = asset_stem(tag)
        gz, needs = f"{stem}.gz", f"{stem}.needs"
        base = self.release_base(tag)
        say(f"Downloading the release files of {tag} ...")
        try:
            fetch(f"{base}/SHA256SUMS", dl / "SHA256SUMS", MAX_SMALL_BYTES, self.box)
            fetch(f"{base}/{needs}", dl / needs, MAX_SMALL_BYTES, self.box)
            fetch(f"{base}/{gz}", dl / gz, MAX_DOWNLOAD_BYTES, self.box)
        except CheckFailed as exc:
            raise CheckFailed(f"{exc} To build the engine from source instead, run: ./crucible update --source-build") from None
        sums = dl / "SHA256SUMS"
        if self.sums_pin and sha256_file(sums) != self.sums_pin:
            raise CheckFailed("SHA256SUMS does not match the hash that you pinned with CRUCIBLE_SUMS_SHA256.")
        text = sums.read_text(errors="replace")
        for name in (gz, needs):
            want, have = sums_hash(text, name), sha256_file(dl / name)
            if want != have:
                raise CheckFailed(f"The sha256 of {name} does not match SHA256SUMS (expected {want}, got {have}). "
                                  "The download is bad or partial.")
            say(f"  sha256 ok: {name}")
        manifest = self.read_manifest(dl / needs, tag)
        if manifest["commit"] != pinned:
            raise CheckFailed(f"The commit in the release manifest ({manifest['commit']}) is not the commit of tag {tag} ({pinned}).")
        engine = self.unpack(dl / gz, dl / "engine", manifest["binary_sha256"])
        say("  engine sha256 ok after unpacking")
        self.check_target(manifest)
        return {"path": engine, "manifest": manifest, "asset": gz}

    def read_manifest(self, path, tag):
        keys, libs = {}, []
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "need":
                libs.append(parts[1])
            elif len(parts) == 2:
                keys.setdefault(parts[0], parts[1])
        if keys.get("format") != "1":
            raise CheckFailed("The release manifest has an unknown format.")
        if keys.get("tag") != tag:
            raise CheckFailed(f"The release manifest is for another release ({keys.get('tag')}), not {tag}.")
        if keys.get("arch") != ARCH:
            raise CheckFailed(f"The release manifest is for another GPU ({keys.get('arch')}), not {ARCH}.")
        if keys.get("rocm") != ROCM_SERIES:
            raise CheckFailed(f"The release manifest is for ROCm {keys.get('rocm')}, not {ROCM_SERIES}.")
        if not HEX40.match(keys.get("commit", "")):
            raise CheckFailed("The release manifest has a bad commit.")
        if not HEX64.match(keys.get("binary_sha256", "")):
            raise CheckFailed("The release manifest has a bad engine hash.")
        for key in ("glibc_min", "glibcxx_min"):
            if not VERSION_RE.match(keys.get(key, "")):
                raise CheckFailed(f"The release manifest has a bad {key}.")
        if any(not LIB_RE.match(lib) for lib in libs):
            raise CheckFailed("The release manifest names a library with a bad name.")
        return {**keys, "needs": libs}

    def unpack(self, gz_path, out_path, want_sha):
        h = hashlib.sha256()
        total = 0
        try:
            with gzip.open(gz_path, "rb") as src, self.box.open_out(out_path, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    total += len(block)
                    if total > MAX_ENGINE_BYTES:
                        raise CheckFailed(f"The unpacked engine is too large (more than {MAX_ENGINE_BYTES} bytes).")
                    h.update(block)
                    dst.write(block)
        except (OSError, EOFError) as exc:
            raise CheckFailed(f"The engine file could not be unpacked ({exc}).") from None
        if h.hexdigest() != want_sha:
            raise CheckFailed("The unpacked engine does not match the sha256 in the release manifest.")
        self.box.chmod(out_path, 0o700)
        return out_path

    def check_target(self, manifest):
        """The libraries and the C library that the engine needs. The engine file is never run; ldd is not used."""
        try:
            libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
        except (ValueError, OSError):
            libc = ""
        m = re.match(r"glibc ([0-9]+(?:\.[0-9]+)*)", libc)
        floor = manifest.get("glibc_min", "")
        if m and floor and tuple(map(int, m.group(1).split("."))) < tuple(map(int, floor.split("."))):
            raise CheckFailed(f"This system has glibc {m.group(1)}. The release engine needs glibc {floor} or newer. "
                              "To build the engine from source instead, run: ./crucible update --source-build")
        missing = [lib for lib in manifest["needs"] if not self.lib_present(lib)]
        if missing:
            raise CheckFailed(f"These libraries are missing: {' '.join(missing)}. Install the packages that hold them "
                              "(the install page of your system lists them), then run this again.")

    def lib_present(self, name):
        listing = ""
        for exe in (self.host.which("ldconfig"), "/sbin/ldconfig", "/usr/sbin/ldconfig"):
            if exe and (os.path.exists(exe) or exe == self.host.which("ldconfig")):
                listing = self.host.runner.run([exe, "-p"], env=dict(self.env)).out
                break
        if re.search(r"(?m)^\s*" + re.escape(name) + r"\s", listing):
            return True
        rocm = Path(self.host.env.get("ROCM_PATH") or self.host.rocm)
        dirs = [rocm / "lib", rocm / "lib64"]
        if not listing:
            dirs += [Path(p) for p in ("/usr/lib", "/usr/lib64", "/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")]
        return any((d / name).exists() for d in dirs)

    # staging ----------------------------------------------------------------------------------------------------
    def clone(self, stage, tag, pinned):
        self.say(f"Getting the source of {tag} ...")
        res = self.host.runner.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--branch", tag, "--",
                                    self.repo_url, str(stage)], env=self.env)
        if res.rc != 0:
            raise CheckFailed(f"git clone of {tag} failed. Check the network and try again.")
        got = self.git("rev-parse", f"refs/tags/{tag}^{{commit}}", cwd=stage).out.strip()
        if got != pinned:
            raise CheckFailed(f"The tag {tag} moved while the update ran (it was {pinned}, it is now {got}). "
                              "Run the update again.")
        head = self.git("rev-parse", "HEAD", cwd=stage).out.strip()
        if head != pinned:
            raise CheckFailed(f"The files that were checked out are not the commit of tag {tag} (a branch with the same name?). "
                              "Run the update again later.")
        if not (stage / "crucible").is_file():
            raise CheckFailed(f"The source of {tag} has no ./crucible command.")
        marker = self.real / ".git/crucible-install"
        text = marker.read_text() if marker.is_file() else f"installed by crucible update, tag {tag}\n"
        self.box.write_text(stage / ".git/crucible-install", text)

    def relation(self, stage, old_commit, target):
        """None when the update may go on, else the sentence that ends it with no change.

        The history of the live checkout is read into the staging clone (objects only; the live repository is not touched)."""
        res = self.host.runner.run(["git", "-c", "protocol.file.allow=always", "fetch", "--quiet", "--no-tags", str(self.real), "HEAD"],
                                   cwd=str(stage), env=self.env)
        if res.rc != 0 or self.git("cat-file", "-e", f"{old_commit}^{{commit}}", cwd=stage).rc != 0:
            raise CheckFailed("The history of this checkout could not be read, so the tool cannot tell how it relates to the release.")
        target_commit = self.git("rev-parse", "HEAD", cwd=stage).out.strip()
        if self.git("merge-base", "--is-ancestor", target_commit, old_commit, cwd=stage).rc == 0 and not self.args.tag:
            return f"This checkout is ahead of the newest release ({target}). Nothing to update."
        if self.git("merge-base", "--is-ancestor", old_commit, target_commit, cwd=stage).rc != 0:
            contains = self.git("tag", "--contains", old_commit, "--list", "v[0-9]*", cwd=stage).out.split()
            if not (self.args.tag and contains):
                raise CheckFailed("This checkout is not on a release line: it has commits that the release does not have.")
        return None

    def carry_list(self, stage, tag):
        """Top-level names of the live tree that are not in git. They move to the new tree at the swap."""
        old_tracked = top_level_names(self.git("ls-tree", "--name-only", "-z", "HEAD").out)
        new_tracked = top_level_names(self.git("ls-tree", "--name-only", "-z", "HEAD", cwd=stage).out)
        carry = sorted(n for n in os.listdir(self.real) if n not in old_tracked and n not in KEEP_OUT)
        clash = sorted(n for n in carry if n in new_tracked or (stage / n).exists() or os.sep in n or n in (".", ".."))
        if clash:
            raise CheckFailed(f"Release {tag} has files named {', '.join(clash)}, and you have your own files with the "
                              "same names. Move your files away, then run this again.")
        return carry

    def place_engine(self, stage, engine, tag):
        eng = stage / "build/engine"
        eng.mkdir(parents=True, exist_ok=True)
        self.box.copyfile(engine["path"], eng / "crucible.tmp")
        self.box.chmod(eng / "crucible.tmp", 0o755)
        self.box.replace(eng / "crucible.tmp", eng / "crucible")
        m = engine["manifest"]
        record = {"format": 1, "tag": tag, "commit": m["commit"], "engine_sha256": m["binary_sha256"], "arch": ARCH,
                  "rocm": ROCM_SERIES, "asset": engine["asset"], "glibc_min": m.get("glibc_min", ""),
                  "glibcxx_min": m.get("glibcxx_min", "")}
        self.box.write_text(eng / "release.json", json.dumps(record) + "\n")

    def build(self, stage, work):
        """Source build install: make in the staging folder. The live tree and the running server are not touched."""
        h, say = self.host, self.say
        make = h.env.get("CRUCIBLE_MAKE") or h.which("make") or "make"
        argv = [make, f"-j{self.s.jobs}"]
        if h.which("nice"):
            argv = ["nice", "-n", "19", *argv]
        if h.which("systemd-run") and service.manager_ok(h):
            argv = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=4G", *argv]
        env = dict(self.env)
        say("Building the engine from source in the staging folder (about 16 minutes). The server keeps running.")
        with self.box.open_log(work / "build.log") as log:
            res = h.runner.run(argv, cwd=str(stage), env=env, log=log)
            if res.rc != 0 and "argument facts drift" in res.out:
                say("The build stopped with 'argument facts drift'. Building again with SPLIT=0 (slower).")
                res = h.runner.run(argv + ["SPLIT=0"], cwd=str(stage), env=env, log=log)
        if res.rc != 0 or not (stage / "build/engine/crucible").is_file():
            raise CheckFailed("The build of the new release failed. " + res.out[-300:].strip())

    def python_change(self, stage):
        """The path of the new requirements file when the Python packages must change, else None. It changes nothing."""
        new_req, old_req = stage / "requirements-cpu.txt", self.real / "requirements-cpu.txt"
        if not new_req.is_file() or not self.s.venv_python.exists():
            return None
        if old_req.is_file() and sha256_file(old_req) == sha256_file(new_req):
            return None
        return new_req

    def pip_freeze(self, work, venv_python=None):
        """The exact pins of the venv (lines name==version), with pip, setuptools and wheel (--all). Nothing is changed."""
        log_path = work / "freeze.txt"
        with self.box.open_log(log_path) as log:
            res = self.host.runner.run([str(venv_python or self.s.venv_python), "-m", "pip", "freeze", "--all",
                                        "--disable-pip-version-check"],
                                       cwd=str(self.real), env=dict(self.env), log=log)
        if res.rc != 0:
            raise CheckFailed("pip could not list the installed Python packages, so the update cannot undo a change to them: "
                              + res.out[-200:].strip())
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(sorted(ln.strip() for ln in lines if PIN_RE.match(ln.strip()))) + "\n"

    def pip_install(self, req, work, name):
        with self.box.open_log(work / name) as log:
            return self.host.runner.run([str(self.s.venv_python), "-m", "pip", "install", "-r", str(req)],
                                        cwd=str(work), env=dict(self.env), log=log)

    def pip_extra(self, freeze, venv_python):
        """The names of the installed packages that the record does not hold (the new requirements added them). Returns a
        list, or a string with the reason when pip cannot list the packages."""
        def norm(name):
            return re.sub(r"[-_.]+", "-", name.split("[")[0]).lower()
        want = {norm(ln.split("==")[0]) for ln in freeze.splitlines() if ln.strip()}
        work = self.real / ".crucible"
        self.box.unlink(work / "freeze.txt", missing_ok=True)
        try:
            now = self.pip_freeze(work, venv_python)
        except CheckFailed as exc:
            return str(exc)
        finally:
            self.box.unlink(work / "freeze.txt", missing_ok=True)
        return sorted({norm(ln.split("==")[0]) for ln in now.splitlines() if ln.strip()} - want)

    def pip_put_back(self, freeze, venv_python):
        """Install the recorded pins again (--no-deps: the list is complete). Returns "" or the reason it failed. On a
        failure the list stays in .crucible/update-pip-before.txt, so that the user can run the shown command."""
        if not freeze or not freeze.strip() or not Path(venv_python).exists():
            return ""
        path = self.real / PIP_BEFORE
        self.box.write_text(path, freeze)
        extra = self.pip_extra(freeze, venv_python)
        if isinstance(extra, str):
            return extra
        if extra:
            with self.box.open_log(path.with_suffix(".log")) as log:
                res = self.host.runner.run([str(venv_python), "-m", "pip", "uninstall", "--yes", *extra],
                                           cwd=str(self.real), env=dict(self.env), log=log)
            if res.rc != 0:
                return res.out[-200:].strip() or "pip could not remove the packages that the update added"
        with self.box.open_log(path.with_suffix(".log")) as log:
            res = self.host.runner.run([str(venv_python), "-m", "pip", "install", "--no-deps", "-r", str(path)],
                                       cwd=str(self.real), env=dict(self.env), log=log)
        if res.rc != 0:
            return res.out[-200:].strip() or "pip failed"
        self.box.unlink(path, missing_ok=True)
        self.box.unlink(path.with_suffix(".log"), missing_ok=True)
        return ""

    def confirm(self, shown, target, release_mode, python_changes=False):
        load = 120
        try:
            load = float(json.loads((self.s.work / "setup-state.json").read_text()).get("load_seconds") or load)
        except (OSError, ValueError, AttributeError):
            pass
        self.say(f"Ready: {target} is downloaded and checked. Your models and settings stay as they are.")
        if python_changes:
            self.say("The Python packages in .venv change after your yes. The update records the old versions and puts them "
                     "back if it rolls back.")
        answer = self.ask(f"Update {shown} to {target} now? The server stops for about {int(FLUSH_SECONDS + load)} s at most", "y")
        return answer.strip().lower() in ("y", "yes")

    # journal ----------------------------------------------------------------------------------------------------
    def write_journal(self, data):
        tmp = self.journal_path.with_name(self.journal_path.name + ".tmp")
        with self.box.open_out(tmp, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        self.box.replace(tmp, self.journal_path)
        fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def read_journal(self):
        try:
            data = json.loads(self.journal_path.read_text())
            ok = (isinstance(data, dict) and data.get("format") == 1 and data.get("dir") == str(self.real)
                  and isinstance(data.get("work"), str) and isinstance(data.get("carry"), list)
                  and isinstance(data.get("ident"), list) and len(data["ident"]) == 2
                  and data.get("mode") in ("service-active", "service-stopped", "pidfile", "none")
                  and all(isinstance(n, str) and n and os.sep not in n and n not in (".", "..") for n in data["carry"])
                  and isinstance(data.get("files"), dict) and isinstance(data.get("pip_freeze"), (str, type(None))))
            work = Path(data["work"]) if ok else None
            if not ok or work.parent != self.parent or not work.name.startswith(WORK_PREFIX.format(name=self.name)):
                raise ValueError("bad fields")
        except (OSError, ValueError, KeyError):
            raise CheckFailed(f"The update journal {self.journal_path} cannot be read, so the tool cannot tell how to undo "
                              "an earlier update. Look at the folders next to the install folder, "
                              "or install again with the installer.") from None
        return data

    # recovery and rollback --------------------------------------------------------------------------------------
    def recover(self):
        """None when no earlier update was cut. Else put the old release back and return the exit code."""
        if not self.journal_path.exists():
            return None
        j = self.read_journal()
        self.say("An earlier update was stopped before it finished. Putting the old release back first.")
        ok = self.restore(j, hold=None)
        if ok:
            self.say("The old release runs again. To try the update again, run: ./crucible update")
            return EXIT_ROLLED_BACK
        return EXIT_ROLLBACK_FAILED

    def clean_stale(self):
        prefix = WORK_PREFIX.format(name=self.name)
        for p in self.parent.iterdir():
            if p.name.startswith(prefix) and p.is_dir() and not p.is_symlink() and p.stat().st_uid == os.getuid():
                self.box.rmtree(p, ignore_errors=True)

    def restore(self, j, hold):
        """Put the old tree, the old unit and the old files back, and start the old release as it was. Returns True when it runs.

        HOLD is the tag to hold (a new release that failed), or None."""
        host, say = self.host, self.say
        stage = Path(j["work"]) / "new"
        try:
            self.stop_all(j)
            heal_swap(self.real, stage, self.box)
            old_root, new_root = self.roots(j, stage)
            for name in j["carry"]:
                if not (old_root / name).exists() and not (old_root / name).is_symlink() and (new_root / name).exists():
                    self.box.rename(new_root / name, old_root / name)
                elif not (old_root / name).exists() and (new_root / name).is_symlink():
                    self.box.rename(new_root / name, old_root / name)
            if old_root != self.real:
                swap_dirs(self.real, stage, self.box)
            self.restore_files(j)
            pip_error = self.restore_pip(j)
            if hold:
                self.box.write_text(self.real / HOLD, hold + "\n")
        except (OSError, RestoreFailed, CheckFailed) as exc:
            say(f"The old release could not be put back: {exc}. The journal {self.journal_path} stays. "
                "Run ./crucible update again to try again.")
            return False
        s = settings_mod.build(self.real, host)
        if pip_error:
            say(f"The old Python packages could not be put back ({pip_error}). To put them back, run: "
                f"{self.real}/.venv/bin/python -m pip install --no-deps -r {self.real}/{PIP_BEFORE}")
        running = self.start_old(j, s)
        self.box.unlink(self.journal_path, missing_ok=True)
        self.box.rmtree(j["work"], ignore_errors=True)
        if not running:
            say("The old release is back, but the server does not run. See: ./crucible service status")
            if j["mode"].startswith("service"):
                service.status(host, s)
            return False
        return True

    def roots(self, j, stage):
        """(old tree, new tree) by identity: the old tree is the folder that has the recorded inode."""
        want = j["ident"]
        for p, other in ((self.real, stage), (stage, self.real)):
            if p.exists() and _ident(p) == want:
                return p, other
        raise RestoreFailed("the old tree was not found")

    def stop_all(self, j):
        host = self.host
        if j["mode"].startswith("service") or service.installed(host):
            service._ctl(host, "stop", service.UNIT)
            service._ctl(host, "reset-failed", service.UNIT)
        pid = self.pid()
        if pid is not None and host.runner.alive(pid):
            host.runner.terminate(pid)

    def pid(self):
        try:
            return int((self.real / ".crucible/server.pid").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def restore_files(self, j):
        for rel, (text, mode) in j["files"].items():
            path = rel_path(self.real, rel)
            if text is None:
                continue
            if path.is_file() and path.read_text() == text and (path.stat().st_mode & 0o7777) == mode:
                continue
            self.box.write_text(path, text, mode=mode)
        unit = j.get("unit")
        if unit and unit.get("text") is not None:
            path = service.unit_path(self.host)
            if not path.is_file() or path.read_text() != unit["text"]:
                self.box.write_text(path, unit["text"], mode=0o644)
            service._ctl(self.host, "daemon-reload")

    def restore_pip(self, j):
        """Put the Python packages of the old release back (the pins that the journal holds). Returns "" or the reason."""
        return self.pip_put_back(j.get("pip_freeze"), settings_mod.build(self.real, self.host).venv_python)

    def start_old(self, j, s):
        mode = j["mode"]
        if mode == "service-active":
            return service.start(self.host, s, restart=True) == 0
        if mode == "pidfile":
            return self.run_new("start", echo=True).rc == 0
        return True

    # quiesce, swap, start, commit -------------------------------------------------------------------------------
    def detect_mode(self):
        host = self.host
        if service.installed(host):
            res = service._ctl(host, "is-active", service.UNIT)
            return "service-active" if res.rc == 0 and res.out.strip().splitlines()[-1:] == ["active"] else "service-stopped"
        pid = self.pid()
        if pid is not None and host.runner.alive(pid):
            return "pidfile"
        return "none"

    def snapshot_files(self):
        out = {}
        for rel in (".crucible/serve.sh", ".crucible/config.json"):
            p = self.real / rel
            try:
                out[rel] = [p.read_text(), p.stat().st_mode & 0o7777]
            except OSError:
                out[rel] = [None, 0]
        return out

    def swap_in(self, work, stage, carry, target, shown, pip_req=None):
        host, say = self.host, self.say
        mode = self.detect_mode()
        unit = None
        if mode.startswith("service"):
            try:
                unit = {"text": service.unit_path(host).read_text()}
            except OSError:
                unit = {"text": None}
        freeze = self.pip_freeze(work) if pip_req is not None else None
        j = {"format": 1, "dir": str(self.real), "work": str(work), "ident": _ident(self.real), "carry": carry,
             "mode": mode, "files": self.snapshot_files(), "unit": unit, "from": shown, "to": target, "pip_freeze": freeze}
        self.write_journal(j)
        old_term = None
        try:
            old_term = signal.signal(signal.SIGTERM, _raise_interrupt)
        except (ValueError, OSError):
            pass
        reason = ""
        try:
            try:
                if pip_req is not None:
                    say("The Python requirements changed. Installing them in .venv ...")
                    res = self.pip_install(pip_req, work, "pip.log")
                    crash_point("pipped")
                    if res.rc != 0:
                        back = self.pip_put_back(freeze, self.s.venv_python)
                        self.box.unlink(self.journal_path, missing_ok=True)
                        raise CheckFailed("pip could not install the new requirements: " + res.out[-200:].strip()
                                          + (f" The old packages could not be put back either: {back}" if back else ""))
                if mode in ("service-active", "pidfile"):
                    code = (service.stop(host, self.s) if mode == "service-active" else steps.stop_pidfile(host, self.s))
                    if code != 0:
                        self.box.unlink(self.journal_path, missing_ok=True)
                        raise CheckFailed("The server did not stop. Wait a minute, or look at ./crucible service status.")
                crash_point("stopped")
                for i, name in enumerate(carry, 1):
                    self.box.rename(self.real / name, stage / name)
                    crash_point(f"carry:{i}")
                crash_point("carried")
                swap_dirs(self.real, stage)
                crash_point("swapped")
                reason = self.start_new(mode, target)
                if not reason:
                    crash_point("started")
                    return self.commit(work, j, target)
            except CheckFailed:
                raise
            except Exception as exc:  # noqa: BLE001 - every failure after the stop must put the old release back
                reason = f"The update failed: {type(exc).__name__}: {exc}"
            except KeyboardInterrupt:
                reason = "The update was stopped."
        finally:
            if old_term is not None:
                try:
                    signal.signal(signal.SIGTERM, old_term)
                except (ValueError, OSError):
                    pass
        say(reason)
        say(f"Going back to {shown} ...")
        ok = self.restore(j, hold=target)
        if ok:
            say(f"The update to {target} was rolled back. {shown} runs again. "
                f"{target} stays on hold until a newer release exists; to try it anyway, run: ./crucible update --tag {target}")
            say(f"If your shell was inside {self.real}, it still sits in the old folder. Run: cd {self.real}")
            return EXIT_ROLLED_BACK
        say("The rollback failed. Look at: ./crucible service status    and: ./crucible service logs")
        return EXIT_ROLLBACK_FAILED

    def start_new(self, mode, target):
        """Refresh the unit with the new code, start the server and prove it. Returns "" or the sentence that says what failed."""
        host = self.host
        self.s = settings_mod.build(self.real, host)
        if mode.startswith("service"):
            res = self.run_new("service", "install", "--no-start")
            if res.rc != 0:
                return "The new release could not write its service unit (service install failed): " + res.out[-200:].strip()
        crash_point("refreshed")
        if mode == "service-active":
            if service.start(host, self.s, restart=True) != 0:
                return f"The service did not start with {target}, or /health did not answer 200."
        elif mode == "pidfile":
            if self.run_new("start", echo=True).rc != 0:
                return f"The server did not start with {target}."
        else:
            return ""
        res = host.http.chat_stream(f"{self.s.base_url}/v1/chat/completions", dict(SMOKE), key_file=self.s.api_key_file)
        if not res.get("ok"):
            return f"The test request got no token with {target}: {res.get('error', '')}"
        return ""

    def commit(self, work, j, target):
        self.box.unlink(self.real / HOLD, missing_ok=True)
        self.box.unlink(self.journal_path, missing_ok=True)
        self.box.rmtree(work, ignore_errors=True)
        mode = j["mode"]
        self.say(f"Updated Crucible {j['from']} to {target}.")
        self.say(f"If your shell was inside {self.real}, it still sits in the old folder. Run: cd {self.real}")
        if mode in ("service-active", "pidfile"):
            self.say("The server runs the new release and answered a test request.")
        else:
            self.say("The server was not running, so it was not started. Start it with: ./crucible start")
        return EXIT_OK


def rel_path(root, rel):
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise RestoreFailed("a bad path in the journal")
    return Path(root) / p


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


ROLLBACK_MESSAGE = ("There is no --rollback. When a new release does not start, the update rolls back by itself and the old "
                    "release runs again. To go back to an older release, name it: ./crucible update --tag vX.Y.Z "
                    "(the release is pinned and checked like a new one, and it rolls back by itself if it does not start). "
                    "Nothing was changed.")


def run_update(host, repo, args, ask):
    if getattr(args, "rollback", False):
        host.out(ROLLBACK_MESSAGE)
        return EXIT_USAGE
    try:
        return Update(host, repo, args, ask).run()
    except Usage as exc:
        host.out(str(exc))
        return EXIT_USAGE


# uninstall -----------------------------------------------------------------------------------------------------------
def _size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for n in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, n)).st_size
            except OSError:
                pass
    return f"{total / 1e9:.1f} GB" if total >= 1e8 else f"{total / 1e6:.0f} MB"


def _within(child, parent):
    child, parent = os.path.realpath(child), os.path.realpath(parent)
    return child != parent and child.startswith(parent + os.sep)


def _keep_names(real, items):
    """(names, problem). NAMES: the top-level names of REAL that hold an item to keep, found by real path (a link, or a
    models folder that is the install folder itself, does not hide an item). PROBLEM: a sentence when an item cannot be
    kept because it is the install folder or holds it."""
    real_s = os.path.realpath(real)
    names = set()
    for item in items:
        rp = os.path.realpath(item)
        if rp == real_s or real_s.startswith(rp + os.sep):
            return set(), f"{item} is the install folder or holds it, so it cannot be kept."
        if rp.startswith(real_s + os.sep):
            names.add(rp[len(real_s) + 1:].split(os.sep)[0])
    return names, None


def run_uninstall(host, repo, args, ask):
    say = host.out
    real = Path(os.path.realpath(repo))
    home = Path(os.path.realpath(host.home))
    if not ((real / ".git/crucible-install").is_file() and (real / "crucible").is_file()):
        say(f"{real} is not an install that the installer made, so uninstall leaves it alone.")
        return EXIT_USAGE
    real_s = str(real)
    if (real == home or real == Path("/") or home.is_relative_to(real)
            or any(real_s == p or real_s.startswith(p + "/") for p in SYSTEM_PREFIXES)):
        say(f"Refusing to remove {real}: it is a system folder or your home folder.")
        return EXIT_USAGE
    if not args.yes and ask is None:
        say("Uninstall asks one question, and no terminal is attached. Add --yes.")
        return EXIT_USAGE
    up = Update(host, real, args, ask)
    if up.box is None:
        say(f"{real} is inside a system folder. Uninstall does not remove files there. Nothing was removed.")
        return EXIT_FAILED
    try:
        lock = os.open(up.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        say(f"Cannot open {up.parent}: {exc}. Nothing was removed.")
        return EXIT_FAILED
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say("An update is running. Nothing was removed.")
            return EXIT_FAILED
        try:
            code = up.recover()
        except CheckFailed as exc:
            say(f"{exc} Nothing was removed.")
            return EXIT_FAILED
        if code == EXIT_ROLLBACK_FAILED:
            say("Nothing was removed.")
            return EXIT_FAILED
        return _uninstall(host, real, home, args, ask, up.box)
    finally:
        os.close(lock)


def _uninstall(host, real, home, args, ask, box):
    say = host.out
    s = settings_mod.build(real, host)
    remove_models = args.remove_models
    extra_roots = [home]
    try:
        extra_roots.append(Sandbox([s.models_dir]).roots[0])
    except ValueError:
        pass
    box = Sandbox(list(box.roots) + [str(r) for r in extra_roots])
    pack_dirs = [Path(s.models_dir) / p.folder for p in packs.PACKS.values()]
    pack_dirs = [p for p in pack_dirs if p.exists() or p.is_symlink()]
    state = Path(s.state_dir) if s.state_dir else None
    state_ok = state is not None and (state.exists() or state.is_symlink())
    # --remove-models removes a state folder only when setup made it (the record in the work folder names this path).
    state_owned = state_ok and settings_mod.state_owned(real, state)
    # Setup may record a folder that did not exist yet (for example ~/.cache), and other programs fill it later. Look inside:
    # the folder is removed only when it holds nothing but what the engine writes there.
    state_why = settings_mod.state_folder_problem(state) if state_owned else None
    if state_why:
        state_owned = False
    models = Path(s.models_dir)
    removable = pack_dirs + ([state] if state_owned else [])
    foreign_state = [state] if state_ok and not state_owned else []
    # What stays. Without --remove-models: the packs, the models folder when it is inside the install folder, and the state
    # folder. With it: only a state folder that setup did not make.
    keep_items = foreign_state if remove_models else pack_dirs + ([models] if _within(models, real) else []) + ([state] if state_ok else [])
    keep_names, problem = _keep_names(real, keep_items)
    if problem:
        say(f"{problem} Nothing was removed. Move it out of the install folder, then run this again.")
        return EXIT_FAILED
    unit_there = service.installed(host)
    say("Uninstall plan:")
    say(f"  Remove:  {real} ({_size(real)})")
    if unit_there:
        say(f"  Remove:  the service {service.UNIT}")
    for p in removable:
        verb = "Remove: " if remove_models else "Keep:   "
        extra = "" if remove_models else "; add --remove-models to remove it"
        say(f"  {verb} {p} ({_size(p)}){extra}")
    for p in foreign_state:
        say(f"  Keep:    {p} ({state_why or 'setup did not make this folder, so it is never removed'})")
    key = s.api_key_file
    if key is not None and Path(key).exists():
        say(f"  Keep:    {key} (the API key)")
    say("  Keep:    the install log of the installer")
    if not args.yes:
        answer = ask("Remove these now?", "n")
        if answer.strip().lower() not in ("y", "yes"):
            say("Nothing was removed.")
            return EXIT_OK
    if unit_there:
        res = service._ctl(host, "stop", service.UNIT)
        if res.rc != 0 and service._ctl(host, "is-active", service.UNIT).rc == 0:
            say("The service did not stop. Nothing was removed. Look at: ./crucible service status")
            return EXIT_FAILED
        service.uninstall(host, s)
    else:
        if (real / ".crucible/server.pid").exists() and steps.stop_pidfile(host, s) != 0:
            say("The server did not stop. Nothing was removed. Wait a minute, then run this again.")
            return EXIT_FAILED
    removed_outside = []
    if remove_models:
        for p in removable:
            if p == state and not _within(p, home):
                say(f"Not removed: {p} is not inside your home folder. Remove it yourself if you want.")
                continue
            _drop(box, p, say)
            removed_outside.append(p)
        if models.is_dir() and not models.is_symlink() and not any(models.iterdir()) and _within(models, real):
            box.rmdir(models)
    kept = set(keep_names)
    entries = sorted(os.listdir(real), key=lambda n: (n in ("crucible", ".git"), n))
    for n in entries:
        if n in kept:
            continue
        _drop(box, real / n, say)
    if kept:
        say(f"Removed the install. Kept inside {real}: {', '.join(sorted(kept))}.")
    else:
        try:
            box.rmdir(real)
        except OSError:
            pass
        say(f"Removed {real}.")
    for p in removed_outside:
        say(f"Removed {p}.")
    say("The systemd linger setting stays on. To turn it off: loginctl disable-linger")
    return EXIT_OK


def _drop(box, path, say):
    """Remove a file, a link or a folder through the sandbox. A path outside its roots is reported and left."""
    try:
        box.rmtree(path)
    except OutsideRoots:
        say(f"Not removed: {path} is outside the folders that this tool may change. Remove it yourself if you want.")
