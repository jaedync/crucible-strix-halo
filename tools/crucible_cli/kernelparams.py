"""The two kernel parameters that the GPU memory limit needs, and how setup puts them on the command line.

Setup finds the boot method (GRUB, systemd-boot or a unified kernel image), prints the exact change and, only after the
user typed "yes", applies it with sudo. The default is to print and stop. The module never restarts the machine. It
applies a change only when exactly one boot file matches with evidence. In any other case it prints the change and names
the files that it found. Every file that it changes gets a backup first.
"""
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from .doctor import GTT_PARAMS
from .files import Sandbox

REQUIRED = GTT_PARAMS.split()
_LOADER_WORDS = ("initrd=", "BOOT_IMAGE=")
METHODS = {"grub": "GRUB", "systemd-boot": "systemd-boot", "uki": "a unified kernel image"}
_BARE = re.compile(r"[A-Za-z0-9_.=:,/@%+-]*")


class Failed(Exception):
    pass


def core(words):
    """The words of a command line without the ones that the boot loader adds (initrd= and BOOT_IMAGE=)."""
    return [w for w in words if not w.startswith(_LOADER_WORDS)]


def _key(word):
    return word.split("=", 1)[0]


def missing(cmdline):
    """The required tokens that the command line does not hold with the right value."""
    have = set(cmdline.split())
    return [t for t in REQUIRED if t not in have]


def set_params(line, tokens):
    """LINE with each token set: a word with the same key gets the new value (the first one; the others go), a new key is
    added at the end. The other words keep their order. Running it twice gives the same line."""
    new = {_key(t): t for t in tokens}
    out, placed = [], set()
    for word in line.split():
        key = _key(word)
        if key not in new:
            out.append(word)
        elif key not in placed:
            out.append(new[key])
            placed.add(key)
    out.extend(t for k, t in new.items() if k not in placed)
    return " ".join(out)


def stamp(host):
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(host.clock()))


@dataclass
class Plan:
    tokens: list
    method: str = ""            # "grub", "systemd-boot", "uki", or "" when no single boot file is known
    file: Path = None
    old_text: str = ""
    new_text: str = ""
    old_line: str = ""
    new_line: str = ""
    mode: str = "644"
    regen: list = None          # the command (without sudo) that makes the change active, or None when none is needed
    can_apply: bool = False
    reason: str = ""            # why the change is print-only
    candidates: list = field(default_factory=list)   # (method, file) of each file that matched, when more than one did
    unreadable: list = field(default_factory=list)

    def commands(self, when, tmp):
        """The root commands of the change, in order (without sudo): the backup, the write, the regeneration."""
        backup = f"{self.file}.crucible-bak-{when}"
        out = [["cp", "-p", str(self.file), backup], ["install", "-m", self.mode, str(tmp), str(self.file)]]
        if self.regen:
            out.append(list(self.regen))
        return out


def _read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return None


# GRUB ----------------------------------------------------------------------------------------------------------------
def _grub_edit(text, want):
    """(new text, old line, new line) or (None, reason, "")."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if re.match(r"\s*GRUB_CMDLINE_LINUX_DEFAULT=", ln)]
    if len(hits) != 1:
        return None, f"GRUB_CMDLINE_LINUX_DEFAULT is not set exactly once in the GRUB file ({len(hits)} lines)", ""
    i = hits[0]
    body = lines[i].rstrip("\r\n")
    end = lines[i][len(body):]
    pre, raw = body.split("=", 1)
    raw = raw.rstrip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"' and not re.search(r'["\\$`]', raw[1:-1]):
        inner = raw[1:-1]
    elif len(raw) >= 2 and raw[0] == raw[-1] == "'" and "'" not in raw[1:-1]:
        inner = raw[1:-1]
    elif _BARE.fullmatch(raw):
        inner = raw
    else:
        return None, "the GRUB_CMDLINE_LINUX_DEFAULT line holds shell syntax ($, a backtick, a backslash or a comment) that this tool does not edit", ""
    quote = raw[0] if raw[:1] == "'" else '"'
    new = f"{pre}={quote}{set_params(inner, want)}{quote}"
    lines[i] = new + end
    return "".join(lines), body, new


def _grub(host, cmd, want):
    f = Path(host.etc) / "default/grub"
    text = _read(f)
    if text is None:
        return []
    cfg = next((b / n for b in host.boot_dirs for n in ("grub/grub.cfg", "grub2/grub.cfg") if (b / n).is_file()), None)
    if cfg is None and "BOOT_IMAGE=" not in cmd:
        return []
    plan = Plan(tokens=want, method="grub", file=f, old_text=text)
    new, old_line, new_line = _grub_edit(text, want)
    if new is None:
        plan.reason = old_line
        return [plan]
    plan.new_text, plan.old_line, plan.new_line = new, old_line, new_line
    which = host.which
    if cfg is not None and which("grub-mkconfig"):
        plan.regen = ["grub-mkconfig", "-o", str(cfg)]
    elif which("update-grub"):
        plan.regen = ["update-grub"]
    elif cfg is not None and which("grub2-mkconfig"):
        plan.regen = ["grub2-mkconfig", "-o", str(cfg)]
    else:
        plan.reason = "no grub-mkconfig, update-grub or grub2-mkconfig was found, or the grub.cfg file is not in /boot"
    return [plan]


# systemd-boot --------------------------------------------------------------------------------------------------------
def _sdboot(host, running, want):
    out, seen = [], set()
    for b in host.boot_dirs:
        d = Path(b) / "loader/entries"
        try:
            entries = sorted(d.glob("*.conf")) if d.is_dir() else []
        except OSError:
            continue
        for f in entries:
            real = os.path.realpath(f)
            text = _read(f)
            if text is None or real in seen:
                continue
            seen.add(real)
            lines = text.splitlines(keepends=True)
            opts = [i for i, ln in enumerate(lines) if re.match(r"options\s", ln)]
            words = [w for i in opts for w in lines[i].split()[1:]]
            if not opts or core(words) != running:
                continue
            plan = Plan(tokens=want, method="systemd-boot", file=f, old_text=text)
            if len(opts) != 1:
                plan.reason = "the entry has more than one options line"
            else:
                i = opts[0]
                body = lines[i].rstrip("\r\n")
                new = "options " + set_params(" ".join(lines[i].split()[1:]), want)
                lines[i] = new + lines[i][len(body):]
                plan.new_text, plan.old_line, plan.new_line = "".join(lines), body, new
            out.append(plan)
    return out


# unified kernel image ------------------------------------------------------------------------------------------------
def _uki(host, running, want):
    f = Path(host.etc) / "kernel/cmdline"
    text = _read(f)
    if text is None or core(text.split()) != running:
        return []
    plan = Plan(tokens=want, method="uki", file=f, old_text=text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        plan.reason = "the file has more than one line"
        return [plan]
    plan.old_line = lines[0]
    plan.new_line = set_params(lines[0], want)
    plan.new_text = plan.new_line + "\n"
    if host.which("mkinitcpio"):
        plan.regen = ["mkinitcpio", "-P"]
    else:
        plan.reason = "mkinitcpio is not installed: build the image with your own tool after the change"
    return [plan]


def plan(host):
    """None when the running command line holds both parameters already. Else a Plan: can_apply only when exactly one
    boot file matched with evidence, the change is plain text, a command to make it active is known and sudo exists."""
    cmd = (_read(Path(host.proc) / "cmdline") or "").strip()
    want = missing(cmd)
    if not want:
        return None
    running = core(cmd.split())
    unreadable = [str(b) for b in host.boot_dirs if Path(b).is_dir() and not os.access(b, os.R_OK | os.X_OK)]
    found = _grub(host, cmd, want) + _sdboot(host, running, want) + _uki(host, running, want)
    if len(found) == 1:
        p = found[0]
        p.unreadable = unreadable
        p.mode = _mode(p.file)
        p.can_apply = not p.reason and bool(p.new_text)
        if p.can_apply and not host.which("sudo"):
            p.can_apply, p.reason = False, "sudo is not installed on this machine"
        return p
    p = Plan(tokens=want, unreadable=unreadable)
    if found:
        p.candidates = [(c.method, c.file) for c in found]
        p.reason = "more than one boot file matches this machine, so this tool does not guess"
    else:
        p.reason = "this tool did not find the boot file of this machine"
    return p


def _mode(path):
    try:
        return format(stat.S_IMODE(os.stat(path).st_mode) & 0o777, "03o")
    except OSError:
        return "644"


# text ----------------------------------------------------------------------------------------------------------------
def describe(p, host):
    """The lines that setup prints for a plan."""
    out = [f"The kernel command line needs these parameters: {' '.join(p.tokens)}"]
    if p.unreadable:
        out.append("This user cannot read " + ", ".join(p.unreadable) + ", so the boot files there were not checked. "
                   "Run the checks as root or use the lines below.")
    if p.method:
        out.append(f"Boot method: {METHODS[p.method]}. File: {p.file}")
        if p.new_line:
            out += ["Change in this file:", f"  - {p.old_line}", f"  + {p.new_line}"]
    else:
        out.append(p.reason + ".")
        for method, f in p.candidates:
            out.append(f"  matches: {METHODS[method]}: {f}")
    if p.can_apply:
        when = "<date>-<time>"
        out.append("After you type yes, setup runs these commands. Each one asks sudo:")
        out += ["  sudo " + " ".join(c) for c in p.commands(when, "<temp file in .crucible>")]
        out.append("A backup of the file is made first. If a command fails, setup restores the backup. "
                   "Setup never restarts the machine: you restart it yourself, then run ./crucible setup again.")
    elif p.method:
        out.append(f"Setup does not apply this change itself: {p.reason}.")
        out.append("To apply it yourself: make the change above" + (", then run: sudo " + " ".join(p.regen) if p.regen else "") + ".")
    else:
        out += ["Keep every parameter that is already there (see: cat /proc/cmdline) and do not add one twice. Where the line is depends on how your system starts:",
                "  - GRUB: GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then: sudo grub-mkconfig -o /boot/grub/grub.cfg",
                "  - systemd-boot with entry files: the options line of your entry in /boot/loader/entries/",
                "  - a unified kernel image: usually /etc/kernel/cmdline, then build the image again (for example: sudo mkinitcpio -P)",
                "  - another boot loader: see its documentation",
                "Keep a working boot entry until the new one starts. Then restart the machine yourself."]
    if p.method == "uki" or any(m == "uki" for m, _ in p.candidates):
        out.append("A new image changes what the TPM measures. If your disk key is bound to the TPM or to Secure Boot, "
                   "keep your recovery key ready before you restart.")
    return out


# apply ---------------------------------------------------------------------------------------------------------------
def apply(host, p, work, say):
    """Make the change of plan P with root commands. Raises Failed. The runner is host.runner and each command goes to its
    run_root. A failed step restores the backup. The machine is never restarted."""
    if not p.can_apply:
        raise Failed("this change cannot be applied by setup: " + (p.reason or "no single boot file"))
    run = host.runner.run_root
    if _read(p.file) != p.old_text:
        raise Failed(f"{p.file} changed since the plan was made. Nothing was changed. Run ./crucible setup again.")
    when = stamp(host)
    p.mode = _mode(p.file)   # the mode of the file now, not of the plan
    backup = f"{p.file}.crucible-bak-{when}"
    if os.path.lexists(backup):
        raise Failed(f"{backup} exists already. Nothing was changed. Run ./crucible setup again in a second.")
    box = Sandbox([work])
    tmp = Path(work) / "kernel-param.new"
    cmds = p.commands(when, tmp)
    box.mkdir(work)
    box.write_text(tmp, p.new_text, mode=0o600)
    try:
        say(f"Backup: {backup}")
        res = run(cmds[0])
        if res.rc != 0:
            raise Failed(f"the backup failed ({res.out.strip()[-200:]}). Nothing was changed.")
        res = run(cmds[1], sources=(str(tmp),))
        if res.rc != 0 or _read(p.file) != p.new_text:
            raise Failed(f"writing {p.file} failed ({res.out.strip()[-200:]}). " + _restore(run, p, backup))
        say(f"Changed {p.file}.")
        if len(cmds) > 2:
            say("Running: sudo " + " ".join(cmds[2]))
            res = run(cmds[2])
            if res.rc != 0:
                back = _restore(run, p, backup)
                more = (" A failed image build can leave an image half built: run sudo mkinitcpio -P yourself before you restart."
                        if p.method == "uki" else "")
                raise Failed(f"{' '.join(cmds[2])} failed ({res.out.strip()[-200:]}). {back}{more}")
    finally:
        box.unlink(tmp, missing_ok=True)
    say(f"Done. The new parameters start with the next boot. Restart the machine yourself, then run ./crucible setup again. "
        f"This tool does not restart it. The backup is {backup}.")


def _restore(run, p, backup):
    """Put the backup back. Returns the sentence that tells what happened."""
    res = run(["cp", "-p", backup, str(p.file)])
    if res.rc == 0 and _read(p.file) == p.old_text:
        return f"The file was restored from the backup {backup}."
    return f"The restore failed. Copy the backup {backup} to {p.file} yourself before you restart."
