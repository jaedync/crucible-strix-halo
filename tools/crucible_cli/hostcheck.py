"""The GTT host rule: is the GPU memory limit (GTT) of this machine large enough? One pure function, gtt_verdict.

./crucible doctor uses it for its "GPU memory limit (GTT)" line. A container host check uses the same rule. It reads
only sysfs and /proc/meminfo, not the kernel command line. tests/data/host_check/gtt_vectors.json holds the test
vectors. tests/test_host_check.py reads them, and a port of the rule to another language can read the same file.

Inputs:
- gtt_total_bytes: the mem_info_gtt_total values (bytes) of /sys/class/drm/card*/device. None or 0: a value that could not
  be read. The largest value counts.
- memtotal_kb: MemTotal of /proc/meminfo in KiB, or None. It gives the memory class (machine.fit); None gives the
  128 GB class.
- pack: a pack name ("4.05", "3.05", "4.18") or None. With a pack, the pack must be one of the packs of the class.
  The doctor gives no pack: its memory line checks the pack.
- container: True for a check that cannot see the kernel command line or the doctor. Then "no value" is a warning,
  not a failure: the engine still checks the real free memory at start.
- params_on_cmdline: True when /proc/cmdline holds amdgpu.gttsize and ttm.pages_limit; None or False when not.

The rule, in order:
1. No value: fail (container: warn), code "no-gtt".
2. A pack that the class does not hold: fail, code "pack".
3. The largest value below the minimum of the class (machine.Fit.gtt_min_bytes): fail, code "gtt-low".
4. Else pass, code "ok".
"""
from dataclasses import dataclass

from . import machine, packs

STATUSES = ("pass", "warn", "fail")
CODES = ("ok", "no-gtt", "pack", "gtt-low")
GB = 1_000_000_000

NO_GTT = "no amdgpu memory value found in /sys/class/drm"
NO_GTT_CONTAINER = NO_GTT + ". The engine checks the free memory at start."
WHY = "The GPU of Strix Halo uses the system memory, and Linux lets it use only part of it by default. "
FULL_NEED = "The 4.05 pack with 4 long sessions needs about 133 GB."
ON_CMDLINE = " The two parameters are on the command line already: restart if you changed them just now."


@dataclass(frozen=True)
class Verdict:
    status: str   # pass, warn or fail
    code: str     # one of CODES
    reason: str   # the text for the user
    params: str   # the two kernel parameters for this machine (the fix of a "gtt-low" failure)

    def as_dict(self):
        return {"status": self.status, "code": self.code, "reason": self.reason, "params": self.params}


def gb_text(n):
    """133143986176 -> "133.1 GB" (decimal GB, one decimal)."""
    return f"{n / GB:.1f} GB"


def gtt_verdict(gtt_total_bytes, memtotal_kb, pack=None, container=False, params_on_cmdline=None):
    """The verdict of the GTT rule (see the module text)."""
    if pack is not None and pack not in packs.PACKS:
        raise ValueError(f"unknown pack {pack!r}; known: {', '.join(packs.PACKS)}")
    fit = machine.fit(memtotal_kb)
    values = [t for t in gtt_total_bytes if t]
    if not values:
        return Verdict("warn" if container else "fail", "no-gtt", NO_GTT_CONTAINER if container else NO_GTT,
                       fit.gtt_params)
    if pack is not None and pack not in fit.packs:
        reason = (f"the {pack} pack needs a 128 GB machine. On a {fit.name} machine, use the 3.05 pack."
                  if fit.supported else machine.refusal(memtotal_kb))
        return Verdict("fail", "pack", reason, fit.gtt_params)
    best = max(values)
    if best >= fit.gtt_min_bytes:
        return Verdict("pass", "ok", gb_text(best), fit.gtt_params)
    gttsize_mib = fit.gtt_params.split()[0].split("=")[1]
    need = FULL_NEED if fit is machine.FULL else f"On a {fit.name} machine, set the limit to {gttsize_mib} MiB."
    note = ON_CMDLINE if params_on_cmdline else ""
    return Verdict("fail", "gtt-low", f"the limit is {gb_text(best)}. {WHY}{need}{note}", fit.gtt_params)
