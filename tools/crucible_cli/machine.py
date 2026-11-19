"""The memory class of this machine and the defaults that fit it (MemTotal of /proc/meminfo).

- 120 GiB or more (a 128 GB machine), or MemTotal unknown: every value as measured (measured.py, the README table). This
  path does not change: the same pack, server command and kernel parameters as before this module.
- 88 GiB to 120 GiB (a 96 GB machine, about 93 GiB MemTotal): the 3.05 pack, 2 sessions of 262,144 tokens, a 4,096 MiB
  conversation cache, and kernel parameters for the size of this machine.
- Below 88 GiB (a 64 GB machine): not supported. The 3.05 pack with one session of 32,768 tokens holds about 63 GB of GPU
  memory, and the engine start keeps 9 GiB free (engine/runtime/batch_admission.h kAdmissionReserve).

The numbers come from the 3.05 starts on the 128 GB test machine. The engine still checks the
real free memory at start and fits the layout, so a machine that has less than the class
expects starts a smaller layout and prints why.
"""
from dataclasses import dataclass, field
from pathlib import Path

from . import measured

KIB_PER_GIB = 1024 * 1024
GIB = 1 << 30
FULL_MIN_KB = 120 * KIB_PER_GIB    # the 128 GB class (doctor.MEM_MIN_KB)
MID_MIN_KB = 88 * KIB_PER_GIB      # the 96 GB class
# The GTT limit may be this much below the value that the kernel parameters set (130 GB of 124 GiB on the 128 GB class).
GTT_SLACK_GIB = 3
# GPU memory of the 3.05 pack at start with one session of 32,768 tokens, from the 128 GB test machine (lowmem.md).
FLOOR_305_GB = 63
START_RESERVE_GIB = 9


@dataclass(frozen=True)
class Fit:
    name: str          # "128 GB", "96 GB" or "small"
    supported: bool
    pack: str          # the default pack
    packs: tuple       # the packs that fit
    streams: int       # the default layout
    ctx: int
    cache_mib: str     # --conversation-cache-mib
    gtt_params: str    # the two kernel parameters
    gtt_min_bytes: int
    min_kb: int        # the smallest MemTotal of the class
    layouts: dict = field(default_factory=dict)   # the named layouts of ./crucible setup --layout


FULL = Fit("128 GB", True, "4.05", ("4.05", "4.18", "3.05"), measured.STREAMS, measured.CTX,
           measured.SERVER_FLAGS["--conversation-cache-mib"], "amdgpu.gttsize=126976 ttm.pages_limit=32505856",
           130_000_000_000, FULL_MIN_KB, measured.LAYOUTS)


def gtt_gib(memtotal_kb):
    """The GTT limit for this machine in GiB: the whole GiB of MemTotal less 1 (124 of 125 on the 128 GB class)."""
    return memtotal_kb // KIB_PER_GIB - 1


def gtt_params(memtotal_kb):
    gib = gtt_gib(memtotal_kb)
    return f"amdgpu.gttsize={gib * 1024} ttm.pages_limit={gib * (GIB // 4096)}"


def _sized(name, supported, memtotal_kb, streams, ctx, cache_mib, layouts):
    gib = gtt_gib(memtotal_kb)
    return Fit(name, supported, "3.05", ("3.05",) if supported else (), streams, ctx, cache_mib, gtt_params(memtotal_kb),
               (gib - GTT_SLACK_GIB) * GIB, MID_MIN_KB, layouts)


def fit(memtotal_kb):
    """The class of a machine with MEMTOTAL_KB (None: unknown, the 128 GB values)."""
    if memtotal_kb is None or memtotal_kb >= FULL_MIN_KB:
        return FULL
    if memtotal_kb >= MID_MIN_KB:
        return _sized("96 GB", True, memtotal_kb, 2, 262144, "4096",
                      {"default": (2, 262144, 0, 0), "long": (1, 524288, 0, 0)})
    return _sized("small", False, memtotal_kb, 1, 32768, "2048", {"default": (1, 32768, 0, 0)})


def memtotal_kb(host):
    """MemTotal in KiB from the host's /proc/meminfo, or None."""
    try:
        text = (Path(host.proc) / "meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "MemTotal:" and parts[1].isdigit():
            return int(parts[1])
    return None


def of_host(host):
    return fit(memtotal_kb(host))


def refusal(memtotal_kb):
    """Why a machine below the 96 GB class cannot run Crucible."""
    got = "an unknown amount of memory" if memtotal_kb is None else f"{memtotal_kb / KIB_PER_GIB:.0f} GiB"
    return (f"Crucible needs a 96 GB or 128 GB machine. This machine has {got}. The 3.05 pack with one session of "
            f"32,768 tokens holds about {FLOOR_305_GB} GB of GPU memory, and the engine keeps {START_RESERVE_GIB} GiB "
            "free for the system at start. If the BIOS gives the GPU a large dedicated memory (UMA frame buffer), set it "
            "to the smallest value (512 MB): the GPU uses the system memory through the GTT.")


# The 96 GB pass line: the class picks the layout on paper; the engine's start-time fit has
# the last word.
START_FIT_NOTE = ("At start the engine measures the free memory. If it is less than this class expects, the engine "
                  "starts a smaller conversation cache or fewer sessions (as the last step, a shorter --ctx) and prints why.")
UMA_ADVICE_MIN_BYTES = 1 << 30


def uma_advice(vram_bytes):
    """The UMA frame buffer advice for the 96 GB pass line when the GPU has more than 1 GiB of VRAM, else "".
    A 128 GB machine with a large frame buffer has a MemTotal of the 96 GB class."""
    if vram_bytes is None or vram_bytes <= UMA_ADVICE_MIN_BYTES:
        return ""
    return (f" The BIOS gives the GPU {vram_bytes / GIB:.0f} GiB of dedicated memory (UMA frame buffer). Set it to the "
            "smallest value (512 MB): the GPU uses the system memory through the GTT. On a 128 GB machine, that gives "
            "the 128 GB defaults.")


def layout_choice(f, name):
    """(streams, ctx, short streams, short ctx) of the layout NAME on this machine, or ValueError."""
    if name not in f.layouts:
        have = ", ".join(f.layouts)
        raise ValueError(f"--layout {name} does not fit a {f.name} machine. Use one of: {have}.")
    return f.layouts[name]
