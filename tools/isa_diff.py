#!/usr/bin/env python3
"""Which device kernels differ between two engine binaries (gfx1151 ISA, no second compile).

  isa_diff.py BEFORE AFTER [--arch gfx1151] [--json] [--show NAME_PART] [--llvm DIR]
  isa_diff.py same BEFORE AFTER [--arch gfx1151] [--show N] [--llvm DIR]
  isa_diff.py asm BASE.s NEW.s

Extracts the device code object of each binary (`llvm-objdump --offloading`), disassembles it, and compares the
text of every function. Addresses, encodings and function-numbered labels do not count; a PC-relative address
(s_getpc_b64, s_add_u32, s_addc_u32) counts as its target symbol, because the literal moves when code in front of
the target moves. Prints the changed, added and removed functions, grouped by base name with counts.
Exit code 0: same ISA. 1: it differs. 2: the tool failed.


Mode `asm BASE.s NEW.s` (the x4-tail gate, replaces tools/isa_compare.py): compares two `--cuda-device-only -S` files.
The default path must keep its device code and its resources: every function of
BASE is in NEW with the same instruction text and the same resource lines. gfx1151 writes a kernel's resources after
`.Lfunc_end`, so the lines from there to the next function label belong to that function (the `.set NAME.field, value`
lines and the `; Name: value` kernel info lines). New functions may be added; they are listed. Exit 0: SAME; 1: a
difference; 2: bad input.

Mode `same BEFORE AFTER [--show N]` (the strict gate of `make isa-check`):
are two builds the same machine code? Per function the instruction text, without the fill after a function
(`s_nop 0` inside a module, `s_code_end` at the end of one: the split build links one object per part), and per
kernel the bytes of its `.kd` descriptor (the 8 bytes at offset 16, the offset to the kernel, are left out: the link
layout sets them). It prints `changed:`, `added:`, `removed:`, `kd changed:` and `kd added or removed:` lines (N names
per kind, 0 for all) and `SAME ISA` or `ISA DIFFERS`. The default mode does not compare descriptors or the fill.
BEFORE and AFTER may be bare code objects (.hsaco) in both modes. Exit codes as above.

Run it where the ROCm LLVM tools are (/opt/rocm); `--llvm` or LLVM_BIN picks a dir.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

LLVM_DIRS = ("/opt/rocm/llvm/bin",)
LOCAL = re.compile(r"\.(LBB|Lfunc_end|Lfunc_begin|Ltmp|LJTI|Lfunc_gep)(\d+)(_\d+)?")
HEADER = re.compile(r"^[0-9a-f]+ <(.+)>:$")
GETPC = re.compile(r"^s_getpc_b64 s\[(\d+):\d+\]$")
ADD = re.compile(r"^s_add_u32 s(\d+), s\1, (0x[0-9a-f]+|-?\d+)$")
ADDC = re.compile(r"^s_addc_u32 s(\d+), s\1, (0x[0-9a-f]+|-?\d+)$")


def find_llvm(explicit=None):
    for candidate in (explicit, os.environ.get("LLVM_BIN"), *LLVM_DIRS):
        if candidate and (Path(candidate) / "llvm-objdump").exists():
            return Path(candidate)
    raise RuntimeError("no llvm-objdump found; pass --llvm DIR or set LLVM_BIN")


def extract_code_object(binary, workdir, llvm, arch="gfx1151"):
    """Unbundle the `arch` code object of `binary` into workdir. The tool writes beside its input, so it reads a
    symlink in workdir (the binary may sit in a read-only folder)."""
    link = Path(workdir) / "bin"
    link.symlink_to(Path(binary).resolve())
    subprocess.run([llvm / "llvm-objdump", "--offloading", str(link)], check=True, capture_output=True, text=True)
    found = [p for p in Path(workdir).iterdir() if p.name.endswith(arch) and p.stat().st_size > 0]
    if len(found) != 1:
        raise RuntimeError(f"{binary}: expected one non-empty {arch} code object, found {[p.name for p in found]}")
    return found[0]


def is_code_object(path, llvm):
    header = subprocess.run([llvm / "llvm-readelf", "-h", str(path)], capture_output=True, text=True).stdout
    return "AMDGPU" in header


def open_code_object(path, workdir, llvm, arch):
    """The code object itself for a bare one (.hsaco), else the unbundled `arch` object of a host binary."""
    return Path(path) if is_code_object(path, llvm) else extract_code_object(path, workdir, llvm, arch)


def symbol_table(text):
    """(address, size, name) of the defined OBJECT and FUNC symbols in `llvm-readelf -sW` text, sorted."""
    syms = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 8 and f[0].endswith(":") and f[3] in ("OBJECT", "FUNC") and f[6] != "UND":
            syms.append((int(f[1], 16), int(f[2], 0), f[7]))
    return sorted(syms)


def where(syms, addr):
    best = max((s for s in syms if s[0] <= addr), default=None)
    return f"{best[2]}+{addr - best[0]:#x}" if best else f"{addr:#x}"


def parse_disassembly(text, syms):
    """Function name -> normalized instruction text, from `llvm-objdump -d --no-show-raw-insn` output."""
    units, name, pc, lo = {}, None, {}, {}
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            name, pc, lo = m.group(1), {}, {}
            units[name] = []
            continue
        if not name or not line.strip():
            continue
        code, _, comment = line.strip().partition("//")
        code = " ".join(code.split())
        addr = re.match(r"\s*([0-9A-Fa-f]+):", comment)
        target = re.search(r"<[^>]+>", comment)
        if (g := GETPC.match(code)) and addr:
            pc[int(g.group(1))] = int(addr.group(1), 16) + 4
        elif (a := ADD.match(code)) and int(a.group(1)) in pc:
            # Masked only when the s_addc_u32 below resolves it; otherwise the literal stays.
            r = int(a.group(1))
            lo[r] = (int(a.group(2), 0) & 0xFFFFFFFF, len(units[name]))
        elif (c := ADDC.match(code)) and int(c.group(1)) - 1 in lo:
            r = int(c.group(1)) - 1
            low, at = lo.pop(r)
            units[name][at] = f"s_add_u32 s{r}, s{r}, <pcrel lo>"
            off = (int(c.group(2), 0) & 0xFFFFFFFF) << 32 | low
            dest = (pc.pop(r) + off) & (2**64 - 1)
            code = f"s_addc_u32 s{r + 1}, s{r + 1}, <pcrel hi> ; {where(syms, dest)}"
        units[name].append(code + (" " + target.group() if target else ""))
    return {k: "\n".join(v) for k, v in units.items()}


def compare_units(before, after):
    """changed (in both, text differs), added (only after), removed (only before): sorted name lists."""
    common = set(before) & set(after)
    return dict(changed=sorted(n for n in common if before[n] != after[n]),
                added=sorted(set(after) - set(before)), removed=sorted(set(before) - set(after)))


def base_name(demangled):
    """'void crucible::Foo<4>(int)' -> 'Foo'. Instantiations and overloads share a base name."""
    head = re.sub(r"<.*>", "", demangled.split("(")[0]).split()
    return head[-1].split("::")[-1] if head else demangled


def group_counts(names, demangled):
    counts = {}
    for n in names:
        base = base_name(demangled.get(n, n))
        counts[base] = counts.get(base, 0) + 1
    return dict(sorted(counts.items()))


def demangle(names, llvm):
    if not names:
        return {}
    out = subprocess.run([llvm / "llvm-cxxfilt"], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, out.splitlines()))


def load_units(binary, llvm, arch):
    """Normalized function texts of one binary, and the sha256 of its code object."""
    with tempfile.TemporaryDirectory(prefix="isa_diff_") as work:
        co = open_code_object(binary, work, llvm, arch)
        syms, units = disassemble(co, llvm)
        return units, hashlib.sha256(co.read_bytes()).hexdigest()


def disassemble(co, llvm):
    """(symbols, function name -> normalized text) of a code object."""
    syms = symbol_table(subprocess.run([llvm / "llvm-readelf", "-sW", str(co)], capture_output=True, text=True,
                                       check=True).stdout)
    dis = subprocess.run([llvm / "llvm-objdump", "-d", "--no-show-raw-insn", str(co)], capture_output=True,
                         text=True, check=True).stdout
    return syms, parse_disassembly(dis, syms)


def report(before, after, llvm):
    delta = compare_units(before, after)
    names = demangle(delta["changed"] + delta["added"] + delta["removed"], llvm)
    out = dict(functions=[len(before), len(after)], identical=len(set(before) & set(after)) - len(delta["changed"]))
    for kind in ("changed", "added", "removed"):
        out[kind] = len(delta[kind])
        out[kind + "_by_base"] = group_counts(delta[kind], names)
    out["same_isa"] = not (delta["changed"] or delta["added"] or delta["removed"])
    out["names"] = {k: [names.get(n, n) for n in delta[k]] for k in ("changed", "added", "removed")}
    return out, delta, names


# --- asm mode ---
ASM_USAGE = """usage: isa_diff.py asm BASE.s NEW.s   (the x4-tail gate, `--cuda-device-only -S` output)"""
LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):\s*(;.*)?$")
RESOURCE = re.compile(r"^\s*; ([A-Za-z][\w :]*?):\s*(.+)$")
SET = re.compile(r"^\s*\.set\s+([A-Za-z_.$][\w.$]*)\.([A-Za-z_]\w*),\s*(.+?)\s*$")


def functions_text(text):
    """Function name -> (instruction lines, resource lines). Only symbols with `.type NAME,@function` count."""
    lines = text.splitlines()
    names = set()
    for line in lines:
        m = re.match(r"^\s*\.type\s+([^,\s]+),@function", line)
        if m:
            names.add(m.group(1))
    out, name, tail = {}, None, False
    for line in lines:
        m = LABEL.match(line)
        if m and m.group(1) in names:
            name, tail = m.group(1), False
            out[name] = ([], [])
            continue
        if name is None:
            continue
        if not tail and (re.match(r"^\.Lfunc_end", line) or line.startswith("\t.end_amdgpu")):
            tail = True
            continue
        m = SET.match(line)
        if m and m.group(1) == name:
            out[name][1].append(f"set {m.group(2)} = {m.group(3)}")
            continue
        r = RESOURCE.match(line)
        if r:
            out[name][1].append(f"{r.group(1).strip()}: {r.group(2).strip()}")
            continue
        if tail:
            continue  # .size, .section, .text and the next function's directives are not this function's code
        # Block labels carry the function's number in the file, which changes when functions are added before them;
        # keep only the block number.
        text_of_line = re.sub(r"\.LBB\d+_(\d+)", r".LBB_\1", line.split(";")[0].rstrip())
        if text_of_line.strip():
            out[name][0].append(text_of_line)
    return out, names


def functions(path):
    with open(path) as source:
        return functions_text(source.read())


def asm_main(argv):
    if len(argv) != 2:
        print(ASM_USAGE, file=sys.stderr)
        return 2
    base, base_names = functions(argv[0])
    new, new_names = functions(argv[1])
    if not base or len(base) != len(base_names) or len(new) != len(new_names):
        print(f"bad input: {len(base)}/{len(base_names)} base and {len(new)}/{len(new_names)} new functions parsed")
        return 2
    missing = sorted(set(base) - set(new))
    added = sorted(set(new) - set(base))
    changed = [n for n in sorted(set(base) & set(new)) if base[n] != new[n]]
    print(f"base functions {len(base)}, new functions {len(new)}")
    for n in missing:
        print(f"MISSING {n}")
    for n in changed:
        if base[n][0] != new[n][0]:
            print(f"CHANGED CODE {n}")
        else:
            print(f"CHANGED RESOURCES {n}")
        for line in sorted(set(base[n][1]) - set(new[n][1])):
            print(f"  - {line}")
        for line in sorted(set(new[n][1]) - set(base[n][1])):
            print(f"  + {line}")
    for n in added:
        print(f"ADDED {n}")
    # The whole file too: the compilation unit id (__hip_cuid_*, from the source path and text) is the only line that
    # may differ when no device code changed.
    cuid = re.compile(r"__hip_cuid_[0-9a-f]+")
    texts = [cuid.sub("__hip_cuid_X", open(path).read()) for path in argv[:2]]
    print(f"file: {'identical' if texts[0] == texts[1] else 'not identical'} except the __hip_cuid name")
    same = not missing and not changed
    print(f"{'SAME' if same else 'DIFFERENT'}: {len(set(base) & set(new)) - len(changed)} equal, "
          f"{len(changed)} changed, {len(missing)} missing, {len(added)} added")
    return 0 if same else 1


# --- same mode ---
PAD = ("s_nop 0", "s_code_end", "...")


def strip_fill(units):
    """Function texts without the trailing fill: a function ends at s_endpgm or s_setpc, never at padding."""
    out = {}
    for name, text in units.items():
        lines = text.split("\n")
        while lines and lines[-1].split(" //")[0] in PAD:
            lines.pop()
        out[name] = "\n".join(lines)
    return out


def kernel_descriptors(co, llvm, syms):
    """name -> sha256 of the bytes of each .kd symbol, read through the LOAD segments.

    Bytes 16 to 24 of a descriptor are the offset from the descriptor to its kernel: the link layout sets it, so
    it is left out.
    """
    data = Path(co).read_bytes()
    ph = subprocess.run([llvm / "llvm-readelf", "-lW", str(co)], capture_output=True, text=True, check=True).stdout
    loads = [(int(f[2], 16), int(f[1], 16), int(f[4], 16)) for f in (l.split() for l in ph.splitlines())
             if f and f[0] == "LOAD"]

    def read(addr, size):
        for vaddr, off, filesz in loads:
            if vaddr <= addr < vaddr + filesz:
                return data[off + addr - vaddr:off + addr - vaddr + size]
        raise RuntimeError(f"{co}: address {addr:#x} is in no LOAD segment")

    def masked(addr, size):
        raw = read(addr, size)
        return raw[:16] + raw[24:]
    return {n: hashlib.sha256(masked(a, s)).hexdigest() for a, s, n in syms if n.endswith(".kd")}


def load_strict(path, llvm, arch):
    """(function texts without fill, kernel descriptor hashes) of a binary or of a bare code object."""
    with tempfile.TemporaryDirectory(prefix="isa_same_") as work:
        co = open_code_object(path, work, llvm, arch)
        syms, units = disassemble(co, llvm)
        return strip_fill(units), kernel_descriptors(co, llvm, syms)


def same_main(argv):
    ap = argparse.ArgumentParser(prog="isa_diff.py same", description="strict same-ISA gate (see isa_diff.py --help)")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--llvm")
    ap.add_argument("--show", type=int, default=10, help="names to list per kind (0: all)")
    args = ap.parse_args(argv)
    try:
        llvm = find_llvm(args.llvm)
        (ub, kb), (ua, ka) = load_strict(args.before, llvm, args.arch), load_strict(args.after, llvm, args.arch)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"isa_diff same: {error}", file=sys.stderr)
        return 2
    delta = compare_units(ub, ua)
    kd_changed = sorted(n for n in set(kb) & set(ka) if kb[n] != ka[n])
    kd_moved = sorted(set(kb) ^ set(ka))
    print(f"functions {len(ub)} -> {len(ua)}, text changed {len(delta['changed'])}, added {len(delta['added'])}, "
          f"removed {len(delta['removed'])}; descriptors {len(kb)} -> {len(ka)}, changed {len(kd_changed)}, "
          f"added or removed {len(kd_moved)}")
    for label, names in (("changed", delta["changed"]), ("added", delta["added"]), ("removed", delta["removed"]),
                         ("kd changed", kd_changed), ("kd added or removed", kd_moved)):
        for name in (names[:args.show] if args.show else names):
            print(f"  {label}: {name}")
    same = not (any(delta.values()) or kd_changed or kd_moved)
    print("SAME ISA" if same else "ISA DIFFERS")
    return 0 if same else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "asm":
        return asm_main(argv[1:])
    if argv and argv[0] == "same":
        return same_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--llvm", help="directory with llvm-objdump, llvm-readelf, llvm-cxxfilt")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show", metavar="PART", help="print the unified diff of changed functions whose name has PART")
    args = ap.parse_args(argv)
    try:
        llvm = find_llvm(args.llvm)
        (b_units, b_sha), (a_units, a_sha) = (load_units(p, llvm, args.arch) for p in (args.before, args.after))
    except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
        print(f"isa_diff: {error}", file=sys.stderr)
        return 2
    rep, delta, names = report(b_units, a_units, llvm)
    rep["code_object_sha256"] = [b_sha, a_sha]
    if args.json:
        print(json.dumps(rep, indent=1))
    else:
        print(f"functions {rep['functions'][0]} -> {rep['functions'][1]}, identical {rep['identical']}, "
              f"changed {rep['changed']}, added {rep['added']}, removed {rep['removed']}")
        for kind in ("changed", "added", "removed"):
            for base, count in rep[kind + "_by_base"].items():
                print(f"  {kind:8} {base} x{count}")
        print("SAME ISA" if rep["same_isa"] else "ISA DIFFERS")
    for n in delta["changed"]:
        if args.show and args.show in names.get(n, n):
            sys.stdout.writelines(difflib.unified_diff(b_units[n].splitlines(True), a_units[n].splitlines(True),
                                                       "before " + n, "after " + n))
    return 0 if rep["same_isa"] else 1


if __name__ == "__main__":
    sys.exit(main())
