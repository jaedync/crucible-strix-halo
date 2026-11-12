#!/usr/bin/env python3
"""The state epoch.

The SSD conversation cache keeps each state under an identity folder. The identity used to hold the sha256 of the engine
binary, so every new binary emptied the cache. It now holds a state epoch: a hash of the things that decide the bytes of a
stored state. This tool computes the epoch at build time and writes it into the slot of engine/runtime/state_epoch.h.

The epoch is the sha256 of four parts:
  1. source:   the state-format sources of tools/state_epoch_sources.txt, with comments and blank space removed;
  2. registry: name, kind and default of every kNumerics, kLayout and kProfile switch of engine/switches.h;
  3. isa:      the gfx1151 machine code of every device function, the kernel descriptors and the device data objects
               (the code that writes state rows; a host-only change does not move it);
  4. pins:     tools/state_epoch_pins.tsv may map a computed epoch to an older one (needs gate evidence).

  state_epoch.py stamp BINARY [--isa-hash HEX]   compute the epoch and write it into BINARY
  state_epoch.py show BINARY                     print the stamp of BINARY
  state_epoch.py parts [--isa BINARY]            print the parts (without a binary: source and registry only)
"""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

PREFIX = b'STATE_EPOCH_SLOT_V1:'
DIGITS = 64
STATE_KINDS = ('numerics', 'layout', 'profile')
SOURCES = ROOT / 'tools/state_epoch_sources.txt'
PINS = ROOT / 'tools/state_epoch_pins.tsv'
REGISTRY = ROOT / 'engine/switches.h'
HEX = re.compile(r'^[0-9a-f]{64}$')


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def strip_source(text):
    """C++ text without comments, with each line trimmed and inner blank runs made one space; empty lines go.
    String and character literals stay as they are, so a `//` inside a string is not a comment."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == '//':
            while i < n and text[i] != '\n':
                i += 1
        elif two == '/*':
            end = text.find('*/', i + 2)
            if end < 0:
                raise ValueError('a block comment is not closed')
            out.append(' ')
            i = end + 2
        elif c in '"\'':
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    lines = (' '.join(line.split()) for line in ''.join(out).split('\n'))
    return '\n'.join(line for line in lines if line)


def source_list(path=SOURCES):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')]
    if not names:
        raise ValueError(f'{path}: no source file listed')
    return names


def source_hash(root=ROOT, sources=SOURCES):
    h = hashlib.sha256()
    for name in source_list(sources):
        text = (Path(root) / name).read_text()
        h.update(name.encode() + b'\0' + sha(strip_source(text)).encode() + b'\n')
    return h.hexdigest()


def registry_hash(header=REGISTRY):
    import switch_registry
    h = hashlib.sha256()
    for s in switch_registry.load(header):
        if s.kind in STATE_KINDS:
            h.update(f'{s.name}\0{s.kind}\0{s.default}\n'.encode())
    return h.hexdigest()


def isa_hash(binary, llvm=None, arch='gfx1151'):
    """Hash of the device code of `binary`: function text, kernel descriptors and the bytes of device data objects."""
    import isa_diff
    import tempfile
    llvm = isa_diff.find_llvm(llvm)
    with tempfile.TemporaryDirectory(prefix='state_epoch_') as work:
        co = isa_diff.open_code_object(binary, work, llvm, arch)
        syms, units = isa_diff.disassemble(co, llvm)
        units = isa_diff.strip_fill(units)
        descriptors = isa_diff.kernel_descriptors(co, llvm, syms)
        objects = device_objects(co, llvm)
    h = hashlib.sha256()
    for name in sorted(units):
        if not name.startswith(isa_diff.STUB):
            h.update(b'F' + name.encode() + b'\0' + sha(units[name]).encode() + b'\n')
    for name in sorted(descriptors):
        if not name.startswith(isa_diff.STUB):
            h.update(b'K' + name.encode() + b'\0' + descriptors[name].encode() + b'\n')
    for name in sorted(objects):
        h.update(b'D' + name.encode() + b'\0' + objects[name].encode() + b'\n')
    return h.hexdigest(), len(units), len(descriptors), len(objects)


def device_objects(co, llvm):
    """name -> sha256 of the bytes of each OBJECT symbol that is not a kernel descriptor (device tables and constants)."""
    data = Path(co).read_bytes()
    table = subprocess.run([llvm / 'llvm-readelf', '-sW', str(co)], capture_output=True, text=True, check=True).stdout
    sections = subprocess.run([llvm / 'llvm-readelf', '-SW', str(co)], capture_output=True, text=True, check=True).stdout
    # [Nr] Name Type Address Off Size ...
    offsets = {}
    for line in sections.splitlines():
        m = re.match(r'\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
        if m:
            offsets[int(m.group(1))] = (m.group(3), int(m.group(4), 16), int(m.group(5), 16), int(m.group(6), 16))
    out = {}
    for line in table.splitlines():
        f = line.split()
        # Num: Value Size Type Bind Vis Ndx Name
        if len(f) < 8 or f[3] != 'OBJECT' or f[7].endswith('.kd') or not f[6].isdigit():
            continue
        section = offsets.get(int(f[6]))
        size = int(f[2], 0)
        if section is None or section[0] == 'NOBITS' or size == 0:
            out[f[7]] = f'size{size}' + (f'-{section[0]}' if section else '')
            continue
        start = section[2] + int(f[1], 16) - section[1]
        out[f[7]] = sha(data[start:start + size])
    return out


def combine(source, registry, isa):
    """The epoch from the three parts; a length prefix keeps one part from sliding into the next."""
    h = hashlib.sha256(b'crucible state epoch v1\n')
    for label, part in (('source', source), ('registry', registry), ('isa', isa)):
        h.update(f'{label}:{len(part)}:{part}\n'.encode())
    return h.hexdigest()


def apply_pins(epoch, pins=PINS):
    """(epoch, pinned): the epoch that a pin row names for `epoch`, else `epoch`."""
    found = None
    for number, line in enumerate(Path(pins).read_text().splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 3 or not HEX.match(cols[0]) or not HEX.match(cols[1]) or not cols[2].strip() or set(cols[1]) == {'0'}:
            raise ValueError(f'{pins}:{number}: a pin row is <computed 64 hex> TAB <adopted 64 hex> TAB <evidence>')
        if cols[0] == epoch:
            found = cols[1]
    return (found, True) if found else (epoch, False)


def slot_positions(data):
    found, start = [], 0
    while (at := data.find(PREFIX, start)) >= 0:
        found.append(at)
        start = at + 1
    return found


def read_slot(path):
    """The 64 digits stamped into `path`, or None while the slot holds zeros."""
    data = Path(path).read_bytes()
    positions = slot_positions(data)
    if len(positions) != 1:
        raise ValueError(f'{path}: expected one epoch slot, found {len(positions)}')
    digits = data[positions[0] + len(PREFIX):positions[0] + len(PREFIX) + DIGITS].decode('latin-1')
    return None if digits == '0' * DIGITS else digits


def write_slot(path, epoch):
    if not HEX.match(epoch) or epoch == '0' * DIGITS:
        raise ValueError('an epoch is 64 lower-case hex digits and not all zeros')
    path = Path(path)
    data = bytearray(path.read_bytes())
    positions = slot_positions(bytes(data))
    if len(positions) != 1:
        raise ValueError(f'{path}: expected one epoch slot, found {len(positions)}')
    at = positions[0] + len(PREFIX)
    data[at:at + DIGITS] = epoch.encode()
    # Write beside the binary and rename: a running engine keeps its old inode.
    temporary = path.with_name(path.name + '.epoch.tmp')
    temporary.write_bytes(bytes(data))
    temporary.chmod(path.stat().st_mode)
    temporary.replace(path)


def compute(isa, root=ROOT):
    source, registry = source_hash(root, Path(root) / 'tools/state_epoch_sources.txt'), registry_hash(Path(root) / 'engine/switches.h')
    raw = combine(source, registry, isa)
    epoch, pinned = apply_pins(raw, Path(root) / 'tools/state_epoch_pins.tsv')
    return {'source': source, 'registry': registry, 'isa': isa, 'computed': raw, 'epoch': epoch, 'pinned': pinned}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    st = sub.add_parser('stamp')
    st.add_argument('binary')
    st.add_argument('--isa-hash', help='use this machine-code hash and do not read the binary (tests)')
    st.add_argument('--llvm')
    sh = sub.add_parser('show')
    sh.add_argument('binary')
    pt = sub.add_parser('parts')
    pt.add_argument('--isa')
    pt.add_argument('--llvm')
    args = ap.parse_args(argv)
    if args.cmd == 'show':
        print(f'epoch={read_slot(args.binary) or "unstamped"}')
        return 0
    if args.cmd == 'parts' and not args.isa:
        print(f'source={source_hash()}\nregistry={registry_hash()}')
        return 0
    if args.cmd == 'parts' or not args.isa_hash:
        digest, functions, descriptors, objects = isa_hash(args.isa if args.cmd == 'parts' else args.binary, args.llvm)
        detail = f' functions={functions} descriptors={descriptors} objects={objects}'
    else:
        digest, detail = args.isa_hash, ''
    parts = compute(digest)
    if args.cmd == 'parts':
        print('\n'.join(f'{k}={v}' for k, v in parts.items()) + detail)
        return 0
    write_slot(args.binary, parts['epoch'])
    print(f'state_epoch stamped epoch={parts["epoch"]} pinned={int(parts["pinned"])} source={parts["source"][:16]} '
          f'registry={parts["registry"][:16]} isa={parts["isa"][:16]}{detail}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
