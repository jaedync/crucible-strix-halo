#!/usr/bin/env python3
"""A digest of the state and block files in a disk-tier folder that leaves out the identity.

A state file and a block file name the identity of the engine that wrote them (engine/runtime/state_format.h), and the
file names hash the identity too. Two engines with another identity never write equal files, even when the state bytes
are equal. The digest drops the identity string and keeps every other byte:

  state file: payload size, the index section after the identity, the resume section, and all bytes after the header
              (the rest of the snapshot, with its zero padding). The header checksums are left out; they cover the identity.
  block file: the tenant, the block digest and the chunk sums, and the payload (the block stream).

  state_digest.py digest DIR            one line per file, sorted: `state SHA payload=N bytes=N`, `block SHA bytes=N`
  state_digest.py compare A B           exit 0 when equal, 1 when different (the lines that differ follow), 2 on an error

DIR is a folder with state files, or a disk-tier root with exactly one identity folder. The tool reads and prints
digests only; no file content or token leaves it.
"""
import hashlib
from pathlib import Path
import re
import struct
import sys

FIXED = 56
STATE_MAGIC, STATE_VERSION = 0x3254415453435243, 2
BLOCK_MAGIC, BLOCK_VERSION = 0x314B4F4C42435243, 1
ALIGN = 4096
IDENTITY_DIR = re.compile(r'^[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}$')
READ = 1 << 20


def fixed_part(head, magic, version, name):
    if len(head) < FIXED:
        raise ValueError(f'{name}: shorter than the fixed header')
    m, v, _, header_bytes, used, payload, _chunk, index_used = struct.unpack('<QIIQQQQQ', head[:FIXED])
    if m != magic or v != version:
        raise ValueError(f'{name}: wrong magic or version')
    if not (FIXED + 8 <= index_used <= used <= header_bytes) or header_bytes % ALIGN:
        raise ValueError(f'{name}: bad header bounds')
    return header_bytes, used, payload, index_used


def hash_tail(handle, start, count, h):
    """Hash `count` bytes from `start` (count None: to the end of the file); return how many bytes were read."""
    handle.seek(start)
    done = 0
    while count is None or done < count:
        chunk = handle.read(READ if count is None else min(READ, count - done))
        if not chunk:
            break
        h.update(chunk)
        done += len(chunk)
    if count is not None and done != count:
        raise ValueError('the file is shorter than its header says')
    return done


def state_line(path):
    size = path.stat().st_size
    with open(path, 'rb') as f:
        head = f.read(ALIGN * 4)
        header_bytes, used, payload, index_used = fixed_part(head, STATE_MAGIC, STATE_VERSION, path.name)
        if size < header_bytes or size % ALIGN:
            raise ValueError(f'{path.name}: shorter than its header, or not a whole number of {ALIGN}-byte blocks')
        if header_bytes > len(head):
            f.seek(0)
            head = f.read(header_bytes)
        (n,) = struct.unpack('<Q', head[FIXED:FIXED + 8])
        after = FIXED + 8 + n
        if n > 4096 or after > index_used - 8:
            raise ValueError(f'{path.name}: bad identity length')
        h = hashlib.sha256(b'crucible-state-digest-v1\0')
        h.update(struct.pack('<Q', payload))
        h.update(head[after:index_used - 8])      # the index after the identity, without its checksum
        h.update(head[index_used:used - 8])       # the resume section, without its checksum
        rest = hash_tail(f, header_bytes, None, h)
    return f'state {h.hexdigest()} payload={payload} bytes={rest + 0}'


def block_line(path):
    size = path.stat().st_size
    with open(path, 'rb') as f:
        head = f.read(ALIGN * 4)
        header_bytes, used, payload, _ = fixed_part(head, BLOCK_MAGIC, BLOCK_VERSION, path.name)
        if size != header_bytes + (payload + ALIGN - 1) // ALIGN * ALIGN:
            raise ValueError(f'{path.name}: its size does not match its header')
        if used > len(head):
            f.seek(0)
            head = f.read(used)
        (n,) = struct.unpack('<Q', head[FIXED:FIXED + 8])
        after = FIXED + 8 + n
        if n > 4096 or after > used - 8:
            raise ValueError(f'{path.name}: bad identity length')
        h = hashlib.sha256(b'crucible-block-digest-v1\0')
        h.update(head[after:used - 8])  # tenant, block digest and chunk sums; the checksum covers the identity
        hash_tail(f, header_bytes, payload, h)
    return f'block {h.hexdigest()} bytes={payload}'


def folder_of(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'{path}: not a folder')
    if any(path.glob('*.state')) or any(path.glob('*.block')):
        return path
    inner = [p for p in path.iterdir() if p.is_dir() and IDENTITY_DIR.match(p.name)]
    if len(inner) != 1:
        raise ValueError(f'{path}: expected state files or one identity folder, found {len(inner)} identity folders')
    return inner[0]


def digest(path):
    """Sorted digest lines of the state and block files in `path`; an error when there is no state file."""
    folder = folder_of(path)
    lines = [state_line(p) for p in sorted(folder.glob('*.state'))] + [block_line(p) for p in sorted(folder.glob('*.block'))]
    if not any(line.startswith('state ') for line in lines):
        raise ValueError(f'{path}: no state file')
    return sorted(lines)


def diff_lines(la, lb):
    """The lines that only one side has, prefixed `A ` or `B `; empty when both hold equal states."""
    out = []
    for kind, left, right in (('A', la, lb), ('B', lb, la)):
        spare = list(right)
        for line in left:
            if line in spare:
                spare.remove(line)
            else:
                out.append(f'{kind} {line}')
    return out


def compare(a, b):
    return diff_lines(digest(a), digest(b))


def main(argv):
    if len(argv) == 2 and argv[0] == 'digest':
        print('\n'.join(digest(argv[1])))
        return 0
    if len(argv) == 3 and argv[0] == 'compare':
        diff = compare(argv[1], argv[2])
        print('DIFFERENT' if diff else 'EQUAL', f'states={sum(l.startswith("state ") for l in digest(argv[1]))}',
              f'blocks={sum(l.startswith("block ") for l in digest(argv[1]))}')
        for line in diff:
            print(line)
        return 1 if diff else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, OSError) as error:
        print(f'ERROR {error}')
        sys.exit(2)
