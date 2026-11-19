#!/usr/bin/env python3
"""Apply one local patch to one vendored upstream file at build time (third_party/README.md).

The tree keeps the upstream file byte for byte (the leak guard pins it); the change is a unified diff
in third_party/patches/. This applier is strict: one file, hunks in order at their exact line numbers, every context
and removed line equal. No offset and no fuzz: a patch that does not fit stops the build. Text before the first `---`
line is the description of the patch. Standard library only, so the build needs no `patch` program.

    python3 -I tools/vendor_patch.py SOURCE PATCH OUT --path REPO_PATH
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_HUNK = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    pass


def apply(source: bytes, patch: bytes, path: str) -> bytes:
    """SOURCE with PATCH applied. PATCH must name PATH as its one file (a/PATH and b/PATH)."""
    lines = patch.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(b"--- "))
    except StopIteration:
        raise PatchError("no file header") from None
    want = path.encode()
    if lines[start].rstrip(b"\r\n") != b"--- a/" + want or start + 1 >= len(lines) \
            or lines[start + 1].rstrip(b"\r\n") != b"+++ b/" + want:
        raise PatchError("the patch does not name the file")
    old = source.splitlines(keepends=True)
    out: list[bytes] = []
    pos = 0             # lines of OLD copied or consumed
    shift = 0           # new line number - old line number, after the hunks so far
    i = start + 2
    if i >= len(lines):
        raise PatchError("no hunk")
    while i < len(lines):
        match = _HUNK.match(lines[i])
        if not match:
            raise PatchError(f"line {i + 1} of the patch is not a hunk header")
        o_start, o_count = int(match[1]), int(match[2] or 1)
        n_start, n_count = int(match[3]), int(match[4] or 1)
        first = o_start - 1 if o_count else o_start
        if first < pos or n_start - o_start != shift:
            raise PatchError("hunks overlap, are out of order or the new line numbers do not follow")
        out += old[pos:first]
        pos = first
        i += 1
        seen_old = seen_new = 0
        while i < len(lines) and not lines[i].startswith(b"@@"):
            line = lines[i]
            tag, body = line[:1], line[1:]
            if tag in (b" ", b"-"):
                if pos >= len(old) or old[pos] != body:
                    raise PatchError(f"hunk at old line {o_start}: line {pos + 1} of the file does not match")
                pos += 1
                seen_old += 1
                if tag == b" ":
                    out.append(body)
                    seen_new += 1
            elif tag == b"+":
                out.append(body)
                seen_new += 1
            elif line.startswith(b"\\ No newline"):
                pass
            else:
                raise PatchError(f"line {i + 1} of the patch is not part of a hunk")
            i += 1
            if seen_old > o_count or seen_new > n_count:
                raise PatchError(f"hunk at old line {o_start} is longer than its counts")
        if (seen_old, seen_new) != (o_count, n_count):
            raise PatchError(f"hunk at old line {o_start} does not match its counts")
        shift += n_count - o_count
    out += old[pos:]
    return b"".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("patch", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--path", required=True, help="the repo path that the patch names")
    args = parser.parse_args(argv)
    try:
        data = apply(args.source.read_bytes(), args.patch.read_bytes(), args.path)
    except PatchError as error:
        print(f"vendor_patch: {args.patch}: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
