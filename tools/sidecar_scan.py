#!/usr/bin/env python3
"""Count the 'mtp sidecar:' lines in the *.log files under a folder. A run on public files must show none.

  sidecar_scan.py DIR [DIR ...] [--min-logs N]

Prints one last line `SIDECAR SCAN PASS|FAIL logs=<n> sidecar_lines=<k>`. Exit 0 when k is 0, 1 when k is more, 2 when DIR
holds fewer than N logs in all (default 1). It prints counts and the relative paths of the logs that hold a line. It prints no log text.
"""
import argparse
from pathlib import Path
import sys

MARK = "mtp sidecar:"


def scan(roots):
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    count, hits = 0, {}
    for root in roots:
        for log in sorted(root.rglob("*.log")):
            count += 1
            with open(log, errors="replace") as f:
                n = sum(1 for line in f if line.startswith(MARK))
            if n:
                hits[f"{root.name}/{log.relative_to(root)}"] = n
    return count, hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="+")
    ap.add_argument("--min-logs", type=int, default=1)
    args = ap.parse_args(argv)
    for d in args.dir:
        if not Path(d).is_dir():
            print(f"sidecar_scan: {d} is not a folder", file=sys.stderr)
            return 2
    count, hits = scan(args.dir)
    if count < args.min_logs:
        print(f"sidecar_scan: {count} *.log file(s) under the folder, need {args.min_logs}", file=sys.stderr)
        return 2
    for name, n in hits.items():
        print(f"{name}: {n}")
    total = sum(hits.values())
    print(f"SIDECAR SCAN {'FAIL' if total else 'PASS'} logs={count} sidecar_lines={total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
