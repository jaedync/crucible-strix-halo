#!/usr/bin/env python3
"""The option table of `./crucible setup` that install.sh checks before any download.

install.sh installs the release PIN_TAG, so its table SETUP_OPTIONS must be the setup parser of that release, not the
parser of this tree: a newer option would pass the early check, and the older setup would then refuse it after the
downloads. tests/data/setup_options/<tag>.txt is the table of each pinned release; tests/test_install_sh.py keeps
SETUP_OPTIONS equal to the file of PIN_TAG.

Use:
    python3 -I tools/setup_options.py table [--repo DIR]
        print the table of the setup parser in DIR (default: this tree), one `NAME KIND` line for each option.
    python3 -I tools/setup_options.py record vX.Y.Z [--repo DIR] [--install FILE] [--out DIR]
        on the release commit: write the table of DIR to tests/data/setup_options/vX.Y.Z.txt and put it into the
        SETUP_OPTIONS block of install.sh. It does not change PIN_TAG (the release constants commit sets it).
Standard library only. Exit 1 on an error, with no change to a file.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "tests/data/setup_options"
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
BLOCK_RE = re.compile(r'^SETUP_OPTIONS="\n(.*?)"\n', re.S | re.M)
PIN_TAG_RE = re.compile(r'^PIN_TAG="(v\d+\.\d+\.\d+)"$', re.M)


class OptionsError(Exception):
    pass


def _kind(action, cli):
    if action.choices is not None:
        if isinstance(action.choices, range):
            return f"int:{action.choices.start}-{action.choices.stop - 1}"
        return "choice:" + ",".join(str(c) for c in action.choices)
    if action.nargs == 0:
        return "flag"
    if action.type is int:
        return "int"
    if action.type is getattr(cli, "_gib_day", object()):
        return "int:0-65536"
    return "value"


def table(repo=ROOT):
    """{option name: kind} of the setup parser in repo/tools/crucible_cli, in parser order."""
    tools = str(Path(repo).resolve() / "tools")
    if "crucible_cli" in sys.modules and not str(Path(sys.modules["crucible_cli"].__file__)).startswith(tools):
        raise OptionsError("crucible_cli of another tree is already loaded; run the tool in its own process")
    sys.path.insert(0, tools)
    try:
        from crucible_cli import cli
    finally:
        sys.path.remove(tools)
    sub = next(a for a in cli.parser(Path(repo))._actions if a.dest == "cmd").choices["setup"]
    return {name: _kind(a, cli) for a in sub._actions for name in a.option_strings}


def text(options):
    return "".join(f"{name} {kind}\n" for name, kind in options.items())


def parse(block):
    return dict(line.split() for line in block.splitlines() if line.strip())


def install_table(install_text):
    match = BLOCK_RE.search(install_text)
    if not match:
        raise OptionsError('install.sh has no SETUP_OPTIONS="..." block')
    return parse(match.group(1))


def pin_tag(install_text):
    match = PIN_TAG_RE.search(install_text)
    if not match:
        raise OptionsError('install.sh has no PIN_TAG="vX.Y.Z" line')
    return match.group(1)


def record(tag, repo=ROOT, install=ROOT / "install.sh", out=RECORDS):
    if not TAG_RE.match(tag):
        raise OptionsError(f"the tag must look like v0.38.0, not {tag!r}")
    body = text(table(repo))
    old = Path(install).read_text()
    install_table(old)
    new = BLOCK_RE.sub(lambda m: f'SETUP_OPTIONS="\n{body}"\n', old, count=1)
    out, rec = Path(out), Path(out) / f"{tag}.txt"
    made, before = not out.exists(), rec.read_text() if rec.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    rec.write_text(body)
    try:
        Path(install).write_text(new)
    except OSError as exc:  # both files or none: put the record back as it was
        if before is None:
            rec.unlink()
            if made:
                out.rmdir()
        else:
            rec.write_text(before)
        raise OptionsError(f"cannot write {install}: {exc.strerror}") from exc
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(prog="setup_options.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--repo", type=Path, default=ROOT)
    r = sub.add_parser("record")
    r.add_argument("tag")
    r.add_argument("--repo", type=Path, default=ROOT)
    r.add_argument("--install", type=Path, default=ROOT / "install.sh")
    r.add_argument("--out", type=Path, default=RECORDS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "table":
            sys.stdout.write(text(table(args.repo)))
        else:
            body = record(args.tag, args.repo, args.install, args.out)
            print(f"recorded {len(body.splitlines())} options of {args.tag}; commit the record and install.sh")
    except OptionsError as exc:
        print(f"setup_options: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
