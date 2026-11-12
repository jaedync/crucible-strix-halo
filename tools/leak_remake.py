#!/usr/bin/env python3
"""Re-make and compare the generated data of a leak guard allow entry (design A).

A data-shaped file that the guard cannot read passes only with an allow entry that names its SOURCE. For data that a
generator in the tree makes from public inputs, the source is a record:

    {"kind": "generated", "generator": "tools/make_x.py", "generator_sha256": "<64 hex>", "output": "x.txt",
     "regen_command": "python3 -I {generator} {out}", "inputs": "text: where the public inputs come from"}

The guard never runs the generator: it checks that the generator in the tree has the pinned sha256. This tool does run
it, so that a person can prove the bytes:

    python3 tools/leak_remake.py check ALLOW.json [PATH...]    re-make each generated entry in a clean temporary folder
                                                               and compare the bytes with the entry sha256 and the file

`check` prints PASS, FAIL or SKIP for each entry and exits 1 on any FAIL. A template holds `{out}` (the clean output
folder), `{generator}`, `{ckpt}` (the public tokenizer folder), `{python}` (LEAK_REMAKE_PYTHON, else this interpreter) and
`{rocm}` (LEAK_REMAKE_ROCM, else /opt/rocm: a public ROCm tree). Only entries whose source has kind "generated" are checked.

Optional fields of a record:
    "compare": "npz"                  compare the ARRAYS of an .npz (a zip has time stamps), not the bytes
    "requires": ["rocm", "torch"]     a tool that this machine may lack; the entry is SKIPPED, and --strict makes a skip a failure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REGEN_TIMEOUT = 900
DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(entry: dict) -> dict:
    """The record of an entry: its source when that is a record, else the entry itself (a bare generator record)."""
    source = entry.get("source")
    return source if isinstance(source, dict) else entry


def checkpoint() -> str:
    """The public tokenizer folder that {ckpt} stands for: LEAK_REMAKE_CKPT, else the one that the cov7 tests name."""
    return os.environ.get("LEAK_REMAKE_CKPT") or os.environ.get("LEAK_R7_CKPT") or str(Path.home() / ".cache/leak-r7/tok/ckpt")


def python() -> str:
    """The interpreter that {python} stands for (a user venv with torch or pyarrow, for a generator that needs them)."""
    return os.environ.get("LEAK_REMAKE_PYTHON") or sys.executable


def rocm() -> str:
    """The public ROCm tree that {rocm} stands for."""
    return os.environ.get("LEAK_REMAKE_ROCM") or "/opt/rocm"


def expand(template: str, root: Path, entry: dict, out: Path, work: Path | None = None) -> list[str]:
    """{out} is the clean output folder, {generator} the generator (in WORK when the record has `copy`), {ckpt} the public
    tokenizer folder, {python} the interpreter, {rocm} the ROCm tree."""
    base = Path(work) if work else Path(root)
    return [part.replace("{out}", str(out)).replace("{generator}", str(base / _source(entry)["generator"]))
            .replace("{ckpt}", checkpoint()).replace("{python}", python()).replace("{rocm}", rocm())
            for part in shlex.split(template)]


def unmet(entry: dict) -> list[str]:
    """The names in `requires` that this machine lacks."""
    missing = []
    for need in _source(entry).get("requires", []):
        if need == "rocm":
            ok = (Path(rocm()) / "bin/hipcc").is_file()
        elif need == "torch":
            ok = subprocess.run([python(), "-I", "-c", "import torch"], capture_output=True).returncode == 0
        else:
            ok = False
        if not ok:
            missing.append(need)
    return missing


def npz_difference(made: bytes, body: bytes) -> str | None:
    """None when two .npz files hold the same arrays (names, dtypes, shapes, bytes); else the first difference."""
    import io

    import numpy as np
    with np.load(io.BytesIO(made)) as a, np.load(io.BytesIO(body)) as b:
        if sorted(a.files) != sorted(b.files):
            return "the array names differ"
        for name in sorted(a.files):
            x, y = a[name], b[name]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return f"the array {name} differs"
    return None


def generator_reason(root: Path, entry: dict) -> str | None:
    """Why the generator in the tree is not the pinned file, or None. An entry with no pin is not checked here."""
    source = _source(entry)
    try:
        body = (Path(root) / source["generator"]).read_bytes()
    except (OSError, KeyError):
        return f"the generator {source.get('generator')} is not in the tree"
    pinned = source.get("generator_sha256")
    if pinned is not None and sha(body) != pinned:
        return f"the generator {source['generator']} does not have the pinned sha256"
    return None


def run_generator(root: Path, entry: dict) -> tuple[dict[str, bytes] | None, str | None]:
    """Run the generator of ENTRY in a clean temporary folder. Return the files that it made, or the reason why not.

    A record with `copy` (a list of paths in the tree) and `output_dir` runs the generator in a temporary copy of those
    paths and the generator, so that a generator that writes next to itself never touches the tree. The files in
    `output_dir` of the copy are what it made."""
    source = _source(entry)
    with tempfile.TemporaryDirectory(prefix="leak-remake-") as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        work = None
        if source.get("copy") is not None:
            work = Path(tmp) / "work"
            for item in [*source["copy"], source.get("generator", "")]:
                rel, target = (item if isinstance(item, list) and len(item) == 2 else (item, item))   # [from, to] renames
                src = Path(root) / rel
                if not rel or not src.exists():
                    return None, f"the copy list names {rel} that is not in the tree"
                dest = work / target
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest) if src.is_dir() else shutil.copy2(src, dest)
        for item in source.get("fetch", []):                  # a public input, pinned by its sha256
            try:
                with urllib.request.urlopen(item["url"], timeout=120) as reply:
                    body = reply.read()
            except (OSError, ValueError) as error:
                return None, f"cannot fetch {item.get('url')} ({type(error).__name__})"
            if sha(body) != item.get("sha256"):
                return None, f"{item['url']} does not have the pinned sha256"
            (work or Path(tmp)).joinpath(item["to"]).write_bytes(body)
        try:
            command = expand(source["regen_command"], root, entry, out, work)
        except KeyError as error:
            return None, f"the entry has no {error.args[0]}"
        try:
            done = subprocess.run(command, cwd=str(work or root), capture_output=True, timeout=REGEN_TIMEOUT)
        except OSError as error:
            return None, f"cannot run the generator ({type(error).__name__})"
        except subprocess.TimeoutExpired:
            return None, f"the generator did not finish in {REGEN_TIMEOUT} seconds"
        if done.returncode != 0:
            tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
            return None, f"the generator exited {done.returncode}" + (f": {tail[0][:200]}" if tail else "")
        where = (work / source["output_dir"]) if work and source.get("output_dir") else out
        made = {path.relative_to(where).as_posix(): path.read_bytes() for path in sorted(where.rglob("*")) if path.is_file()}
        if not made:
            return None, "the generator made no file"
        return made, None


def check_entry(root: Path, entry: dict) -> str | None:
    """None when the generator of ENTRY makes exactly the bytes of the entry (its sha256) and of the file in the tree;
    else the reason."""
    source = _source(entry)
    why = generator_reason(root, entry)
    if why:
        return why
    made, error = run_generator(root, entry)
    if error:
        return error
    name = source.get("output") or Path(entry.get("path", "")).name
    if name not in (made or {}):
        return f"the generator does not make {name}"
    try:
        body = (Path(root) / entry["path"]).read_bytes()
    except (OSError, KeyError):
        return "the file is not in the tree"
    if source.get("compare") == "npz":                    # the file in the tree is pinned by its sha256 in the entry
        digest = entry.get("sha256")
        if digest is not None and sha(body) != digest:
            return "the file in the tree does not have the entry sha256"
        why = npz_difference(made[name], body)
        return f"the arrays in the file are not what the generator makes ({why})" if why else None
    digest = entry.get("sha256")
    if digest is not None and sha(made[name]) != digest:
        return "the generator makes other bytes than the entry sha256"
    if body != made[name]:
        return "the file in the tree is not what the generator makes"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check",))
    parser.add_argument("allow", type=Path, help="an allow list (JSON) or a publish.toml")
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--strict", action="store_true", help="a SKIP (a tool that the machine lacks) counts as a failure")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.allow.suffix == ".toml":
        import tomllib
        entries = tomllib.loads(args.allow.read_text()).get("data_allow", [])
    else:
        doc = json.loads(args.allow.read_text())
        entries = doc.get("files", []) if isinstance(doc, dict) else []
    failures = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("source"), dict) \
                or entry["source"].get("kind") != "generated":
            continue
        if args.paths and entry.get("path") not in args.paths:
            continue
        lacking = unmet(entry)
        if lacking:
            print(f"SKIP {entry.get('path')}: this machine lacks {', '.join(lacking)}")
            failures += bool(args.strict)
            continue
        why = check_entry(root, entry)
        print(f"PASS {entry.get('path')}" if why is None else f"FAIL {entry.get('path')}: {why}")
        failures += why is not None
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
