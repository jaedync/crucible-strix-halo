#!/usr/bin/env python3
"""The self-test of `make update` (tools/update.sh).

  selftest.py cpu VERSION         checks that need no GPU and no model
  selftest.py gpu CKPT            a few greedy tokens on the model in CKPT, against tools/update_selftest/<pack>.json
  selftest.py record CKPT         write that file from the build in build/engine/crucible (a maintainer does this)

Exit codes: 0 passed, 1 failed, 3 the GPU part has no expected file for this pack (skipped, not failed).

The expected file holds token IDs that this release produced on a qualified machine. Greedy decoding is exact, so a
different engine build or a different GPU stack that changes one ID has changed the output, and the update rolls back.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIR = ROOT / "tools" / "update_selftest"
ENGINE = ROOT / "build" / "engine" / "crucible"
PACK_PREFIX = "Qwen3.8-Flash-Next-exl3-"
# "Hello, world!" in the Qwen tokenizer. Any valid IDs work: the test compares the engine with itself.
PROMPT_IDS = [9707, 11, 1879, 0]
TOKENS = 16
CONTEXT = 1024
GEN_TIMEOUT_SECONDS = 1800
SKIPPED = 3


def pack_name(ckpt):
    name = Path(ckpt).resolve().name
    return name[len(PACK_PREFIX):] if name.startswith(PACK_PREFIX) else name


def expected_path(ckpt):
    pack = pack_name(ckpt)
    # The name comes from a folder name: it must stay one plain file name inside the directory.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", pack) or pack.startswith("."):
        return None
    return EXPECTED_DIR / f"{pack}.json"


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f": {detail}" if detail and not ok else ""))
    return ok


def cpu(version, root=ROOT, engine=ENGINE):
    print(f"CPU self-test for v{version}")
    results = []
    file_version = (root / "VERSION").read_text().strip() if (root / "VERSION").is_file() else None
    results.append(check("VERSION file matches the release tag", file_version == version, f"file says {file_version}"))
    header = root / "engine" / "cli" / "version.h"
    found = re.search(r'#define ENGINE_BUILD_VERSION "([^"]+)"', header.read_text()) if header.is_file() else None
    results.append(check("engine/cli/version.h matches", bool(found) and found.group(1) == version,
                         found.group(1) if found else "no version line"))
    sys.path.insert(0, str(root))
    try:
        import server.http  # noqa: F401
        import server.update_check  # noqa: F401
        loaded = True
    except Exception as error:  # any failure to load the server code is a failed self-test
        loaded = False
        results.append(check("server modules load", False, repr(error)))
    if loaded:
        results.append(check("server modules load", True))
    if not (engine.is_file() and engine.stat().st_mode & 0o111):
        results.append(check("engine binary exists", False, str(engine)))
    else:
        done = subprocess.run([str(engine), "--version"], capture_output=True, text=True, timeout=60)
        results.append(check("engine binary reports the version", done.returncode == 0 and
                             done.stdout.strip() == f"crucible {version}", (done.stdout + done.stderr).strip()[:200]))
    return 0 if all(results) else 1


def generate(ckpt, ids, count, engine=ENGINE):
    command = [str(engine), "gen", str(ckpt), "--ids", ",".join(map(str, ids)), "--n", str(count), "--ctx", str(CONTEXT)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=GEN_TIMEOUT_SECONDS)
    if done.returncode:
        raise RuntimeError(f"the engine stopped with code {done.returncode}: {done.stderr.strip()[-300:]}")
    for line in done.stdout.splitlines():
        if line.startswith("ids:"):
            return [int(token) for token in line[4:].split()]
    raise RuntimeError("the engine printed no token IDs")


def load_expected(path):
    data = json.loads(path.read_text())
    ids, count, expected = data["prompt_ids"], data["n"], data["expected"]
    if not (isinstance(ids, list) and isinstance(expected, list) and isinstance(count, int) and
            all(isinstance(token, int) for token in ids + expected) and len(expected) > 0):
        raise ValueError("the expected file is malformed")
    return ids, count, expected


def gpu(ckpt, engine=ENGINE):
    path = expected_path(ckpt)
    if path is None or not path.is_file():
        print(f"GPU self-test skipped: this release has no expected token IDs for the pack '{pack_name(ckpt)}'.")
        return SKIPPED
    try:
        ids, count, expected = load_expected(path)
        got = generate(ckpt, ids, count, engine)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"GPU self-test failed: {error}")
        return 1
    if got != expected:
        print(f"GPU self-test failed: the tokens differ from {path.name}.\n  expected {expected}\n  got      {got}")
        return 1
    print(f"GPU self-test passed: {len(got)} greedy tokens match {path.name}.")
    return 0


def record(ckpt, engine=ENGINE):
    path = expected_path(ckpt)
    if path is None:
        print("The model folder name is not a usable pack name.")
        return 1
    got = generate(ckpt, PROMPT_IDS, TOKENS, engine)
    path.write_text(json.dumps(dict(pack=pack_name(ckpt), prompt_ids=PROMPT_IDS, n=TOKENS, expected=got), indent=2) + "\n")
    print(f"wrote {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="part", required=True)
    sub.add_parser("cpu").add_argument("version")
    sub.add_parser("gpu").add_argument("ckpt")
    sub.add_parser("record").add_argument("ckpt")
    args = parser.parse_args(argv)
    if args.part == "cpu":
        return cpu(args.version)
    return {"gpu": gpu, "record": record}[args.part](args.ckpt)


if __name__ == "__main__":
    sys.exit(main())
