#!/usr/bin/env python3
"""Print the server arguments and the environment of the measured run (tools/crucible_cli/measured.py) for a container
recipe, so that a recipe copies these lines and never the values by hand.

    python3 tools/recipe_env.py                      # the working tree
    python3 tools/recipe_env.py --ref v0.38.1        # measured.py at a release tag (any git ref)
    python3 tools/recipe_env.py --ref v0.38.1 --format cpp

Formats: text (one "arg NAME VALUE" or "env NAME=VALUE" line each, after one "#" line), json, cpp (two initializer
lists of string pairs). The file is parsed, not run: only the literal dicts ENV and SERVER_FLAGS count, in their order.
A recipe can set its own --ctx and --streams (the user's choice); the printed values are the measured ones.
tests/test_recipe_env.py checks that the printed set equals measured.py.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASURED = "tools/crucible_cli/measured.py"
TABLES = ("SERVER_FLAGS", "ENV")


class TableError(Exception):
    pass


def source(repo, ref):
    """The text of measured.py in REPO at REF (None: the working tree)."""
    if ref is None:
        return (Path(repo) / MEASURED).read_text()
    r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{MEASURED}"], capture_output=True, text=True,
                       timeout=60)
    if r.returncode != 0:
        raise TableError(f"cannot read {MEASURED} at {ref}: {r.stderr.strip()}")
    return r.stdout


def tables(text):
    """{"SERVER_FLAGS": [(name, value), ...], "ENV": [...]} from the literal dicts of measured.py."""
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in TABLES:
            name = node.targets[0].id
            try:
                value = ast.literal_eval(node.value)
            except ValueError as e:
                raise TableError(f"{name} in {MEASURED} is not a literal dict: {e}") from None
            if not isinstance(value, dict):
                raise TableError(f"{name} in {MEASURED} is not a dict")
            for k, v in value.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise TableError(f"{name} in {MEASURED}: {k!r}: {v!r} is not a string pair")
            found[name] = list(value.items())
    missing = [name for name in TABLES if name not in found]
    if missing:
        raise TableError(f"{MEASURED} has no {' and no '.join(missing)}")
    return found


def render(t, ref, fmt):
    where = ref if ref is not None else "the working tree"
    if fmt == "json":
        return json.dumps({"ref": ref, "args": [list(kv) for kv in t["SERVER_FLAGS"]],
                           "env": [list(kv) for kv in t["ENV"]]}, indent=2) + "\n"
    if fmt == "cpp":
        def pairs(items):
            return "".join(f"    {{{json.dumps(k)}, {json.dumps(v)}}},\n" for k, v in items)
        command = "python3 tools/recipe_env.py" + (f" --ref {ref}" if ref is not None else "") + " --format cpp"
        return (f"// From {MEASURED} at {where} ({command}).\n"
                "static const std::vector<std::pair<std::string, std::string>> kCrucibleArgs = {\n"
                + pairs(t["SERVER_FLAGS"]) + "};\n"
                "static const std::vector<std::pair<std::string, std::string>> kCrucibleEnv = {\n"
                + pairs(t["ENV"]) + "};\n")
    lines = [f"# {MEASURED} at {where}"]
    lines += [f"arg {k} {v}" for k, v in t["SERVER_FLAGS"]]
    lines += [f"env {k}={v}" for k, v in t["ENV"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", help="a git ref, for example a release tag (default: the working tree)")
    p.add_argument("--repo", default=str(ROOT), help="the repository (default: this one)")
    p.add_argument("--format", choices=("text", "json", "cpp"), default="text")
    args = p.parse_args(argv)
    try:
        text = render(tables(source(args.repo, args.ref)), args.ref, args.format)
    except (TableError, OSError, SyntaxError) as e:
        print(f"recipe_env: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
