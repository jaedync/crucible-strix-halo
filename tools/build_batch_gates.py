#!/usr/bin/env python3
"""Freeze one source tree and build its engine and batch probe with the same numeric flags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# tests load this file as tools.build_batch_gates, so tools/ may be missing from sys.path.
sys.path.append(str(Path(__file__).resolve().parent))
from proc_tree import run_tree  # noqa: E402

COMPILER = Path("/opt/rocm/bin/hipcc")
FLAGS = ["-O3", "-std=c++20", "--offload-arch=gfx1151", "--rocm-path=/opt/rocm",
         "-I/opt/rocm/include", "--no-default-config"]
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as file: return hashlib.file_digest(file, "sha256").hexdigest()


def source_hashes(root):
    files = [path for folder in ("engine", "kernels", "bench") for path in (root / folder).rglob("*")
             if path.suffix in (".h", ".hip", ".inc") and path.is_file()]
    if any(path.is_symlink() for path in files): raise ValueError("source files cannot be symbolic links")
    return {str(path.relative_to(root)): digest(path) for path in sorted(files)}


def source_digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compiler_identity():
    version = subprocess.run([str(COMPILER), "--version"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    return dict(path=str(COMPILER), hipcc_sha256=digest(COMPILER),
                clang_sha256=digest(Path("/opt/rocm/lib/llvm/bin/clang++").resolve()), version=version)


def build(source, output, source_commit):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink(): raise ValueError("build output must be a new directory")
    original = source_hashes(source)
    output.mkdir(parents=True)
    frozen = output / "source"
    for name in original:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    if source_hashes(frozen) != original or source_hashes(source) != original:
        raise ValueError("source changed during the freeze")
    toolchain = compiler_identity()
    fingerprint = source_digest(original)
    env = dict(PATH="/opt/rocm/bin:/usr/bin:/bin", HOME=str(Path.home()), LANG="C.UTF-8")
    candidate, probe = output / "crucible", output / "batch_gates"
    commands = []
    for kind, binary, entry in (("candidate", candidate, "engine/engine.hip"),
                                ("batch-probe", probe, "bench/batch_gates.hip")):
        binding = [] if kind == "candidate" else [f'-DBATCHING_ENGINE_SHA="{digest(candidate)}"',
                                                   f'-DBATCHING_SOURCE_SHA="{fingerprint}"']
        command = [str(COMPILER), *FLAGS, *binding, "-o", str(binary), str(frozen / entry), "-lpthread"]
        completed = run_tree(command, env=env, timeout=300)  # hipcc runs clang and the offload tools: end them all at the limit
        (output / f"{kind}.stdout.log").write_bytes(completed.stdout)
        (output / f"{kind}.stderr.log").write_bytes(completed.stderr)
        if completed.returncode != 0: raise ValueError(f"{kind} build failed; read its stderr log")
        if source_hashes(frozen) != original or compiler_identity() != toolchain:
            raise ValueError("source or compiler changed during compilation")
        record = dict(schema=1, builder="crucible-batch-v1", builder_sha256=digest(Path(__file__)),
                      kind=kind, binary_sha256=digest(binary), source_hashes=original,
                      source_root=str(frozen), source_commit=source_commit, numeric_flags=FLAGS,
                      compiler=toolchain, command=command)
        if kind == "batch-probe": record["candidate_binary_sha256"] = digest(candidate)
        (output / f"{binary.name}.build.json").write_text(json.dumps(record, indent=2) + "\n")
        commands.append(command)
    return dict(engine=candidate, probe=probe, engine_record=output / "crucible.build.json",
                probe_record=output / "batch_gates.build.json", commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = build(args.source, args.output, args.source_commit)
    print(json.dumps({key: str(value) for key, value in result.items() if key != "commands"}, indent=2))

if __name__ == "__main__": main()
