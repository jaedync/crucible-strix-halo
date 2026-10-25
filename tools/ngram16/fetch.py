#!/usr/bin/env python3
"""Fetch the source files of the 16-bit n-gram table and build the BF16 file, in one command.

Usage:
  fetch.py [--out DIR] [--dry-run] [--delete-source] [--rate-mb 100]

Steps:
  1. Read model.safetensors.index.json of Qwen/Qwen3.8-Flash-Next at one fixed revision and
     pick only the files that hold *.ngram_embedding.shard_*.weight tensors.
  2. Print the files and the download size. Stop if the disk has too little free space
     (downloads that are still missing + the output file). --dry-run stops here.
  3. Download the files into DIR/src (a stopped download resumes), run convert.py, and check
     the size and the sha256 of DIR/ngram_embedding_bf16.safetensors.
  4. Print the two environment lines for the engine. --delete-source also removes the
     downloaded files in DIR/src, but only after the checks pass. After a normal run, run the
     command again with --delete-source: it checks the finished file again and then deletes.

Needs the Python package huggingface_hub for the download (not for --dry-run).
Exit codes: 0 ok, 2 refused before the work (disk, memory, existing output, bad input),
3 stopped by --stop-at or an incomplete download (run again to resume), 4 the output failed its checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import convert  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
# 8 + header + 320,001,536 rows x 160 x 2 bytes; the sha256 is that of the converted file.
OUT_BYTES = 102_400_496_832
OUT_SHA256 = "71205584be130551594a3e1f43949f47c576a3cc65486f1e51f4e7ba45fa42be"
OUT_NAME = "ngram_embedding_bf16.safetensors"
NGRAM_MARK = ".ngram_embedding.shard_"
HF = "https://huggingface.co"
TIMEOUT_S = 60
GB = 1e9


def http_get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "crucible-ngram16-fetch"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.load(r)


def pick_files(index: dict) -> list[str]:
    """The files of the index that hold at least one n-gram shard tensor, sorted by name."""
    files = {f for key, f in index["weight_map"].items() if NGRAM_MARK in key and key.endswith(".weight")}
    if not files:
        raise ValueError("the index has no *.ngram_embedding.shard_*.weight tensor")
    for name in files:
        # A name from the network becomes a path under src, so it must stay a plain file name.
        if os.path.isabs(name) or "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"unsafe file name in the index: {name!r}")
    return sorted(files)


def plan_sizes(files: list[str], tree: list[dict]) -> list[tuple[str, int]]:
    sizes = {t["path"]: t["size"] for t in tree if t.get("type", "file") == "file" and "size" in t}
    missing = [f for f in files if f not in sizes]
    if missing:
        raise ValueError(f"no size in the file listing for {', '.join(missing[:3])}")
    return [(f, sizes[f]) for f in files]


def remaining_download(plan: list[tuple[str, int]], src: str) -> int:
    """Bytes still to download: a file that is complete in src counts as zero."""
    left = 0
    for name, size in plan:
        path = os.path.join(src, name)
        if not (os.path.isfile(path) and os.path.getsize(path) == size):
            left += size
    return left


def free_bytes(path: str) -> int:
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return shutil.disk_usage(path).free


def hub_download(repo_id: str, revision: str, files: list[str], local_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: the download needs huggingface_hub. Run: python -m pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(2)
    snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=files, local_dir=local_dir)


def fail(text: str, code: int = 2) -> int:
    print(f"error: {text}", file=sys.stderr, flush=True)
    return code


def verify(out: str, expected_bytes: int, expected_sha: str) -> str | None:
    """Return a message when the output is wrong, else None.

    The sha256 is the hash that convert.py computed while it wrote the bytes; the finished
    file is not read again.
    """
    size = os.path.getsize(out)
    if size != expected_bytes:
        return f"size {size} bytes, expected {expected_bytes}"
    with open(out + ".sha256") as f:
        digest = f.read().split()[0]
    if digest != expected_sha:
        return f"sha256 {digest}, expected {expected_sha}"
    return None


def delete_source(src: str, files: list[str]) -> int:
    """Remove the downloaded files and the hub cache folder inside src. Returns bytes freed."""
    freed = 0
    for name in files:
        path = os.path.join(src, name)
        if os.path.isfile(path):
            freed += os.path.getsize(path)
            os.unlink(path)
    shutil.rmtree(os.path.join(src, ".cache"), ignore_errors=True)
    try:
        os.rmdir(src)  # only an empty folder: never remove files that are not ours
    except OSError:
        pass
    return freed


def main(argv=None, *, repo_id=REPO, revision=REVISION, expected_bytes=OUT_BYTES,
         expected_sha=OUT_SHA256, get_json=http_get_json, download=hub_download,
         free_bytes=free_bytes) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="ngram16", help="work folder (default ./ngram16)")
    ap.add_argument("--dry-run", action="store_true", help="list the files and the sizes, then stop")
    ap.add_argument("--delete-source", action="store_true",
                    help="after the checks pass, delete the downloaded source files")
    ap.add_argument("--rate-mb", type=float, default=100.0, help="write rate cap of convert.py, MB/s")
    ap.add_argument("--chunk-mb", type=float, default=64.0, help=argparse.SUPPRESS)
    ap.add_argument("--stop-at", help="stop the conversion at this local time, YYYY-MM-DD HH:MM")
    a = ap.parse_args(argv)

    out_dir = os.path.abspath(a.out)
    src, final = os.path.join(out_dir, "src"), os.path.join(out_dir, OUT_NAME)
    try:
        index = get_json(f"{HF}/{repo_id}/resolve/{revision}/model.safetensors.index.json")
        files = pick_files(index)
        tree = get_json(f"{HF}/api/models/{repo_id}/tree/{revision}?limit=1000")
        plan = plan_sizes(files, tree)
    except (ValueError, KeyError, OSError) as e:
        return fail(f"cannot read the file list of {repo_id}: {e}")

    total = sum(s for _, s in plan)
    sidecar, part = final + ".sha256", final + ".part"
    have_final, have_sidecar = os.path.exists(final), os.path.exists(sidecar)
    if have_final != have_sidecar:
        return fail(f"{final} and {sidecar} must exist together or not at all (found only "
                    f"{'the file' if have_final else 'the .sha256 file'}). A new build writes both and "
                    f"stops if one is there. Delete the one that is left, then run again.")
    left = remaining_download(plan, src)
    part_bytes = os.path.getsize(part) if os.path.exists(part) else 0
    need = left + (0 if have_final else max(0, expected_bytes - part_bytes))
    free = free_bytes(out_dir)
    print(f"{repo_id} at {revision}")
    print(f"{len(plan)} files hold the n-gram tensors:")
    for name, size in plan:
        print(f"  {name}  {size / GB:.2f} GB")
    print(f"download: {total / GB:.1f} GB in total, {left / GB:.1f} GB still to fetch")
    print(f"output:   {expected_bytes / GB:.1f} GB ({final})")
    print(f"disk:     {need / GB:.1f} GB needed, {free / GB:.1f} GB free")
    files_done = [n for n, _ in plan]
    if have_final:
        if a.dry_run:
            return 0
        if not a.delete_source:
            return fail(f"{final} exists; refusing to overwrite. To check it and delete the source "
                        f"files in {src}, run again with --delete-source.")
        # No disk check: this run only reads the finished file and deletes files.
        return finish(final, src, files_done, total, expected_bytes, expected_sha, True)
    if free < need:
        return fail(f"not enough free disk space in {out_dir}: need {need / GB:.1f} GB, "
                    f"have {free / GB:.1f} GB. Free space or use --out on a larger disk.")
    if a.dry_run:
        return 0
    mem_kb = common.mem_available_kb()
    if mem_kb < common.MEM_START_KB:
        return fail(f"MemAvailable is {mem_kb} kB; convert.py starts only above {common.MEM_START_KB} kB. "
                    "Free memory and run again.")

    os.makedirs(src, exist_ok=True)
    try:
        download(repo_id, revision, files_done, src)
    except Exception as e:  # network, auth and disk errors of the hub client all end here
        return fail(f"download stopped: {type(e).__name__}: {e}. Run the command again to resume.", 3)
    short = [n for n, s in plan if not os.path.isfile(os.path.join(src, n)) or os.path.getsize(os.path.join(src, n)) != s]
    if short:
        return fail(f"download incomplete: {', '.join(short[:3])}. Run the command again to resume.", 3)

    args = [src, final, "--revision", revision, "--rate-mb", str(a.rate_mb), "--chunk-mb", str(a.chunk_mb)]
    if a.stop_at:
        args += ["--stop-at", a.stop_at]
    rc = convert.main(args)
    if rc != 0:
        return rc
    return finish(final, src, files_done, total, expected_bytes, expected_sha, a.delete_source)


def finish(final, src, files, total, expected_bytes, expected_sha, delete) -> int:
    """Check the finished file, delete the source if asked, and print the engine lines."""
    problem = verify(final, expected_bytes, expected_sha)
    if problem:
        return fail(f"{final} (with {final}.sha256) failed its check: {problem}. The source files stay in "
                    f"{src}. Delete both output files before you run again.", 4)
    print(f"checked: {os.path.getsize(final)} bytes, sha256 {expected_sha}")
    if delete:
        print(f"source deleted: freed {delete_source(src, files) / GB:.1f} GB")
    else:
        print(f"The source files in {src} use {total / GB:.1f} GB. "
              f"Run the command again with --delete-source to check the file and remove them.")
    print("Add these two lines to the environment of the engine:")
    print("CRUCIBLE_PLE_HOST=1")
    print(f"CRUCIBLE_PLE_BF16={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
