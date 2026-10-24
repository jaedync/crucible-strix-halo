"""Fail-closed batch supplement. The shared gate runner invokes this after primary gates."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

# tests load this file as tools.batch_gate_report, so tools/ may be missing from sys.path.
sys.path.append(str(Path(__file__).resolve().parent))
import layout  # noqa: E402

WIDTH = 10240
VOCAB = 248320
DUMPS = tuple(f"h_{layer:02d}.f32" for layer in range(49))
LOGITS = tuple(f"logits_{step}.f32" for step in range(8))
REQUIRED_SOURCES = {"engine/engine.hip", "engine/layers/ops.h", "engine/runtime/batch_forward.h", "engine/layers/packed_state.h", "bench/batch_gates.hip"}
BUILDER = Path(__file__).with_name("build_batch_gates.py")
BUILD_FLAGS = ["-O3", "-std=c++20", "--offload-arch=gfx1151", "--rocm-path=/opt/rocm",
               "-I/opt/rocm/include", "--no-default-config"]
# Prefix-based collection could publish a future credential variable.
# The CRUCIBLE_* names are the kRecord switches of engine/switches.h (tests/test_switch_lists.py).
RUNTIME_KNOBS = (
    "HSA_USERPTR_FOR_PAGED_MEM", "HSA_USE_SVM", "GPU_PINNED_MIN_XFER_SIZE",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CRUCIBLE_WARMUP", "CRUCIBLE_PROFILE",
    "CRUCIBLE_TUNE_ROWS", "CRUCIBLE_FUSE", "CRUCIBLE_LOOKAHEAD",
    "CRUCIBLE_LOOKAHEAD_MIB", "CRUCIBLE_STATE_CACHE", "CRUCIBLE_STATIC_CACHE",
    "CRUCIBLE_DECODE_RING",
    "CRUCIBLE_PERSISTENT_GDN",
    "CRUCIBLE_PF_EXACT_SPLIT", "CRUCIBLE_PF_HC_FUSED", "CRUCIBLE_PF_GDN_MICRO", "CRUCIBLE_PF_GDN_EAGER",
    "CRUCIBLE_PF_QSA_ROWS", "CRUCIBLE_PF_MOE_RM",
    "CRUCIBLE_PF_TILES", "CRUCIBLE_PF_FX_LEAD",
    "CRUCIBLE_PF_PAD_H", "CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR",
    "CRUCIBLE_PF_OUT", "CRUCIBLE_PF_LAUNCH_ROWS", "CRUCIBLE_PF_STAGE_SHARE", "CRUCIBLE_ABLATE", "CRUCIBLE_ABLATE_ANY_CHECKPOINT",
    "CRUCIBLE_COMPACT_STATE",
    "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY",
    "CRUCIBLE_HC_WMMA", "CRUCIBLE_PLE_BF16", "CRUCIBLE_PLE_HOST",

    "CRUCIBLE_PF_PRUNE", "CRUCIBLE_PLE_TRACE",
    "CRUCIBLE_MTP_SIDECAR",
    "CRUCIBLE_EXACT",
    "CRUCIBLE_PACKED_ROWS",

)


def ple_bf16_public(value):
    """The BF16 n-gram table as a report shows it: file name and size. The path names the user's home, and
    the 102 GB file is too large to hash for each gate."""
    path = Path(value)
    try:
        info = path.stat()
    except OSError:
        return "set"
    return f"{path.name} ({info.st_size} bytes)" if stat.S_ISREG(info.st_mode) else "set"


def runtime_environment(env):
    # The sidecar path names the user's home; mtp_sidecar_record publishes its sha256 and size instead.
    out = {name: env.get(name) for name in RUNTIME_KNOBS}
    if out.get("CRUCIBLE_MTP_SIDECAR"):
        out["CRUCIBLE_MTP_SIDECAR"] = "set (see mtp_sidecar)"
    if out.get("CRUCIBLE_PLE_BF16"):
        out["CRUCIBLE_PLE_BF16"] = ple_bf16_public(out["CRUCIBLE_PLE_BF16"])
    return out


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def mtp_sidecar_record(env):
    """Size and sha256 of the CRUCIBLE_MTP_SIDECAR file: a path is not provenance, and it names the user."""
    value = env.get("CRUCIBLE_MTP_SIDECAR")
    if not value:
        return None
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("CRUCIBLE_MTP_SIDECAR must name a regular file")
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def file_record(path, started_ns=None, size=None):
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"expected a regular artifact: {path}")
    info = path.stat()
    if started_ns is not None and (info.st_mtime_ns < started_ns or info.st_ctime_ns < started_ns):
        raise ValueError(f"stale batch artifact: {path}")
    if size is not None and info.st_size != size:
        raise ValueError(f"wrong artifact size: {path}")
    return dict(path=str(path), bytes=info.st_size, sha256=digest(path),
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def recheck(records):
    for name, record in records.items():
        if file_record(Path(name)) != record:
            raise ValueError(f"batch input or artifact changed during capture: {name}")


def ids_record(path, count, started_ns):
    result = file_record(path, started_ns)
    text = path.read_text()
    if not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", text):
        raise ValueError(f"invalid IDs artifact: {path}")
    ids = [int(value) for value in text.split(",")]
    if len(ids) != count or any(value >= VOCAB for value in ids):
        raise ValueError(f"wrong ID count or vocabulary: {path}")
    return {**result, "ids": ids}


def directory(path, names):
    if path.is_symlink() or not path.is_dir() or {p.name for p in path.iterdir()} != set(names):
        raise ValueError(f"unexpected batch artifact set: {path}")


def stream_record(path, started_ns, batch):
    ids_names = ("ids-dump.txt", "ids-perf.txt") if batch else ("ids.txt",)
    directory(path, (*DUMPS, *LOGITS, *ids_names))
    result = dict(dumps={name: file_record(path / name, started_ns, 8 * WIDTH * 4) for name in DUMPS},
                  logits={name: file_record(path / name, started_ns, VOCAB * 4) for name in LOGITS})
    if batch:
        result["ids_dump"] = ids_record(path / "ids-dump.txt", 8, started_ns)
        result["ids_perf"] = ids_record(path / "ids-perf.txt", 33, started_ns)
    else:
        result["ids"] = ids_record(path / "ids.txt", 33, started_ns)
    return result


def equal_files(left, right):
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            av, bv = a.read(1 << 20), b.read(1 << 20)
            if av != bv: raise ValueError(f"batch contents differ: {left}")
            if not av: return


def validate_artifacts(root, started_ns):
    singles = {}
    directory(root / "single", (f"stream_{stream}" for stream in range(8)))
    for stream in range(8):
        singles[str(stream)] = stream_record(root / "single" / f"stream_{stream}", started_ns, False)
    batches = {}
    for size in range(1, 9):
        folder = root / f"batch{size}"
        directory(folder, (f"stream_{stream}" for stream in range(size)))
        rows = {}
        for stream in range(size):
            row = stream_record(folder / f"stream_{stream}", started_ns, True)
            reference = singles[str(stream)]
            if (row["ids_dump"]["ids"] != reference["ids"]["ids"][:8] or
                    row["ids_perf"]["ids"] != reference["ids"]["ids"]):
                raise ValueError(f"batch IDs differ: B={size}, stream={stream}")
            for family in ("dumps", "logits"):
                for name, artifact in row[family].items():
                    equal_files(artifact["path"], reference[family][name]["path"])
            rows[str(stream)] = row
        batches[str(size)] = rows
    return dict(single=singles, batches=batches)


def flattened_artifacts(value):
    result = {}
    def visit(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                result[node["path"]] = {k: v for k, v in node.items() if k != "ids"}
            else:
                for item in node.values(): visit(item)
    visit(value)
    return result


def validate_builds(engine, probe, build_record, probe_build_record, engine_built=None):
    """engine is the file that runs; its sha must equal the record's. engine_built is the path that the
    dual builder wrote (the record's -o), when gates.py runs a pinned copy from elsewhere."""
    candidate = json.loads(build_record.read_text())
    batch = json.loads(probe_build_record.read_text())
    source = candidate.get("source_hashes")
    # Keys follow tools/layout_map.tsv, so records from before the layout move still verify.
    if (not isinstance(source, dict) or not layout.source_keys(REQUIRED_SOURCES) <= layout.source_keys(source) or
            batch.get("source_hashes") != source or batch.get("source_root") != candidate.get("source_root") or
            batch.get("compiler") != candidate.get("compiler") or
            batch.get("source_commit") != candidate.get("source_commit")):
        raise ValueError("batch probe source or compiler differs from the selected engine")
    tracked = {str(BUILDER): file_record(BUILDER)}
    frozen = Path(candidate["source_root"])
    for name, expected in source.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in ("engine", "kernels", "bench"):
            raise ValueError("invalid frozen source path")
        artifact = file_record(frozen / path)
        if artifact["sha256"] != expected: raise ValueError("frozen engine source changed")
        tracked[artifact["path"]] = artifact
    actual = {str(path.relative_to(frozen)) for folder in ("engine", "kernels", "bench")
              for path in (frozen / folder).rglob("*") if path.suffix in (".h", ".hip", ".inc") and path.is_file()}
    if actual != set(source): raise ValueError("frozen source manifest is incomplete")
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    built = {"candidate": Path(engine_built) if engine_built is not None else engine, "batch-probe": probe}
    for record, kind, binary, entry in ((candidate, "candidate", engine, "engine/engine.hip"),
                                        (batch, "batch-probe", probe, "bench/batch_gates.hip")):
        if (record.get("schema") != 1 or record.get("builder") != "crucible-batch-v1" or
                record.get("builder_sha256") != digest(BUILDER) or record.get("kind") != kind or
                record.get("binary_sha256") != digest(binary) or record.get("numeric_flags") != BUILD_FLAGS or
                record.get("compiler", {}).get("path") != "/opt/rocm/bin/hipcc"):
            raise ValueError("batch build record does not match the fixed dual builder")
        binding = [] if kind == "candidate" else [f'-DBATCHING_ENGINE_SHA="{digest(engine)}"',
                                                   f'-DBATCHING_SOURCE_SHA="{fingerprint}"']
        expected = ["/opt/rocm/bin/hipcc", *BUILD_FLAGS, *binding, "-o", str(built[kind].absolute()),
                    str(frozen / entry), "-lpthread"]
        if record.get("command") != expected: raise ValueError("batch build command has unqualified flags or sources")
    if batch.get("candidate_binary_sha256") != digest(engine): raise ValueError("probe names a different candidate")
    return source, fingerprint, tracked


def run(**kwargs):
    """Run after primary PASS. prompts is a UTF-8 file with eight paths, one per line."""
    output = Path(kwargs["output"])
    existing = output.exists() or output.is_symlink()
    try:
        return _run(**kwargs)
    except Exception as error:
        if not existing and output.is_dir() and not output.is_symlink():
            (output / "report.json").write_text(json.dumps(dict(status="FAIL", error=str(error)), indent=2) + "\n")
        raise


def _run(*, checkpoint, engine, probe, build_record, probe_build_record, prompts, primary_ids, output, baseline_dump,
         gpu_run, priority=2, queue_wait=7200, engine_built=None):
    sidecar = mtp_sidecar_record(os.environ)  # refuse a bad file before any subprocess or GPU wait
    engine, probe, build_record, prompts, output = map(Path, (engine, probe, build_record, prompts, output))
    if output.exists() or output.is_symlink(): raise ValueError("batch output must be a new directory")
    primary_ids = Path(primary_ids)
    probe_build_record = Path(probe_build_record)
    tracked = {str(path): file_record(path) for path in (engine, probe, build_record, probe_build_record, prompts, primary_ids)}
    source, source_hash, frozen_files = validate_builds(engine, probe, build_record, probe_build_record, engine_built)
    tracked.update(frozen_files)
    info = subprocess.run([str(probe), "--build-info"], capture_output=True, text=True, timeout=30, check=True)
    expected = dict(candidate_sha256=digest(engine), source_sha256=source_hash)
    if json.loads(info.stdout) != expected: raise ValueError("batch probe is not bound to the candidate build")
    names = [name for name in prompts.read_text().splitlines() if name]
    if len(names) != 8: raise ValueError("batch prompt list must contain exactly eight paths")
    for name in names:
        path = Path(name).resolve()
        tracked[str(path)] = file_record(path)
    equal_files(Path(names[0]).resolve(), primary_ids)
    # The resident probe retains its baseline comparison. Recheck those immutable files too.
    baseline_dump = Path(baseline_dump)
    directory(baseline_dump, DUMPS)
    for name in DUMPS:
        path = baseline_dump / name
        tracked[str(path)] = file_record(path)
    output.mkdir(parents=True)
    output_stat = output.stat()
    started_ns = min(output_stat.st_mtime_ns, output_stat.st_ctime_ns)
    root = output / "raw"
    command = [str(gpu_run), "-p", str(priority), "-w", str(queue_wait), "--", "timeout", "-k", "10", "300",
               str(probe), str(checkpoint), str(prompts), str(root), str(baseline_dump)]
    # The lock wait, the 300 s run, the 10 s kill grace and a small margin.
    completed = subprocess.run(command, capture_output=True, timeout=queue_wait + 315, check=False)
    (output / "stdout.log").write_bytes(completed.stdout)
    (output / "stderr.log").write_bytes(completed.stderr)
    recheck(tracked)
    if mtp_sidecar_record(os.environ) != sidecar: raise ValueError("MTP sidecar changed during the batch run")
    if completed.returncode != 0: raise ValueError(f"batch probe failed: exit {completed.returncode}")
    artifacts = validate_artifacts(root, started_ns)
    recheck(flattened_artifacts(artifacts))
    environment = runtime_environment(os.environ)
    result = dict(status="PASS", supplemental=True, command=command, environment=environment,
                  mtp_sidecar=sidecar, started_ns=started_ns,
                  finished_ns=time.time_ns(), inputs=tracked, build_binding=expected, artifacts=artifacts,
                  stdout=file_record(output / "stdout.log", started_ns),
                  stderr=file_record(output / "stderr.log", started_ns))
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
