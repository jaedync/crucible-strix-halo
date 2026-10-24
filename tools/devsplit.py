#!/usr/bin/env python3
"""Split-device build of one HIP program (the engine) with the ISA of one hipcc call.

  python3 -I tools/devsplit.py --out build/engine/crucible --work build/engine/split -j 8 -- \\
      <hipcc flags> engine/engine.hip -lpthread

Why: hipcc compiles the device code of engine/engine.hip in one clang process on one core, and 95 percent of
that time is LLVM optimization and code generation of about 640 kernel instances. This tool runs the same driver
steps (planned by `hipcc -###`, so every Makefile flag reaches every step) but cuts the device compile:

  1. device front end only (clang -cc1 -disable-llvm-passes, device libraries linked in): one bitcode file, 15 to 25 s;
  2. kernels are assigned to --parts parts; llvm-extract --recursive cuts each part out with every function and
     constant that its kernels call (the helper functions are copied, so inlining sees what it saw before);
  3. each part goes through clang -cc1 -x ir with the original -O3 and back-end options, in parallel. The object
     is cached under the hash of the part and of its command line: a part that no edit touched is not compiled;
  4. lld links the objects; the host compile runs at the same time with a placeholder fat binary, and
     llvm-objcopy puts the real one into the host object; the host link follows.

No kernel source moves. Kernels go to the lightest part, heaviest first; the kernels in tools/devsplit_groups.tsv
(see --derive) always share a part, because the optimizer judges a shared helper by all its callers.
Exit codes: 0 built, 1 a step failed, 2 the tool could not plan the build.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import devsplit_ir as ir  # noqa: E402  (-I drops the script directory from sys.path)

GROUPS = Path(__file__).with_name("devsplit_groups.tsv")


def log(msg):
    print(f"devsplit: {msg}", file=sys.stderr, flush=True)


class StepFailed(Exception):
    pass


def run(cmd, what, stdin=None):
    """Run one step; returns (seconds, stdout). A failure prints the tool output and raises StepFailed."""
    t0 = time.monotonic()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, input=stdin)
    dt = time.monotonic() - t0
    if r.returncode:
        sys.stderr.write(f"devsplit: {what} failed (rc {r.returncode}):\n{r.stdout[-3000:]}{r.stderr[-6000:]}")
        raise StepFailed(what)
    return dt, r.stdout


def driver_plan(hipcc, flags, out, probe_dir):
    """The five driver commands (device cc1, lld, bundler, host cc1, host link) of `hipcc flags -###`."""
    r = subprocess.run([hipcc, *flags, "-###", "-o", str(out)], capture_output=True, text=True, cwd=probe_dir)
    if r.returncode:
        raise RuntimeError(f"hipcc -### failed:\n{r.stderr}")
    plan = {}
    for line in r.stderr.splitlines():
        if not line.startswith(' "'):
            continue
        cmd = shlex.split(line)
        name = Path(cmd[0]).name
        if "-cc1" in cmd:
            plan["device" if "-fcuda-is-device" in cmd else "host"] = cmd
        elif name == "lld":
            plan["lld"] = cmd
        elif name.startswith("clang-offload-bundler"):
            plan["bundler"] = cmd
        elif name.startswith("ld"):
            plan["link"] = cmd
    missing = {"device", "lld", "bundler", "host", "link"} - set(plan)
    if missing:
        raise RuntimeError(f"driver plan lacks {sorted(missing)}; got {sorted(plan)}")
    return plan


def strip_opts(cmd, with_arg=(), alone=()):
    """cmd without the options in `with_arg` (and their value) and `alone`."""
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] in with_arg:
            i += 2
        elif cmd[i] in alone:
            i += 1
        else:
            out.append(cmd[i])
            i += 1
    return out


def split_source(cmd):
    """(options, source) of a cc1 command that ends in `-x hip SOURCE`."""
    if cmd[-3] != "-x":
        raise RuntimeError(f"cc1 command does not end in -x LANG SOURCE: {cmd[-4:]}")
    return cmd[:-3], cmd[-1]


# Dependency output belongs to the host compile only: the device front end reads the same files.
DEP_WITH_ARG = ("-dependency-file", "-MT")
DEP_ALONE = ("-MP", "-sys-header-deps", "-show-skipped-includes")


def device_front_end(cmd, out_bc):
    """The device compile up to optimization: same options, bitcode out, no LLVM passes."""
    opts, src = split_source(cmd)
    opts = ["-emit-llvm-bc" if t == "-emit-obj" else t for t in strip_opts(opts, ("-o",) + DEP_WITH_ARG, DEP_ALONE)]
    return [*opts, "-disable-llvm-passes", "-o", out_bc, "-x", "hip", src]


def device_back_end(cmd, part, out_obj, trace):
    """Optimizer and code generator over one part. The device libraries are linked in already."""
    opts, _ = split_source(cmd)
    opts = strip_opts(opts, ("-o", "-mlink-builtin-bitcode") + DEP_WITH_ARG, DEP_ALONE)
    return [*opts, *(["-ftime-trace"] if trace else []), "-o", out_obj, "-x", "ir", part]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def part_key(part, cmd):
    """Cache key: the part bytes, the compiler binary, and the command line without the file names."""
    h = hashlib.sha256()
    h.update(sha_file(part).encode())
    h.update(sha_file(cmd[0]).encode())
    keep = [str(t) for t in cmd[1:] if str(t) != str(part) and not str(t).endswith((".o", ".json")) and "-dumpdir" not in str(t)]
    h.update("\0".join(keep).encode())
    return h.hexdigest()[:40]


CACHE_KEEP_BUILDS = 3  # the cache holds the parts of this many full builds, so switching branches back and forth stays fast


def prune_cache(cache, used, nparts):
    """Delete the oldest cached parts beyond CACHE_KEEP_BUILDS * nparts. Without it, every edit leaves parts behind for good."""
    now = time.time()
    for obj in used:
        os.utime(obj, (now, now))  # a hit counts as use, so the parts of the current build are the newest
    files = sorted(cache.glob("*.o"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[CACHE_KEEP_BUILDS * nparts:]:
        f.unlink()


def nm_defined(llvm_nm, bc):
    """[(type, name)] of the symbols that `bc` defines."""
    _, out = run([llvm_nm, "--defined-only", bc], f"llvm-nm {Path(bc).name}")
    return [tuple(line.split()[-2:]) for line in out.splitlines() if len(line.split()) >= 2]


def extract_parts(llvm, bc, mod, groups, globs, outs, pool):
    """llvm-extract each group, then restore the linkage and make every strong global live in one part only."""
    ext = [str(llvm / "llvm-extract"), "--recursive", "--keep-const-init"]
    raw = [o.with_suffix(".raw.bc") for o in outs]
    futures = [pool.submit(run, [*ext, *[f"--func={k}" for k in g], *[f"--glob={x}" for x in (globs if i == 0 else [])],
                                 bc, "-o", raw[i]], f"llvm-extract {outs[i].name}") for i, g in enumerate(groups)]
    for f in futures:
        f.result()
    head, strong = mod.head, mod.strong
    texts = list(pool.map(lambda p: run([llvm / "llvm-dis", p, "-o", "-"], f"llvm-dis {p.name}")[1].split("\n"), raw))
    uses = list(pool.map(lambda t: ir.mentioned(t, strong), texts))
    owner = {}
    for i, names in enumerate(uses):
        for n in names:
            owner.setdefault(n, i)

    def finish(i):
        owned = {n for n, o in owner.items() if o == i}
        run([llvm / "llvm-as", "-o", outs[i]], f"llvm-as {outs[i].name}", stdin=ir.restore(texts[i], head, strong, owned, mod.order, mod.lds))
        raw[i].unlink()
    list(pool.map(finish, range(len(groups))))
    return len(strong)


def shared_symbols(llvm_nm, bcs):
    """Strong data symbols that more than one part defines: lld would see them twice. -> {symbol: [part index]}."""
    seen = {}
    for i, bc in enumerate(bcs):
        for typ, name in nm_defined(llvm_nm, bc):
            if typ in "DBRTSGC":  # uppercase: strong, external
                seen.setdefault(name, []).append(i)
    return {n: ix for n, ix in seen.items() if len(ix) > 1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="final executable")
    ap.add_argument("--work", required=True, type=Path, help="work and cache directory")
    ap.add_argument("--hipcc", default=os.environ.get("HIPCC", "/opt/rocm/bin/hipcc"))
    ap.add_argument("--parts", type=int, default=64, help="device parts: the parallel and the cache unit")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--trace", action="store_true", help="write -ftime-trace JSON of each part back end")
    ap.add_argument("--groups", default=str(GROUPS), help="kernels that are compiled in one part (group, kernel); 'none' for no groups")
    ap.add_argument("--derive", metavar="CHANGED", help="write --groups from a list of kernels whose ISA changed in a build with no groups, then stop")
    ap.add_argument("--only", help="experiment: build only the kernels whose names contain one of these (comma list)")
    ap.add_argument("--device-only", action="store_true", help="stop after the device link (for timing)")
    ap.add_argument("flags", nargs=argparse.REMAINDER, help="after --: the hipcc command line without -o")
    args = ap.parse_args(argv)
    args.device_only = args.device_only or bool(args.derive)
    flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    work = args.work
    t_all = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    (work / "cache").mkdir(exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        plan = driver_plan(args.hipcc, flags, args.out, Path.cwd())
    except (RuntimeError, OSError) as error:
        print(f"devsplit: {error}", file=sys.stderr)
        return 2
    llvm = Path(plan["device"][0]).parent
    try:
        return build(args, plan, llvm, work, t_all)
    except StepFailed:
        return 1
    except (OSError, RuntimeError) as error:
        print(f"devsplit: {error}", file=sys.stderr)
        return 1


def build(args, plan, llvm, work, t_all):
    # -j 1 means one compiler process at a time (the test machine rule). The host compile then runs after the parts.
    serial = args.jobs == 1
    pool = concurrent.futures.ThreadPoolExecutor(1 if serial else max(args.jobs, 2))
    host_o = work / "host.o"
    host_future = None
    if not args.device_only:
        # The host compile needs no device code except the fat binary; a placeholder holds its place.
        placeholder = work / "placeholder.hipfb"
        placeholder.write_bytes(b"placeholder\0")
        host_opts, host_src = split_source(plan["host"])
        host_opts = strip_opts(host_opts, ("-o", "-fcuda-include-gpubinary"))
        host_cmd = [*host_opts, "-fcuda-include-gpubinary", placeholder, "-o", host_o, "-x", "hip", host_src]
        if not serial:
            host_future = pool.submit(run, host_cmd, "host compile")
    bc = work / "dev0.bc"
    dt, _ = run(device_front_end(plan["device"], bc), "device front end")
    log(f"device front end {dt:.1f}s")
    t_parse = time.monotonic()
    nm = llvm / "llvm-nm"
    names = nm_defined(nm, bc)
    cuid = [n for t, n in names if n.startswith("__hip_cuid_")]
    _, text = run([llvm / "llvm-dis", bc, "-o", "-"], "llvm-dis dev0.bc")
    mod = ir.Module(text)
    del text
    log(f"read the module {time.monotonic() - t_parse:.1f}s")
    if args.derive:
        derived = ir.culprits(mod, Path(args.derive).read_text().split())
        Path(args.groups).write_text("# group\tkernel: kernels that one part must hold together (tools/devsplit.py --derive)\n"
                                     + "".join(f"{g}\t{k}\n" for k, g in sorted(derived.items(), key=lambda kv: (kv[1], kv[0]))))
        log(f"{len(derived)} kernels in {len(set(derived.values()))} groups -> {args.groups}")
        return 0
    kgroups = ir.groups_of(mod, {} if args.groups == "none" else ir.read_groups(Path(args.groups)))
    if args.only:
        kgroups = [(ks, w) for ks, w in kgroups if any(w_ in k for k in ks for w_ in args.only.split(","))]
    groups, load = ir.pack(kgroups, args.parts)
    kernels = [k for g in groups for k in g]
    log(f"{len(kgroups)} kernel groups, largest {kgroups[0][1] if kgroups else 0} lines; part loads {sorted(load, reverse=True)[:4]}")
    parts_dir = work / "parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir()
    bcs = [parts_dir / f"p{i}.bc" for i in range(len(groups))]
    t0 = time.monotonic()
    nstrong = extract_parts(llvm, bc, mod, groups, cuid, bcs, pool)
    dup = shared_symbols(nm, bcs)
    if dup:
        raise RuntimeError(f"parts define the same strong symbol twice: {dup}")
    log(f"{len(kernels)} kernels in {len(groups)} parts ({nstrong} strong globals), extract {time.monotonic() - t0:.1f}s")
    jobs = []
    for i, part in enumerate(bcs):
        obj = parts_dir / f"p{i}.o"
        cmd = device_back_end(plan["device"], part, obj, args.trace)
        jobs.append((part, obj, cmd, work / "cache" / (part_key(part, cmd) + ".o")))
    todo = [j for j in jobs if not j[3].exists()]
    log(f"{len(jobs) - len(todo)} parts cached, {len(todo)} to compile")

    def compile_part(job):
        part, obj, cmd, cached = job
        dt, _ = run(cmd, f"back end of {part.name}")
        os.replace(obj, cached)
        return part.name, dt

    todo.sort(key=lambda j: -j[0].stat().st_size)
    for name, dt in pool.map(compile_part, todo):
        log(f"  {name} {dt:.1f}s")
    objs = [j[3] for j in jobs]
    hsaco = work / "dev.hsaco"
    lld = strip_opts(plan["lld"], ("-o",))
    first = next(i for i, t in enumerate(lld) if t.endswith(".o"))
    lld[first:first + 1] = objs
    t_link = time.monotonic()
    run([*lld, "-o", hsaco], "lld")
    if args.device_only:
        log(f"device only: {time.monotonic() - t_all:.1f}s -> {hsaco}")
        return 0
    hipfb = work / "dev.hipfb"
    bundler = [t for t in plan["bundler"] if not t.startswith("-output=") and t.startswith("-input=") <= (t == "-input=/dev/null")]
    run([*bundler, f"-input={hsaco}", f"-output={hipfb}"], "bundler")
    log(f"lld and bundler {time.monotonic() - t_link:.1f}s")
    dt, _ = host_future.result() if host_future else run(host_cmd, "host compile")
    log(f"host compile {dt:.1f}s")
    run([llvm / "llvm-objcopy", "--update-section", f".hip_fatbin={hipfb}", host_o], "objcopy")
    link = strip_opts(plan["link"], ("-o",))
    stem = Path(plan["host"][-1]).stem + "-"
    objs_in_link = [i for i, t in enumerate(link) if t.endswith(".o") and Path(t).name.startswith(stem)]
    if len(objs_in_link) != 1:
        raise RuntimeError(f"expected one host object in the link command, found {len(objs_in_link)}")
    link[objs_in_link[0]] = str(host_o)
    run([*link, "-o", args.out], "link")
    prune_cache(work / "cache", objs, len(objs))
    log(f"done {time.monotonic() - t_all:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
