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
import collections
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
FACTS = Path(__file__).with_name("devsplit_facts.tsv")


def log(msg):
    print(f"devsplit: {msg}", file=sys.stderr, flush=True)


class StepFailed(Exception):
    pass


def run(cmd, what, stdin=None, out=None):
    """Run one step; returns (seconds, stdout). A failure prints the tool output and raises StepFailed.

    `out` is the file that the step writes. It is removed before the step, and the step fails when it exits 0 and leaves
    no output (a killed compiler child does that under hipcc, and a wrapper can hide the signal)."""
    if out is not None:
        Path(out).unlink(missing_ok=True)
    t0 = time.monotonic()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, input=stdin)
    dt = time.monotonic() - t0
    if r.returncode:
        sys.stderr.write(f"devsplit: {what} failed (rc {r.returncode}):\n{r.stdout[-3000:]}{r.stderr[-6000:]}")
        raise StepFailed(what)
    if out is not None and (not Path(out).is_file() or Path(out).stat().st_size == 0):
        Path(out).unlink(missing_ok=True)
        sys.stderr.write(f"devsplit: {what} exited 0 but wrote no output ({out}); the compiler may have been killed (out of memory?):\n"
                         f"{r.stdout[-3000:]}{r.stderr[-6000:]}")
        raise StepFailed(what)
    return dt, r.stdout


def commands(text):
    """The command lines that a driver prints with -### (or a wrapper with --dry-run): lines that start ` \"`."""
    return [shlex.split(line) for line in text.splitlines() if line.startswith(' "')]


def tool_stderr(cmd, what, cwd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    if r.returncode:
        raise RuntimeError(f"{what} failed:\n{r.stderr}")
    return r.stderr


def driver_plan(hipcc, flags, out, probe_dir):
    """The five driver commands (device cc1, lld, bundler, host cc1, host link) of `hipcc flags -###`.

    The old offload driver (AMD clang 22, ROCm 7.2) prints all five. The new one (AMD clang 24, ROCm 10.1) prints
    the device cc1 (LTO bitcode), llvm-offload-binary, `clang-linker-wrapper --emit-fatbin-only`, the host cc1 and
    the host link (clang-linker-wrapper). The device link and the bundler run inside the first wrapper:
    new_driver_steps asks for them. plan["driver"] is "old" or "new"."""
    plan = {"driver": "old"}
    for cmd in commands(tool_stderr([hipcc, *flags, "-###", "-o", str(out)], "hipcc -###", probe_dir)):
        name = Path(cmd[0]).name
        if "-cc1" in cmd:
            plan["device" if "-fcuda-is-device" in cmd else "host"] = cmd
        elif name == "lld":
            plan["lld"] = cmd
        elif name.startswith("clang-offload-bundler"):
            plan["bundler"] = cmd
        elif name == "llvm-offload-binary":
            plan["packager"] = cmd
        elif name == "clang-linker-wrapper":
            plan["fatbin" if "--emit-fatbin-only" in cmd else "link"] = cmd
        elif name.startswith("ld"):
            plan["link"] = cmd
    if "fatbin" in plan and "packager" in plan and "device" in plan:
        plan["driver"] = "new"
        plan.update(new_driver_steps(plan, Path(probe_dir)))
    missing = {"device", "lld", "bundler", "host", "link"} - set(plan)
    if missing:
        raise RuntimeError(f"driver plan lacks {sorted(missing)}; got {sorted(set(plan) - {'driver'})}")
    return plan


def new_driver_steps(plan, probe_dir):
    """{lld, bundler} of the new offload driver: the commands that `clang-linker-wrapper --emit-fatbin-only` runs.

    A dry run of the wrapper over a probe offload binary (the image spec of the plan's llvm-offload-binary, with an
    empty file) prints the device clang command and the bundler command; `-###` of that clang command gives the
    ld.lld line (full LTO: -plugin-opt=mcpu, -plugin-opt=O3, --lto-CGO3). No step compiles anything."""
    probe_dir = probe_dir / ".devsplit-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    image = next(t for t in plan["packager"] if t.startswith("--image="))
    spec = dict(kv.split("=", 1) for kv in image[len("--image="):].split(","))
    (probe_dir / "probe.o").write_bytes(b"")
    spec["file"] = "probe.o"
    tool_stderr([plan["packager"][0], "-o", "probe.out", "--image=" + ",".join(f"{k}={v}" for k, v in spec.items())],
                "llvm-offload-binary probe", probe_dir)
    wrapper = strip_opts(plan["fatbin"], ("-o",))
    wrapper = [t for t in wrapper if not t.endswith(".out")]
    dry = commands(tool_stderr([*wrapper, "-o", "probe.hipfb", "probe.out", "--dry-run", "--wrapper-verbose"],
                               "clang-linker-wrapper --dry-run", probe_dir))
    steps = {}
    for cmd in dry:
        name = Path(cmd[0]).name
        if name.startswith("clang-offload-bundler"):
            steps["bundler"] = cmd
        elif name.startswith("clang") and "-cc1" not in cmd:
            # the image path of the dry run does not exist: -### needs an input file
            clang = ["probe.o" if t.endswith(".o") and not t.startswith("-") else t for t in cmd]
            for line in commands(tool_stderr([*clang, "-###"], "device clang -###", probe_dir)):
                if Path(line[0]).name in ("ld.lld", "lld"):
                    steps["lld"] = line
    return steps


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


def lto_steps(plan, pre_bc, opt_bc, obj):
    """The two ld.lld steps that turn one part's prelinked bitcode into an object, for the new offload driver.

    The plain build runs one full-LTO ld.lld over the whole module (internalize, the LTO O3 pipeline, code
    generation) and writes the final code object. A part must give a relocatable object that the device link
    joins with the other parts. Step 1 is the plain command with -plugin-opt=emit-llvm: the same symbol
    resolution (so the same internalization) and the same LTO pipeline, but it writes the optimized bitcode.
    --no-undefined goes, because a part declares the strong globals that another part defines. Step 2 (-r, LTO
    pipeline at O0, code generation at the plain command's --lto-CGO level) only generates the code."""
    lld = [t for t in strip_opts(plan["lld"], ("-o",)) if t.startswith("-") or not t.endswith(".o")]
    shared = [t for t in lld if t != "--no-undefined"]
    step1 = [*shared, "-plugin-opt=emit-llvm", pre_bc, "-o", opt_bc]
    keep = [re.sub(r"^-plugin-opt=O\d$", "-plugin-opt=O0", t) for t in lld[1:]
            if t not in ("--no-undefined", "-shared") and not t.startswith("-L")]
    step2 = [lld[0], "-r", *keep, opt_bc, "-o", obj]
    return step1, step2


def extract_by_opt(llvm):
    """True when the compiler tree has no llvm-extract but has opt (the ROCm 10.1 wheels)."""
    return not (llvm / "llvm-extract").exists() and (llvm / "opt").exists()


def extract_cmd(llvm, funcs, globs, bc, out):
    """The command that cuts one part (the kernels `funcs` and all that they use) out of the module `bc`.

    llvm-extract --recursive where the compiler tree has it (ROCm 7.2). The ROCm 10.1 wheels ship no
    llvm-extract: there, opt internalizes every symbol but the part's kernels and drops what they do not use.
    restore() puts the linkage words of the module back on what is left, as it does for llvm-extract."""
    if not extract_by_opt(llvm):
        return [llvm / "llvm-extract", "--recursive", "--keep-const-init", *[f"--func={k}" for k in funcs],
                *[f"--glob={x}" for x in globs], bc, "-o", out]
    keep = Path(out).with_suffix(".keep")
    keep.write_text("".join(f"{n}\n" for n in [*funcs, *globs]))
    return [llvm / "opt", "-passes=internalize,globaldce", f"-internalize-public-api-file={keep}", bc, "-o", out]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def part_key(part, cmd, more=()):
    """Cache key: the part bytes, the compiler binary, and the command line without the file names.

    `more`: the later commands of the part (the ld.lld steps of the new driver), each with its tool binary."""
    h = hashlib.sha256()
    h.update(sha_file(part).encode())
    h.update(sha_file(cmd[0]).encode())
    keep = [str(t) for t in cmd[1:] if str(t) != str(part) and not str(t).endswith((".o", ".json")) and "-dumpdir" not in str(t)]
    h.update("\0".join(keep).encode())
    for step in more:
        h.update(sha_file(step[0]).encode())
        h.update("\0".join(str(t) for t in step[1:] if not str(t).endswith((".o", ".bc"))).encode())
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


def extract_parts(llvm, bc, mod, groups, globs, outs, pool, facts=None):
    """llvm-extract each group, then restore the linkage and make every strong global live in one part only.

    With `facts` (devsplit_ir.read_facts), each part also gets a stub kernel that passes the one-module build's
    argument facts to the functions that kernels of other parts call too (see devsplit_ir.fact_stub).
    -> (number of strong globals, number of stub calls)."""
    raw = [o.with_suffix(".raw.bc") for o in outs]
    by_opt = extract_by_opt(llvm)
    keep_used = []
    if by_opt:
        # a -shared link exports the functions of default visibility (__cxa_pure_virtual): part 0 keeps them
        globs = [*globs, *mod.exported]
        # internalize keeps what @llvm.compiler.used names, in every part. Parts are cut from the module without
        # it; part 0 gets the line back with its members, as llvm-extract gives it the cuid.
        keep_used = list(mod.used_lines)
        if keep_used:
            globs = [*globs, *(n for line in keep_used for n in ir.REF.findall(line.split(" = ", 1)[1]) if n not in globs)]
            _, text = run([llvm / "llvm-dis", bc, "-o", "-"], f"llvm-dis {Path(bc).name}")
            drop = set(keep_used)
            bc = Path(outs[0]).parent / "module-unused.bc"
            run([llvm / "llvm-as", "-o", bc], "llvm-as module-unused.bc",
                stdin="\n".join(line for line in text.split("\n") if line not in drop))
            del text
    futures = [pool.submit(run, extract_cmd(llvm, g, globs if i == 0 else [], bc, raw[i]), f"extract {outs[i].name}")
               for i, g in enumerate(groups)]
    for f in futures:
        f.result()
    head, strong = mod.head, mod.strong
    texts = list(pool.map(lambda p: run([llvm / "llvm-dis", p, "-o", "-"], f"llvm-dis {p.name}")[1].split("\n"), raw))
    uses = list(pool.map(lambda t: ir.mentioned(t, strong), texts))
    owner = {}
    for i, names in enumerate(uses):
        for n in names:
            owner.setdefault(n, i)

    users = mod.users() if facts else {}
    lds = set(mod.lds)

    def finish(i):
        owned = {n for n, o in owner.items() if o == i}
        text = ir.restore(texts[i], head, strong, owned, mod.order, mod.lds, mod.comdat if by_opt else None)
        if i == 0 and keep_used:
            text += "\n" + "\n".join(keep_used) + "\n"
        calls = 0
        if facts:
            mine = set(groups[i])
            # A stub kernel that reaches shared memory would change the LDS layout of the part: no stub call.
            wanted = {f for f in facts if users.get(f, set()) - mine and not mod.reach(f) & lds}
            stub, used = ir.fact_stub(f"{ir.STUB_PREFIX}{i}", texts[i], facts, wanted)
            text += "\n" + "\n".join(stub)
            calls = len(used)
        run([llvm / "llvm-as", "-o", outs[i]], f"llvm-as {outs[i].name}", stdin=text)
        raw[i].unlink()
        return calls
    calls = sum(pool.map(finish, range(len(groups))))
    return len(strong), calls


ARG_REF = re.compile(r"%(\d+)\b")


def ipsccp_defines(back_end, lto=False):
    """The define lines of the module after IPSCCP: the back-end command runs until that pass, then it is stopped.

    The argument ranges in these lines are the facts that tools/devsplit_facts.tsv pins (devsplit_ir.derive_facts).
    lto: back_end is the first ld.lld step of the new driver (lto_steps), and the dump is of the LTO pipeline's IPSCCP.
    """
    cmd = ["-emit-llvm-bc" if str(t) == "-emit-obj" else str(t) for t in back_end]
    at = 1 if lto else 2
    cmd = cmd[:at] + ["-mllvm", "-print-after=ipsccp", "-mllvm", "-print-module-scope"] + cmd[at:]
    out, started, complete, tail = [], False, False, collections.deque(maxlen=20)
    nargs, used = 0, set()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    try:
        for line in proc.stderr:
            if not started:
                tail.append(line)
            if line.startswith("; *** IR Dump After"):
                if started:
                    complete = True
                    break
                started = True
            elif started and line.startswith("define "):
                out.append(line.rstrip("\n"))
                if lto:
                    nargs, used = len(ir.define_params(line)[2]), set()
            elif lto and started and out and line.startswith("}"):
                # LTO: the prelink left range attributes on the arguments, so a constant that IPSCCP finds in a
                # part does not show in the define line. Mark the arguments that the body still uses.
                out[-1] += f" {ir.USED_MARK}{','.join(map(str, sorted(used)))}"
                nargs = 0
            elif lto and nargs and "%" in line:
                used.update(k for k in map(int, ARG_REF.findall(line)) if k < nargs)
            elif started and line.startswith("attributes #"):
                complete = True
                break
    finally:
        proc.kill()
        proc.wait()
    if not started:
        sys.stderr.write(f"devsplit: IPSCCP dump failed (rc {proc.returncode}):\n{''.join(tail)}")
        raise StepFailed("IPSCCP dump")
    if not complete:
        sys.stderr.write("devsplit: IPSCCP dump cut off before its last line (the compiler was killed?); the facts would be partial.\n")
        raise StepFailed("IPSCCP dump")
    return "\n".join(out)


def check_facts(module_defs, part_defs, facts):
    """Raise RuntimeError when a part's argument facts after IPSCCP differ from the one-module build's.

    module_defs and part_defs are define lines after IPSCCP (ipsccp_defines); facts is the --facts value."""
    drift = ir.derive_facts(module_defs, part_defs)
    if drift:
        table = FACTS.relative_to(FACTS.parents[1]) if facts == "none" else facts
        names = ", ".join(sorted(drift)[:8])
        raise RuntimeError(f"argument facts drift: {len(drift)} functions see other IPSCCP facts in a part than in "
                           f"one module ({names}); the ISA would differ from plain hipcc. Run `make split-facts` "
                           f"and commit {table}. If the drift remains after make split-facts, set SPLIT_CHECK=0, "
                           f"run make isa-check, and report it")


def facts_tables(path, plan):
    """(prelink table, LTO table) of the --facts value.

    The old offload driver has one IPSCCP run per part (the cc1 back end): the table is `path`, and there is no LTO
    table. The new driver has two: the prelink cc1 and the LTO pipeline of ld.lld, where the hidden functions are
    internal. Its tables carry the compiler's name, because another compiler finds other facts:
    tools/devsplit_facts.tsv -> tools/devsplit_facts.clang-24.tsv and tools/devsplit_facts.clang-24.lto.tsv."""
    if plan["driver"] != "new":
        return path, None
    if path == "none":
        return "none", "none"
    p, tag = Path(path), Path(plan["device"][0]).name
    return str(p.with_name(f"{p.stem}.{tag}{p.suffix}")), str(p.with_name(f"{p.stem}.{tag}.lto{p.suffix}"))


def add_lto_stub(llvm, pre, name, facts, wanted):
    """Append a fact stub kernel (devsplit_ir.fact_stub) to one part's prelinked bitcode, for the LTO pipeline.

    Why: the prelink stub calls are inlined in the prelink pipeline, so the LTO pipeline's IPSCCP sees only the
    part's own call sites of an internalized function. This stub passes the module's LTO facts there.
    Only a part with wanted functions is rewritten. -> seconds."""
    if not wanted:
        return 0.0
    t0 = time.monotonic()
    _, text = run([llvm / "llvm-dis", pre, "-o", "-"], f"llvm-dis {Path(pre).name}")
    # The prelink writes a module summary (lines that start with ^); llvm-as wants the stub in it, and the
    # full-LTO link does not use it.
    lines = [line for line in text.split("\n") if not line.startswith("^")]
    stub, used = ir.fact_stub(name, lines, facts, wanted)
    text = "\n".join(lines)
    if used:
        run([llvm / "llvm-as", "-o", pre], f"llvm-as {Path(pre).name}", stdin=text + "\n" + "\n".join(stub) + "\n")
    return time.monotonic() - t0


def stub_wanted(mod, facts, group):
    """The functions of `facts` that kernels outside `group` also reach, and that reach no shared memory."""
    users, lds, mine = mod.users(), set(mod.lds), set(group)
    return {f for f in facts if users.get(f, set()) - mine and not mod.reach(f) & lds}


def facts_of(plan, bcs, work, pool):
    """[define lines after IPSCCP] of each bitcode file, in parallel."""
    def one(i):
        return ipsccp_defines(device_back_end(plan["device"], bcs[i], work / f"ipsccp{i}.bc", False))
    return list(pool.map(one, range(len(bcs))))


def shared_symbols(llvm_nm, bcs, kinds="DBRTSGC"):
    """Strong data symbols that more than one part defines: lld would see them twice. -> {symbol: [part index]}.

    The new offload driver passes kinds without T: its LTO step internalizes the hidden functions that several
    parts copy, and the device link stops on a function that two parts export."""
    seen = {}
    for i, bc in enumerate(bcs):
        for typ, name in nm_defined(llvm_nm, bc):
            if typ in kinds:  # uppercase: strong, external
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
    ap.add_argument("--facts", default=str(FACTS), help="argument facts of the one-module build (function, argument, fact); 'none' for no stubs")
    ap.add_argument("--derive-facts", action="store_true", help="write --facts from every kernel in a part of its own, then stop")
    ap.add_argument("--check", action="store_true", help="fail when a part's argument facts after IPSCCP differ from the one-module build's (adds one whole-module IPSCCP run)")
    ap.add_argument("--only", help="experiment: build only the kernels whose names contain one of these (comma list)")
    ap.add_argument("--device-only", action="store_true", help="stop after the device link (for timing)")
    ap.add_argument("flags", nargs=argparse.REMAINDER, help="after --: the hipcc command line without -o")
    args = ap.parse_args(argv)
    args.device_only = args.device_only or bool(args.derive) or args.derive_facts
    flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    work = args.work
    t_all = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    (work / "cache").mkdir(exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not (args.derive or args.derive_facts or args.device_only):
        args.out.unlink(missing_ok=True)  # an old engine must not stay in place when this build fails
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
            host_future = pool.submit(run, host_cmd, "host compile", out=host_o)
    bc = work / "dev0.bc"
    dt, _ = run(device_front_end(plan["device"], bc), "device front end", out=bc)
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
    if args.derive_facts:
        return derive_facts(args, plan, llvm, mod, bc, cuid, work, pool)
    args.facts, lto_table = facts_tables(args.facts, plan)
    lto_facts = {} if lto_table in (None, "none") else ir.read_facts(Path(lto_table))
    if lto_table not in (None, "none") and not Path(lto_table).exists():
        log(f"warning: no LTO facts file {lto_table}: the ISA can differ from plain hipcc (run `make split-facts`)")
    if args.facts != "none" and not Path(args.facts).exists():
        log(f"warning: no facts file {args.facts}: no fact stubs, the ISA can differ from plain hipcc "
            f"(run `make split-facts`)")
    facts = {} if args.facts == "none" else ir.read_facts(Path(args.facts))
    mono_facts = pool.submit(ipsccp_defines, device_back_end(plan["device"], bc, work / "ipsccp-module.bc", False)) if args.check else None
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
    nstrong, ncalls = extract_parts(llvm, bc, mod, groups, cuid, bcs, pool, facts)
    dup = shared_symbols(nm, bcs, "DBRSGC" if plan["driver"] == "new" else "DBRTSGC")
    if dup:
        raise RuntimeError(f"parts define the same strong symbol twice: {dup}")
    log(f"{len(kernels)} kernels in {len(groups)} parts ({nstrong} strong globals, {ncalls} fact stub calls), extract {time.monotonic() - t0:.1f}s")
    if mono_facts:
        t_check = time.monotonic()
        check_facts(mono_facts.result(), facts_of(plan, bcs, work, pool), args.facts)
        log(f"argument facts of every part equal the module's, check {time.monotonic() - t_check:.1f}s")
    jobs = []
    for i, part in enumerate(bcs):
        obj = parts_dir / f"p{i}.o"
        salt = ""
        if plan["driver"] == "new":
            # new driver: the cc1 back end writes prelinked LTO bitcode, the LTO fact stub goes in, then the two
            # ld.lld steps (lto_steps)
            pre = parts_dir / f"p{i}.pre.bc"
            wanted = stub_wanted(mod, lto_facts, groups[i]) if lto_facts else set()
            salt = repr(sorted((f, sorted(lto_facts[f].items())) for f in wanted))
            stub = (lambda pre=pre, i=i, wanted=wanted:
                    add_lto_stub(llvm, pre, f"{ir.STUB_PREFIX}lto_{i}", lto_facts, wanted))
            steps = [device_back_end(plan["device"], part, pre, args.trace), stub,
                     *lto_steps(plan, pre, parts_dir / f"p{i}.opt.bc", obj)]
        else:
            steps = [device_back_end(plan["device"], part, obj, args.trace)]
        cmds = [c for c in steps if not callable(c)]
        key = part_key(part, cmds[0], cmds[1:]) if not salt else \
            hashlib.sha256((part_key(part, cmds[0], cmds[1:]) + salt).encode()).hexdigest()[:40]
        jobs.append((part, obj, steps, work / "cache" / (key + ".o")))
    todo = [j for j in jobs if not j[3].exists()]
    log(f"{len(jobs) - len(todo)} parts cached, {len(todo)} to compile")

    def compile_part(job):
        part, obj, steps, cached = job
        # the last step writes the part object: run() removes an old one first and fails when the step leaves none
        dt = sum(cmd() if callable(cmd) else run(cmd, f"back end of {part.name}", out=obj if cmd is steps[-1] else None)[0]
                 for cmd in steps)
        os.replace(obj, cached)
        for tmp in (part.with_suffix(".pre.bc"), part.with_suffix(".opt.bc")):
            tmp.unlink(missing_ok=True)
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
    run([*lld, "-o", hsaco], "lld", out=hsaco)
    if args.device_only:
        log(f"device only: {time.monotonic() - t_all:.1f}s -> {hsaco}")
        return 0
    hipfb = work / "dev.hipfb"
    bundler = [t for t in plan["bundler"] if not t.startswith("-output=") and t.startswith("-input=") <= (t == "-input=/dev/null")]
    run([*bundler, f"-input={hsaco}", f"-output={hipfb}"], "bundler", out=hipfb)
    log(f"lld and bundler {time.monotonic() - t_link:.1f}s")
    dt, _ = host_future.result() if host_future else run(host_cmd, "host compile", out=host_o)
    log(f"host compile {dt:.1f}s")
    run([llvm / "llvm-objcopy", "--update-section", f".hip_fatbin={hipfb}", host_o], "objcopy")
    link = strip_opts(plan["link"], ("-o",))
    stem = Path(plan["host"][-1]).stem + "-"
    objs_in_link = [i for i, t in enumerate(link) if t.endswith(".o") and Path(t).name.startswith(stem)]
    if len(objs_in_link) != 1:
        raise RuntimeError(f"expected one host object in the link command, found {len(objs_in_link)}")
    link[objs_in_link[0]] = str(host_o)
    run([*link, "-o", args.out], "link", out=args.out)
    prune_cache(work / "cache", objs, len(objs))
    log(f"done {time.monotonic() - t_all:.1f}s -> {args.out}")
    return 0


def derive_facts(args, plan, llvm, mod, bc, cuid, work, pool):
    """--derive-facts: every kernel alone in a part (the fewest call sites, so the narrowest facts), IPSCCP on each
    part and on the whole module; every argument whose fact differs gets the module's fact in --facts.

    New offload driver: also the LTO table (facts_tables). The parts get the prelink stubs of the new prelink
    table, every part and the whole module go through the prelink cc1, and the facts come from the IPSCCP of the
    first ld.lld step. The whole-module prelink is the long step (about 11 min on the test machine)."""
    t0 = time.monotonic()
    prelink_table, lto_table = facts_tables(args.facts, plan)
    new = lto_table not in (None, "none")
    mono = pool.submit(ipsccp_defines, device_back_end(plan["device"], bc, work / "ipsccp-module.bc", False))
    mono_pre = work / "module.pre.bc"
    if new:
        # the whole-module prelink is kept with the key of its input and command: a second derive reuses it
        prelink = device_back_end(plan["device"], bc, mono_pre, False)
        key, key_file = part_key(bc, prelink), work / "module.pre.key"
        if mono_pre.exists() and key_file.exists() and key_file.read_text() == key:
            log("module prelink: reused")
            mono_prelink = pool.submit(lambda: None)
        else:
            key_file.unlink(missing_ok=True)

            def module_prelink():
                run(prelink, "module prelink")
                key_file.write_text(key)
            mono_prelink = pool.submit(module_prelink)
    alone = sorted(mod.kernels)
    fdir = work / "facts"
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir()
    bcs = [fdir / f"k{i}.bc" for i in range(len(alone))]
    extract_parts(llvm, bc, mod, [[k] for k in alone], cuid, bcs, pool)
    log(f"{len(alone)} one-kernel parts, extract {time.monotonic() - t0:.1f}s")
    facts = ir.derive_facts(mono.result(), facts_of(plan, bcs, fdir, pool))
    ir.write_facts(Path(prelink_table), facts)
    log(f"{sum(len(v) for v in facts.values())} argument facts of {len(facts)} functions -> {prelink_table} ({time.monotonic() - t0:.1f}s)")
    if new:
        extract_parts(llvm, bc, mod, [[k] for k in alone], cuid, bcs, pool, facts)

        def lto_dump(i):
            pre = bcs[i].with_suffix(".pre.bc")
            run(device_back_end(plan["device"], bcs[i], pre, False), f"prelink {bcs[i].name}")
            step1, _ = lto_steps(plan, pre, bcs[i].with_suffix(".opt.bc"), bcs[i].with_suffix(".o"))
            out = ipsccp_defines(step1, lto=True)
            pre.unlink()
            return out
        parts = list(pool.map(lto_dump, range(len(bcs))))
        log(f"LTO IPSCCP of {len(bcs)} parts ({time.monotonic() - t0:.1f}s); waiting for the module prelink")
        mono_prelink.result()
        step1, _ = lto_steps(plan, mono_pre, work / "module.opt.bc", work / "module.o")
        lto = ir.derive_facts(ipsccp_defines(step1, lto=True), parts)
        ir.write_facts(Path(lto_table), lto)
        log(f"{sum(len(v) for v in lto.values())} LTO argument facts of {len(lto)} functions -> {lto_table} ({time.monotonic() - t0:.1f}s)")
    shutil.rmtree(fdir, ignore_errors=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
