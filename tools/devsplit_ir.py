"""IR text work of tools/devsplit.py: the call graph of the front-end module, the grouping of kernels, and the
repair of the parts that llvm-extract cuts out of it. All of it reads and writes the textual IR of `llvm-dis`."""
from collections import defaultdict
from pathlib import Path
import re

HEAD_WORDS = frozenset(
    "private internal linkonce_odr weak_odr linkonce weak appending extern_weak external available_externally common "
    "hidden protected default dso_local dso_preemptable unnamed_addr local_unnamed_addr".split())
LOCAL_WORDS = frozenset("private internal linkonce_odr weak_odr linkonce weak available_externally appending extern_weak".split())


def split_head(text):
    """(linkage and visibility words, rest) of the text after `define ` or `@name = `."""
    words = text.split(" ")
    n = 0
    while n < len(words) and words[n] in HEAD_WORDS:
        n += 1
    return words[:n], " ".join(words[n:])



REF = re.compile(r'@((?:[\w.$-]+|"[^"]*"))')


class Module:
    """What the grouping and the repair need to know about the whole module."""

    def __init__(self, text):
        chunks = text.split("\ndefine ")
        self.head, self.strong = {}, {}  # name -> linkage words; strong globals: name -> (words, rest of line)
        self.local = set()  # names with local linkage (internal, private): the optimizer sees all their uses
        self.kernels, self.size, self.refs = set(), {}, {}
        self.order = {}  # global variable name -> position in the module
        self.lds = {}  # shared-memory (addrspace 3) variable name -> its definition line as the front end wrote it
        self.reached = {}  # memo of reach()
        for line in chunks[0].split("\n"):
            if line.startswith("@") and " = " in line:
                name, rest = line[1:].split(" = ", 1)
                words, tail = split_head(rest)
                self.head[name] = words
                self.order[name] = len(self.order)
                if " addrspace(3) " in rest and not rest.startswith("external "):
                    self.lds[name] = line
                if {"internal", "private"} & set(words):
                    self.local.add(name)
                elif not LOCAL_WORDS & set(words) and "external" not in words:
                    self.strong[name] = (words, tail)
        for chunk in chunks[1:]:
            header, _, body = chunk.partition("\n")
            at = header.index("@")
            name = REF.match(header, at).group(1)
            words = split_head(header[:at].rstrip())[0]
            self.head[name] = words
            if {"internal", "private"} & set(words):
                self.local.add(name)
            if "amdgpu_kernel" in header[:at]:
                self.kernels.add(name)
            body = body.split("\n}\n", 1)[0]
            self.size[name] = body.count("\n")
            self.refs[name] = frozenset(REF.findall(body))

    def reach(self, kernel):
        """Every function and global that `kernel` uses, directly or through calls."""
        if kernel in self.reached:
            return self.reached[kernel]
        seen, todo = set(), [kernel]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(r for r in self.refs.get(node, ()) if r in self.head and r not in seen)
        self.reached[kernel] = seen
        return seen

    def users(self):
        """{function or global: kernels that reach it}."""
        out = defaultdict(set)
        for k in self.kernels:
            for n in self.reach(k):
                out[n].add(k)
        return out


def read_groups(path):
    """{kernel: group name} from a two-column file (group, kernel). A missing file means no groups."""
    pinned = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            if line and not line.startswith("#"):
                group, kernel = line.split("\t")
                pinned[kernel] = group
    return pinned


def groups_of(mod, pinned):
    """Kernels that must be compiled together, with the cost of each group in lines of IR.

    Why: the optimizer treats the module as a whole. A function that several kernels call is judged by all its
    callers, so a part that holds only some of the callers can compile a kernel differently from the one-module
    build (different inlining, so different ISA). `pinned` names the kernels where that happens. Every other
    kernel is a group of its own. A name that is not in the module is ignored, so a stale file does no harm.
    """
    reach = {k: mod.reach(k) for k in sorted(mod.kernels)}
    members = defaultdict(list)
    for k in reach:
        members[pinned.get(k, k)].append(k)
    groups = []
    for ks in members.values():
        nodes = set().union(*(reach[k] for k in ks))
        groups.append((sorted(ks), sum(mod.size.get(n, 0) for n in nodes)))
    return sorted(groups, key=lambda g: (-g[1], g[0][0]))


def culprits(mod, changed, min_size=100, max_outside=8):
    """Group the kernels that need one part. `changed` are kernels whose ISA moved when every kernel had a part of its own.

    A function of `min_size` lines or more that at least two changed kernels call is what moved them (the
    smaller ones are the thread-index and math helpers that hundreds of kernels call, and they never move
    anything). Its callers are joined, because the optimizer counts all callers. A function with more than
    `max_outside` callers joins them only when all of them changed: one that many unchanged kernels also call is
    not what moved these.
    -> {kernel: group name}, one name per set of joined kernels; kernels that join nothing are left out.
    """
    changed = set(changed) & mod.kernels
    users = defaultdict(set)
    for k in sorted(mod.kernels):
        for n in mod.reach(k):
            if n not in mod.kernels:
                users[n].add(k)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for n, ks in users.items():
        if mod.size.get(n, 0) >= min_size and (ks <= changed or len(ks) <= max_outside) and len(ks & changed) >= 2:
            ks = sorted(ks)
            for k in ks[1:]:
                parent[find(k)] = find(ks[0])
    joined = defaultdict(list)
    for k in sorted(parent):
        joined[find(k)].append(k)
    return {k: f"g{i:02d}" for i, ks in enumerate(sorted(g for g in joined.values() if len(g) > 1)) for k in ks}


WEIGHT_STEP = 5000  # lines of IR; a group is about 50000, so the load of a part is right within a few percent


def pack(groups, nparts):
    """Parts of kernel lists, heaviest group first onto the lightest part. -> (parts, load of each part).

    Why the weights are rounded: an edit changes the size of a group by a few lines. With exact weights that can
    swap two groups in the order, and then every group after them lands on another part, so the part cache misses
    on most parts. With rounded weights the same edit leaves the assignment as it was.
    """
    parts, load = [[] for _ in range(nparts)], [0] * nparts
    rounded = [(ks, w, round(w / WEIGHT_STEP) * WEIGHT_STEP) for ks, w in groups]
    for ks, _, step in sorted(rounded, key=lambda g: (-g[2], g[0][0])):
        i = min(range(nparts), key=lambda p: (load[p], p))
        parts[i].extend(ks)
        load[i] += step
    return [sorted(p) for p in parts if p], load


def scalar_declaration(name, words, tail):
    """The declaration line of a strong global, from its definition. Only scalar types: the initializer is cut."""
    kind = "constant" if " constant " in f" {tail} " else "global"
    pre, _, post = tail.partition(f" {kind} ")
    typ = post.split(" ", 1)[0].rstrip(",")
    if not (typ == "ptr" or (typ[:1] == "i" and typ[1:].isdigit())):
        raise RuntimeError(f"global @{name} is shared by parts and has type {typ}: only scalars are supported")
    align = post[post.index(", align"):] if ", align" in post else ""
    keep = [w for w in words if w in ("protected", "hidden", "default")]
    return f"@{name} = external dso_local {' '.join(keep)} {pre} {kind} {typ}{align}".replace("  ", " ")


def mentioned(lines, strong):
    """Names of the strong globals that one part text mentions."""
    pats = {name: re.compile(rf"@{re.escape(name)}(?![\w.$])") for name in strong}
    seen = set()
    for line in lines:
        if "@" in line:
            seen.update(name for name, pat in pats.items() if name not in seen and pat.search(line))
    return seen


def restore(lines, head, strong, owned, order, lds):
    """The part text with original linkage words, and with each strong global defined only in the part that owns it."""
    out, comdats = [], set()
    for line in lines:
        if line.startswith("define "):
            name = REF.match(line, line.index("@")).group(1)
            _, rest = split_head(line[7:])
            if name in head:
                line = "define " + " ".join([*head[name], rest])
        elif line.startswith("@") and " = " in line:
            name, rest = line[1:].split(" = ", 1)
            if name in lds and rest.startswith("external "):
                # llvm-extract turns a shared-memory variable into a bodiless declaration. The layout pass
                # sorts and places definitions and declarations differently, so the definition has to come back.
                line = lds[name]
                if ", comdat" in line:
                    comdats.add(name)
            elif name in strong:
                words, tail = strong[name]
                line = f"@{name} = {' '.join([*words, tail])}" if name in owned else scalar_declaration(name, words, tail)
            elif name in head and not rest.startswith("external "):
                _, tail = split_head(rest)
                line = f"@{name} = " + " ".join([*head[name], tail])
        out.append(line)
    # llvm-extract writes the globals in the order it found them. The LDS layout pass breaks ties between
    # equal variables by module order, so the original order has to come back or kernel LDS offsets move.
    at = [i for i, line in enumerate(out) if line.startswith("@") and " = " in line]
    key = lambda line: order.get(line[1:line.index(" = ")], len(order))
    for i, line in zip(at, sorted((out[i] for i in at), key=key)):
        out[i] = line
    have = {line.split(" = ", 1)[0] for line in out if line.startswith("$")}
    first = next((i for i, line in enumerate(out) if line[:1] in "@$"), len(out))
    out[first:first] = [f"${n} = comdat any" for n in sorted(comdats) if f"${n}" not in have]
    return "\n".join(flatten_srcloc(out))


# Argument facts (tools/devsplit_facts.tsv). IPSCCP gives an argument of an internal function a range (or a
# constant) when every call site in the module passes a value in it. A part holds only some of the call sites, so
# it can find a narrower fact than the one-module build, and the inlined body then compiles to other code
# (LaneWindow<5>::LaneWindow(int) on trunk 27e0f62b: range(i32 0, 32) in a part, none in the module, because
# the TileBody call sites of the prefill MoE tile kernels pass a lane that IPSCCP cannot bound).
# A part therefore gets a stub kernel that calls each such function with the module's fact: the stub's value
# widens the lattice to exactly that fact, because the part's own call sites are a subset of the module's.
RANGE_ATTR = re.compile(r"\brange\(i(\d+) (-?\d+), (-?\d+)\)")
DEFINE_NAME = re.compile(r"@((?:[\w.$-]+|\"[^\"]*\"))\(")
NO_FACT = "@"
STUB_PREFIX = "__devsplit_facts_"  # tools/isa_diff.py does not compare these kernels
UNSUPPORTED_PARAM = ("byval(", "sret(", "inalloca(", "preallocated(", "byref(", "...")


def split_params(text):
    """The parameters of a parameter list text, split at the commas outside brackets."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "({[<":
            depth += 1
        elif ch in ")}]>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def param_type(param):
    """The type at the start of one parameter text."""
    if param[:1] in "{<[":
        close = {"{": "}", "<": ">", "[": "]"}[param[0]]
        depth = 0
        for i, ch in enumerate(param):
            depth += ch == param[0]
            depth -= ch == close
            if depth == 0:
                return param[:i + 1]
    m = re.match(r"ptr addrspace\(\d+\)|\S+", param)
    return m.group(0)


def define_params(line):
    """(name, prefix before @name, parameter texts) of a `define` line."""
    m = DEFINE_NAME.search(line)
    depth, i = 0, m.end() - 1
    for j in range(i, len(line)):
        depth += line[j] == "("
        depth -= line[j] == ")"
        if depth == 0:
            return m.group(1), line[:m.start()], split_params(line[i + 1:j])
    raise ValueError(f"no parameter list in {line[:120]}")


def argument_ranges(line):
    """Per argument of a `define` line: 'BITS:LO:HI' of its range attribute, or NO_FACT."""
    out = []
    for p in define_params(line)[2]:
        m = RANGE_ATTR.search(p)
        out.append(f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else NO_FACT)
    return out


def define_lines(text):
    """{function: its define line} of an IR text (only the define lines matter)."""
    out = {}
    for line in text.split("\n"):
        if line.startswith("define "):
            out[define_params(line)[0]] = line
    return out


def derive_facts(module_defs, part_defs):
    """{function: {argument index: module fact}} where a part's IPSCCP result differs from the module's.

    Inputs are the define lines after IPSCCP of the one-module build and of each part. A fact is the module's
    range ('BITS:LO:HI') or 'any' (the module knows nothing). A part that has no range where the module has one
    holds a constant (fewer call sites); a part with a range where the module has none is narrower.
    """
    mono = {n: argument_ranges(l) for n, l in define_lines(module_defs).items()}
    facts = {}
    for text in part_defs:
        for name, line in define_lines(text).items():
            if name not in mono:
                continue
            part = argument_ranges(line)
            if len(part) != len(mono[name]):
                continue
            if mono[name] != part:
                # The other arguments keep the module's range: the stub passes a value in it, because poison
                # would make the range "may be undef", and IPSCCP then drops the attribute.
                row = facts.setdefault(name, {})
                for i, (m, p) in enumerate(zip(mono[name], part)):
                    if m != p:
                        row[i] = "any" if m == NO_FACT else m
                    elif m != NO_FACT:
                        row[i] = m
    return facts


def read_facts(path):
    """{function: {argument index: fact}} from a three-column file (function, argument, fact). Missing: none."""
    facts = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line and not line.startswith("#"):
                name, arg, fact = line.split("\t")
                facts.setdefault(name, {})[int(arg)] = fact
    return facts


def write_facts(path, facts):
    rows = "".join(f"{n}\t{a}\t{facts[n][a]}\n" for n in sorted(facts) for a in sorted(facts[n]))
    Path(path).write_text("# function\targument\tfact of the one-module build after IPSCCP (tools/devsplit.py --derive)\n" + rows)


def range_width(bits, lo, hi):
    """Number of values in the range [lo, hi) of a `bits` wide integer (wrapping); 0 for the full set."""
    return (hi - lo) % (1 << bits)


def signed(value, bits):
    value %= 1 << bits
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def fact_stub(name, lines, facts, wanted):
    """A kernel `name` that calls each function of `wanted` (defined in `lines`, with facts) with the module's facts.

    -> (the stub's lines or [], the functions it calls). An argument with a fact gets a volatile load (any value)
    or a load folded into [lo, hi) by urem and add; one without gets poison, which adds nothing to a constant or
    an unknown value. A result goes to a volatile store, or the early passes delete the call of a function without
    side effects before IPSCCP runs. Each call has its own block behind a switch on a loaded value, so a call that
    the optimizer finds undefined (poison for a noundef argument) cannot remove the others.
    A function with an aggregate-in-memory or variadic parameter, or a calling convention, gets no call.
    """
    body, used, attrs = [], [], ""
    for line in lines:
        if not line.startswith("define "):
            continue
        if not attrs and " amdgpu_kernel " in line:
            # The stub takes the attributes of a kernel of the part. Without the target attributes the inliner
            # keeps the callees as functions of their own, and they would be added to the code object.
            group = re.findall(r" (#\d+)", line.rsplit(")", 1)[1])
            attrs = f" {group[-1]}" if group else ""
        fn, prefix, params = define_params(line)
        if fn not in wanted or fn not in facts:
            continue
        if any(u in p for p in params for u in UNSUPPORTED_PARAM):
            continue
        words, rest = split_head(prefix[len("define "):].strip())
        rest = re.sub(r"\b(range|nofpclass|dereferenceable|dereferenceable_or_null)\([^)]*\)|\balign \d+", "", rest)
        rtype = " ".join(w for w in rest.split() if w not in ("noundef", "zeroext", "signext", "inreg", "noalias", "nonnull"))
        if not rtype or rtype.split()[0].endswith("cc") or rtype.split()[0].startswith("amdgpu_"):
            continue
        k = len(used)
        body.append(f"k{k}:")
        args = []
        for i, p in enumerate(params):
            t = param_type(p)
            fact = facts[fn].get(i)
            if fact is None:
                args.append(f"{t} poison")
                continue
            body.append(f"  %a{k}.{i} = load volatile {t}, ptr addrspace(1) %p, align 1")
            value = f"%a{k}.{i}"
            if fact != "any":
                bits, lo, hi = (int(x) for x in fact.split(":"))
                width = range_width(bits, lo, hi)
                if width:
                    body.append(f"  %r{k}.{i} = urem {t} %a{k}.{i}, {signed(width, bits)}")
                    body.append(f"  %a{k}.{i}.r = add {t} %r{k}.{i}, {signed(lo, bits)}")
                    value = f"%a{k}.{i}.r"
            args.append(f"{t} {value}")
        if rtype == "void":
            body.append(f"  call void @{fn}({', '.join(args)}) alwaysinline")
        else:
            body.append(f"  %c{k} = call {rtype} @{fn}({', '.join(args)}) alwaysinline")
            body.append(f"  store volatile {rtype} %c{k}, ptr addrspace(1) %p, align 1")
        body.append("  br label %done")
        used.append(fn)
    if not used:
        return [], []
    cases = " ".join(f"i32 {k}, label %k{k}" for k in range(len(used)))
    return [f"define amdgpu_kernel void @{name}(ptr addrspace(1) noundef %p){attrs} {{",
            "  %sel = load volatile i32, ptr addrspace(1) %p, align 4",
            f"  switch i32 %sel, label %done [ {cases} ]", *body, "done:", "  ret void", "}"], used


SRCLOC_USE = re.compile(r"!srcloc !(\d+)")
SRCLOC_NODE = re.compile(r"^!(\d+) = !\{i\d+ -?\d+(?:, i\d+ -?\d+)*\}$")


def flatten_srcloc(lines):
    """Replace the source offsets of inline assembly (!srcloc) with their position in the node.

    Why: an offset counts bytes from the start of all sources. One added byte in a header moves every later offset,
    so the text of nearly every part changes and the part cache misses. The offsets only select the source line
    in a diagnostic of the assembler; the machine code does not depend on them.
    """
    used = {m.group(1) for line in lines if "!srcloc" in line for m in SRCLOC_USE.finditer(line)}
    if not used:
        return lines
    out = []
    for line in lines:
        m = SRCLOC_NODE.match(line) if line[:1] == "!" else None
        if m and m.group(1) in used:
            n = line.count(",") + 1
            line = f"!{m.group(1)} = !{{" + ", ".join(f"i64 {i + 1}" for i in range(n)) + "}"
        out.append(line)
    return out


