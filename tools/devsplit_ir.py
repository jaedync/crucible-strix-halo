"""IR text work of tools/devsplit.py: the call graph of the front-end module, the grouping of kernels, and the
repair of the parts that llvm-extract cuts out of it. All of it reads and writes the textual IR of `llvm-dis`."""
from collections import defaultdict
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
        seen, todo = set(), [kernel]
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            todo.extend(r for r in self.refs.get(node, ()) if r in self.head and r not in seen)
        return seen


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


