"""The regex subset of a client `pattern`, and the form of it that the engine reads exactly
.

xgrammar 0.2.9 turns a pattern into one automaton (third_party/xgrammar/cpp/fsm_builder.cc ParseRegexToIR and
RegexIR::Build). Two of its steps take strings that the pattern refuses, and the vendored files stay byte-equal to
upstream:

- Optional (fsm.cc FSMWithStartEnd::Optional) adds one epsilon edge from the start of X to the first end of X, with
  no new state. When the start of X is on a loop, or that end has edges out, the skip lands inside X:
  `^[A-Z]{2}(-[0-9]+)?$` took "AB1", and `(ab+)?` took "b". Upstream an earlier change (not merged) adds a new start state.
- A repeat {n,} (RegexIR::visit(Repeat), "Handling {n,}") loops back to the first end of its first n copies. When that
  end has edges inside its copy, the loop goes back into the copy: `(ab+|c){1,}` took "ccb".

A third step merges states (fsm.cc FSMWithStartEnd::MergeEquivalentStates, after SimplifyEpsilon). Its rule 1 merges
two states that one state enters with the same edges; its last rule merges all end states without edges out. In one
pass the two chain: an end state with edges out (merged by rule 1 with an end state without edges) gets the
predecessors of every other end state without edges. So `a{1,2}|a` took "aaa", and `ab{0,2}|a` took "abbb". The chain
needs two end states without edges out. engine_pattern counts an upper bound of them (_Atom.sinks), and when it is 2
or more, it writes the pattern P as `(?:P)()`: all ends of P go to the one end state of `()`, so the last rule has
nothing to merge. That form leaves out the anchors at the ends of the top branches (the engine ignores them, and
fullmatch makes them hold). It is not used for all patterns: each end of P gets an edge to the new end, and
`.{1,128}`, with 128 ends, took about 35% longer to compile.

engine_pattern also writes each group with such a quantifier in a form with the same strings that avoids the first two
steps:
`(G)?` and `(G)??` become `(G|)` (FSMWithStartEnd::Union adds a new start state), `(G){0,}` becomes `(G)*`, `(G){1,}`
becomes `(G)+`, and `(G){n,}` becomes `(G){n-1}(G)+`: G^(n-1) G+ is G{n,}, and the automaton has n copies of G, as
xgrammar's own {n,} has (`(G){n}(G)*` has n+1). Star and Plus are safe: Star adds a new state, and Plus loops from
every end. A quantifier on one character, a class or an escape is safe: that automaton has no loop and one end.

refusal names a part of a pattern that the engine reads in another way than Python's re (the reference of the tests);
server/schema_exact.check_pattern makes it a 400 constraint_keyword. The parser of fsm_builder.cc drops a lookahead
and an anchor inside the pattern, reads a quantifier after a quantifier as a nested repeat (`(xb)?+` is possessive
in Python 3.11 and later), and refuses other group modifiers, a `{` that is not a repetition, and broken groups and
classes with its own error. An anchor inside a group is served when it can hold only at an end of the string
(_check_anchors): then fullmatch makes it true, and the engine can ignore it.

Both functions read the pattern in one pass with an explicit stack (no recursion: test_structured_guard.py sends
40,000 nested groups), and they copy the text of a group only when it changes.
"""
import re
from .grammar_limits import MAX_PATTERN_BYTES

# The longest engine form of a pattern. Each (G){n,} with n of 2 or more writes G twice: 11 nested levels (the most
# that MAX_PATTERN_EXPANSION allows) give 18,424 characters; more levels would double the text again at each level.
MAX_ENGINE_PATTERN = 32 * MAX_PATTERN_BYTES
# A strict repetition as fsm_builder.cc RegexIR::CheckRepeat reads it, without the spaces that it skips (Python reads
# "a{ 2}" as text) and with at most 9 digits (its limit).
_REPEAT = re.compile(r"\{(\d{1,9})(,(\d{1,9})?)?\}")
# A named group: ParseRegexToIR takes a name of letters, digits and "_" and ignores it.
_NAMED = re.compile(r"\(\?P?<[A-Za-z0-9_]+>")
_FLAGS = "(?i)"


class _Refused(Exception):
    pass


class _Atom:
    """One item of a sequence: a leaf (a character, an escape or a class), a group, or an anchor. text is the engine
    text of the item with its quantifier, or None when it is pattern[start:end] unchanged. With its quantifier, the
    automaton of the item (as RegexIR::Build makes it) has at most `sinks` end states without edges out, after
    SimplifyEpsilon; `edges` is False when it reads no character (an anchor, an empty group, or a repeat {0}).
    `anchors` holds the anchors ^ and $ of an anchor, or the anchors inside a group (_check_anchors)."""
    __slots__ = ("kind", "start", "end", "text", "quantified", "sinks", "edges", "anchors")

    def __init__(self, kind, start, end, text=None, sinks=1, edges=True, anchors=""):
        self.kind, self.start, self.end, self.text, self.quantified = kind, start, end, text, False
        self.sinks, self.edges, self.anchors = sinks, edges, anchors

    def size(self):
        return self.end - self.start if self.text is None else len(self.text)


def refusal(pattern, max_bytes=MAX_PATTERN_BYTES):
    """A reason when the engine reads `pattern` in another way than Python's re does, else None. A client pattern
    over grammar_limits.MAX_PATTERN_BYTES is not read: the grammar limits refuse it (400 constraint_limit)."""
    try:
        _read(pattern, max_bytes)
    except _Refused as refused:
        return str(refused)
    return None


def engine_pattern(pattern, max_bytes=MAX_PATTERN_BYTES):
    """The pattern in the form that the engine reads exactly, `(?:P)()` with the groups of P rewritten; `pattern`
    itself when it is refused (refusal) or over max_bytes (None: no limit, for the server's own regexes)."""
    try:
        return _read(pattern, max_bytes)
    except _Refused:
        return pattern


def _read(pattern, max_bytes):
    if max_bytes is not None and len(pattern.encode("utf-8", "surrogatepass")) > max_bytes:
        return pattern
    at = len(_FLAGS) if pattern.startswith(_FLAGS) else 0
    frames = [[[]]]  # the open groups, each a list of branches (lists of _Atom); frames[0] is the whole pattern
    openers = []     # (start, body start) of each open group
    while at < len(pattern):
        char = pattern[at]
        branch = frames[-1][-1]
        if char == "(":
            body = _group_body(pattern, at)
            frames.append([[]])
            openers.append((at, body))
            at = body
        elif char == ")":
            if not openers:
                raise _Refused("an unmatched )")
            start, body = openers.pop()
            branches = frames.pop()
            for each in branches:
                _check_anchors(pattern, each)
            anchors = "".join(sorted({char for each in branches for atom in each for char in atom.anchors}))
            text = None
            if any(atom.text is not None for each in branches for atom in each):
                text = pattern[start:body] + "|".join(_text(pattern, each) for each in branches) + ")"
            at += 1
            frames[-1][-1].append(_Atom("group", start, at, text, *_ends(branches), anchors=anchors))
        elif char == "|":
            frames[-1].append([])
            at += 1
        elif char in "^$":
            branch.append(_Atom("anchor", at, at + 1, edges=False, anchors=char))
            at += 1
        elif char in "*+?{":
            at = _quantify(pattern, at, branch)
        else:
            end = _leaf_end(pattern, at)
            branch.append(_Atom("leaf", at, end))
            at = end
    if openers:
        raise _Refused("an unclosed group")
    for branch in frames[0]:
        _check_anchors(pattern, branch)
    flags = _FLAGS if pattern.startswith(_FLAGS) else ""
    if _ends(frames[0])[0] >= 2:
        return f"{flags}(?:{'|'.join(_text(pattern, branch, anchors=False) for branch in frames[0])})()"
    if not any(atom.text is not None for branch in frames[0] for atom in branch):
        return pattern
    return flags + "|".join(_text(pattern, branch) for branch in frames[0])


def _text(pattern, branch, anchors=True):
    """The engine text of a branch; anchors=False leaves out its anchors (at the ends of a top branch only)."""
    return "".join(pattern[atom.start:atom.end] if atom.text is None else atom.text for atom in branch
                   if anchors or atom.kind != "anchor")


def _ends(branches):
    """(sinks, edges) of a group or of the whole pattern (_Atom). A branch ends with the ends of its last item. An empty
    branch adds none when another branch reads a character: SimplifyEpsilon (rule 2) merges its state, which only the
    new start state of the union enters, into that start state, and the start state has edges out."""
    edges = any(atom.edges for branch in branches for atom in branch)
    sinks = 0
    for branch in branches:
        items = [atom for atom in branch if atom.kind != "anchor"]
        if items:
            sinks += items[-1].sinks if any(atom.edges for atom in items) else max(1, items[-1].sinks)
        elif not edges:
            sinks += 1
    return sinks, edges


def _group_body(pattern, at):
    """The index where the body of the group that opens at `at` starts; refuses the modifiers that the engine does not
    read as Python does."""
    if not pattern.startswith("(?", at):
        return at + 1
    if pattern.startswith("(?:", at):
        return at + 3
    named = _NAMED.match(pattern, at)
    if named:
        return named.end()
    if pattern.startswith(("(?=", "(?!"), at):
        raise _Refused("a lookahead assertion: the engine ignores it")
    if pattern.startswith(("(?<=", "(?<!"), at):
        raise _Refused("a lookbehind assertion")
    if pattern.startswith("(?>", at):
        raise _Refused("an atomic group")
    if pattern.startswith(_FLAGS, at):
        raise _Refused("the flag (?i) after the start of the pattern")
    return _refuse(f"the group modifier (?{pattern[at + 2:at + 3]}")


def _refuse(reason):
    raise _Refused(reason)


def _leaf_end(pattern, at):
    char = pattern[at]
    if char == "[":
        return _class_end(pattern, at)
    if char != "\\":  # a ] outside a class is a literal, in Python and in the engine
        return at + 1
    escaped = pattern[at + 1:at + 2]
    if not escaped:
        raise _Refused("a trailing backslash")
    if escaped == "x":
        return at + 4
    if escaped == "u" and pattern.startswith("{", at + 2):
        close = pattern.find("}", at + 3)
        return _refuse("an unclosed \\u{") if close < 0 else close + 1
    if escaped == "u":
        return at + 6
    if escaped == "c":
        return at + 3
    return at + 2


def _class_end(pattern, at):
    """The index after the class that opens at `at` (fsm_builder.cc SkipCharacterClass)."""
    index = at + 1
    if pattern.startswith("^", index):
        index += 1
    if pattern.startswith("]", index):
        raise _Refused("an empty class [] or [^]: the engine refuses it, and Python reads ] as a member")
    while index < len(pattern):
        if pattern[index] == "\\":
            index += 2
        elif pattern[index] == "]":
            return index + 1
        else:
            index += 1
    raise _Refused("an unclosed [")


def _quantify(pattern, at, branch):
    """Apply the quantifier at `at` to the last atom of `branch`; returns the index after it and its lazy `?`."""
    if not branch:
        raise _Refused("a quantifier with nothing to repeat")
    atom = branch[-1]
    if atom.kind == "anchor":
        raise _Refused("a quantifier after an anchor")
    if atom.anchors:
        raise _Refused(f"a quantifier on a group with an anchor {atom.anchors[0]} inside: the engine ignores it")
    if atom.quantified:
        raise _Refused("a quantifier after a quantifier: Python 3.11 and later read it as possessive")
    low, high, end = None, {"?": 1, "*": None, "+": None}.get(pattern[at]), at + 1
    if pattern[at] == "{":
        repeat = _REPEAT.match(pattern, at)
        if not repeat:
            raise _Refused("a { that is not a repetition {m}, {m,} or {m,n} of at most 9 digits")
        low, end = int(repeat[1]), repeat.end()
        high = None if repeat[2] is not None and repeat[3] is None else int(repeat[3] or repeat[1])
        if high is not None and high < low:
            raise _Refused("a repetition {m,n} with n below m")
    if pattern.startswith("?", end):
        end += 1  # a lazy quantifier: the engine skips the "?" and takes the same strings
    if atom.kind == "group" and (pattern[at] == "?" or (low is not None and high is None)):
        atom.text = _safe_form(pattern, atom, "?" if pattern[at] == "?" else low)
    elif atom.text is not None:
        atom.text += pattern[atom.end:end]
    atom.end, atom.quantified = end, True
    # The ends with the quantifier (_Atom). A loop (* + {n,} and their forms) ends with edges out. ? and {m,n} keep
    # the ends of the last copy, and (G|) adds none, as G reads a character.
    if high == 0:
        atom.sinks, atom.edges = 1, False  # {0} and {0,0}: one state
    elif not atom.edges:
        atom.sinks = 2  # a repeat of an item that reads nothing: no bound, so the pattern gets the end group
    elif high is None:
        atom.sinks = 0
    return end


def _safe_form(pattern, group, quantifier):
    """The engine text of a group with `?` (quantifier "?") or with {n,} (quantifier n)."""
    size = group.size()
    if quantifier == "?" or quantifier < 2:
        needed = size + 1
    else:
        needed = 2 * size + len(str(quantifier - 1)) + 3
    if needed > MAX_ENGINE_PATTERN:
        raise _Refused(f"nested repeats {{n,}} whose engine form is over {MAX_ENGINE_PATTERN} characters")
    text = pattern[group.start:group.end] if group.text is None else group.text
    if quantifier == "?":
        return text[:-1] + "|)"
    if quantifier < 2:
        return text + ("*" if quantifier == 0 else "+")
    return f"{text}{{{quantifier - 1}}}{text}+"


def _check_anchors(pattern, branch):
    """The engine ignores an anchor that is not at the start or the end of the pattern. In each branch, only anchors ^
    stand before an anchor ^ or a group with one, and only anchors $ after an anchor $ or a group with one. _read checks
    each branch of a group and of the whole pattern, and _quantify refuses a group with an anchor inside: so ^ holds
    only at the start of the string, and $ only at its end, where fullmatch makes them true."""
    for index, atom in enumerate(branch):
        if "^" in atom.anchors and not all(_is_anchor(before, "^") for before in branch[:index]):
            raise _Refused(_anchor_reason(atom, "^"))
        if "$" in atom.anchors and not all(_is_anchor(after, "$") for after in branch[index + 1:]):
            raise _Refused(_anchor_reason(atom, "$"))


def _is_anchor(atom, char):
    return atom.kind == "anchor" and atom.anchors == char


def _anchor_reason(atom, char):
    if atom.kind == "anchor":
        return f"an anchor {char} inside the pattern: the engine ignores it"
    end = "start" if char == "^" else "end"
    return f"an anchor {char} in a group that is not at the {end} of the pattern: the engine ignores it"
