"""Grammar limits: bound the JSON schemas of one request before the grammar compiler
(xgrammar, in the engine) sees them. A request over a limit gets HTTP 400, code constraint_limit, with the limit's
name and value; the message never echoes schema text.

These limits come before the compile helper (engine/runtime/constraint_helper.h), which turns a compiler crash,
timeout or out-of-memory into a 400 as well. The limits give the friendly message early and keep the compile of
every accepted request fast (the test machine, the 248,320-token vocabulary: every shape at its limit compiles in 0.7 s or less).
The size, depth, node and reference limits of each schema are in server/schema_limits.py (MAX_SCHEMA_BYTES,
MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES, MAX_REFERENCE_CHAIN); cyclic references are refused in server/tool_schema.py.
No client EBNF or regex text reaches the engine: the only regular expressions are `pattern` (checked here) and the
built-in `format` grammars (counted here).
"""
import re
from .api import ApiError

# One `pattern` (a regular expression).
MAX_PATTERN_BYTES = 1024
MAX_PATTERN_GROUP_DEPTH = 16       # nesting of ( ) groups (40,000 levels crash the compiler)
MAX_PATTERN_GROUPS = 128           # ( characters in the text, escaped or not: a bound that needs no parse
# Each bound of a counted repeat {n}, {n,}, {n,m}. xgrammar 0.2.9 unrolls a repeat of at most 128 (fsm_builder.cc
# kLargeRepeatThreshold); a larger one stays a counted repeat outside the token mask cache, and its mask then costs
# about 300 ms a token on the serve loop (".{1,129}": 312 ms mean, ".{1,128}": 0.15 ms; fuzz test, test machine).
MAX_PATTERN_REPEAT = 128
MAX_PATTERN_ALTERNATIVES = 256     # | characters in the text, plus one
MAX_PATTERN_EXPANSION = 2048       # the size of the pattern with every counted repeat written out
# All patterns and formats of one request.
MAX_PATTERNS = 64
MAX_PATTERN_TOTAL_EXPANSION = 8192
MAX_FORMATS = 32                   # each `format` is a built-in grammar; email or uri cost about 20 ms each
# Schema repeats and branches.
MAX_LENGTH_BOUND = 65536           # minLength and maxLength
MAX_ITEMS_BOUND = 65536            # minItems and maxItems
MAX_OPTIONAL_PROPERTIES = 128      # properties that are not required, in one object (1,024 take 4.4 s)
MAX_OPTIONAL_PROPERTIES_TOTAL = 512
MAX_ANYOF_BRANCHES = 256
MAX_ENUM_VALUES = 1024

REPEAT = re.compile(r"\{\s*(\d*)\s*(,?)\s*(\d*)\s*\}")


def over(name, limit, field, what):
    raise ApiError(f"The JSON schema exceeds the structured output limit {name} = {limit} ({what}).", field,
                   code="constraint_limit")


class _Pattern:
    """A structural read of one regular expression: group depth, alternatives, counted repeats and the size with
    the repeats written out. Escapes skip one character; a character class is one unit."""

    def __init__(self, text):
        self.text, self.at, self.depth, self.max_depth = text, 0, 0, 0
        self.max_repeat = 0

    def expansion(self):
        return self._alternation()

    def _alternation(self):
        total = self._sequence()
        while self.at < len(self.text) and self.text[self.at] == "|":
            self.at += 1
            total += self._sequence()
        return total

    def _sequence(self):
        total = 0
        while self.at < len(self.text) and self.text[self.at] not in "|)":
            total += self._quantified(self._atom())
        return total

    def _atom(self):
        char = self.text[self.at]
        if char == "(":
            self.at += 1
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            if self.text.startswith("?", self.at):
                # (?: (?= (?! (?<= (?<name> : skip the group's prefix up to its body.
                while self.at < len(self.text) and self.text[self.at] not in ":=!>":
                    self.at += 1
                self.at += 1
            size = self._alternation()
            if self.at < len(self.text) and self.text[self.at] == ")":
                self.at += 1
            self.depth -= 1
            return max(size, 1)
        if char == "[":
            self.at += 1
            if self.text.startswith("^", self.at):
                self.at += 1
            if self.text.startswith("]", self.at):
                self.at += 1
            while self.at < len(self.text) and self.text[self.at] != "]":
                self.at += 2 if self.text[self.at] == "\\" else 1
            self.at += 1
            return 1
        self.at += 2 if char == "\\" else 1
        return 1

    def _quantified(self, size):
        while self.at < len(self.text):
            char = self.text[self.at]
            if char in "*+?":
                self.at += 1
                continue
            match = REPEAT.match(self.text, self.at) if char == "{" else None
            if not match or not (match.group(1) or match.group(3)):
                return size
            low = int(match.group(1) or 0)
            high = int(match.group(3)) if match.group(3) else low
            self.max_repeat = max(self.max_repeat, low, high)
            size *= max(high, low, 1)
            self.at = match.end()
        return size


def pattern_expansion(text, field):
    """Check one pattern against the per-pattern limits; returns its expansion."""
    if len(text.encode("utf-8")) > MAX_PATTERN_BYTES:
        over("MAX_PATTERN_BYTES", MAX_PATTERN_BYTES, field, "bytes of one pattern")
    if text.count("(") > MAX_PATTERN_GROUPS:
        over("MAX_PATTERN_GROUPS", MAX_PATTERN_GROUPS, field, "groups in one pattern")
    if text.count("|") + 1 > MAX_PATTERN_ALTERNATIVES:
        over("MAX_PATTERN_ALTERNATIVES", MAX_PATTERN_ALTERNATIVES, field, "alternatives in one pattern")
    # Every counted repeat in the text, also one that the structural read takes as a literal.
    for match in REPEAT.finditer(text):
        for bound in (match.group(1), match.group(3)):
            if bound and int(bound) > MAX_PATTERN_REPEAT:
                over("MAX_PATTERN_REPEAT", MAX_PATTERN_REPEAT, field, "bound of a counted repeat in a pattern")
    read = _Pattern(text)
    size = 0
    while read.at < len(text):
        size += read.expansion()
        read.at += 1  # an unbalanced ) : the compiler refuses the pattern; keep reading for the limits
    if read.max_depth > MAX_PATTERN_GROUP_DEPTH:
        over("MAX_PATTERN_GROUP_DEPTH", MAX_PATTERN_GROUP_DEPTH, field, "nesting of groups in one pattern")
    if size > MAX_PATTERN_EXPANSION:
        over("MAX_PATTERN_EXPANSION", MAX_PATTERN_EXPANSION, field, "size of one pattern with its repeats written out")
    return size


def _int(value):
    return type(value) is int


class GrammarBudget:
    """The limits of all schemas that go into one request's grammar."""

    def __init__(self):
        self.patterns = self.expansion = self.formats = self.optional = 0

    def check(self, schema, field):
        """Walk every object in `schema` (also definitions, enum values and unknown keywords: a reference can point
        anywhere in the document). A key with a value of the keyword's type counts; a property named `pattern`
        holds an object, not text, so it does not count."""
        pending = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            pending.extend(node.values())
            if isinstance(node.get("pattern"), str):
                self.patterns += 1
                if self.patterns > MAX_PATTERNS:
                    over("MAX_PATTERNS", MAX_PATTERNS, field, "patterns in one request")
                self.expansion += pattern_expansion(node["pattern"], field)
                if self.expansion > MAX_PATTERN_TOTAL_EXPANSION:
                    over("MAX_PATTERN_TOTAL_EXPANSION", MAX_PATTERN_TOTAL_EXPANSION, field,
                         "size of all patterns in one request with their repeats written out")
            if isinstance(node.get("format"), str):
                self.formats += 1
                if self.formats > MAX_FORMATS:
                    over("MAX_FORMATS", MAX_FORMATS, field, "string formats in one request")
            for key in ("minLength", "maxLength"):
                if _int(node.get(key)) and node[key] > MAX_LENGTH_BOUND:
                    over("MAX_LENGTH_BOUND", MAX_LENGTH_BOUND, field, key)
            for key in ("minItems", "maxItems"):
                if _int(node.get(key)) and node[key] > MAX_ITEMS_BOUND:
                    over("MAX_ITEMS_BOUND", MAX_ITEMS_BOUND, field, key)
            if isinstance(node.get("properties"), dict):
                required = node.get("required")
                required = set(item for item in required if isinstance(item, str)) if isinstance(required, list) else set()
                optional = sum(1 for key in node["properties"] if key not in required)
                if optional > MAX_OPTIONAL_PROPERTIES:
                    over("MAX_OPTIONAL_PROPERTIES", MAX_OPTIONAL_PROPERTIES, field,
                         "properties that are not required, in one object")
                self.optional += optional
                if self.optional > MAX_OPTIONAL_PROPERTIES_TOTAL:
                    over("MAX_OPTIONAL_PROPERTIES_TOTAL", MAX_OPTIONAL_PROPERTIES_TOTAL, field,
                         "properties that are not required, in one request")
            if isinstance(node.get("anyOf"), list) and len(node["anyOf"]) > MAX_ANYOF_BRANCHES:
                over("MAX_ANYOF_BRANCHES", MAX_ANYOF_BRANCHES, field, "branches of one anyOf")
            if isinstance(node.get("enum"), list) and len(node["enum"]) > MAX_ENUM_VALUES:
                over("MAX_ENUM_VALUES", MAX_ENUM_VALUES, field, "values of one enum")
