"""Exact grammars for the string formats whose xgrammar 0.2.9 regex takes values that the format refuses
.

xgrammar's converter writes each builtin `format` as a regex (third_party/xgrammar/cpp/json_schema_converter.cc,
JSONFormatToRegexPattern). For the formats of EXACT that regex is looser than the reference validator
(python-jsonschema 4.26.0 with its format checkers): `date` took 2027-02-30, `time` took the second 60, `ipv4` took
010.0.4.17. The vendored files stay byte-equal to upstream, so the server gives the engine its own regex: in the
spec that goes to the engine, each such `format` becomes a `pattern` with the exact regex, at the same place. The
client's schema is not changed, and a schema without these formats gives the same spec bytes as before.

Each regex takes only values that the reference takes. Where the reference is stricter than the RFC (the year 0000,
the leap second, a relative-json-pointer integer such as 100), the regex follows the reference (schema_exact.STRICTER).

An optional group is written `(X|)`, never `(X)?`: xgrammar's Optional (third_party/xgrammar/cpp/fsm.cc:1064) can skip
only a part of X, so the optional fraction of time took 23:21:080Z.
"""
import re
from .pattern_grammar import engine_pattern

_OCTET = r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]\d|\d)"
_IPV4 = rf"({_OCTET}\.){{3}}{_OCTET}"
_YEAR = r"(\d{3}[1-9]|\d{2}[1-9]0|\d[1-9]00|[1-9]000)"
_LEAP_YEAR = r"(\d{2}(0[48]|[2468][048]|[13579][26])|(0[48]|[2468][048]|[13579][26])00)"
_MONTH_DAY = r"((0[13578]|1[02])-(0[1-9]|[12]\d|3[01])|(0[469]|11)-(0[1-9]|[12]\d|30)|02-(0[1-9]|1\d|2[0-8]))"
_DATE = rf"({_YEAR}-{_MONTH_DAY}|{_LEAP_YEAR}-02-29)"
_TIME = r"([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d+|)(Z|[+-]([01]\d|2[0-3]):[0-5]\d)"
_H16 = "[0-9a-fA-F]{1,4}"
# RFC 3986 IPv6address (section 3.2.2), with an IPv4 address in its last 32 bits (no leading zero, as _IPV4).
_LS32 = f"({_H16}:{_H16}|{_IPV4})"
_IPV6 = "|".join((
    f"({_H16}:){{6}}{_LS32}",
    f"::({_H16}:){{5}}{_LS32}",
    f"({_H16}|)::({_H16}:){{4}}{_LS32}",
    f"(({_H16}:){{0,1}}{_H16}|)::({_H16}:){{3}}{_LS32}",
    f"(({_H16}:){{0,2}}{_H16}|)::({_H16}:){{2}}{_LS32}",
    f"(({_H16}:){{0,3}}{_H16}|)::{_H16}:{_LS32}",
    f"(({_H16}:){{0,4}}{_H16}|)::{_LS32}",
    f"(({_H16}:){{0,5}}{_H16}|)::{_H16}",
    f"(({_H16}:){{0,6}}{_H16}|)::",
))
# A JSON pointer token character: '~' only in ~0 and ~1. The engine writes a pattern without JSON escapes (it leaves
# out '"', '\\' and the control characters), so the class leaves them out too. The engine's regex reads \u{...}; the
# range leaves out the UTF-16 surrogates, which have no UTF-8 form.
_POINTER_CHAR = r"([\x20\x21\x23-\x2E\x30-\x5B\x5D-\x7D\x7F-\u{D7FF}\u{E000}-\u{10FFFF}]|~[01])"


def _label(most):
    """A hostname label (RFC 1123) of 1 to `most` characters: letters, digits and inner hyphens."""
    return f"[a-z0-9]([a-z0-9-]{{0,{most - 2}}}[a-z0-9]|)"


# A hostname has labels of at most 63 characters and at most 253 characters in total. A regex cannot count the total
# across labels, so the grammar takes at most HOSTNAME_LABELS labels of at most HOSTNAME_LABEL characters (8 x 30 + 7
# dots = 247; STRICTER). A union of shapes (4 labels of 62, 8 of 30, ...) took more than the 5 s compile deadline: the
# compile cost grows with the regex positions.
HOSTNAME_LABELS, HOSTNAME_LABEL = 8, 30
_HOSTNAME = rf"{_label(HOSTNAME_LABEL)}(\.{_label(HOSTNAME_LABEL)}){{0,{HOSTNAME_LABELS - 1}}}"
# RFC 6570 level 4, less what python-jsonschema's uri_template 1.3.0 refuses: the reserved operators (= , ! @ |), a
# percent escape in a variable name, and a prefix of 4 digits.
_VARNAME = r"\w(\.?\w)*"
_VARSPEC = rf"{_VARNAME}(:[1-9]\d{{0,2}}|\*|)"
_LITERAL = r"([\x21\x23-\x24\x26\x28-\x3B\x3D\x3F-\x5B\x5D\x5F\x61-\x7A\x7E]|%[0-9A-Fa-f][0-9A-Fa-f])"
_URI_TEMPLATE = rf"({_LITERAL}|\{{[+#./;?&]?{_VARSPEC}(,{_VARSPEC})*\}})*"

EXACT = {
    "date": f"({_DATE})",
    "time": f"({_TIME})",
    "date-time": f"({_DATE}T{_TIME})",
    "ipv4": f"({_IPV4})",
    "ipv6": f"({_IPV6})",
    # The reference refuses a 0 before another digit (100, 205): the integer is 0, or digits 1-9 and one last 0.
    "relative-json-pointer": rf"((0|[1-9]+0?)(#|(/{_POINTER_CHAR}*)*))",
    "hostname": f"({_HOSTNAME})",
    "uri-template": f"({_URI_TEMPLATE})",
}

# The engine form of each regex (pattern_grammar.engine_pattern): the regex, or (?:R)() when its automaton can have
# two end states without edges out (date, time, date-time, ipv4, ipv6).
EXACT_ENGINE = {fmt: engine_pattern(regex, max_bytes=None) for fmt, regex in EXACT.items()}

# The schema places that hold a schema (server/schema_exact.py Walk.children): a value elsewhere is data.
_SCHEMA_MAPS = ("properties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "prefixItems")
_SCHEMA_ONE = ("items", "additionalProperties")


def grammar_schema(schema):
    """The schema for the engine: each `format` of EXACT becomes `pattern` with its exact regex, at the same key
    position, and each `pattern` is in its engine form (pattern_grammar.engine_pattern). Returns `schema` itself when
    nothing changes; it never changes `schema`."""
    if not isinstance(schema, dict):
        return schema
    out = None
    for key, value in schema.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            new = {name: grammar_schema(child) for name, child in value.items()}
            changed = any(new[name] is not child for name, child in value.items())
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            new = [grammar_schema(child) for child in value]
            changed = any(a is not b for a, b in zip(new, value))
        elif key in _SCHEMA_ONE:
            new = grammar_schema(value)
            changed = new is not value
        else:
            continue
        if changed:
            out = dict(schema) if out is None else out
            out[key] = new
    pattern = schema.get("pattern")
    engine = engine_pattern(pattern) if isinstance(pattern, str) else pattern
    if engine is not pattern:
        out = dict(schema) if out is None else out
        out["pattern"] = engine
    fmt = schema.get("format")
    if isinstance(fmt, str) and fmt in EXACT and "pattern" not in schema:
        base = schema if out is None else out
        out = {("pattern" if key == "format" else key): (EXACT_ENGINE[fmt] if key == "format" else value)
               for key, value in base.items()}
    return schema if out is None else out


def python_regex(regex):
    """The regex in Python re syntax (tests and tools): Python writes a code point as \\U00XXXXXX."""
    return re.sub(r"\\u\{([0-9A-Fa-f]{1,6})\}", lambda m: "\\U%08X" % int(m.group(1), 16), regex)
