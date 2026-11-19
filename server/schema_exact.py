"""The keyword allow-list of structured output (review H2 round 2).

A schema is served only when the engine grammar takes exactly the documents that the schema allows, except for the
named grammar-stricter policies of STRICTER. xgrammar's converter (third_party/xgrammar/cpp/json_schema_converter.cc,
SchemaParser::Parse) reads one branch of each schema node: the first of $ref, const, enum, anyOf, type, properties
(or additionalProperties), items (or prefixItems), else any value. A value keyword that the node's branch drops is a
400 constraint_keyword that names the keyword and the node's JSON pointer. The server never rewrites the schema, except
in the engine spec: the formats of server/format_grammar.EXACT get their exact regex in place of xgrammar's looser one,
and each pattern gets its engine form (server/pattern_grammar.py).
"""
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from .api import ApiError
from .pattern_grammar import refusal
from .tool_schema import matches

# The grammar takes fewer documents than the schema in these cases only (docs/OPENAI_CLIENTS.md).
STRICTER = (
    "an object with properties and no additionalProperties holds no other property",
    "a pattern matches the whole string, not a part of it",
    "a format of BUILTIN_FORMATS is an assertion",
    "a string with minLength or maxLength holds no escape sequence",
    "a number with a bound holds only the decimals of the bound's grid",
    "a format grammar takes a part of the format's values only (server/format_grammar.py)",
)
# Keywords that the grammar cannot enforce: a 400 that names the keyword, anywhere in the schema.
REFUSED = ("allOf", "oneOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "dependencies",
           "patternProperties", "propertyNames", "unevaluatedProperties", "unevaluatedItems", "contains",
           "minContains", "maxContains", "uniqueItems", "multipleOf", "minProperties", "maxProperties",
           "additionalItems", "$anchor", "$dynamicRef", "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor",
           "$vocabulary")
# Keywords that change where a reference binds: the schema root only.
ROOT_ONLY = ("$id", "id")
DEFINITIONS = ("$defs", "definitions")
NUMBER = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
STRING = ("minLength", "maxLength", "pattern", "format")
ARRAY = ("items", "prefixItems", "minItems", "maxItems")
OBJECT = ("properties", "required", "additionalProperties")
# The value keywords: each one limits the documents. Any other keyword is an annotation (title, description,
# default, examples, $comment, ...) or unknown, and the schema, the grammar and tool_schema.matches all ignore it.
VALUE = ("$ref", "const", "enum", "anyOf", "type") + NUMBER + STRING + ARRAY + OBJECT
# The formats that the converter enforces (JSONFormatToRegexPattern); it ignores any other format, as the schema does.
# The server gives the engine its own exact regex for some of them (server/format_grammar.py EXACT).
BUILTIN_FORMATS = ("date", "date-time", "duration", "email", "hostname", "ipv4", "ipv6", "json-pointer",
                   "relative-json-pointer", "time", "uri", "uri-reference", "uri-template", "uuid")
# Pattern escapes that cannot name a quote, a backslash or a control character. The converter writes the other
# characters of a pattern into the JSON string as they are: `\\`, `\"`, `\n`, `\x01` or `\u005c` gave invalid JSON or
# a value that the pattern forbids (probes through tests/constraint_text_check.cpp).
PATTERN_ESCAPES = frozenset("dDwWsS.-^$|?*+()[]{}/,:;=!<>#&%'@`~_")
MAX_DEPTH = 64
# The converter keeps an integer literal of this range; it writes any other one as a float64, so the grammar would take
# another literal (18446744073709551617 became 1.8446744073709552e+19; probes through tests/constraint_text_check.cpp).
INT64 = (-(2 ** 63), 2 ** 63 - 1)
# The number grammar of a bounded number (JSONSchemaConverter::GenerateNumber): decimals on a grid of 10^-6, and an
# empty range gives a grammar that takes no digit at all (it took `{"x":}`). Bounds past 1e15 break its range.
NUMBER_GRID = Decimal("0.000001")
MAX_NUMBER_BOUND = 10 ** 15
# The keys that the converter's cache key skips (SchemaParser::ComputeCacheKey).
CONVERTER_ANNOTATIONS = frozenset(("title", "default", "description", "examples", "deprecated", "readOnly",
                                   "writeOnly", "$comment", "$schema"))


def pointer(path):
    return "#" + "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in path)


def refuse(keyword, path, why=""):
    detail = f" ({why})" if why else ""
    raise ApiError(f"Structured output does not support the JSON schema keyword {keyword}{detail} at {pointer(path)}.",
                   None, code="constraint_keyword")


def impossible(message, path):
    """No document matches the schema."""
    raise ApiError(f"{message} at {pointer(path)}.", None, code="constraint_impossible")


def empty_branch(message, path):
    """The schema has valid documents, but this node has none: the grammar could enter it and not end."""
    raise ApiError(f"{message} at {pointer(path)}. The schema has other valid documents, but the grammar cannot end "
                   "a value that enters this node. Remove the node, or make it unreachable.", None,
                   code="constraint_empty_branch")


def ref_path(ref):
    """The tokens of a local reference, as tool_schema.resolve reads them."""
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in ref[2:].split("/"))


class Walk:
    """One walk of a schema document. tool: a strict tool's parameters (each value gets its own grammar, beside the
    root definitions only)."""

    def __init__(self, root, tool):
        self.root, self.tool = root, tool
        self.nodes = {}
        self.refs = []
        self.dead = {}  # path: {type or None: why} for a type (None: the whole node) that has no value

    def run(self):
        self.node(self.root, ())
        for ref, path in self.refs:
            target = ref_path(ref)
            if target not in self.nodes:
                refuse("$ref", path, "the pointer does not name a schema of this document")
            self.target(self.nodes[target], target)
        self.productive()

    @staticmethod
    def target(schema, path):
        """The converter generates a reference's target without its cache (JSONSchemaConverter::GenerateRef), so
        strict mode closes even the plain {"type": "array"} and {"type": "object"} there."""
        if not isinstance(schema, dict) or not isinstance(schema.get("type", []), (list, str)):
            return
        declared = schema.get("type", [])
        types = set(declared) if isinstance(declared, list) else {declared}
        if "array" in types and "items" not in schema:
            refuse("items", path, "absent in a $ref target: the grammar holds no item")
        if "object" in types and not {"properties", "additionalProperties"} & set(schema):
            refuse("additionalProperties", path, "absent in a $ref target with no properties: the grammar holds "
                   "no property")

    def node(self, schema, path, depth=0):
        if depth > MAX_DEPTH:
            raise ApiError("The JSON schema is too deep for structured output.", None, code="constraint_depth")
        self.nodes[path] = schema
        if not isinstance(schema, dict):
            return
        self.common(schema, path)
        if self.tool and not path:
            # The call grammar reads the root (server/structured.py tool_root): only a root $ref has a branch here.
            if "$ref" in schema:
                self.beside(schema, path, "$ref")
        else:
            self.branch(schema, path)
        self.children(schema, path, depth)

    def common(self, schema, path):
        for keyword in REFUSED:
            if keyword in schema:
                refuse(keyword, path)
        for keyword in ROOT_ONLY:
            if keyword in schema and path:
                refuse(keyword, path, "below the root")
        if self.tool and path:
            for keyword in DEFINITIONS:
                if keyword in schema:
                    refuse(keyword, path, "below the root of a strict tool: a reference could bind to it")
        if isinstance(schema.get("items"), list):
            refuse("items", path, "array form")
        if "$ref" in schema:
            self.reference(schema["$ref"], path)
        if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
            impossible("A JSON schema enum must be a nonempty list", path)

    def reference(self, ref, path):
        if not isinstance(ref, str) or not ref.startswith("#/"):
            refuse("$ref", path, "remote")
        if "%" in ref:
            refuse("$ref", path, "percent-encoded")
        if self.tool and path and not ref.startswith(("#/$defs/", "#/definitions/")):
            refuse("$ref", path, "outside $defs in a strict tool")
        self.refs.append((ref, path))

    def branch(self, schema, path):
        """The converter's branch of this node (SchemaParser::Parse) must hold every value keyword of the node."""
        if "$ref" in schema:
            self.beside(schema, path, "$ref")
        elif "const" in schema or "enum" in schema:
            self.finite(schema, path)
        elif "anyOf" in schema:
            self.beside(schema, path, "anyOf")
        elif "type" in schema:
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not types:
                refuse("type", path, "an empty list")
            self.typed(schema, path, set(types))
        elif "properties" in schema:
            self.typed(schema, path, {"object"})
        else:
            # The any branch: no keyword limits the value. tool_schema.matches reads properties as an object type, so
            # properties is the one keyword that implies a type; additionalProperties and items do not.
            for keyword in VALUE:
                if keyword in schema:
                    refuse(keyword, path, "without type")

    @staticmethod
    def beside(schema, path, head):
        """$ref and anyOf: the converter reads the head keyword only, so no other value keyword may stand beside it."""
        for keyword in VALUE:
            if keyword in schema and keyword != head:
                refuse(keyword, path, f"beside {head}")

    def finite(self, schema, path):
        """const or enum: the grammar holds exactly the listed values. The schema holds the values that also match
        every other keyword of the node, so each value must match them (checked here, not rewritten)."""
        head = "const" if "const" in schema else "enum"
        rest = {key: value for key, value in schema.items() if key != head}
        keyword = uncheckable(rest, self.root)
        if keyword:
            refuse(keyword, path, f"beside {head}: the server cannot check the {head} values against it")
        values = [schema["const"]] if head == "const" else schema["enum"]
        if any(wide_integer(value) for value in values):
            refuse(head, path, "an integer outside the 64-bit range: the converter writes it as a float64, so the "
                   "grammar would take another literal")
        for value in values:
            if not matches(value, rest, self.root):
                failed = next((key for key in VALUE if key in rest and not matches(value, {key: rest[key]}, self.root)),
                              "type")
                if head == "const":
                    self.dead[path] = {None: f"The JSON schema const fails its {failed}: no value matches it"}
                    return
                refuse(failed, path, "beside enum: an enum value fails it")

    def typed(self, schema, path, types):
        if "string" in types:
            builtin = schema.get("format") in BUILTIN_FORMATS
            for length in ("minLength", "maxLength"):
                if length in schema and ("pattern" in schema or builtin):
                    refuse(length, path, "beside " + ("pattern" if "pattern" in schema else "format"))
            if "pattern" in schema and builtin:
                refuse("pattern", path, "beside format")
            if "pattern" in schema:
                check_pattern(schema["pattern"], path)
        # A keyword of a type that the node does not have is ignored (a length beside type integer).
        dead = {}
        for kind, low, high in (("string", "minLength", "maxLength"), ("array", "minItems", "maxItems")):
            if kind in types and type(schema.get(low)) is int and type(schema.get(high)) is int and \
                    schema[low] > schema[high]:
                dead[kind] = f"The JSON schema has {low} above {high}"
        if "number" in types and number_range(schema, path):
            dead["number"] = "The JSON schema number range is empty"
        if dead:
            self.dead[path] = dead
        if "object" in types:
            properties = schema.get("properties", {})
            if any(name not in properties for name in schema.get("required", [])):
                refuse("required", path, "a name that is not in properties")
        # The converter's strict mode closes an array without items and an object without additionalProperties,
        # except the plain {"type": "array"} and {"type": "object"} (its cache key skips CONVERTER_ANNOTATIONS).
        others = set(schema) - CONVERTER_ANNOTATIONS - {"type"}
        if "array" in types and "items" not in schema and others:
            refuse("prefixItems" if "prefixItems" in schema else "items", path,
                   "items is absent beside other keywords: the grammar holds no other item")
        if "object" in types and "additionalProperties" not in schema and "properties" not in schema and others:
            refuse("additionalProperties", path,
                   "absent beside other keywords and no properties: the grammar holds no property")

    def children(self, schema, path, depth):
        for key in ("properties",) + DEFINITIONS:
            if isinstance(schema.get(key), dict):
                for name, child in schema[key].items():
                    self.node(child, path + (key, name), depth + 1)
        for key in ("anyOf", "prefixItems"):
            if isinstance(schema.get(key), list):
                for index, child in enumerate(schema[key]):
                    self.node(child, path + (key, str(index)), depth + 1)
        for key in ("items", "additionalProperties"):
            if isinstance(schema.get(key), (dict, bool)):
                self.node(schema[key], path + (key,), depth + 1)

    def node_types(self, schema, path):
        """The types of a node's typed branch; None for a node that is not one ($ref, const, enum, anyOf, any)."""
        if any(key in schema for key in ("$ref", "const", "enum", "anyOf")):
            return None
        if self.tool and not path:
            return {"object"}
        declared = schema.get("type", "object" if "properties" in schema else None)
        if declared is None:
            return None
        return set(declared) if isinstance(declared, list) else {declared}

    def alternatives(self, schema, path):
        """The ways to a value of this node: each one is the list of nodes that it needs, all of them."""
        if "$ref" in schema:
            return [[ref_path(schema["$ref"])]]
        if "anyOf" in schema and "const" not in schema and "enum" not in schema:
            return [[path + ("anyOf", str(index))] for index in range(len(schema["anyOf"]))]
        dead = self.dead.get(path, {})
        types = self.node_types(schema, path)
        if None in dead:
            return []
        if types is None:
            return [[]]
        ways = []
        for kind in sorted(types - set(dead)):
            need = []
            if kind == "object":
                properties = schema.get("properties", {})
                # A required false parameter of a strict tool: call_format refuses it with its own message.
                need = [path + ("properties", name) for name in schema.get("required", [])
                        if name in properties and not (self.tool and not path and self.false_ref(properties[name]))]
            elif kind == "array":
                prefix = len(schema.get("prefixItems", []))
                least = schema.get("minItems", 0)
                need = [path + ("prefixItems", str(index)) for index in range(min(least, prefix))]
                need += [path + ("items",)] if least > prefix else []
            ways.append(need)
        return ways

    def edges(self, schema, path):
        """The nodes whose grammar a value of this node can enter."""
        if "$ref" in schema:
            return [ref_path(schema["$ref"])]
        if "const" in schema or "enum" in schema:
            return []
        if "anyOf" in schema:
            return [path + ("anyOf", str(index)) for index in range(len(schema["anyOf"]))]
        types = self.node_types(schema, path) or set()
        out = []
        if "object" in types:
            out += [path + ("properties", name) for name in schema.get("properties", {})]
            out += [path + ("additionalProperties",)] if "additionalProperties" in schema else []
        if "array" in types:
            out += [path + ("prefixItems", str(index)) for index in range(len(schema.get("prefixItems", [])))]
            out += [path + ("items",)] if "items" in schema else []
        return out

    def productive(self):
        """A node has a value when one of its alternatives has every node that it needs. With no value at the root,
        no document matches (constraint_impossible). Else a node that the grammar can enter and that has no value is
        an empty branch: a value that enters it has no end (constraint_empty_branch)."""
        table = {path: schema for path, schema in self.nodes.items() if isinstance(schema, bool)}
        rules, users = {}, {}
        for path, schema in self.nodes.items():
            if isinstance(schema, dict):
                rules[path] = self.alternatives(schema, path)
                for need in {need for way in rules[path] for need in way}:
                    users.setdefault(need, []).append(path)
        work = list(rules)
        while work:
            path = work.pop()
            if table.get(path) or path not in rules:
                continue
            if any(all(table.get(need, False) for need in way) for way in rules[path]):
                table[path] = True
                work.extend(users.get(path, ()))
        reach, stack = set(), [()]
        while stack:
            path = stack.pop()
            if path in reach or not isinstance(self.nodes.get(path), dict):
                continue
            reach.add(path)
            stack.extend(self.edges(self.nodes[path], path))
        fault = empty_branch if table.get(()) else impossible
        for path, schema in reversed(self.nodes.items()):  # the deepest node first
            if path in reach and not self.false_ref(schema) and (not table.get(path) or (
                    fault is empty_branch and path in self.dead)):  # a union type with no value: it has no end
                dead = self.dead.get(path, {})
                fault(dead.get(None) or next(iter(dead.values()), "The JSON schema node has no finite value that "
                                             "matches it"), path)

    def false_ref(self, schema):
        """A reference to the schema false: the call grammar never emits it (structured.call_format) and the
        converter refuses it, so it is not a node without an end."""
        seen = set()
        while isinstance(schema, dict) and "$ref" in schema and set(schema) - CONVERTER_ANNOTATIONS == {"$ref"}:
            target = ref_path(schema["$ref"])
            if target in seen:
                return False
            seen.add(target)
            schema = self.nodes.get(target)
        return schema is False


def number_range(schema, path):
    """True when no number lies between the bounds. A range that holds numbers but no point of the converter's grid is
    a 400 constraint_range: the schema has values that the grammar cannot write."""
    low = high = None
    low_open = high_open = False
    for key in ("minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"):
        value = schema.get(key)
        if type(value) in (int, float) and abs(value) >= MAX_NUMBER_BOUND:
            refuse(key, path, "a number bound of 1e15 or more: the number grammar cannot hold its range")
    if type(schema.get("minimum")) in (int, float):
        low = Decimal(str(schema["minimum"]))
    if type(schema.get("exclusiveMinimum")) in (int, float):
        value = Decimal(str(schema["exclusiveMinimum"]))
        if low is None or value >= low:
            low, low_open = value, True
    if type(schema.get("maximum")) in (int, float):
        high = Decimal(str(schema["maximum"]))
    if type(schema.get("exclusiveMaximum")) in (int, float):
        value = Decimal(str(schema["exclusiveMaximum"]))
        if high is None or value <= high:
            high, high_open = value, True
    if low is None or high is None:
        return False
    if low > high or (low == high and (low_open or high_open)):
        return True
    first = (low / NUMBER_GRID).to_integral_value(ROUND_CEILING)
    last = (high / NUMBER_GRID).to_integral_value(ROUND_FLOOR)
    first += 1 if low_open and first * NUMBER_GRID == low else 0
    last -= 1 if high_open and last * NUMBER_GRID == high else 0
    if first > last:
        raise ApiError(f"The number grammar writes decimals with at most 6 digits after the point, and none lies "
                       f"between the bounds at {pointer(path)}. Widen the range to hold such a decimal.", None,
                       code="constraint_range")
    return False


def wide_integer(value):
    """True when value holds an integer outside INT64, at any depth."""
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is int and not INT64[0] <= item <= INT64[1]:
            return True
        stack.extend(item.values() if isinstance(item, dict) else item if isinstance(item, list) else ())
    return False


def check_pattern(pattern, path):
    if not isinstance(pattern, str):
        refuse("pattern", path, "not text")
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == '"' or ord(char) < 32:
            refuse("pattern", path, "a quote or a control character")
        if char == "\\":
            escaped = pattern[index + 1:index + 2]
            if escaped not in PATTERN_ESCAPES:
                refuse("pattern", path, f"the escape \\{escaped}")
            index += 1
        index += 1
    # A part that the engine reads in another way than the pattern means (server/pattern_grammar.py).
    reason = refusal(pattern)
    if reason:
        refuse("pattern", path, reason)


def uncheckable(schema, root):
    """The first keyword (pattern or format) that tool_schema.matches does not check, in `schema` or in any schema
    that it holds or references; None when matches checks all of it."""
    seen, stack = set(), [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        for keyword in ("pattern", "format"):
            if keyword in node:
                return keyword
        for key in ("properties",) + DEFINITIONS:
            if isinstance(node.get(key), dict):
                stack.extend(node[key].values())
        for key in ("anyOf", "prefixItems", "allOf", "oneOf"):
            if isinstance(node.get(key), list):
                stack.extend(node[key])
        for key in ("items", "additionalProperties", "not"):
            stack.append(node.get(key))
        if isinstance(node.get("$ref"), str) and node["$ref"].startswith("#/"):
            target = root
            for part in ref_path(node["$ref"]):
                target = target.get(part) if isinstance(target, dict) else None
            stack.append(target)
    return None


def check_exact(schema, field, tool=False):
    """Refuse a schema whose grammar would take a document that the schema forbids (or one that is impossible).
    field: the request field of the 400 (response_format or tools)."""
    try:
        Walk(schema, tool).run()
    except ApiError as error:
        error.param = field
        raise
