"""A seeded random test of the keyword allow-list (review H2 round 2, server/schema_exact.py).

Each seed makes small schemas from the keywords of the allow-list (also combinations that the server must refuse),
and random JSON values for each schema that the server serves. Every value goes through the real engine grammar
(build/tests/constraint_text_check) and through server/tool_schema.matches:

- the grammar takes a value only when matches takes it, on every value (zero tolerance: this is review H2);
- matches takes a value only when the grammar takes it, except for the named grammar-stricter policies of
  schema_exact.STRICTER (the test models policy 1 in the reference and skips values that policies 4 and 5 can touch).

Patterns and formats are not generated: matches does not check them (server/tests/test_structured_exact.py has
fixed cases). Both paths run: a response_format answer and a strict tool call."""
import copy
import itertools
import json
import random
import unittest
from server.api import ApiError
from server.tool_schema import matches, types
from server.tests.test_structured_exact import answer_request, tool_request
from server.tests.test_structured_strict import CHECK, call, engine

SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
SCHEMAS = 50
VALUES = 24
NAMES = ("a", "b", "c")
EXTRA = ("a", "b", "c", "d")
CHARS = ("a", "b", " ", '"', "\\", "\n", "\u00e9", "\U0001F600")
MAX_TEXTS = 24


class Gen:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def pick(self, *items):
        return self.rng.choice(items)

    def chance(self, p):
        return self.rng.random() < p

    # Values.
    def string(self):
        return "".join(self.rng.choice(CHARS) for _ in range(self.rng.randint(0, 3)))

    def scalar(self):
        return self.pick(None, True, False, self.rng.randint(-3, 3), self.rng.randint(-6, 6) / 2 + 0.25,
                         self.string())

    def value(self, depth=0):
        kind = self.rng.randint(0, 5 if depth < 2 else 3)
        if kind <= 3:
            return self.scalar()
        if kind == 4:
            return [self.value(depth + 1) for _ in range(self.rng.randint(0, 3))]
        return {key: self.value(depth + 1) for key in self.rng.sample(EXTRA, self.rng.randint(0, 3))}

    def fit(self, schema, root, depth=0):
        """A value that may match `schema` (so that served schemas see accepted values too)."""
        if depth > 3 or not isinstance(schema, dict) or self.chance(0.15):
            return self.value(depth)
        if "$ref" in schema:
            target = root
            for part in schema["$ref"][2:].split("/"):
                target = target[part]
            return self.fit(target, root, depth + 1)
        if "const" in schema:
            return copy.deepcopy(schema["const"])
        if "enum" in schema:
            return copy.deepcopy(self.rng.choice(schema["enum"]))
        if "anyOf" in schema:
            return self.fit(self.rng.choice(schema["anyOf"]), root, depth + 1)
        kinds = types(schema, root) or {"object", "array", "string", "integer"}
        kind = self.rng.choice(sorted(kinds))
        if kind == "object":
            properties = schema.get("properties", {})
            out = {}
            for name in NAMES:
                if name in properties and (name in schema.get("required", []) or self.chance(0.6)):
                    out[name] = self.fit(properties[name], root, depth + 1)
            if self.chance(0.3):
                out["d"] = self.fit(schema.get("additionalProperties", {}), root, depth + 1)
            return out
        if kind == "array":
            prefix = schema.get("prefixItems", [])
            out = [self.fit(item, root, depth + 1) for item in prefix[:self.rng.randint(0, len(prefix))]]
            if len(out) == len(prefix):
                out += [self.fit(schema.get("items", {}), root, depth + 1) for _ in range(self.rng.randint(0, 2))]
            return out
        if kind == "string":
            return self.string()
        if kind == "number":
            return self.rng.randint(-6, 6) / 2 + self.pick(0, 0.25)
        if kind in ("integer",):
            return self.rng.randint(-3, 3)
        return self.pick(None, True, False)

    # Schemas.
    def node(self, depth=0, defs=()):
        leaf = depth >= 2
        kind = self.rng.randint(0, 6 if leaf else 11)
        if kind == 0:
            node = {"type": "integer"} | self.bounds(int)
        elif kind == 1:
            node = {"type": "number"} | self.bounds(lambda v: v / 2)
        elif kind == 2:
            node = {"type": "string"} | self.lengths()
        elif kind == 3:
            node = {"type": self.pick("boolean", "null", ["integer", "string"], ["string", "null"])}
        elif kind == 4:
            values = [self.value(2) for _ in range(self.rng.randint(1, 3))]
            node = {"enum": values} | self.siblings(values)
        elif kind == 5:
            value = self.value(2)
            node = {"const": value} | self.siblings([value])
        elif kind == 6:
            node = self.pick({}, {"minimum": 1}, {"maxLength": 1}, {"required": ["a"]}, {"items": {}}, True)
            if defs and self.chance(0.5):
                node = {"$ref": "#/$defs/" + self.rng.choice(defs)}
                if self.chance(0.2):
                    node["minimum"] = 0
        elif kind in (7, 8):
            node = self.obj(depth, defs)
        elif kind == 9:
            node = {"type": "array"} | self.counts()
            if self.chance(0.4):
                node["prefixItems"] = [self.node(depth + 1, defs) for _ in range(self.rng.randint(1, 2))]
            if self.chance(0.7) or ("prefixItems" in node and self.chance(0.8)):
                node["items"] = self.pick(False, True, self.node(depth + 1, defs))
        else:
            node = {"anyOf": [self.node(depth + 1, defs) for _ in range(self.rng.randint(1, 3))]}
            if self.chance(0.15):
                node["type"] = "integer"
        if isinstance(node, dict) and self.chance(0.1):
            node["description"] = "d"
        return node

    def obj(self, depth, defs):
        node = {} if self.chance(0.2) else {"type": "object"}
        names = [name for name in NAMES if self.chance(0.6)]
        node["properties"] = {name: self.node(depth + 1, defs) for name in names}
        required = [name for name in names if self.chance(0.5)] + (["c"] if self.chance(0.1) else [])
        if required:
            node["required"] = list(dict.fromkeys(required))
        if self.chance(0.5):
            node["additionalProperties"] = self.pick(False, True, {"type": "integer"})
        return node

    def bounds(self, scale):
        out = {}
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if self.chance(0.3):
                out[key] = scale(self.rng.randint(-4, 4))
        return out

    def lengths(self):
        out = {}
        for key in ("minLength", "maxLength"):
            if self.chance(0.4):
                out[key] = self.rng.randint(0, 3)
        return out

    def counts(self):
        out = {}
        for key in ("minItems", "maxItems"):
            if self.chance(0.3):
                out[key] = self.rng.randint(0, 3)
        return out

    def siblings(self, values):
        """Siblings of an enum or const: often the type of the values, sometimes a keyword that some value fails."""
        out = {}
        kinds = sorted({"null" if v is None else "boolean" if type(v) is bool else "integer" if type(v) is int
                        else "number" if type(v) is float else "string" if isinstance(v, str)
                        else "array" if isinstance(v, list) else "object" for v in values})
        if self.chance(0.5):
            out["type"] = kinds if len(kinds) > 1 else kinds[0]
        extra = self.rng.randint(0, 4)
        if extra == 1:
            out["minimum"] = self.rng.randint(-2, 2)
        elif extra == 2:
            out["maxLength"] = self.rng.randint(0, 2)
        elif extra == 3:
            out["maxItems"] = self.rng.randint(0, 2)
        return out

    def schema(self):
        defs = {}
        if self.chance(0.4):
            defs = {"D0": self.node(1)}
        root = self.obj(0, tuple(defs)) if self.chance(0.7) else self.node(0, tuple(defs))
        if defs and isinstance(root, dict):
            root["$defs"] = defs
        return root


def closed(schema):
    """The reference of policy 1: an object node with properties and no additionalProperties holds no other one."""
    if isinstance(schema, list):
        return [closed(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key in ("properties", "$defs"):
            out[key] = {name: closed(child) for name, child in value.items()}
        elif key in ("anyOf", "prefixItems", "items", "additionalProperties"):
            out[key] = closed(value)
        else:
            out[key] = value
    if "properties" in out and "additionalProperties" not in out and not {"$ref", "const", "enum", "anyOf"} & set(out):
        out["additionalProperties"] = False
    return out


def texts(value):
    """Every compact JSON text of `value` (each order of the keys of each object), or None when there are too many."""
    if isinstance(value, dict):
        keys = list(value)
        parts = [texts(value[key]) for key in keys]
        if any(part is None for part in parts):
            return None
        out = []
        for order in itertools.permutations(range(len(keys))):
            for chosen in itertools.product(*(parts[i] for i in order)):
                out.append("{" + ",".join(json.dumps(keys[i], ensure_ascii=False) + ":" + text
                                          for i, text in zip(order, chosen)) + "}")
                if len(out) > MAX_TEXTS:
                    return None
        return out
    if isinstance(value, list):
        parts = [texts(item) for item in value]
        if any(part is None for part in parts):
            return None
        out = ["[" + ",".join(chosen) + "]" for chosen in itertools.product(*parts)]
        return out if len(out) <= MAX_TEXTS else None
    return [json.dumps(value, ensure_ascii=False)]


def tight_value(value, schema):
    """True when policy 4 or 5 can refuse this value: a string with an escape under a length bound, or a decimal
    under a number bound, anywhere in the schema (a coarse test: the reference skips the value)."""
    flat = json.dumps(schema)
    text = json.dumps(value, ensure_ascii=False)
    lengths = "Length" in flat and any(char in text for char in ('\\"', "\\\\", "\\n"))
    decimals = '"number"' in flat and any(type(v) is float for v in leaves(value))
    return lengths or decimals


def leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    else:
        yield value


@unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
class RandomTests(unittest.TestCase):
    def compare(self, schema, values, constraint, render):
        """Run each value's texts through the grammar; return the disagreements."""
        lines, spans = [], []
        for value in values:
            options = render(value)
            if options is None:
                continue
            spans.append((value, len(lines), len(options)))
            lines.extend(options)
        if not lines:
            return [], 0
        results = engine(constraint, lines)
        if any(result.startswith("refused") for result in results):
            # The engine refuses the grammar at open (a 400, no output), for example an impossible branch.
            self.assertEqual(set(results), {results[0]}, results[0])
            return None, 0
        reference = closed(schema)
        loose, tight = [], []
        for value, start, count in spans:
            grammar = "accept" in results[start:start + count]
            allowed = matches(value, schema, schema)
            if grammar and not allowed:
                loose.append(value)
            elif allowed and not grammar and matches(value, reference, reference) and not tight_value(value, schema):
                tight.append(value)
        return loose + tight, len(spans)

    def test_answers(self):
        totals = dict(served=0, refused=0, engine_refused=0, values=0)
        for seed in SEEDS:
            gen = Gen(seed)
            for index in range(SCHEMAS):
                schema = gen.schema()
                if not isinstance(schema, dict):
                    continue
                try:
                    constraint = answer_request(schema).constraint
                except ApiError:
                    totals["refused"] += 1
                    continue
                totals["served"] += 1
                values = [gen.fit(schema, schema) for _ in range(VALUES)] + [gen.value() for _ in range(4)]
                bad, count = self.compare(schema, values, constraint, texts)
                if bad is None:
                    totals["engine_refused"] += 1
                    continue
                totals["values"] += count
                with self.subTest(seed=seed, index=index):
                    self.assertEqual(bad, [], json.dumps(schema))
        print("answers:", totals)
        self.assertGreater(totals["served"], len(SEEDS) * SCHEMAS // 4)
        self.assertGreater(totals["refused"], len(SEEDS) * 4)

    def test_tool_calls(self):
        totals = dict(served=0, refused=0, engine_refused=0, values=0)
        for seed in SEEDS:
            gen = Gen(seed + 1000)
            for index in range(SCHEMAS):
                parameters = gen.obj(0, ())
                defs = {"D0": gen.node(1)} if gen.chance(0.4) else {}
                if defs:
                    parameters["$defs"] = defs
                    parameters["properties"]["c"] = {"$ref": "#/$defs/D0"}
                try:
                    constraint = tool_request(parameters).constraint
                except ApiError:
                    totals["refused"] += 1
                    continue
                totals["served"] += 1
                properties = parameters["properties"]

                def render(arguments):
                    if not isinstance(arguments, dict) or set(arguments) - set(properties):
                        return None  # the call grammar names the schema's parameters only (policy 1)
                    out = {}
                    for key in properties:
                        if key not in arguments:
                            continue
                        raw = types(properties[key], parameters) == {"string"}
                        if raw:
                            if not isinstance(arguments[key], str) or "\n" in arguments[key]:
                                return None  # a raw string parameter holds the text as it is
                            out[key] = [arguments[key]]
                        else:
                            out[key] = texts(arguments[key])
                            if out[key] is None:
                                return None
                    rendered = [call(**dict(zip(out, chosen))) for chosen in itertools.product(*out.values())]
                    return rendered if len(rendered) <= MAX_TEXTS else None
                values = [gen.fit(parameters, parameters) for _ in range(VALUES)]
                bad, count = self.compare(parameters, values, constraint, render)
                if bad is None:
                    totals["engine_refused"] += 1
                    continue
                totals["values"] += count
                with self.subTest(seed=seed, index=index):
                    self.assertEqual(bad, [], json.dumps(parameters))
        print("tool calls:", totals)
        self.assertGreater(totals["served"], len(SEEDS) * SCHEMAS // 4)


if __name__ == "__main__":
    unittest.main()
