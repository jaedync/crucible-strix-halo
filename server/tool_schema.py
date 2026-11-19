"""Validate common tool argument schema rules without an extra dependency."""
from .schema_limits import MAX_REFERENCE_CHAIN, WorkBudget

TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}

def resolve(schema, root, depth=0, budget=None):
    budget = budget if budget is not None else WorkBudget()
    budget.visit()
    if depth > MAX_REFERENCE_CHAIN:
        raise ValueError("schema reference chain exceeds 32 levels")
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError("only local JSON schema references are supported")
        target = root
        for part in ref[2:].split("/"):
            budget.visit()
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or key not in target:
                raise ValueError("a local schema reference is missing")
            target = target[key]
        return resolve(target, root, depth + 1, budget)
    return schema

def types(schema, root, depth=0, budget=None):
    budget = budget if budget is not None else WorkBudget()
    result = budget.memo("types", (id(root), id(schema)), depth,
                         lambda: frozenset(schema_types(schema, root, depth, budget)))
    return set(result)


def schema_types(schema, root, depth, budget):
    schema = resolve(schema, root, budget=budget)
    if not isinstance(schema, dict):
        return set()
    declared = schema.get("type")
    if declared is not None:
        choices = declared if isinstance(declared, list) else [declared]
        if any(not isinstance(item, str) or item not in TYPES for item in choices):
            raise ValueError("unsupported JSON schema type")
        return set(choices)
    branches = schema.get("anyOf", schema.get("oneOf", []))
    if branches:
        # A branch without a type (true, {}, a const) takes any type: then so does the whole anyOf.
        found = [types(branch, root, depth + 1, budget) for branch in branches]
        return set() if any(not item for item in found) else set().union(*found)
    if "properties" in schema:
        return {"object"}
    return set()

def check_schema(schema, root, depth=0, budget=None):
    budget = budget if budget is not None else WorkBudget()
    return budget.memo("schema", (id(root), id(schema)), depth,
                       lambda: validate_schema_node(schema, root, depth, budget))


def validate_schema_node(schema, root, depth, budget):
    if type(schema) is bool: return
    if not isinstance(schema, dict): raise ValueError("a schema must be an object or boolean")
    types(schema, root, budget=budget)
    resolved = resolve(schema, root, budget=budget)
    if resolved is not schema:
        check_schema(resolved, root, depth + 1, budget)
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError("schema required must be a list of names")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict): raise ValueError("schema properties must be an object")
    children = list(properties.values())
    for key in ("$defs", "definitions"):
        if key in schema:
            if not isinstance(schema[key], dict): raise ValueError("schema definitions must be an object")
            children.extend(schema[key].values())
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]: raise ValueError("schema alternatives must be nonempty lists")
            children.extend(schema[key])
    for key in ("additionalProperties", "not", "additionalItems"):
        if key in schema: children.append(schema[key])
    if "items" in schema:
        children.extend(schema["items"] if isinstance(schema["items"], list) else [schema["items"]])
    for child in children: check_schema(child, root, depth + 1, budget)
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and type(schema[key]) not in (int, float): raise ValueError("schema numeric bounds must be numbers")
    for key in ("minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0): raise ValueError("schema size bounds must be nonnegative integers")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("schema enum must be a nonempty list")

def category(value):
    if value is None: return "null"
    if type(value) is bool: return "boolean"
    if type(value) is int or (type(value) is float and value.is_integer()): return "integer"
    if type(value) is float: return "number"
    if isinstance(value, str): return "string"
    if isinstance(value, dict): return "object"
    if isinstance(value, list): return "array"
    return "invalid"

def matches(value, schema, root, depth=0, budget=None, scope=None, path=()):
    budget = budget if budget is not None else WorkBudget()
    scope = budget.new_instance() if scope is None else scope
    key = (id(root), id(schema), scope, path)
    return budget.memo("matches", key, depth,
                       lambda: match_schema_node(value, schema, root, depth, budget, scope, path))


def match_schema_node(value, schema, root, depth, budget, scope, path):
    schema = resolve(schema, root, budget=budget)
    if type(schema) is bool: return schema
    if not isinstance(schema, dict): raise ValueError("a schema must be an object or boolean")
    declared = types(schema, root, budget=budget)
    actual = category(value)
    if declared and actual not in declared and not (actual == "integer" and "number" in declared): return False
    if "enum" in schema and not any(same_json(value, item, budget, depth) for item in schema["enum"]): return False
    if "const" in schema and not same_json(value, schema["const"], budget, depth): return False
    for name, rule in (("anyOf", any), ("allOf", all)):
        if name in schema and not rule(matches(value, part, root, depth + 1, budget, scope, path) for part in schema[name]): return False
    if "oneOf" in schema and sum(matches(value, part, root, depth + 1, budget, scope, path) for part in schema["oneOf"]) != 1: return False
    if "not" in schema and matches(value, schema["not"], root, depth + 1, budget, scope, path): return False
    if isinstance(value, dict): return object_matches(value, schema, root, depth, budget, scope, path)
    if isinstance(value, list): return array_matches(value, schema, root, depth, budget, scope, path)
    if isinstance(value, str): return string_matches(value, schema)
    if type(value) in (int, float): return number_matches(value, schema)
    return True

def object_matches(value, schema, root, depth, budget, scope, path):
    if any(key not in value for key in schema.get("required", [])): return False
    properties = schema.get("properties", {})
    for key, item in value.items():
        child = properties.get(key, schema.get("additionalProperties", True))
        if not matches(item, child, root, depth + 1, budget, scope, path + (key,)): return False
    return len(value) >= schema.get("minProperties", 0) and len(value) <= schema.get("maxProperties", len(value))

def array_matches(value, schema, root, depth, budget, scope, path):
    if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)): return False
    items = schema.get("items", True)
    prefix = schema.get("prefixItems")
    if isinstance(prefix, list):
        # JSON Schema 2020-12: prefixItems checks the first items, and items checks only the items after them.
        return all(matches(item, prefix[index] if index < len(prefix) else items, root, depth + 1, budget, scope,
                           path + (index,)) for index, item in enumerate(value))
    if isinstance(items, list):
        return all(matches(item, items[index] if index < len(items) else schema.get("additionalItems", True),
                           root, depth + 1, budget, scope, path + (index,)) for index, item in enumerate(value))
    return all(matches(item, items, root, depth + 1, budget, scope, path + (index,)) for index, item in enumerate(value))

def string_matches(value, schema):
    if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)): return False
    # The template advertises patterns. Do not run untrusted regular expressions here.
    return True

def same_json(left, right, budget, depth=0):
    budget.visit(depth)
    first, second = category(left), category(right)
    if first != second and {first, second} != {"integer", "number"}: return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(same_json(item, right[key], budget, depth + 1) for key, item in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(same_json(a, b, budget, depth + 1) for a, b in zip(left, right))
    return left == right

def number_matches(value, schema):
    if "minimum" in schema and value < schema["minimum"]: return False
    if "maximum" in schema and value > schema["maximum"]: return False
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]: return False
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]: return False
    return True
