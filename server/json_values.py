"""Strict bounded JSON values for templates and generated tool arguments."""
import json
import math
from .api import ApiError

def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result

def json_value(value, field, depth=0):
    if depth > 64:
        raise ApiError("JSON nesting must not exceed 64 levels.", field)
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ApiError("Text must contain valid Unicode.", field) from None
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [json_value(item, field, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {json_value(key, field, depth + 1): json_value(item, field, depth + 1)
                for key, item in value.items()}
    raise ApiError("Use finite JSON values only.", field)

def history_arguments(value, field):
    """Arguments of an assistant tool call in the history. A streamed call that the server later judged invalid
    reaches the client as a prefix of its JSON object: "", or a cut after a value or inside a string value (the
    stream sends a key, a value, or an escaped string piece at a time and never a duplicate key). A client that
    joins the deltas sends that prefix back, and refusing it stopped the session (an earlier soak run). So close
    exactly these prefixes; anything else still needs valid JSON."""
    if isinstance(value, str) and value and not value.startswith("{"):
        return json_arguments(value, field)
    if value == "":
        return {}
    try:
        return json_arguments(value, field)
    except ApiError:
        if not isinstance(value, str):
            raise
        for suffix in ("}", '"}'):
            try:
                return json_arguments(value + suffix, field)
            except ApiError:
                pass
        raise

def json_arguments(value, field):
    try:
        parsed = json.loads(value, object_pairs_hook=json_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON"))) if isinstance(value, str) else value
    except (ValueError, RecursionError):
        raise ApiError("Tool arguments must contain valid JSON without duplicate fields.", field) from None
    parsed = json_value(parsed, field)
    if not isinstance(parsed, dict):
        raise ApiError("Tool arguments must be a JSON object.", field)
    return parsed
