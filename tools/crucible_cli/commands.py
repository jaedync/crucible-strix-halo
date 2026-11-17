"""Render commands at the output boundary, including hints in structured results."""
import json
from pathlib import Path
import shlex


def command_text(repo, args=""):
    command = shlex.quote(str(Path(repo).resolve() / "crucible"))
    return command + (" " + args if args else "")


def output_text(repo, text):
    command = command_text(repo)

    def replace(value):
        if isinstance(value, str):
            return value.replace("./crucible", command)
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return replace(text)
    return json.dumps(replace(value))
