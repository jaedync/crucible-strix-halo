#!/usr/bin/env python3
"""Re-make the two draft prompt ids that tools/draft_prepare.py does not write: the long copy prompt and the original
Fibonacci prompt. Both prompts are plain text in this file. The ids come from tools/tok.py (chat format, empty think block).

    python3 tools/draft_long_ids.py CHECKPOINT OUTPUT_DIR

Writes code_copy_long.ids and fibonacci_original.ids. The leak guard allow entries of tests/data/draft-long and
tests/data/draft-original name this script as their generator (tools/leak_remake.py compares the bytes).
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "code_copy_long": "Repeat the following Python code exactly, inside one Python code block:\n\n" + "\n".join(
        f"def transform_{i}(value):\n    result = value * {i + 2}\n    return result + {i + 1}\n" for i in range(100)),
    "fibonacci_original": "Write a Python function that returns the n-th Fibonacci number.",
}


def main(argv):
    checkpoint, output = Path(argv[1]), Path(argv[2])
    output.mkdir(parents=True, exist_ok=True)
    for name, prompt in PROMPTS.items():
        result = subprocess.run([sys.executable, str(ROOT / "tools/tok.py"), str(checkpoint), "encode", prompt, "--chat"],
                                text=True, capture_output=True, check=True)
        (output / (name + ".ids")).write_text(result.stdout)


if __name__ == "__main__":
    main(sys.argv)
