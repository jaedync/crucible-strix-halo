"""tools/gates.py reads the engine's sidecar success line; the start guard's stop line must not match it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gates  # noqa: E402


def test_success_line_matches():
    m = gates.SIDECAR_LINE.search('mtp sidecar: 17 tensors from side-d2.safetensors (42202168 bytes)\n')
    assert m and m[1] == '17' and m[2] == '42202168'


def test_stop_lines_do_not_match():
    for line in ('mtp sidecar error: 0 tensors from side.safetensors (10 bytes), expected 17 or more; start stopped',
                 'mtp sidecar error: cannot read side.safetensors: No such file or directory; start stopped',
                 'mtp sidecar: 0 tensors from side.safetensors (10 bytes)'):
        assert gates.SIDECAR_LINE.search(line + '\n') is None
