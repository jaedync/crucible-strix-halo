"""The engine's YaRN tables are the reference bits (tools/yarn/rope_header.py)."""
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.yarn import rope_header  # noqa: E402


def test_header_equals_rendered_snapshots():
    assert rope_header.HEADER.read_text() == rope_header.render(rope_header.load_snapshots())


def test_factor4_snapshot_is_the_validated_one():
    # yarn4-reference.json is checked against Hugging Face (tools/yarn/reference.py, a torch test).
    snap = json.loads((ROOT / "tools" / "yarn" / "yarn4-reference.json").read_text())
    assert snap["attention_factor"] == 1 + 0.1 * math.log(4)


def test_factor2_multiplier_and_ramp():
    snaps = rope_header.load_snapshots()
    two, four = snaps[2], snaps[4]
    assert two["attention_factor"] == 1 + 0.1 * math.log(2)
    # Channels below the ramp (j < 14) keep the plain frequency; above it (j >= 22) the frequency is divided by
    # the factor. Both factors share the plain part, and factor 2 halves the plain value exactly.
    assert two["inv_freq"][:14] == four["inv_freq"][:14]
    for j in range(22, 32):
        assert two["inv_freq"][j] == struct.unpack("<f", struct.pack("<f", four["inv_freq"][j] * 2))[0]


def header_values(text):
    """The float literals of the header, in order (C++ hex floats such as 0x1.356656p-1f)."""
    body = text.split("kYarnTable[kYarnTables][33] = {", 1)[1].split("};", 1)[0]
    words = [w.strip() for line in body.splitlines() for w in line.split("//")[0].split(",")]
    return [float.fromhex(w[:-1]) for w in words if w.startswith("0x")]


def test_render_layout():
    text = rope_header.render(rope_header.load_snapshots())
    assert "constexpr float kYarnTable[kYarnTables][33]" in text
    assert len(header_values(text)) == 2 * 33


def test_header_literals_are_the_float32_reference_bits():
    """Each hex float literal is exact: its float32 bits are the snapshot value rounded to float32 (the bits of the old
    uint32 table of 24b419a5, which rope_table_test.hip checked on the device)."""
    snaps = rope_header.load_snapshots()
    want = [rope_header.f32_bits(v) for f in sorted(snaps)
            for v in list(snaps[f]["inv_freq"]) + [snaps[f]["attention_factor"]]]
    got = header_values(rope_header.HEADER.read_text())
    assert [struct.unpack("<f", struct.pack("<f", v))[0] for v in got] == got  # float32 values, no rounding
    assert [rope_header.f32_bits(v) for v in got] == want
