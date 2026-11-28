"""v0.40.0 release notes: the "## v0.40.0 (2026-11-28)" section is in the house style of v0.39.0 (an intro, topic
sections). It holds the user-visible changes after v0.39.0: the MoE kernels of bulk prefill decode the weights in a
scheduled order (faster prefill, the same output bytes), and the README numbers of this release, with the 32K chart
point after one warm-up run (the v0.39.0 point was a cold first run). The v0.39.0 section stays equal to its release
notes. It leaves the Hy4 work and the host tools out and has no team labels."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CHANGELOG.md").read_text()
HEAD = "## v0.40.0 (2026-11-28)"
# sha256 of the v0.39.0 section (from "## v0.39.0 " to the next "## ") in the CHANGELOG of the public v0.39.0 release.
V0390_SHA256 = "65d1d1b2dbd6b3342f7fdae47fc8d94adeb0f2d3d08c06da7b86d92467d8a134"


def _section(head: str = HEAD) -> str:
    start = TEXT.index(f"{head}\n")
    return TEXT[start : TEXT.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _topic(head: str) -> str:
    return _flat(_section().split(f"### {head}\n", 1)[1].split("\n### ", 1)[0])


def test_v0400_is_the_newest_release_above_v0390() -> None:
    # An "## Unreleased" section for the changes after v0.40.0 can stand above the releases (trunk, after the merge).
    heads = [line for line in TEXT.splitlines() if line.startswith("## ")]
    releases = [line for line in heads if line != "## Unreleased"]
    assert heads[: len(heads) - len(releases)] in ([], ["## Unreleased"])
    assert releases[:3] == [HEAD, "## v0.39.0 (2026-11-26)", "## v0.38.1 (2026-11-23)"]


def test_the_v0390_section_equals_the_v0390_release_notes() -> None:
    start = TEXT.index("## v0.39.0 ")
    section = TEXT[start : TEXT.index("\n## ", start + 1)]
    assert hashlib.sha256(section.encode()).hexdigest() == V0390_SHA256


def test_the_section_has_an_intro_and_the_topic_sections() -> None:
    section = _section()
    assert section.startswith(f"{HEAD}\n\nRelease v0.40.0 ")
    assert re.findall(r"^### (.+)$", section, re.M) == ["Engine and build", "README"]


def test_the_intro_names_the_faster_prefill_and_the_same_bytes() -> None:
    intro = _flat(_section().split("\n\n", 2)[1])
    for words in ("bulk prefill", "faster", "output bytes do not change"):
        assert words in intro, words


def test_the_engine_line_names_the_scheduled_weight_decode() -> None:
    # kTileDecodeSched = true in engine/layers/prefill_moe_tile.h: the PcFatPair and TileDown MoE kernels of bulk
    # prefill; the tokens equal the old build on both packs, the kernel bench outputs are bitwise equal. Decode (token
    # by token) does not use these kernels.
    text = _topic("Engine and build")
    for words in ("MoE kernels of bulk prefill", "decode the weights", "output bytes do not change",
                  "Decode speed does not change"):
        assert words in text, words


def test_the_readme_lines_name_the_numbers_and_the_warm_32k_point() -> None:
    # The chart tool runs one discarded 32K warm-up unit per session before the first counted cell; the
    # v0.39.0 32K chart point was the cold first unit of its session.
    text = _topic("README")
    for words in ("speed table and the charts", "measured numbers of this release", "32K point", "warm-up run",
                  "does not count", "v0.39.0", "cold"):
        assert words in text, words


def test_no_hy4_and_no_team_labels_and_no_exact_output_headline() -> None:
    section = _section()
    assert not re.search(r"hy4|hunyuan", section, re.I)
    for word in ("review", "owner", "desk", "coordinator", "RED", "wip/", "Kiln", "train", "item", "night", "soak",
                 "scoreboard", "lease", "gate", "gpu-run", "warden", "kTileDecodeSched", "P1", "ABAB"):
        assert not re.search(rf"\b{re.escape(word)}\b", section), word
    assert "\u2014" not in section
    for head in re.findall(r"^#+ .+$", section, re.M):
        assert "exact output" not in head.lower(), head


def test_each_bullet_is_one_change_of_a_few_lines_and_each_line_fits() -> None:
    bullets = re.split(r"\n(?=- )", _section())[1:]
    assert len(bullets) >= 2
    for bullet in bullets:
        body = bullet.split("\n\n", 1)[0]
        assert len(body.splitlines()) <= 4, body
    for line in _section().splitlines():
        assert len(line) <= 120, line
