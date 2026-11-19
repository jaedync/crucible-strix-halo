"""v0.37.3 release notes: the CHANGELOG section is in the house style of v0.37.2 (an intro, then
topic sections and the known limits), user facing, with no team labels; the docs it names say the same."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CHANGELOG.md").read_text()


def _section(tag: str) -> str:
    start = TEXT.index(f"## {tag} ")
    return TEXT[start : TEXT.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_v0373_section_has_an_intro_and_the_topic_sections() -> None:
    section = _section("v0.37.3")
    assert section.startswith("## v0.37.3 (2026-11-17)\n\nRelease v0.37.3 ")
    heads = re.findall(r"^### (.+)$", section, re.M)
    assert heads == ["Network access and the API key", "Start, stop and memory", "Install and update", "Setup and build",
                     "Known limits"]


def test_the_v0373_section_has_no_team_labels() -> None:
    section = _section("v0.37.3")
    for word in ("r373", "secrev", "review M", "review L", "owner", "desk", "coordinator", "RED", "wip/", "unreleased"):
        assert word not in section, word
    assert "\u2014" not in section


def test_the_v0373_section_keeps_the_key_rule_and_the_facts_of_the_release() -> None:
    text = _flat(_section("v0.37.3"))
    for words in ("A key is required for a public address, except in the Docker image.",
                  "The Docker image on `0.0.0.0` or `::` still has no default Host list, as in v0.37.2.",
                  "exits with code 3", "exits with code 4", "`--tag vX.Y.Z`", "`--version vX.Y.Z`", "`CRUCIBLE_TAG`",
                  "sends no API key", "`./crucible setup --api-key`", "`./crucible setup --no-api-key`",
                  "The engine kernels are the kernels of v0.37.2."):
        assert words in text, words


def test_the_v0373_section_names_no_readme_text_that_the_public_readme_lacks() -> None:
    # The public README is a template of the publish tool, not this README.md: the notes point to pages that ship.
    assert "The README explains" not in _section("v0.37.3")


def test_the_docker_guide_keeps_its_run_notes_in_one_list() -> None:
    lines = (ROOT / "docs/DOCKER.md").read_text().splitlines()
    host = next(i for i, line in enumerate(lines) if line.startswith("- `--host 0.0.0.0` is necessary in the container"))
    assert lines[host + 1].startswith("- `-p 127.0.0.1:8790:8790` publishes the port")


def test_the_runtime_rule_names_session_replay_as_its_exception() -> None:
    text = _flat((ROOT / "engine/runtime/README.md").read_text())
    assert "Session replay is the exception (docs/SESSION_REPLAY.md)." in text
