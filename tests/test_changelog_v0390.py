"""v0.39.0 release notes: the "## v0.39.0 (2026-11-26)" section is in the house style of v0.38.1 (an intro, topic
sections). It holds the changes after v0.38.1: the HC mix default, the kernels that no longer spill registers, the free
CMA memory fix (host memory checks, doctor, setup and start), the OpenAI `stop_reason` field, the two stop and
state folder fixes, and the disk limit that covers the whole state folder. It leaves the Hy4 work out (no user can run it
yet) and has no team labels."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CHANGELOG.md").read_text()
HEAD = "## v0.39.0 (2026-11-26)"


def _section(head: str = HEAD) -> str:
    start = TEXT.index(f"{head}\n")
    return TEXT[start : TEXT.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _topic(head: str) -> str:
    return _flat(_section().split(f"### {head}\n", 1)[1].split("\n### ", 1)[0])


def test_v0390_is_the_newest_release_above_v0381() -> None:
    # An "## Unreleased" section for the changes after v0.39.0 can stand above the releases (trunk, after the merge).
    heads = [line for line in TEXT.splitlines() if line.startswith("## ")]
    releases = [line for line in heads if line != "## Unreleased"]
    assert heads[: len(heads) - len(releases)] in ([], ["## Unreleased"])
    assert releases[:3] == [HEAD, "## v0.38.1 (2026-11-23)", "## v0.38.0 (2026-11-19)"]


def test_the_section_has_an_intro_and_the_topic_sections() -> None:
    section = _section()
    assert section.startswith(f"{HEAD}\n\nRelease v0.39.0 ")
    assert re.findall(r"^### (.+)$", section, re.M) == ["Engine and build", "Host memory", "OpenAI API",
                                                       "Stop and the state folder"]


def test_the_intro_names_the_cma_fix_and_the_state_folder_limit() -> None:
    intro = _flat(_section().split("\n\n", 2)[1])
    for words in ("CMA memory", "state folder"):
        assert words in intro, words


def test_the_engine_lines_name_the_hc_mix_default_and_its_rollback() -> None:
    # The HC mix default (CRUCIBLE_HC_MIX_HI): Q1 KL to BF16 on the 32-chunk ruler moved by +0.00018 (4.05 pack).
    text = _topic("Engine and build")
    for words in ("fp16 high half of t", "+0.00018", "`CRUCIBLE_HC_MIX_HI=0`", "`CRUCIBLE_EXACT=1`"):
        assert words in text, words
    assert "CRUCIBLE_HC_MIX_HI" not in TEXT[TEXT.index("## v0.38.1 ") :]


def test_the_engine_lines_name_the_kernels_that_no_longer_spill() -> None:
    # The retune of Nov 23: 0 scratch in the full builds of both compilers (kernel register dumps); the kernel bench
    # outputs are bitwise equal to the old kernels. The HC dot kernel spilled only with ROCm 10.1; the QSA score kernel
    # changed only for ROCm 10.1 (its ROCm 7.2.4 code is the same).
    text = _topic("Engine and build")
    for words in ("dense GEMV", "two MoE stage kernels", "five draft GEMV kernels", "ROCm 7.2.4", "ROCm 10.1",
                  "HC dot", "QSA score", "output bytes do not change", "README"):
        assert words in text, words


def test_the_stop_lines_name_the_8_s_stop_and_the_ledger_tmp_file() -> None:
    text = _topic("Stop and the state folder")
    for words in ("8 s after Ctrl-C or SIGTERM", "exits 0", "90 s", "code 143", "`lost.ledger.tmp`", "full disk"):
        assert words in text, words


def test_no_hy4_and_no_team_labels_and_no_exact_output_headline() -> None:
    section = _section()
    assert not re.search(r"hy4|hunyuan", section, re.I)
    for word in ("review", "owner", "desk", "coordinator", "RED", "wip/", "Kiln", "train", "item", "night", "soak",
                 "scoreboard", "row 7", "lease", "gate", "retune"):
        assert not re.search(rf"\b{re.escape(word)}\b", section), word
    assert "\u2014" not in section
    for head in re.findall(r"^#+ .+$", section, re.M):
        assert "exact output" not in head.lower(), head


def test_each_bullet_is_one_change_of_a_few_lines_and_each_line_fits() -> None:
    bullets = re.split(r"\n(?=- )", _section())[1:]
    assert len(bullets) >= 4
    for bullet in bullets:
        body = bullet.split("\n\n", 1)[0]
        assert len(body.splitlines()) <= 4, body
    for line in _section().splitlines():
        assert len(line) <= 120, line


def test_the_host_memory_lines_name_the_cma_fix_of_issue_18() -> None:
    # free CMA pages (Kexec HandOver on Ubuntu 26.04, kernel 7.0) are in MemAvailable, but GPU memory cannot
    # use them. Every engine memory guard subtracts CmaFree; doctor, setup and start warn `memory-cma` with `kho=off`.
    text = _topic("Host memory")
    for words in ("`CmaFree`", "`MemAvailable`", "every memory check", "smaller layout",
                  "Ubuntu 26.04", "Kexec HandOver", "`kho=off`", "`memory-cma`", "`./crucible doctor`",
                  "`./crucible setup`", "`./crucible start`", "more than 1 GiB", "nothing changes"):
        assert words in text, words


def test_the_openai_line_names_stop_reason() -> None:
    text = _topic("OpenAI API")
    for words in ("`stop_reason`", "other engines", "`null`", "`finish_reason`", "`token_ids`"):
        assert words in text, words


def test_the_state_folder_lines_name_the_whole_folder_limit() -> None:
    # The state tier disk cap covers the whole state root: other identity folders that no live engine holds count and
    # leave first, oldest first; doctor shows their bytes.
    text = _topic("Stop and the state folder")
    for words in ("`--state-disk-gib`", "all the states in the state folder", "older builds or settings",
                  "no running engine uses", "oldest first", "its own oldest states", "`./crucible doctor`",
                  "fill the disk"):
        assert words in text, words
