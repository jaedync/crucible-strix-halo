"""v0.38.1 release notes: the "## v0.38.1 (2026-11-23)" section is in the house style of v0.38.0 (an intro, topic
sections), names each user-facing change of the patch release with its measured number and method, leaves the Hy4 work
out (no user can run it yet), and has no team labels."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CHANGELOG.md").read_text()
HEAD = "## v0.38.1 (2026-11-23)"


def _section() -> str:
    start = TEXT.index(f"{HEAD}\n")
    return TEXT[start : TEXT.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _topic(head: str) -> str:
    return _flat(_section().split(f"### {head}\n", 1)[1].split("\n### ", 1)[0])


def test_v0381_is_the_newest_release_above_v0380() -> None:
    heads = [line for line in TEXT.splitlines() if line.startswith("## ")]
    assert heads[:3] == [HEAD, "## v0.38.0 (2026-11-19)", "## v0.37.3 (2026-11-17)"]
    assert "## Unreleased" not in heads


def test_the_section_has_an_intro_and_the_topic_sections() -> None:
    section = _section()
    assert section.startswith(f"{HEAD}\n\nRelease v0.38.1 ")
    assert re.findall(r"^### (.+)$", section, re.M) == [
        "Sessions and the conversation cache", "Tool calls", "Images", "Engine", "Stop and the state folder"]


def test_the_release_table_line_names_the_words_the_paths_and_the_measured_row() -> None:
    # tools/crucible_cli/measured.py ENV; the A/B run of Nov 19 (v0.38.0 engine, 4.05 pack, 4 ABBA pairs): last first
    # token of 4 sessions on one new prompt 79.48 -> 17.39 s (32K), 228.85 -> 54.92 s (100K).
    text = _topic("Sessions and the conversation cache")
    for words in ("`CRUCIBLE_EXACT_RESUME=1`", "`CRUCIBLE_TWIN=1`", "`./crucible start`", "the user service",
                  "docs/DOCKER.md", "79.5 s to 17.4 s", "228.9 s to 54.9 s", "4.05 pack", "v0.38.0 engine",
                  "4 pairs", "`CRUCIBLE_TWIN=0`"):
        assert words in text, words


def test_the_cache_lines_name_the_think_cap_and_the_bypass_fixes() -> None:
    text = _topic("Sessions and the conversation cache")
    for words in ("think cap", "`thinking_budget`", "`\"cache_control\": \"bypass\"`"):
        assert words in text, words


def test_the_cache_lines_name_the_queue_fix_of_cancelled_waiting_requests() -> None:
    # server/batch_backend.py _sweep_cancelled: a cancelled waiting turn
    # gives back its admission tokens at once; before, new work got HTTP 503 while the head turn read its state (60 to
    # 180 s).
    text = _topic("Sessions and the conversation cache")
    for words in ("cancels while it waits", "HTTP 503", "180 s"):
        assert words in text, words


def test_the_tool_call_lines_name_the_reasoning_rule_and_its_switch() -> None:
    # server/README.md: calls in the reasoning before an end of turn are made; the count field; the
    # end-of-turn guard switch turns it off.
    text = _topic("Tool calls")
    for words in ("`reasoning_content`", "`tool_calls`", "`usage.completion_tokens_details.tool_call_from_reasoning`",
                  "`--end-of-turn-guard off`"):
        assert words in text, words


def test_the_tool_call_lines_name_the_three_sibling_fixes() -> None:
    # server/README.md: a call in the reasoning, then the end tag and a white-space answer, is made
    # (F1); only the end-of-thinking token ends the reasoning (F3); text that names the call tag before a real call no
    # longer loses that call (F2).
    text = _topic("Tool calls")
    for words in ("an answer of only white space", "end-of-thinking token", "plain text", "think cap",
                  "names `<tool_call>`", "`/v1/responses`"):
        assert words in text, words


def test_the_engine_lines_name_the_prefill_changes_and_the_pinned_memory() -> None:
    text = _topic("Engine")
    for words in ("`--ngram16 yes`", "160 MiB", "20 MiB", "The speed table in the README gives the measured numbers"):
        assert words in text, words


def test_the_stop_lines_name_sigint_and_the_link_rule() -> None:
    text = _topic("Stop and the state folder")
    for words in ("Ctrl-C", "SIGINT like SIGTERM", "60 s at most (was 75 s)", "one name at a time",
                  "never follow a link"):
        assert words in text, words


def test_the_stop_and_state_folder_lines_left_v0380() -> None:
    start = TEXT.index("## v0.38.0 (2026-11-19)\n")
    v0380 = _flat(TEXT[start : TEXT.index("\n## ", start + 1)])
    assert "Ctrl-C" not in v0380 and "one name at a time" not in v0380


def test_no_hy4_and_no_team_labels_and_no_exact_output_headline() -> None:
    section = _section()
    assert not re.search(r"hy4|hunyuan", section, re.I)
    for word in ("review", "owner", "desk", "coordinator", "RED", "wip/", "Kiln", "train", "item", "night", "soak",
                 "scoreboard", "row 7", "lease", "gate"):
        assert not re.search(rf"\b{re.escape(word)}\b", section), word
    assert "\u2014" not in section
    for head in re.findall(r"^#+ .+$", section, re.M):
        assert "exact output" not in head.lower(), head


def test_each_bullet_is_one_change_of_a_few_lines_and_each_line_fits() -> None:
    bullets = re.split(r"\n(?=- )", _section())[1:]
    assert len(bullets) >= 8
    for bullet in bullets:
        body = bullet.split("\n\n", 1)[0]
        assert len(body.splitlines()) <= 4, body
    for line in _section().splitlines():
        assert len(line) <= 120, line


def test_no_line_is_only_a_reference_in_parentheses() -> None:
    # The export cuts a parenthesized reference to a private page. A line that held only such a reference became a lone
    # "." in the public CHANGELOG of v0.38.0 (line 142). Each reference now follows a word on its line.
    for line in TEXT.splitlines():
        assert not re.fullmatch(r"\s*\([\w./-]+\)[.,;:]?", line), line


def _v0380() -> str:
    start = TEXT.index("## v0.38.0 (2026-11-19)\n")
    return _flat(TEXT[start : TEXT.index("\n## ", start + 1)])


def test_the_engine_lines_name_the_start_reserve_flag_of_pr16_once_with_the_thanks() -> None:
    # Public an earlier change by enncoded: --admission-reserve-mib sets the start
    # reserve, default 9216 MiB unchanged. Trunk put the bullet under v0.38.0, which shipped without it.
    text = _topic("Engine")
    for words in ("`--admission-reserve-mib N`", "9,216 MiB", "`--host-reserve-mib`", "thanks @enncoded"):
        assert words in text, words
    assert TEXT.count("thanks @enncoded") == 1
    assert "--admission-reserve-mib" not in _v0380()


def test_the_cache_lines_name_the_shared_system_prompt_rule() -> None:
    # The system prompt prefix that most sessions extend
    # outlasts other cache states (kSharedFloor); a waiting resume point of another system prompt with no disk copy
    # stays (kWaitingAlone). GPU load proof load-proof-7c3bf0dd: 0 full misses.
    text = _topic("Sessions and the conversation cache")
    for words in ("system prompt", "full prefill", "other system prompts"):
        assert words in text, words


def test_the_image_lines_name_the_shm_fallback_and_left_v0380() -> None:
    # server/pixel_files.py: pixel files use /dev/shm only when the request fits, else
    # the temp folder; HTTP 503 only when both are full (was HTTP 500 on a 64 MB container /dev/shm). Trunk put the
    # bullet under v0.38.0, which shipped without it.
    text = _topic("Images")
    for words in ("`/dev/shm`", "64 MB", "temp folder", "HTTP 500", "HTTP 503", "the tokens do not change"):
        assert words in text, words
    assert "/dev/shm" not in _v0380()
    assert TEXT.count("An image request whose pixels do not fit") == 1


def test_the_engine_lines_name_the_new_method_of_the_same_prompt_again_row() -> None:
    # The README cell "same 32K prompt
    # again" waits 0.25 s after the cold prompt and takes the median of 6 re-sends 1 s apart. About 0.017 s before, about
    # 0.009 s now: the method changed, not the engine.
    text = _topic("Engine")
    for words in ('"Same 32K prompt again: first token"', "0.25 s", "median of 6 re-sends", "1 s apart", "0.009 s",
                  "0.017 s", "not the engine", "tools/repro/README.md"):
        assert words in text, words
