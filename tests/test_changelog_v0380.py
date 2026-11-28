"""v0.38.0 release notes: the "## v0.38.0 (2026-11-19)" section is in the house style of v0.37.3 (an intro,
topic sections, the known limits), names each user-facing change of the release, credits the public contributions,
and has no team labels."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CHANGELOG.md").read_text()
HEAD = "## v0.38.0 (2026-11-19)"


def _section() -> str:
    start = TEXT.index(f"{HEAD}\n")
    return TEXT[start : TEXT.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_v0380_has_its_date_below_v0381() -> None:
    # An "## Unreleased" section for the changes after v0.38.1 can stand above the releases (trunk, after the merge).
    heads = [line for line in TEXT.splitlines() if line.startswith("## ")]
    releases = [line for line in heads if line != "## Unreleased"]
    assert heads[: len(heads) - len(releases)] in ([], ["## Unreleased"])
    i = releases.index("## v0.38.1 (2026-11-23)")
    assert releases[:i] in ([], ["## v0.39.0 (2026-11-26)"], ["## v0.40.0 (2026-11-28)", "## v0.39.0 (2026-11-26)"])
    assert releases[i : i + 3] == ["## v0.38.1 (2026-11-23)", HEAD, "## v0.37.3 (2026-11-17)"]


def test_the_section_has_an_intro_and_the_topic_sections() -> None:
    section = _section()
    assert section.startswith(f"{HEAD}\n\nRelease v0.38.0 ")
    heads = re.findall(r"^### (.+)$", section, re.M)
    assert heads == ["New APIs", "Structured output", "Tool calls and request fields", "Engine and build",
                     "Setup and install", "Start, stop and memory", "Known limits"]


def test_the_section_names_the_new_api_and_structured_output_changes() -> None:
    text = _flat(_section())
    for words in ("`POST /v1/responses`", "`POST /v1/messages`", "`text.format`", "`output_config.format`",
                  "`response_format`", "`strict: true`", "`constraint_keyword`", "`<invoke>`", "`--lenient-fields off`",
                  "`CRUCIBLE_LENIENT_FIELDS=0`", "`timings`", "`./crucible service disable`", "`CRUCIBLE_HC_WMMA=1`",
                  "--ngram16", "--abliterated", "`--state-write-gib-day N`", "`--download-anyway`",
                  "`--no-disk-check`"):
        assert words in text, words


def test_the_section_credits_the_public_contributions() -> None:
    text = _flat(_section())
    for credit in ("(thanks @mikenavi)", "(thanks @TheBeaninator)", "(thanks @dineshr93)"):
        assert credit in text, credit
    assert not re.search(r"\(#\d", text), "one form for the credits: (PR #N, thanks @name)"


def test_a_spill_claim_names_its_kernels() -> None:
    # The ROCm 10.1 compiler still gives some kernels more scratch than the 7.2.4 compiler, so each
    # "no longer spill" claim names its kernels, never a whole build.
    claims = re.findall(r"((?:[^.:;,]|\.(?=\d))*)no longer spills?\b", _flat(_section()))  # 10.1 is one word
    assert claims
    for clause in claims:
        assert "kernel" in clause, clause


def test_the_responses_line_names_the_strict_default() -> None:
    # server/responses_strict.py: a Responses function tool without strict gets the strict grammar when its schema
    # allows it; Chat Completions keep no grammar for it, so the tokens can differ from the equal chat request.
    assert (ROOT / "server/responses_strict.py").is_file()
    text = _flat(_section())
    assert "same prompt and tokens" not in text
    assert "a function tool without `strict` (or with `null`) gets the strict call grammar" in text
    assert "`strict: false`" in text


def test_the_known_limit_gives_the_two_measurements() -> None:
    limits = _flat(_section().split("### Known limits", 1)[1])
    assert "about 1.5% slower at a 32K prefill" in limits
    assert "(two measurements)" in limits
    assert "In v0.37.1 to v0.37.3 it was about 42% slower." in limits


def test_the_lenient_line_names_the_default_and_each_start_path_of_the_strict_mode() -> None:
    # server/lenient.py DEFAULT = True: the mode is on with no flag and no (or an empty) variable; "--lenient-fields off"
    # or CRUCIBLE_LENIENT_FIELDS=0 is the strict mode. The launcher passes no lenient setting
    # (tools/crucible_cli/launch.py server_command); ./crucible start without the service passes the caller's
    # environment (launch.process_env), and the user service needs a drop-in.
    text = _flat(_section())
    for words in ("Lenient request fields are on by default", "`--lenient-fields off` (a server flag)",
                  "`CRUCIBLE_LENIENT_FIELDS=0 ./crucible start`", "`systemctl --user edit crucible-server.service`",
                  "`Environment=CRUCIBLE_LENIENT_FIELDS=0`", "`./crucible service restart`"):
        assert words in text, words
    for words in ("off by default", "`CRUCIBLE_LENIENT_FIELDS=1`", "With `--lenient-fields on`"):
        assert words not in text, words


def test_the_section_has_no_team_labels_and_no_exact_output_headline() -> None:
    section = _section()
    for word in ("r373", "secrev", "review", "owner", "desk", "coordinator", "RED", "wip/", "Kiln",
                 "train", "stage 1", "item", "night", "soak"):
        assert not re.search(rf"\b{re.escape(word)}\b", section), word
    assert "\u2014" not in section
    for head in re.findall(r"^#+ .+$", section, re.M):
        assert "exact output" not in head.lower(), head


def test_each_bullet_is_one_change_of_a_few_lines() -> None:
    bullets = re.split(r"\n(?=- )", _section())[1:]
    assert len(bullets) >= 20
    for bullet in bullets:
        body = bullet.split("\n\n", 1)[0]
        assert len(body.splitlines()) <= 4, body


def test_the_section_names_the_tools_that_the_public_tree_drops() -> None:
    # The v0.38.0 export manifest drops these v0.37.3 public files: each one needed a library that the public tree
    # does not have, so it stopped at start. The line gives each file name and its folder, not a full path: the export
    # check counts a full path of a dropped file as a dead reference.
    text = _flat(_section())
    tools = ("ab_order.py", "hc8_speed.py", "ids_compare.py", "lease_holder_check.py", "ple_abab.py", "tail_cost.py")
    assert "six tools in `tools/`: " + ", ".join(f"`{name}`" for name in tools[:-1]) + f" and `{tools[-1]}`" in text
    assert "`abab.sh` and `abab_report.py` in `training/mtp_tune/`" in text
    assert "/ab_order.py" not in text and "mtp_tune/abab" not in text


def test_the_tool_tail_line_names_the_switch_and_the_wait() -> None:
    # server/tool_tail.py: after a tool-call turn the server prefills the reply (--extend-tool-tails, default 1 with
    # session replay), and a next request waits for it at most WAIT_SECONDS (120 s) with no byte to a stream.
    text = _flat(_section().split("### Tool calls and request fields", 1)[1].split("\n### ", 1)[0])
    for words in ("`--extend-tool-tails`", "default 1 with session replay", "at most 120 s",
                  "a stream gets no byte", "`--extend-tool-tails 0`"):
        assert words in text, words


def test_the_penalty_line_gives_the_fields_and_the_ranges() -> None:
    # server/README.md (merge-pen): the engine applies the three penalties; v0.37.3 refused a value that was not
    # neutral; repeat_last_n (a lenient field) must be null with a penalty, because the penalties have no window.
    text = _flat(_section().split("### Tool calls and request fields", 1)[1].split("\n### ", 1)[0])
    for words in ("`presence_penalty` and `frequency_penalty` (from `-2` to `2`)",
                  "`repetition_penalty` (from `0.01` to `2`)", "v0.37.3 refused", "`repeat_last_n` must be null"):
        assert words in text, words


def _topic(head: str) -> str:
    return _flat(_section().split(f"### {head}\n", 1)[1].split("\n### ", 1)[0])


def test_the_setup_lines_name_the_418_pack_and_the_memory_classes() -> None:
    # tools/crucible_cli/packs.py "4.18" (122 GB download, 125 GB disk check); machine.py fit(): from 88 GiB the 96 GB
    # class (3.05 pack, 2 sessions of 262,144 tokens, a 4,096 MiB cache), below 88 GiB a refusal; install.sh fails
    # the 4.05 and 4.18 packs below 120 GiB.
    text = _topic("Setup and install")
    for words in ("`./crucible setup --pack 4.18`", "`4.18bpw_h6_ng8`", "122 GB download", "125 GB",
                  "A 96 GB machine (88 GiB to 120 GiB) gets the 3.05 pack, 2 sessions of 262,144 tokens",
                  "4,096 MiB conversation cache", "below 88 GiB"):
        assert words in text, words


def test_the_memory_lines_name_the_start_fit_and_the_new_limits() -> None:
    # engine/runtime/batch_admission.h FitServeLayout, server/__main__.py --fit (auto, off), server/http.py health
    # counts; doctor.py: a machine below 88 GiB, or the 4.05 or 4.18 pack below 120 GiB, stops a start.
    text = _topic("Start, stop and memory")
    for words in ("`--fit off`", "fewer sessions", "a shorter `--ctx`", '`{"status":"ok","sessions":N,"context":C}`',
                  "a machine below 88 GiB, the 4.05 or 4.18 pack below 120 GiB"):
        assert words in text, words
    assert "a machine below 120 GiB, and" not in text
