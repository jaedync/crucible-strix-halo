"""Train 20: the crucible_turn line also carries numbers about the engine, the decode passes, the stall and the
token classes. Logging only: no command to the engine and no token changes.

Fields: engine_id (first 12 hex digits of the engine executable's sha256), passes, decoders_per_pass, rows_per_pass,
stall_s, think_tokens, tool_tokens, answer_tokens. Every value is a number or a hex identifier, never text.
"""
import hashlib
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from server import metrics
from server.api import validate_request
from server.tests.test_backend_paths import collect, request, start_backend  # noqa: F401 (fixture)

THINK_END, TOOL_START, TOOL_END = 248069, 248058, 248059
FIELDS = ("engine_id", "passes", "decoders_per_pass", "rows_per_pass", "stall_s", "think_tokens", "tool_tokens",
          "answer_tokens")


def turn_lines(lines):
    return [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_turn ")]


def chat_request(count, **options):
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": count, "spec": 0, **options}
    return validate_request(body, "chat", "crucible")


def classes(turn, ids):
    for token in ids: metrics.count_token(turn, token)
    return {key: turn.get(key, 0) for key in ("think_tokens", "tool_tokens", "answer_tokens")}


def test_thinking_tokens_end_at_the_end_tag():
    turn = dict(think_open=True)
    assert classes(turn, [5, 6, THINK_END, 7, 8, 9]) == dict(think_tokens=3, tool_tokens=0, answer_tokens=3)


def test_without_thinking_every_token_is_answer_or_tool():
    turn = dict(think_open=False)
    assert classes(turn, [1, TOOL_START, 2, 3, TOOL_END, 4, TOOL_START, 5, TOOL_END]) == \
        dict(think_tokens=0, tool_tokens=7, answer_tokens=2)


def test_an_unclosed_think_block_counts_all_tokens_as_thinking():
    assert classes(dict(think_open=True), [1, 2, 3])["think_tokens"] == 3


def test_a_turn_that_is_not_tracked_gets_no_class_fields():
    fields = metrics.load_fields((metrics.new_turn(1),))
    assert not set(FIELDS) & set(fields)


def test_load_fields_are_numbers_from_the_turns():
    first, cont = metrics.new_turn(1), metrics.new_turn(2)
    first.update(engine_id="abcdef012345", passes=4, pass_decoders=8, pass_rows=12, stall_s=0.5, think_tokens=6)
    cont.update(passes=2, pass_decoders=2, pass_rows=2, stall_s=0.25, tool_tokens=1, answer_tokens=3)
    fields = metrics.load_fields((first, cont))
    assert fields == dict(engine_id="abcdef012345", passes=6, decoders_per_pass=10 / 6, rows_per_pass=14 / 6,
                          stall_s=0.75, think_tokens=6, tool_tokens=1, answer_tokens=3)


def test_the_stage_reply_of_the_first_job_is_a_turn_field():
    """An earlier change: the turn line names its last stage reply, so a log check links each turn to its disk read."""
    first, cont = metrics.new_turn(1), metrics.new_turn(2)
    first.update(state_disk="ready", state_disk_wait_s=1.5, state_disk_reason="start", state_disk_polls=3,
                 state_disk_tokens=110592, state_disk_tag=4, state_disk_job=9)
    cont.update(state_disk="none", state_disk_reason="floor", state_disk_polls=1, state_disk_job=10)
    fields = metrics.load_fields((first, cont))
    # Fix 2 (cause.md item 10): state_disk_job is the tag on the engine's stage and host_evict lines.
    assert {key: fields[key] for key in fields if key.startswith("state_disk")} == dict(
        state_disk="ready", state_disk_wait_s=1.5, state_disk_reason="start", state_disk_polls=3,
        state_disk_tokens=110592, state_disk_tag=4, state_disk_job=9)
    # Only fixed words: a bad state or reason is left out.
    bad = metrics.new_turn(1)
    bad.update(state_disk="none", state_disk_reason="Some Text")
    assert metrics.load_fields((bad,)) == dict(state_disk="none")
    assert "state_disk" not in metrics.load_fields((dict(metrics.new_turn(1), state_disk="other"),))


def test_engine_id_must_be_a_short_hex_string():
    turn = metrics.new_turn(1)
    turn["engine_id"] = "SECRET text with spaces"
    assert "engine_id" not in metrics.load_fields((turn,))


def test_engine_id_is_the_sha256_prefix_of_the_executable(tmp_path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"synthetic engine bytes")
    assert metrics.file_id(binary) == hashlib.sha256(b"synthetic engine bytes").hexdigest()[:12]
    assert metrics.file_id(tmp_path / "missing") is None


def test_solo_spec_turn_reports_passes_and_rows(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128"})
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        collect(backend.submit([1, 2, 3], request(10, spec=7)))
    (turn,) = turn_lines(lines)
    # Three solo speculative steps: one decoder each, 1 + the draft depth bound of 7 rows each.
    assert (turn["passes"], turn["decoders_per_pass"], turn["rows_per_pass"]) == (3, 1.0, 8.0)
    assert turn["stall_s"] == 0.0
    assert "engine_id" not in turn  # the executable of this fixture is not a file


def test_packed_passes_count_every_decoder(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_DELAY": ".002"})
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        jobs = [backend.submit([1, 2, 3], request(9, spec=7)), backend.submit([4, 5, 6], request(9, spec=7))]
        for job in jobs: collect(job)
    packed = [e for e in backend.trace.since(0) if e["op"] == "verify_step" and len(e["request_ids"]) == 2]
    assert packed
    for turn in turn_lines(lines):
        assert turn["passes"] == turn["spec_rounds"]
        assert 1.0 < turn["decoders_per_pass"] <= 2.0
        assert turn["rows_per_pass"] <= 8.0  # the pass row limit


def test_prefill_of_another_session_stalls_a_decoder(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_PREFILL_ROWS": "16",
                                                  "FAKE_DELAY": ".002"})
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        decoder = backend.submit([1], request(60))
        assert decoder.events.get(timeout=3)["event"] == "token"
        long = backend.submit(list(range(48)), request(2))
        collect(decoder, prefix=[0]); collect(long)
    chunks = [e for e in backend.trace.since(0) if e["op"] == "prefill" and e["request_ids"] == [long.turn["request_id"]]]
    expected = sum(e["t_end"] - e["t_start"] for e in chunks)
    by_id = {t["request_id"]: t for t in turn_lines(lines)}
    assert by_id[decoder.turn["request_id"]]["stall_s"] == pytest.approx(expected, abs=1e-3) and expected > 0
    assert by_id[long.turn["request_id"]]["stall_s"] == 0.0


def test_token_classes_reach_the_line_through_the_backend(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128"})
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        # The fake engine counts up from the prompt's last ID: 248068 <think>, 248069 </think>, then 248070 and on.
        collect(backend.submit([248067], chat_request(6, enable_thinking=True)))
        collect(backend.submit([248056], chat_request(5, enable_thinking=False)))
    thinking, tools = turn_lines(lines)
    assert (thinking["think_tokens"], thinking["tool_tokens"], thinking["answer_tokens"]) == (2, 0, 4)
    # Tokens 248057, 248058 (tool start), 248059 (tool end), 248060, 248061.
    assert (tools["think_tokens"], tools["tool_tokens"], tools["answer_tokens"]) == (0, 2, 3)
    for line in (thinking, tools):
        assert line["think_tokens"] + line["tool_tokens"] + line["answer_tokens"] == line["completion_tokens"]


def test_closed_think_turn_sums_both_jobs():
    first, cont = metrics.new_turn(1), metrics.new_turn(2)
    first.update(t_first_token=10.0, passes=50, pass_decoders=50, pass_rows=50, think_tokens=151)
    cont.update(t_first_token=20.5, t_done=21.0, passes=12, pass_decoders=24, pass_rows=24, tool_tokens=4,
                answer_tokens=32)
    line = metrics.request_line(first, dict(prompt_tokens=100, completion_tokens=212, finish_reason="stop"),
                                SimpleNamespace(tokens=150, by="answer_room"), cont,
                                dict(prompt_tokens=276, cached_tokens=100, completion_tokens=36))
    fields = json.loads(line.split(" ", 1)[1])
    assert fields["passes"] == 62 and fields["decoders_per_pass"] == pytest.approx(74 / 62, abs=1e-4)
    # 150 generated tokens before the close plus 26 forced ones are inside the think block.
    assert (fields["think_tokens"], fields["tool_tokens"], fields["answer_tokens"]) == (176, 4, 32)


def run_scenario(start_backend, tmp_path, name, track):
    log = tmp_path / name
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_COMMAND_LOG": str(log)})
    patches = [] if track else [mock.patch.object(metrics, "count_token"), mock.patch.object(metrics, "load_fields")]
    with mock.patch.object(metrics, "emit", lambda line: None):
        for patch in patches: patch.start()
        try:
            ids = [collect(backend.submit([248067], chat_request(12, spec=7, enable_thinking=True)))[0]]
            ids.append(collect(backend.submit([5, 6], request(9, spec=0)))[0])
        finally:
            for patch in patches: patch.stop()
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    for command in commands: command.get("request", {}).pop("seed", None)  # random per request
    return ids, commands


def test_the_counters_change_no_engine_command_and_no_token(start_backend, tmp_path):
    with_counters = run_scenario(start_backend, tmp_path, "a.log", True)
    without = run_scenario(start_backend, tmp_path, "b.log", False)
    assert with_counters == without
    assert with_counters[1]  # the commands were logged


def pass_rows(op, specs, packed_rows="", streams=8):
    """The rows that _count_rounds records for one pass over decoders with these draft depths."""
    from server.batch_backend import BatchedEngineBackend
    worker = BatchedEngineBackend.__new__(BatchedEngineBackend)
    worker.packed_rows, worker.streams = packed_rows, streams
    turns = {id: {} for id, _ in enumerate(specs)}
    worker.active = {id: SimpleNamespace(job=SimpleNamespace(turn=turns[id], payload=dict(spec=spec, max_tokens=64)))
                     for id, spec in enumerate(specs)}
    worker._count_rounds({id: [1] for id in turns}, op)
    return {turn["pass_rows"] for turn in turns.values()}, {turn["pass_decoders"] for turn in turns.values()}


def test_a_packed_pass_counts_the_rows_of_every_speculative_decoder():
    assert pass_rows("verify_step", [2, 0]) == ({4}, {2})  # 3 + 1
    assert pass_rows("verify_step", [3, 3, 0]) == ({8}, {3})  # 4 + 4 + 1 is capped at the 8 rows of a pass
    assert pass_rows("verify_step", [0, 0, 0]) == ({3}, {3})
    assert pass_rows("step", [7, 7]) == ({2}, {2})  # a plain step has one row per decoder
    assert pass_rows("spec_step", [7]) == ({8}, {1})


def test_a_decoder_with_one_token_left_is_not_speculative():
    from server.batch_backend import BatchedEngineBackend
    worker = BatchedEngineBackend.__new__(BatchedEngineBackend)
    entries = [SimpleNamespace(job=SimpleNamespace(turn={}, payload=dict(spec=5, max_tokens=max_tokens)))
               for max_tokens in (1, 9)]
    worker.active = dict(enumerate(entries))
    worker._count_rounds({0: [1], 1: [2]}, "verify_step")
    assert entries[0].job.turn["pass_rows"] == 1 + 6


def test_classes_are_left_out_when_they_do_not_add_up_to_the_completion_tokens():
    # A stop string ends the turn before the engine's done event: the engine thread counted more tokens (33) than the
    # client got (3).
    turn = metrics.new_turn(1)
    turn.update(think_tokens=33)
    for tokens, expected in ((3, False), (33, True)):
        fields = json.loads(metrics.request_line(turn, dict(prompt_tokens=5, completion_tokens=tokens,
                                                          finish_reason="stop")).split(" ", 1)[1])
        assert ("think_tokens" in fields) is expected


def test_a_closed_turn_without_a_continuation_count_has_no_classes():
    first, cont = metrics.new_turn(1), metrics.new_turn(2)
    first.update(think_tokens=30)
    cont.update(answer_tokens=40)  # the continuation stopped on a stop string: its done event never came
    line = metrics.request_line(first, dict(prompt_tokens=5, completion_tokens=9, finish_reason="stop"),
                                SimpleNamespace(tokens=4, by="answer_room"), cont, dict(finish_reason="stop"))
    fields = json.loads(line.split(" ", 1)[1])
    assert not {"think_tokens", "tool_tokens", "answer_tokens"} & set(fields)


def test_a_failing_counter_never_fails_the_request():
    assert metrics.count_token(None, 5) is None  # no turn dict
    assert metrics.count_token(dict(think_tokens="x", think_open=True), 5) is None
    assert metrics.count_pass(dict(passes="x"), 1, 1) is None
    assert metrics.load_fields((dict(passes=3, pass_decoders="x"), None)) == {}


def test_a_payload_without_spec_still_counts_one_row_each():
    from server.batch_backend import BatchedEngineBackend
    worker = BatchedEngineBackend.__new__(BatchedEngineBackend)
    entry = SimpleNamespace(job=SimpleNamespace(turn={}, payload={}))
    worker.active = {0: entry}
    worker._count_rounds({0: [1]}, "verify_step")
    assert entry.job.turn["pass_rows"] == 1


def test_the_default_pass_is_16_rows_from_4_decoders():
    # CRUCIBLE_PACKED_ROWS unset: engine/runtime/packed_rows_option.h, 16 rows from 4 decoders, else 8.
    assert pass_rows("verify_step", [7, 7, 7, 7]) == ({16}, {4})
    assert pass_rows("verify_step", [7, 7, 7]) == ({8}, {3})
    assert pass_rows("verify_step", [0, 0, 0, 0]) == ({4}, {4})
    assert pass_rows("verify_step", [7, 7, 7, 7], streams=3) == ({8}, {4})  # fewer than 4 slots: no wide pass


def test_a_set_pass_row_cap_is_fixed_for_any_decoder_count():
    assert pass_rows("verify_step", [7, 7], packed_rows="12") == ({12}, {2})
    assert pass_rows("verify_step", [7, 7, 7, 7], packed_rows="8") == ({8}, {4})
    assert pass_rows("verify_step", [7, 7, 7, 7], packed_rows="bad") == ({8}, {4})  # the engine refuses to start
    assert pass_rows("verify_step", [7, 7, 7, 7], packed_rows="99") == ({8}, {4})
