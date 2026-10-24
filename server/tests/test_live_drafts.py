"""crucible_decode: the live draft counters of a decoding speculative turn (live drafts for a log reader).

The engine puts a speculative request's cumulative counters (draft_proposed, draft_accepted, spec_steps) on the first
token event of each speculative step. The server writes at most one crucible_decode line a second for each such turn,
and one final line with the values of its crucible_turn line. A turn without speculation writes none.
"""
import json
import time
from types import SimpleNamespace
from unittest import mock

from server import metrics
from server.tests.test_backend_paths import collect, request, start_backend, wait_for  # noqa: F401 (fixture)
from server.tests.test_think_budget import ANSWER, CLOSE, IM_END, REASON, LinkedBackend, chat, run

KEYS = ["request_id", "completion_tokens", "spec_rounds", "draft_proposed", "draft_accepted", "elapsed_s"]
LIVE = {"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_LIVE_DRAFTS": "1"}


def parsed(lines, kind):
    return [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith(kind + " ")]


def decode_values(fields):
    return {key: fields[key] for key in KEYS if key in fields} | {"elapsed_s": fields.get("elapsed_s", fields.get("decode_s"))}


def job(turn, after=None, think_cap=None):
    return SimpleNamespace(turn=turn, after=after, think_cap=think_cap)


def live_turn(request_id=7, **fields):
    turn = metrics.new_turn(request_id)
    turn.update(speculative=True, t_first_token=100.0, **fields)
    return turn


def test_line_format_names_cumulative_counts_and_no_ids():
    turn = live_turn(spec_rounds=4, round_tokens=11, decode_tokens=12,
                     draft_live=dict(draft_proposed=14, draft_accepted=7, spec_steps=4))
    line = metrics.decode_line(job(turn), 102.5)
    assert line.startswith("crucible_decode ")
    fields = json.loads(line.split(" ", 1)[1])
    assert list(fields) == KEYS
    assert fields == dict(request_id=7, completion_tokens=12, spec_rounds=4, draft_proposed=14, draft_accepted=7,
                          elapsed_s=2.5)


def test_note_token_keeps_only_well_formed_step_counters():
    turn = metrics.new_turn(1)
    metrics.note_token(turn, dict(event="token", id=5, draft_proposed=3, draft_accepted=2, spec_steps=1))
    metrics.note_token(turn, dict(event="token", id=6))
    metrics.note_token(turn, dict(event="token", id=7, draft_proposed=-1, draft_accepted=2, spec_steps=1))
    metrics.note_token(turn, dict(event="token", id=8, draft_proposed=True, draft_accepted=2, spec_steps=1))
    assert turn["decode_tokens"] == 4
    assert turn["draft_live"] == dict(draft_proposed=3, draft_accepted=2, spec_steps=1)


def test_rate_limit_is_one_line_a_second_from_the_first_token():
    turn = live_turn(spec_rounds=1, round_tokens=3, decode_tokens=4,
                     draft_live=dict(draft_proposed=3, draft_accepted=2, spec_steps=1))
    times = [100.4, 100.99, 101.0, 101.5, 101.99, 102.0, 102.3, 103.7]
    written = [t for t in times if metrics.decode_line(job(turn), t) is not None]
    assert written == [101.0, 102.0, 103.7]


def test_no_line_without_speculation_or_before_a_round():
    plain = live_turn(spec_rounds=5, round_tokens=5, decode_tokens=6)
    plain["speculative"] = False
    assert metrics.decode_line(job(plain), 105.0) is None
    first_token_only = live_turn(decode_tokens=1)
    assert metrics.decode_line(job(first_token_only), 105.0) is None
    assert metrics.final_decode_line('crucible_turn {"request_id": 1, "spec_rounds": 4, "decode_s": 1.0}',
                                     dict(speculative=False)) is None
    assert metrics.final_decode_line('crucible_turn {"request_id": 1, "spec_rounds": 0, "decode_s": 1.0}',
                                     dict(speculative=True)) is None


def test_no_line_without_draft_proposed():
    """a log parser needs draft_proposed: an engine without step counters writes no line."""
    turn = live_turn(spec_rounds=3, round_tokens=9, decode_tokens=10)
    assert metrics.decode_line(job(turn), 101.0) is None
    assert metrics.final_decode_line('crucible_turn {"request_id": 1, "spec_rounds": 3, "draft_accepted": 6, '
                                     '"completion_tokens": 10, "decode_s": 1.0}', dict(speculative=True)) is None


def test_no_live_line_after_the_final_line():
    """A stop string with a think cap ends the client before its job is cancelled."""
    turn = live_turn(spec_rounds=3, round_tokens=9, decode_tokens=10,
                     draft_live=dict(draft_proposed=9, draft_accepted=6, spec_steps=3))
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        metrics.emit_turn('crucible_turn {"request_id": 7, "completion_tokens": 10, "spec_rounds": 3, '
                          '"draft_proposed": 9, "draft_accepted": 6, "decode_s": 1.0}', turn)
    assert [line.split(" ", 1)[0] for line in lines] == ["crucible_decode", "crucible_turn"]
    assert metrics.decode_line(job(turn), 105.0) is None


def test_malformed_turns_write_no_line():
    for turn in (live_turn(spec_rounds="3", decode_tokens=4), live_turn(spec_rounds=3, decode_tokens=None,
                 draft_live=dict(draft_proposed=9, draft_accepted=6, spec_steps=3))):
        assert metrics.decode_line(job(turn), 105.0) is None
    assert metrics.decode_line(SimpleNamespace(turn=None, after=None), 105.0) is None
    broken = live_turn(spec_rounds=3, decode_tokens=4, draft_live="x")
    broken["t_first_token"] = "late"
    assert metrics.decode_line(job(broken), 105.0) is None


def test_unloggable_crucible_turn_writes_no_final_line():
    unloggable = metrics.turn_line(dict(request_id=7), dict(completion_tokens=float("nan")),
                                   dict(spec_rounds=3, draft_proposed=9, draft_accepted=6))
    assert "unloggable metrics" in unloggable
    assert metrics.final_decode_line(unloggable, dict(speculative=True)) is None
    for text in ("crucible_turn {not json", "crucible_turn [1]", "crucible_prefill {}", None):
        assert metrics.final_decode_line(text, dict(speculative=True)) is None


def run_turn(backend, lines, *args, **kwargs):
    with mock.patch.object(metrics, "emit", lines.append):
        return collect(backend.submit(*args, **kwargs))


def test_solo_spec_turn_writes_live_lines_and_a_final_line_equal_to_crucible_turn(start_backend):
    backend = start_backend(True, streams=2, env=LIVE)
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0):
        ids, done = run_turn(backend, lines, [1, 2, 3], request(10, spec=7))
    assert len(ids) == 10
    decode, (turn,) = parsed(lines, "crucible_decode"), parsed(lines, "crucible_turn")
    # Three solo steps of three tokens: a line after each of the first two, then the final line.
    assert [line["completion_tokens"] for line in decode] == [4, 7, 10]
    assert [line["draft_proposed"] for line in decode] == [3, 6, 9]
    assert decode[-1] == decode_values(turn)
    assert (turn["spec_rounds"], turn["draft_proposed"], turn["draft_accepted"]) == (3, 9, 6)
    assert lines.index(next(line for line in lines if line.startswith("crucible_turn "))) == len(lines) - 1
    assert all(line["request_id"] == turn["request_id"] for line in decode)


def test_packed_turns_write_lines_for_speculative_requests_only(start_backend):
    backend = start_backend(True, streams=4, env=dict(LIVE, FAKE_SLOT_CONTEXTS="128,128,128,128", FAKE_DELAY=".002"))
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0), mock.patch.object(metrics, "emit", lines.append):
        jobs = [backend.submit([1, 2, 3 + n], request(24, spec=7 if n % 2 else 0)) for n in range(4)]
        for one in jobs: collect(one)
    turns = {line["request_id"]: line for line in parsed(lines, "crucible_turn")}
    speculative = {one.turn["request_id"] for one in jobs if one.turn["speculative"]}
    decode = parsed(lines, "crucible_decode")
    assert len(speculative) == 2 and {line["request_id"] for line in decode} == speculative
    for request_id in speculative:
        mine = [line for line in decode if line["request_id"] == request_id]
        assert len(mine) >= 2
        for before, after in zip(mine, mine[1:]):
            assert all(after[key] >= before[key] for key in KEYS[1:])
        assert mine[-1] == decode_values(turns[request_id])


def test_live_lines_keep_the_rate_limit_on_the_backend(start_backend):
    backend = start_backend(True, streams=2, env=dict(LIVE, FAKE_DELAY=".01"))
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.05):
        run_turn(backend, lines, [1, 2, 3], request(60, spec=7))
    decode = parsed(lines, "crucible_decode")
    live = [line["elapsed_s"] for line in decode[:-1]]
    assert len(live) >= 2
    assert live[0] >= 0.05 - 1e-4 and all(b - a >= 0.05 - 2e-4 for a, b in zip(live, live[1:]))


def test_plain_turn_writes_no_decode_line(start_backend):
    backend = start_backend(True, streams=2, env=LIVE)
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0):
        run_turn(backend, lines, [1, 2, 3], request(10))
    assert parsed(lines, "crucible_decode") == [] and len(parsed(lines, "crucible_turn")) == 1


def test_old_engine_writes_no_decode_line(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128"})
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0):
        run_turn(backend, lines, [1, 2, 3], request(10, spec=7))
    (turn,) = parsed(lines, "crucible_turn")
    assert "draft_proposed" not in turn and parsed(lines, "crucible_decode") == []


def test_closed_think_turn_keeps_one_request_id_and_cumulative_counts(start_backend):
    """The two engine jobs of a closed think turn write their lines under the first job's request_id (the id of the
    turn's crucible_turn line and of its first job's crucible_prefill lines), with counts over both jobs."""
    backend = start_backend(True, streams=2, env=dict(LIVE, FAKE_CHECKPOINTS="1", FAKE_DELAY=".002"))
    lines, cap, close = [], 5, 3
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0), mock.patch.object(metrics, "emit", lines.append):
        first = backend.submit([1, 2, 3, 4], request(40, spec=7), log_turn=False, think_cap=cap)
        seen = 0
        while seen < cap:
            seen += first.events.get(timeout=3)["event"] == "token"
        first.turn["closed_count"] = cap + close  # http.close_thinking: the cap tokens and the close text
        cont = backend.continue_job(first, [1, 2, 3, 4, 5, 6], request(6, spec=7), log_turn=False)
        ids, cont_event = collect(cont)
        counts = dict(prompt_tokens=4, completion_tokens=cap + close + len(ids), finish_reason="length")
        metrics.emit_turn(metrics.request_line(first.turn, counts, SimpleNamespace(tokens=cap, by="answer_room"),
                                               cont.turn, cont_event), first.turn)
    decode, (turn,) = parsed(lines, "crucible_decode"), parsed(lines, "crucible_turn")
    assert turn["request_id"] == first.turn["request_id"] != turn["cont_request_id"] == cont.turn["request_id"]
    assert {line["request_id"] for line in decode} == {first.turn["request_id"]}
    # The continuation's prefill lines carry its own id; turn_request_id names the turn for a log reader (train 23).
    prefill = parsed(lines, "crucible_prefill")
    own = [line for line in prefill if line["request_id"] == cont.turn["request_id"]]
    base = [line for line in prefill if line["request_id"] == first.turn["request_id"]]
    assert own and all(line["turn_request_id"] == first.turn["request_id"] for line in own), prefill
    assert base and all("turn_request_id" not in line for line in base), prefill
    for before, after in zip(decode, decode[1:]):
        assert all(after[key] >= before[key] for key in KEYS[1:]), (before, after)
    # The continuation's lines start from the client's count at the close.
    cont_lines = [line for line in decode[:-1] if line["completion_tokens"] > cap]
    assert cont_lines and cont_lines[0]["completion_tokens"] > cap + close
    # The engine counters of both jobs: the first job's last step counters stand in for its done event.
    live, final = first.turn["draft_live"], cont_event
    assert turn["draft_proposed"] == live["draft_proposed"] + final["draft_proposed"]
    assert turn["spec_rounds"] == live["spec_steps"] + final["spec_steps"]
    assert decode[-1] == decode_values(turn)


class SpeculativeBackend(LinkedBackend):
    """LinkedBackend whose first job speculated: its last step counters, as note_token keeps them."""

    def submit(self, ids, request, **options):
        first = not self.jobs
        job = super().submit(ids, request, **options)
        if first:
            job.turn.update(speculative=True, spec_rounds=400, round_tokens=1100,
                            draft_live=dict(draft_proposed=1500, draft_accepted=700, spec_steps=400))
        return job


def test_http_close_writes_the_final_line_for_the_whole_turn():
    backend = SpeculativeBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", 9))
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        _, reply = run(backend, chat(max_tokens=2048))
    assert backend.jobs[0].turn["closed_count"] == 1024 + len(CLOSE)
    (decode,), (turn,) = parsed(lines, "crucible_decode"), parsed(lines, "crucible_turn")
    assert decode == decode_values(turn)
    assert (decode["request_id"], decode["completion_tokens"]) == (1, reply["usage"]["completion_tokens"])
    assert (decode["spec_rounds"], decode["draft_proposed"], decode["draft_accepted"]) == (400, 1500, 700)


def test_prefill_line_names_the_turn_only_for_a_continuation():
    """A log reader pairs a closed think turn's continuation by turn_request_id; every other line keeps today's keys."""
    turn = dict(request_id=9, t_first_gpu=metrics.now())
    plain = json.loads(metrics.prefill_line(turn, 4, 8, metrics.now()).split(" ", 1)[1])
    assert sorted(plain) == ["done", "elapsed_s", "request_id", "total"]
    cont = json.loads(metrics.prefill_line(turn, 4, 8, metrics.now(), turn_request_id=3).split(" ", 1)[1])
    assert cont.pop("turn_request_id") == 3 and cont.keys() == plain.keys()
