"""presence_penalty, frequency_penalty and repetition_penalty.

The server checks the range and sends a value to the engine only when it is not neutral, so a request with the neutral
values (or none) gives the engine the same payload as before. A value outside the range is a 400 that names the field.
"""
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from server.api import ApiError, Request, validate_request
from server.backend import EngineBackend
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, invoke, token_events

FAKE = Path(__file__).parent / "fake_engine.py"
FIELDS = ("presence_penalty", "frequency_penalty", "repetition_penalty")
NEUTRAL = dict(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)


def bodies():
    return (("chat", dict(model="crucible", messages=[{"role": "user", "content": "hi"}], max_tokens=8, seed=7)),
            ("completion", dict(model="crucible", prompt="hello", max_tokens=8, temperature=0, seed=7)))


@pytest.mark.parametrize("kind,body", bodies())
def test_accepted_values_reach_the_request(kind, body):
    for field, values in (("presence_penalty", (-2, -0.5, 0.25, 1, 2.0)), ("frequency_penalty", (-2.0, -1, 0.5, 2)),
                          ("repetition_penalty", (0.01, 0.9, 1.1, 1.5, 2))):
        for value in values:
            request = validate_request(body | {field: value}, kind, "crucible")
            assert getattr(request, field) == value and type(getattr(request, field)) is float
            for other in FIELDS:
                if other != field:
                    assert getattr(request, other) == NEUTRAL[other]
    request = validate_request(body | dict(presence_penalty=0.5, frequency_penalty=-0.5, repetition_penalty=1.1),
                               kind, "crucible")
    assert (request.presence_penalty, request.frequency_penalty, request.repetition_penalty) == (0.5, -0.5, 1.1)


@pytest.mark.parametrize("kind,body", bodies())
def test_neutral_and_null_are_the_request_without_the_field(kind, body):
    expected = validate_request(body, kind, "crucible")
    for field, values in (("presence_penalty", (0, 0.0, -0.0, None)), ("frequency_penalty", (0, 0.0, -0.0, None)),
                          ("repetition_penalty", (1, 1.0, None))):
        for value in values:
            assert validate_request(body | {field: value}, kind, "crucible") == expected
    for name, value in NEUTRAL.items():
        assert getattr(expected, name) == value
    assert validate_request(body | dict(presence_penalty=-0.0), kind, "crucible").presence_penalty == 0.0
    assert math.copysign(1, validate_request(body | dict(presence_penalty=-0.0), kind, "crucible").presence_penalty) == 1


@pytest.mark.parametrize("kind,body", bodies())
def test_values_outside_the_range_are_refused(kind, body):
    bad = (True, False, "0", "", [], [0], {}, math.nan, math.inf, -math.inf, 10 ** 400)
    for field, low, high, outside in (("presence_penalty", "[-2", "2]", (-2.01, 2.5, 100)),
                                      ("frequency_penalty", "[-2", "2]", (-3, 2.0001)),
                                      ("repetition_penalty", "[0.01", "2]",
                                       (0, 0.0, -0.0, -1, 2.01, 3, 1e-50, 1e-38, 0.0099))):
        for value in bad + outside:
            with pytest.raises(ApiError) as caught:
                validate_request(body | {field: value}, kind, "crucible")
            assert caught.value.param == field and caught.value.status == 400
            assert str(caught.value) == f"{field} must be a finite number in {low},{high}."


def served(payload):
    backend = ScriptBackend(token_events(["hi"]))
    response = invoke(backend, PiecesTokenizer(["hi"]), payload)
    return backend, response


def test_http_error_shape_names_the_field():
    _, response = served(chat(repetition_penalty=2.5))
    assert response.statuses[0] == 400
    error = json.loads(response.body)["error"]
    assert error["param"] == "repetition_penalty" and error["type"] == "invalid_request_error"
    assert error["message"] == "repetition_penalty must be a finite number in [0.01,2]."


def test_tiny_repetition_penalty_is_refused_before_the_engine():
    # 1e-50 is 0 as a float, so a prompt token's positive logit became +inf and the
    # engine's batch session stopped for every user. The floor refuses it with a 400; the backend never sees it.
    for value in (1e-50, 1e-38, 0.001):
        backend, response = served(chat(repetition_penalty=value))
        assert response.statuses[0] == 400 and not backend.submissions
        error = json.loads(response.body)["error"]
        assert error["param"] == "repetition_penalty"
        assert error["message"] == "repetition_penalty must be a finite number in [0.01,2]."


def test_http_accepts_a_client_repetition_penalty():
    # The test suite of another server sends repetition_penalty 1.1.
    _, response = served(chat(repetition_penalty=1.1, presence_penalty=0.5, frequency_penalty=0.5))
    assert response.statuses[0] == 200


class TestEnginePayload:
    def setup_method(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = EngineBackend("unused", "unused", context=32)

    def teardown_method(self):
        self.backend.close()

    def payload(self, **options):
        request = validate_request(dict(prompt="x", max_tokens=8, seed=7) | options, "completion", "crucible")
        job = self.backend.submit([1], request)
        while job.events.get(timeout=5)["event"] != "done":
            pass
        return job.payload

    def test_neutral_payload_is_unchanged(self):
        plain = self.payload()
        assert not set(plain) & set(FIELDS) and "penalty_prompt" not in plain
        neutral = self.payload(presence_penalty=0, frequency_penalty=None, repetition_penalty=1)
        assert neutral == plain

    def test_penalties_reach_the_engine(self):
        payload = self.payload(presence_penalty=0.5, repetition_penalty=1.1)
        assert payload["presence_penalty"] == 0.5 and payload["repetition_penalty"] == 1.1
        assert "frequency_penalty" not in payload and "penalty_prompt" not in payload
        assert self.payload(frequency_penalty=-1)["frequency_penalty"] == -1.0

    def test_a_continuation_names_its_prompt_length(self):
        # A server continuation job (a think cap, the end-of-turn guard) prefills the earlier output again: the
        # engine counts the IDs after the client's prompt as output, as one job would have counted them.
        request = validate_request(dict(prompt="x", max_tokens=8, repetition_penalty=1.1), "completion", "crucible")
        job = self.backend.submit([1, 5, 6], dataclasses.replace(request, max_tokens=4, penalty_prompt=1))
        while job.events.get(timeout=5)["event"] != "done":
            pass
        assert job.payload["penalty_prompt"] == 1
        plain = validate_request(dict(prompt="x", max_tokens=8), "completion", "crucible")
        job = self.backend.submit([1, 5, 6], dataclasses.replace(plain, max_tokens=4, penalty_prompt=1))
        while job.events.get(timeout=5)["event"] != "done":
            pass
        assert "penalty_prompt" not in job.payload

    def test_penalties_and_a_constraint_reach_the_engine_together(self):
        # Structured output and the penalties are independent fields of one job: the engine applies the penalties
        # first and the grammar mask after them (engine/runtime/constraint.h ChoosePenalized).
        self.backend.constraint_ready = True
        body = dict(messages=[{"role": "user", "content": "hi"}], max_tokens=8, seed=7,
                    response_format=dict(type="json_object"), enable_thinking=False)
        request = validate_request(body | dict(presence_penalty=0.5, repetition_penalty=1.1), "chat", "crucible")
        job = self.backend.submit([1, 5, 6], dataclasses.replace(request, penalty_prompt=1), constraint_from=1)
        job.cancelled.set()
        assert job.payload["constraint"] == dict(kind="json_object", spec="", reasoning=0)
        assert job.payload["constraint_from"] == 1 and job.payload["penalty_prompt"] == 1
        assert job.payload["presence_penalty"] == 0.5 and job.payload["repetition_penalty"] == 1.1
        neutral = self.backend.submit([1, 5, 6], validate_request(body, "chat", "crucible"), constraint_from=1)
        neutral.cancelled.set()
        assert not set(neutral.payload) & (set(FIELDS) | {"penalty_prompt"}) and "constraint" in neutral.payload


def test_request_fields_have_neutral_defaults():
    names = {field.name: field.default for field in dataclasses.fields(Request)}
    assert {name: names[name] for name in FIELDS} == NEUTRAL
    assert names["penalty_prompt"] is None


def test_continuation_jobs_name_the_client_prompt_length():
    # A think-cap close and the end-of-turn guard continue on a new job from the prompt and the output so far: the
    # engine counts the earlier output as output (penalty_prompt), as one job would have counted it.
    from server.tests.test_end_of_turn_guard import GuardTokenizer
    from server.tests.test_think_budget import (ANSWER, BREAK, END, IM_END, PROMPT, REASON, BudgetTokenizer,
                                                SequenceBackend)
    from server.tests.test_think_budget import chat as thinking_chat
    for tokenizer, script in ((BudgetTokenizer(), (([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))),
                              (GuardTokenizer(), (([REASON] * 3 + [IM_END], "stop"), ([REASON, END, BREAK, ANSWER, IM_END], "stop")))):
        for penalties, expected in ((dict(repetition_penalty=1.1), len(PROMPT)), ({}, None)):
            backend = SequenceBackend(*script)
            response = invoke(backend, tokenizer, thinking_chat(max_tokens=2048, **penalties))
            assert response.statuses == [200] and len(backend.submissions) == 2
            assert backend.submissions[0][1].penalty_prompt is None
            assert backend.submissions[1][1].penalty_prompt == expected


def test_a_constrained_continuation_names_both_starts():
    # A think-cap continuation of a penalized request with response_format: the job names the client's prompt length
    # twice, as the output start of the grammar replay (constraint_from) and of the penalty counts (penalty_prompt).
    from server.tests.test_think_budget import ANSWER, IM_END, PROMPT, REASON, BudgetTokenizer, SequenceBackend
    from server.tests.test_think_budget import chat as thinking_chat
    backend = SequenceBackend(([REASON] * 1500, "length"), ([ANSWER, IM_END], "stop"))
    backend.constraint_ready = True  # the engine named the constraint key in its ready event
    body = thinking_chat(max_tokens=2048, repetition_penalty=1.1, response_format=dict(type="json_object"))
    response = invoke(backend, BudgetTokenizer(), body)
    assert response.statuses == [200] and len(backend.submissions) == 2
    (_, first, first_options), (_, follow, follow_options) = backend.submissions
    assert first.penalty_prompt is None and "constraint_from" not in first_options
    assert follow.penalty_prompt == len(PROMPT) and follow_options["constraint_from"] == len(PROMPT)
    assert follow.constraint is not None and follow.repetition_penalty == 1.1
