"""--lenient-fields (CRUCIBLE_LENIENT_FIELDS): an allow-list of request fields that cannot change the tokens.

Default on: the server ignores only the named fields of server/lenient.py (IGNORED at any value,
NEUTRAL at the value that changes nothing, DEPENDENT as a number). --lenient-fields off is the strict mode. Every other
unknown field is a 400 that names the field, in both modes: a request never gets 200 with a constraint not applied. The
log names the ignored fields, never their values.
"""
import contextlib
import io
import json

import pytest

from server import lenient
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, invoke, token_events

ON = dict(lenient_fields=True)
STRICT = dict(lenient_fields=False)
ALLOW_LISTED = (("user", "u1"), ("repeat_penalty", 1.0), ("cache_prompt", True), ("typical_p", 1), ("repeat_last_n", 64))


def served(payload, options=None, path="/v1/chat/completions"):
    backend = ScriptBackend(token_events(["hi"]))
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        response = invoke(backend, PiecesTokenizer(["hi"]), payload, path=path, service_options=options)
    return backend, response, log.getvalue()


def status(response):
    return response.statuses[0]


def error(response):
    return json.loads(response.body)["error"]


@pytest.mark.parametrize("field,value", ALLOW_LISTED, ids=[field for field, _ in ALLOW_LISTED])
def test_default_is_lenient(field, value):
    backend, response, _ = served(chat(**{field: value}))
    assert status(response) == 200
    plain, _, _ = served(chat())
    assert backend.submissions[0][:2] == plain.submissions[0][:2]


@pytest.mark.parametrize("field,value", ALLOW_LISTED, ids=[field for field, _ in ALLOW_LISTED])
def test_off_is_strict(field, value):
    backend, response, _ = served(chat(**{field: value}), STRICT)
    assert status(response) == 400 and error(response)["param"] == field
    assert backend.submissions == []


def test_the_service_default_is_the_serve_default():
    import inspect
    from server.http import Service
    assert inspect.signature(Service).parameters["lenient_fields"].default is lenient.resolve(None, {}) is True


@pytest.mark.parametrize("field,value", [("user", "someone"), ("cache_prompt", True), ("metadata", {"a": "b"}),
                                         ("service_tier", "auto"), ("safety_identifier", "s1"), ("prompt_cache_key", "k"),
                                         ("repeat_last_n", 64), ("mirostat_tau", 5.0), ("mirostat_eta", 0.1),
                                         ("dynatemp_exponent", 1), ("xtc_threshold", 0.1), ("dry_base", 1.75),
                                         ("dry_allowed_length", 2), ("dry_penalty_last_n", -1), ("repeat_last_n", None)])
def test_allow_listed_fields_are_ignored(field, value):
    backend, response, _ = served(chat(**{field: value}), ON)
    assert status(response) == 200
    plain, _, _ = served(chat(), ON)
    assert backend.submissions[0][:2] == plain.submissions[0][:2]  # the same prompt IDs and the same Request


@pytest.mark.parametrize("field,value", [("repeat_penalty", 1), ("repeat_penalty", 1.0), ("repeat_penalty", None),
                                         ("typical_p", 1.0), ("tfs_z", 1), ("mirostat", 0), ("dynatemp_range", 0),
                                         ("xtc_probability", 0.0), ("dry_multiplier", 0), ("top_n_sigma", -1),
                                         ("logit_bias", {}), ("ignore_eos", False), ("n_probs", 0), ("store", False)])
def test_neutral_sampling_values_are_accepted(field, value):
    backend, response, _ = served(chat(**{field: value}), ON)
    assert status(response) == 200
    request = backend.submissions[0][1]
    assert (request.temperature, request.top_k, request.top_p, request.min_p) == (0, 0, 1, 0)


@pytest.mark.parametrize("field,value", [("repeat_penalty", 1.1), ("repeat_penalty", 2.0), ("repeat_penalty", True),
                                         ("repeat_penalty", "1"), ("typical_p", 0.9), ("mirostat", 2), ("tfs_z", 0.5),
                                         ("xtc_probability", 0.1), ("dry_multiplier", 0.8), ("top_n_sigma", 1.0),
                                         ("logit_bias", {"42": -100}), ("ignore_eos", True), ("n_probs", 5),
                                         ("store", True), ("dynatemp_range", 0.5)])
def test_a_sampling_field_that_crucible_cannot_apply_is_refused_by_name(field, value):
    backend, response, _ = served(chat(**{field: value}), ON)
    assert status(response) == 400
    assert error(response)["param"] == field and field in error(response)["message"]
    assert backend.submissions == []


# response_format is a known field (structured output): server/tests/test_structured_adapters.py checks it in both modes.
@pytest.mark.parametrize("field,value", [("functions", []),
                                         ("function_call", "auto"), ("n_predict", 8), ("grammar", "root ::= \"a\""),
                                         ("json_schema", {}), ("samplers", ["top_k"])])
def test_fields_that_change_the_output_stay_refused(field, value):
    backend, response, _ = served(chat(**{field: value}), ON)
    assert status(response) == 400 and error(response)["param"] == field
    assert backend.submissions == []


def test_completion_echo_is_refused_and_false_is_accepted():
    payload = dict(prompt="x", max_tokens=4, seed=1)
    _, response, _ = served(payload | dict(echo=True), ON, path="/v1/completions")
    assert status(response) == 400 and error(response)["param"] == "echo"
    _, response, _ = served(payload | dict(echo=False), ON, path="/v1/completions")
    assert status(response) == 200


def test_known_fields_keep_their_checks():
    _, response, _ = served(chat(temperature=9, user="u"), ON)
    assert status(response) == 400 and error(response)["param"] == "temperature"
    _, response, _ = served(chat(presence_penalty=2.5), ON)
    assert status(response) == 400 and error(response)["param"] == "presence_penalty"


def test_nested_objects_stay_strict():
    _, response, _ = served(chat(stream_options=dict(include_usage=True, other=1)), ON)
    assert status(response) == 400 and error(response)["param"] == "stream_options"


def test_the_log_names_the_ignored_fields_once_never_their_values():
    secret = "VALUECANARY8812"
    _, response, log = served(chat(user=secret, prompt_cache_key=secret, metadata={"x": secret}, repeat_penalty=1), ON)
    assert status(response) == 200
    lines = [line for line in log.splitlines() if line.startswith("crucible_lenient ")]
    assert len(lines) == 1
    assert json.loads(lines[0].split(" ", 1)[1])["ignored"] == ["metadata", "prompt_cache_key", "repeat_penalty", "user"]
    assert secret not in log


def test_a_field_name_that_is_not_a_plain_name_is_refused_and_not_logged():
    for odd in ("x" * 65, "bad name\n"):
        backend, response, log = served(chat(**{odd: 1, "user": "u"}), ON)
        assert status(response) == 400 and backend.submissions == []
        assert "crucible_lenient" not in log


# Before the fix, each of these returned 200 with the constraint not applied.
@pytest.mark.parametrize("field,value", [("allowed_token_ids", [21]), ("stop_token_ids", [20]), ("min_tokens", 10),
                                         ("guided_regex", "[0-9]+")])
def test_output_controls_are_refused_with_the_mode_on(field, value):
    backend, response, log = served(chat(return_token_ids=True, **{field: value}), ON)
    assert status(response) == 400
    assert error(response)["param"] == field and field in error(response)["message"]
    assert backend.submissions == [] and "crucible_lenient" not in log


# Fields of OpenAI-compatible, other engines, other engines, TGI and OpenAI that are not on the allow-list (and not fields that
# validate_request knows), each with a value that a client can send. All are 400 in both modes.
OUTSIDE = [("allowed_token_ids", [21]), ("stop_token_ids", [20]), ("min_tokens", 10), ("guided_regex", "[0-9]+"),
           ("guided_json", {}), ("guided_choice", ["a"]), ("guided_grammar", "root ::= \"a\""),
           ("guided_decoding_backend", "x"), ("structured_outputs", {}), ("bad_words", ["x"]),
           ("logits_processors", []), ("include_stop_str_in_output", True), ("skip_special_tokens", False),
           ("spaces_between_special_tokens", False), ("length_penalty", 1.0), ("use_beam_search", False),
           ("truncate_prompt_tokens", 8), ("prompt_logprobs", 1), ("min_keep", 0), ("n_keep", 0), ("n_indent", 0),
           ("t_max_predict_ms", 100), ("t_max_prompt_ms", 100), ("lora", []), ("id_slot", 0), ("post_sampling_probs", False),
           ("dry_sequence_breakers", ["\n"]), ("samplers", ["top_k"]), ("grammar", "root ::= \"a\""), ("json_schema", {}),
           ("n_predict", 8), ("functions", []), ("function_call", "auto"),
           ("prediction", {}), ("audio", {}), ("web_search_options", {}), ("ebnf", "x"), ("regex", "x"), ("zeta", 1),
           ("think", True), ("min_new_tokens", 1), ("max_new_tokens", 4), ("do_sample", False),
           ("watermark", False), ("decoder_input_details", False), ("top_n_tokens", 1), ("verbosity", "low"),
           ("reasoning", {}), ("instructions", "x"), ("max_output_tokens", 4)]


@pytest.mark.parametrize("options", [STRICT, None, ON], ids=["strict", "default", "lenient"])
@pytest.mark.parametrize("field,value", OUTSIDE, ids=[field for field, _ in OUTSIDE])
def test_every_field_outside_the_allow_list_is_refused_in_both_modes(options, field, value):
    assert field not in lenient.ALLOWED
    backend, response, _ = served(chat(**{field: value}), options)
    assert status(response) == 400 and error(response)["param"] == field
    assert backend.submissions == []


def test_the_allow_list_is_the_three_tables():
    assert lenient.ALLOWED == set(lenient.IGNORED) | set(lenient.NEUTRAL) | set(lenient.DEPENDENT)
    assert not lenient.ALLOWED & set(lenient.REFUSED)
    for name, reason in lenient.IGNORED.items():
        assert isinstance(reason, str) and reason, name
    for name, parent in lenient.DEPENDENT.items():
        assert parent in lenient.NEUTRAL, name


def test_the_client_guide_names_each_allowed_field():
    from pathlib import Path
    guide = (Path(__file__).parents[2] / "docs/OPENAI_CLIENTS.md").read_text()
    section = guide.split("### Lenient request fields", 1)[1].split("\n## ", 1)[0]
    for name in sorted(lenient.ALLOWED):
        assert f"`{name}`" in section, name


@pytest.mark.parametrize("value", [True, "64", [64], {"n": 64}])
def test_a_dependent_field_must_be_a_number(value):
    backend, response, _ = served(chat(repeat_last_n=value), ON)
    assert status(response) == 400 and error(response)["param"] == "repeat_last_n"
    assert backend.submissions == []


def test_no_ignored_field_no_line():
    _, _, log = served(chat(), ON)
    assert "crucible_lenient" not in log


def test_the_token_count_utility_uses_the_same_rule():
    _, response, _ = served(chat(user="u"), ON, path="/tokenize")
    assert status(response) == 200
    _, response, _ = served(chat(user="u"), STRICT, path="/tokenize")
    assert status(response) == 400


@pytest.mark.parametrize("flag,environ,expected", [
    (None, {}, True), (None, {lenient.ENV: ""}, True), (None, {lenient.ENV: " "}, True), (None, {lenient.ENV: "1"}, True),
    (None, {lenient.ENV: "on"}, True), (None, {lenient.ENV: "true"}, True), (None, {lenient.ENV: "0"}, False),
    (None, {lenient.ENV: "off"}, False), (None, {lenient.ENV: "no"}, False), (None, {lenient.ENV: "false"}, False),
    ("on", {lenient.ENV: "0"}, True), ("off", {lenient.ENV: "1"}, False), ("off", {}, False)])
def test_flag_and_environment(flag, environ, expected):
    assert lenient.resolve(flag, environ) is expected


def test_a_bad_environment_value_is_an_error():
    with pytest.raises(ValueError, match=lenient.ENV):
        lenient.resolve(None, {lenient.ENV: "maybe"})


def test_serve_has_the_flag():
    from pathlib import Path
    source = (Path(__file__).parents[1] / "__main__.py").read_text()
    assert '"--lenient-fields"' in source and "lenient.resolve(" in source and "lenient_fields=" in source
    flag = source.split('"--lenient-fields"', 1)[1].split("parser.add_argument", 1)[0]
    assert "Default on" in flag and "off:" in flag and "Default off" not in flag


# OpenAI-compatible's repeat_last_n is the window of its presence and frequency penalties too, and its server puts the prompt
# tokens in that window (tools/server/server-context.cpp: common_sampler_accept for each prompt token; 0 turns the
# penalties off). No value of it gives the rule of this server (the whole output, prompt tokens only for
# repetition_penalty). So a request that sets a penalty and a repeat_last_n other than null is refused.
@pytest.mark.parametrize("penalty,value", [("presence_penalty", 0.5), ("frequency_penalty", -0.3),
                                           ("repetition_penalty", 1.1)])
@pytest.mark.parametrize("window", [0, 64, -1, 4096.0])
@pytest.mark.parametrize("options", [None, ON], ids=["default", "lenient"])
def test_a_window_with_an_active_penalty_is_refused(penalty, value, window, options):
    backend, response, _ = served(chat(**{penalty: value, "repeat_last_n": window}), options)
    assert status(response) == 400
    assert error(response)["param"] == "repeat_last_n" and error(response)["type"] == "invalid_request_error"
    assert "repeat_last_n" in error(response)["message"] and penalty in error(response)["message"]
    assert backend.submissions == []


@pytest.mark.parametrize("extra", [dict(presence_penalty=0, frequency_penalty=0.0, repetition_penalty=1),
                                   dict(presence_penalty=None, repetition_penalty=None), {}])
@pytest.mark.parametrize("options", [None, ON], ids=["default", "lenient"])
def test_a_window_with_neutral_penalties_is_ignored(extra, options):
    backend, response, _ = served(chat(repeat_last_n=64, **extra), options)
    assert status(response) == 200
    plain, _, _ = served(chat(), options)
    assert backend.submissions[0][:2] == plain.submissions[0][:2]


@pytest.mark.parametrize("options", [None, ON], ids=["default", "lenient"])
def test_a_null_window_with_an_active_penalty_is_accepted(options):
    backend, response, _ = served(chat(presence_penalty=0.5, repeat_last_n=None), options)
    assert status(response) == 200 and backend.submissions


@pytest.mark.parametrize("penalty,value", [("presence_penalty", 0.5), ("repetition_penalty", 1.1)])
def test_strict_mode_refuses_any_window(penalty, value):
    backend, response, _ = served(chat(**{penalty: value, "repeat_last_n": 64}), STRICT)
    assert status(response) == 400 and error(response)["param"] == "repeat_last_n"
    assert backend.submissions == []
