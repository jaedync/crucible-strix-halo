"""tools/constraint_fuzz.py: the CPU fuzz of the structured output boundary.

The pure parts: the request generator, the server step and the rules that make a runner line a finding."""
import json
import random
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import constraint_fuzz as cf  # noqa: E402


def test_the_generator_is_deterministic_json():
    ra, rb = random.Random(5), random.Random(5)
    a = [cf.random_body(ra, i) for i in range(80)]
    b = [cf.random_body(rb, i) for i in range(80)]
    assert json.dumps(a) == json.dumps(b)
    kinds = {cf.body_kind(body) for body in a}
    assert {"answer", "tools"} <= kinds


def test_the_generator_reaches_the_hostile_shapes():
    rng = random.Random(1)
    text = json.dumps([cf.random_body(rng, i) for i in range(400)])
    for word in ('"pattern"', '"format"', '"anyOf"', '"$ref"', '"enum"', '"maxItems"', '"maxLength"', '"strict"'):
        assert word in text, word


def test_a_seed_mutation_stays_json():
    rng = random.Random(3)
    seed = {"type": "object", "properties": {"a": {"type": "string", "pattern": "(a|b){1,9}"}}, "required": ["a"]}
    for _ in range(50):
        json.dumps(cf.mutate(rng, seed))


def test_json_depth_needs_no_recursion():
    deep = []
    for _ in range(50000):
        deep = [deep]
    assert cf.json_depth(deep) == 50001 and cf.json_depth({"a": [1]}) == 3


def test_the_server_step_sorts_accepted_refused_and_faults():
    accepted = cf.server_step({"messages": [{"role": "user", "content": "x"}],
                               "response_format": {"type": "json_object"}})
    assert accepted[0] == "ok" and accepted[1]["kind"] == "json_object"
    refused = cf.server_step({"messages": [{"role": "user", "content": "x"}],
                              "response_format": {"type": "json_schema", "json_schema": {"schema": {"oneOf": []}}}})
    assert refused[0] == "refused"
    bad = cf.server_step({"messages": [{"role": "user", "content": "x"}], "response_format": "x"})
    assert bad[0] == "refused"


def test_a_server_fault_is_a_finding(monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError("x")
    monkeypatch.setattr(cf.api, "validate_request", boom)
    assert cf.server_step({"messages": []})[0] == "fault"


def test_runner_rules():
    ok = cf.parse_result("R 3 ok 120 50000 4096 4096 0.020 64 0.900 40.0")
    assert ok["status"] == "ok" and ok["make_ms"] == 120 and ok["max_token_ms"] == 0.9
    assert cf.findings(ok) == []
    assert cf.findings(cf.parse_result("R 3 constraint_limit/time 5004 0 0 0 0 0 0 0")) == ["accepted_spec_hits_deadline"]
    assert cf.findings(cf.parse_result("R 3 constraint_limit/memory 900 0 0 0 0 0 0 0")) == ["accepted_spec_hits_memory"]
    assert cf.findings(cf.parse_result("R 3 constraint_compile/crash 30 0 0 0 0 0 0 0")) == ["compiler_crash"]
    assert cf.findings(cf.parse_result("R 3 ok 10 0 0 0 150.0 4 0.5 3")) == ["slow_open"]
    assert cf.findings(cf.parse_result("R 3 ok 10 0 0 0 0.1 4 150.0 3")) == ["slow_token"]
    assert cf.findings(cf.parse_result("R 3 ok 10 0 0 600000000 0.1 4 0.5 3")) == ["cache_over_limit"]
    # A schema that does not compile is a plain 400 for its client, not a finding.
    assert cf.findings(cf.parse_result("R 3 constraint_compile 10 0 0 0 0 0 0 0")) == []
    assert cf.findings(cf.parse_result("R 3 ok 1500 0 0 0 0.1 4 0.5 3")) == ["slow_compile"]
    assert "slow_compile" in cf.NOTES and not {"compiler_crash", "accepted_spec_hits_deadline"} & cf.NOTES


def test_the_runner_reader_keeps_two_lines_of_one_write(tmp_path):
    # Both lines come in one write: a buffered readline after select() held the second one and timed out (a false hang).
    fake = tmp_path / "runner.sh"
    fake.write_text("#!/bin/sh\nread line\nprintf 'S 0\\nR 0 ok 1 0 0 0 0.001 0 0.001 0.1\\n'\nexec sleep 30\n")
    fake.chmod(0o755)
    runner = cf.Runner(fake, "x", 1)
    runner.argv = [str(fake)]
    start = time.monotonic()
    result, event = runner.case({"kind": "json_object", "spec": ""})
    runner.stop()
    assert event == "ok" and result["status"] == "ok" and time.monotonic() - start < 5


def test_the_slowest_cases_are_kept():
    slowest = []
    for key in [3, 9, 1, 7, 5, 8, 2, 6, 4, 10]:
        cf.keep_slowest(slowest, key, {"k": key})
    assert [case["k"] for _, case in slowest] == [10, 9, 8, 7, 6, 5, 4, 3]
