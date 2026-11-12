"""CPU proof of tools/serve_parity.py with fake servers (no engine, no GPU)."""
import base64
import json
from pathlib import Path
import socket
import sys
import threading
import zlib
from http.server import ThreadingHTTPServer

import pytest

from tools import serve_parity as parity
from tools import serve_parity_cases as cases_module
from server.tests import fake_parity_server as fake

FAKE = Path(__file__).parent / "fake_parity_server.py"


def ok(ids, finish="length", cached=0, seconds=1.0):
    return dict(status=200, ids=ids, finish=finish, text="x", seconds=seconds, cached=cached, label=None,
                usage=dict(prompt_tokens=5, completion_tokens=len(ids)))


def refused(status=400):
    return dict(status=status, error="no", seconds=0.1)


def case(name="c", gap=False):
    return cases_module.Case(name, "gap" if gap else "greedy", {}, gap=gap)


def test_first_difference():
    assert parity.first_difference([1, 2], [1, 2]) is None
    assert parity.first_difference([1, 2], [1, 3]) == 1
    assert parity.first_difference([1, 2], [1, 2, 3]) == 2


def test_judge_equal_ids_pass_and_cache_difference_is_not_a_failure():
    result = parity.judge(case(), ok([1, 2], cached=0), ok([1, 2], cached=64))
    assert result["verdict"] == "PASS"


@pytest.mark.parametrize("other", [ok([1, 3]), ok([1, 2], finish="stop"), ok([1, 2, 3])])
def test_judge_fails_on_ids_finish_or_length(other):
    assert parity.judge(case(), ok([1, 2]), other)["verdict"] == "FAIL"


def test_judge_one_sided_error_fails_and_same_error_passes():
    assert parity.judge(case(), ok([1]), refused())["verdict"] == "FAIL"
    assert parity.judge(case(), refused(413), refused(413))["verdict"] == "PASS"
    assert parity.judge(case(), refused(0), ok([1]))["verdict"] == "FAIL"


def test_judge_gap_states():
    gap = case(gap=True)
    assert parity.judge(gap, ok([1]), refused())["verdict"] == "KNOWN_GAP"
    assert parity.judge(gap, ok([1]), ok([1]))["verdict"] == "GAP_CLOSED"
    assert parity.judge(gap, ok([1]), ok([2]))["verdict"] == "FAIL"
    # A gap case where the batch server fails with a server error is not the known refusal.
    assert parity.judge(gap, ok([1]), refused(500))["verdict"] == "FAIL"


def test_compare_summary_and_slow_cases():
    cases = [case("a"), case("b"), case("g", gap=True)]
    serial = {"a": ok([1] * 10, seconds=1.0), "b": ok([1] * 10, seconds=1.0), "g": ok([1])}
    batch = {"a": ok([1] * 10, seconds=1.0), "b": ok([1] * 10, seconds=2.0), "g": refused()}
    report = parity.compare(cases, serial, batch)
    assert report["status"] == "PASS"
    assert report["summary"] == dict(PASS=2, FAIL=0, KNOWN_GAP=1, GAP_CLOSED=0)
    assert report["slow_cases"] == ["b"]
    batch["a"] = ok([9] * 10)
    assert parity.compare(cases, serial, batch)["status"] == "FAIL"


def test_compare_with_no_common_case_fails():
    assert parity.compare([case()], {}, {})["status"] == "FAIL"


def test_run_cases_skips_vision_and_replay_when_off():
    sent = []
    def send(base, body):
        sent.append(body)
        return ok([1])
    everything = cases_module.default_cases()
    results = parity.run_cases("u", everything, vision=False, cache=False, send=send)
    assert not any(name.startswith(("vision", "replay")) for name in results)
    assert len(results) == len([c for c in everything if not c.needs_vision and not c.needs_cache])
    assert len(parity.run_cases("u", everything, vision=True, cache=True, send=send)) == len(everything)


def test_replay_turn_repeats_the_reply_of_this_server():
    turn = next(c for c in cases_module.default_cases() if c.name == "replay-t2")
    results = {"replay-t1": dict(text="the lighthouse reply")}
    messages = turn.body_for(results)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == "the lighthouse reply"


def test_every_case_is_unique_seeded_and_synthetic():
    names = [c.name for c in cases_module.default_cases()]
    assert len(names) == len(set(names))
    assert {c.group for c in cases_module.default_cases()} == set(cases_module.GROUPS)
    for item in cases_module.default_cases():
        body = item.body
        if body.get("temperature", 0) > 0:
            assert "seed" in body, item.name


def test_png_is_valid():
    url = cases_module.png_data_url(8, 4)
    data = base64.b64decode(url.split(",", 1)[1])
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    start = data.index(b"IDAT") + 4
    length = int.from_bytes(data[start - 8:start - 4], "big")
    assert len(zlib.decompress(data[start:start + length])) == 4 * (1 + 8 * 3)


def test_estimate_counts_only_selected_cases():
    everything = cases_module.default_cases()
    assert parity.estimate(everything, vision=False, cache=False) < parity.estimate(everything)
    assert 0 < parity.estimate(everything) < 1800


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def port_base(count, pick=None):
    """A first port for --port-base: serve_parity uses this port and the count - 1 ports after it, so all must be free."""
    pick = pick or (lambda: free_ports(1)[0])
    for _ in range(200):
        base = pick()
        if base + count > 65536:
            continue
        sockets = [socket.socket() for _ in range(count)]
        try:
            for port, item in zip(range(base, base + count), sockets):
                item.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
        finally:
            for item in sockets:
                item.close()
    raise RuntimeError(f"no {count} free ports that follow each other")


def test_port_base_gives_ports_that_follow_each_other_and_are_free():
    # CPU workflow, Nov 12 (container run b3): the heads test took free_ports(2)[0] as the base, but the server of the
    # second head binds base + 1, which nobody checked. Under pytest -n 8 another worker held it: "the server stopped".
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        taken = held.getsockname()[1]
        picks = iter([taken - 1])
        base = port_base(2, pick=lambda: next(picks, None) or free_ports(1)[0])
        for port in range(base, base + 2):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))


def serve_in_thread(batch):
    handler = type("H", (fake.Handler,), dict(batch=batch, vision=True, seen=set()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_run_against_two_fake_servers(tmp_path, capsys):
    servers = [serve_in_thread(False), serve_in_thread(True)]
    try:
        urls = [f"http://127.0.0.1:{item.server_address[1]}/v1" for item in servers]
        out = tmp_path / "report.json"
        code = parity.main(["run", "--vision", "--serial-url", urls[0], "--batch-url", urls[1], "--out", str(out)])
    finally:
        for item in servers:
            item.shutdown()
    report = json.loads(out.read_text())
    assert code == 0 and report["status"] == "PASS"
    assert report["summary"]["KNOWN_GAP"] == 3 and report["summary"]["GAP_CLOSED"] == 0
    assert report["summary"]["FAIL"] == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_run_reports_a_difference(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PARITY_DIVERGE", "lighthouse")
    servers = [serve_in_thread(False), serve_in_thread(True)]
    try:
        urls = [f"http://127.0.0.1:{item.server_address[1]}/v1" for item in servers]
        out = tmp_path / "report.json"
        code = parity.main(["run", "--serial-url", urls[0], "--batch-url", urls[1], "--out", str(out)])
    finally:
        for item in servers:
            item.shutdown()
    report = json.loads(out.read_text())
    assert code == 1 and report["status"] == "FAIL"
    assert {row["name"] for row in report["cases"] if row["verdict"] == "FAIL"} >= {"greedy-short"}


def test_launch_starts_each_server_in_turn_and_stops_it(tmp_path):
    first = port_base(2)
    out = tmp_path / "launch" / "report.json"
    out.parent.mkdir()
    code = parity.main(["launch", "ENGINE", "CKPT", "--out", str(out), "--vision", "--port-base", str(first),
                        "--server-command", f"{sys.executable} {FAKE}"])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["summary"]["KNOWN_GAP"] == 3
    for role in ("serial", "batch"):
        assert out.with_suffix(f".{role}.log").exists()
    for port in (first, first + 1):
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0  # both processes are stopped


def test_serve_args_batch_adds_one_stream_only():
    args = parity.parser_for().parse_args(["launch", "E", "C", "--out", "o.json", "--vision"])
    serial, batch = (parity.serve_args(args, role, 9000) for role in ("serial", "batch"))
    # A missing --streams now means --streams 1, so the serial role must say --serial.
    assert "--streams" not in serial and "--serial" in serial and "--serial" not in batch
    assert batch[batch.index("--streams") + 1] == "1"
    assert batch[:-2] == serial[:-1] and (serial[-1], batch[-2:]) == ("--serial", ["--streams", "1"])
    assert "--vision" in serial and "--session-replay" in serial


def test_heads_runs_the_default_then_each_block_file(tmp_path):
    # serve --draft-blocks: one batch server per head, in turn; every head must give the IDs of the
    # built-in head.
    first = port_base(3)
    heads = []
    for name, text in (("full", ",".join(map(str, range(1940)))), ("one", "1937")):
        path = tmp_path / f"{name}.txt"
        path.write_text(text + "\n")
        heads.append(f"{name}={path}")
    out = tmp_path / "heads.json"
    argv = ["heads", "ENGINE", "CKPT", "--out", str(out), "--port-base", str(first),
            "--server-command", f"{sys.executable} {FAKE}"]
    for head in heads:
        argv += ["--blocks", head]
    assert parity.main(argv) == 0
    report = json.loads(out.read_text())
    assert report["status"] == "PASS" and set(report["runs"]) == {"full", "one"}
    for run in report["runs"].values():
        assert run["status"] == "PASS" and run["summary"]["FAIL"] == 0 and run["summary"]["PASS"] > 10
    logs = {name: out.with_suffix(f".{name}.log") for name in ("default", "full", "one")}
    assert all(path.exists() for path in logs.values())
    for port in (first, first + 1, first + 2):
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_heads_fails_when_a_head_changes_an_output(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_HEADS_DIVERGE", "lighthouse")
    path = tmp_path / "one.txt"
    path.write_text("1937\n")
    out = tmp_path / "heads.json"
    code = parity.main(["heads", "ENGINE", "CKPT", "--out", str(out), "--port-base", str(port_base(2)),
                        "--server-command", f"{sys.executable} {FAKE}", "--blocks", f"one={path}"])
    report = json.loads(out.read_text())
    assert code == 1 and report["status"] == "FAIL" and report["runs"]["one"]["summary"]["FAIL"] > 0


def test_heads_server_args_add_the_block_file_to_a_batch_server():
    args = parity.parser_for().parse_args(["heads", "E", "C", "--out", "o.json", "--blocks", "one=/x/one.txt"])
    assert args.blocks == [("one", Path("/x/one.txt"))]
    command = parity.serve_args(args, "batch", 9000) + ["--draft-blocks", "/x/one.txt"]
    assert command[command.index("--streams") + 1] == "1" and "--serial" not in command
    for bad in ("one", "=/x", "One=/x/a.txt", "one=x.txt"):
        with pytest.raises(SystemExit):
            parity.parser_for().parse_args(["heads", "E", "C", "--out", "o.json", "--blocks", bad])


def test_batch_against_batch_has_no_known_gap_and_needs_mostly_200():
    # two batch servers must agree on a gap case too; a 200 against a 400 is a FAIL.
    gap = case("g", gap=True)
    assert parity.judge(gap, ok([1]), refused(), gaps=False)["verdict"] == "FAIL"
    assert parity.judge(gap, refused(), refused(), gaps=False)["verdict"] == "PASS"
    assert parity.judge(gap, ok([1]), ok([1]), gaps=False)["verdict"] == "PASS"
    # A set where most cases are refused on both sides proves nothing: FAIL.
    cases = [case("a"), case("b"), case("c")]
    one_ok = {"a": ok([1]), "b": refused(), "c": refused()}
    report = parity.compare(cases, one_ok, dict(one_ok))
    assert report["status"] == "FAIL" and report["summary"]["FAIL"] == 0 and report["answered"] == 1
    two_ok = {"a": ok([1]), "b": ok([2]), "c": refused()}
    assert parity.compare(cases, two_ok, dict(two_ok))["status"] == "PASS"


def test_heads_with_a_baseline_engine_proves_the_default_path(tmp_path, monkeypatch):
    # the old binary's default path against the new binary's default path, then each head file against
    # the new default. FAKE_ENGINE_DIVERGE makes the server of that engine differ.
    path = tmp_path / "one.txt"
    path.write_text("1937\n")
    out = tmp_path / "heads.json"
    argv = ["heads", "NEW", "CKPT", "--baseline-engine", "OLD", "--out", str(out), "--port-base",
            str(port_base(3)), "--server-command", f"{sys.executable} {FAKE}", "--blocks", f"one={path}"]
    assert parity.main(argv) == 0
    report = json.loads(out.read_text())
    assert set(report["runs"]) == {"default", "one"} and report["status"] == "PASS"
    assert report["runs"]["default"]["reference"] == "baseline" and report["runs"]["one"]["reference"] == "default"
    assert all(out.with_suffix(f".{tag}.log").exists() for tag in ("baseline", "default", "one"))
    monkeypatch.setenv("FAKE_ENGINE_DIVERGE", "OLD")
    assert parity.main(argv) == 1
    report = json.loads(out.read_text())
    assert report["runs"]["default"]["status"] == "FAIL" and report["runs"]["one"]["status"] == "PASS"
