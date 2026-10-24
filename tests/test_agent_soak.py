"""CPU tests for tools/agent_soak.py: the soak rule, synthetic content, and stream parsing (no server)."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import agent_soak as soak

# The long session of the Oct 5 incident (the logged numbers: prompt, cached), the served build, replay off.
INCIDENT = [(12834, 0), (17510, 12288), (22749, 16384), (23468, 20480), (24994, 20480), (25558, 24576),
            (34299, 32768), (54752, 32768), (74721, 53248), (94222, 53248), (113374, 53248), (219071, 53248)]


def turns(pairs, completion=300, session="long"):
    return [dict(session=session, turn=i, prompt_tokens=p, cached_tokens=c, completion_tokens=completion)
            for i, (p, c) in enumerate(pairs)]


class RuleTest(unittest.TestCase):
    def test_incident_fails_from_94222(self):
        report = soak.check(turns(INCIDENT))
        self.assertEqual(report["verdict"], "FAIL")
        first = report["failures"][0]
        self.assertEqual((first["prompt_tokens"], first["prefilled"], first["allowed"]), (94222, 40974, 19501 + 4096))

    def test_fixed_pattern_passes(self):
        fixed = [(p, 0 if i == 0 else (INCIDENT[i - 1][0] - 4) // 4096 * 4096) for i, (p, _) in enumerate(INCIDENT)]
        self.assertEqual(soak.check(turns(fixed))["verdict"], "PASS")

    def test_sessions_are_independent(self):
        rows = turns([(12000, 0), (30000, 8192)], session="a") + turns([(12000, 0), (13000, 8192)], session="b")
        report = soak.check(rows)
        self.assertEqual(report["verdict"], "PASS")  # b re-prefills 4808 of 1000 appended + 4096
        self.assertEqual(report["sessions"], {"a": 2, "b": 2})

    def test_replay_rule(self):
        # The previous turn ended at 20000 + 300 generated; a replay hit restores 20299 of the next 21000.
        hit = turns([(20000, 0), (21000, 20299)])
        self.assertEqual(soak.check(hit, expect_replay=True)["verdict"], "PASS")
        frontier = turns([(20000, 0), (21000, 16384)])
        self.assertEqual(soak.check(frontier)["verdict"], "PASS")
        report = soak.check(frontier, expect_replay=True)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["failures"][0]["rule"], "replay")

    def test_cold_first_turn_may_omit_the_cache_count(self):
        # Halogen 0.16.2 omits cached_tokens on a cold first turn; a later turn without it is still an error.
        rows = turns([(20000, 0), (21000, 20000)])
        rows[0]["cached_tokens"] = None
        self.assertEqual(soak.check(rows)["verdict"], "PASS")

    def test_server_error_is_named(self):
        rows = turns([(20000, 0), (21000, 20000)])
        rows[1].update(prompt_tokens=None, cached_tokens=None, completion_tokens=None, error="Malformed model tool call")
        report = soak.check(rows)
        self.assertEqual(report["verdict"], "ERROR")
        self.assertIn("Malformed model tool call", report["error"])

    def test_missing_cache_count_is_an_error(self):
        rows = turns([(20000, 0), (21000, 0)])
        rows[1]["cached_tokens"] = None
        self.assertEqual(soak.check(rows)["verdict"], "ERROR")


class ContentTest(unittest.TestCase):
    def test_deterministic_and_sized(self):
        a, b = soak.tool_result(7, 2000), soak.tool_result(7, 2000)
        self.assertEqual(a, b)
        self.assertNotEqual(a, soak.tool_result(8, 2000))
        self.assertAlmostEqual(len(a), 2000 * soak.CHARS_PER_TOKEN, delta=200)

    def test_tools_and_system(self):
        tools = soak.tool_specs()
        self.assertEqual(len(tools), 23)
        self.assertEqual(len({tool["function"]["name"] for tool in tools}), 23)
        self.assertGreater(len(soak.system_prompt(10000)), 9000 * soak.CHARS_PER_TOKEN)

    def test_schedule_reaches_the_target(self):
        sizes = soak.schedule("long", 220000, 12000)
        self.assertGreaterEqual(12000 + sum(sizes), 200000)
        self.assertTrue(all(500 <= size <= 25000 for size in sizes))
        self.assertLess(12000 + sum(soak.schedule("short", 40000, 12000)), 45000)


class StreamTest(unittest.TestCase):
    def test_stable_ids_and_hash(self):
        reply = dict(content="c", reasoning="", tool_calls=[dict(id="call_9f", type="function",
                                                                  function=dict(name="read", arguments="{}"))])
        fixed = soak.stable_call_ids(reply, "agent-0_3")
        self.assertEqual(fixed["tool_calls"][0]["id"], "call_agent-0_3_0")
        self.assertEqual(reply["tool_calls"][0]["id"], "call_9f")  # the reply itself is unchanged
        self.assertEqual(soak.content_hash(reply), soak.content_hash(fixed))

    def test_agent_schedule(self):
        sizes = soak.schedule("agent", 44000, 9000 + 24000)
        self.assertTrue(sizes and max(sizes) < 10000 and 9000 + 24000 + sum(sizes) <= 44000)
        self.assertEqual(soak.AGENT_SCENARIOS["agents"][0], 4)
        self.assertIn("--streams 4 --ctx 131072", soak.AGENT_SCENARIOS["agents"][1])
        self.assertEqual(soak.AGENT_SCENARIOS["served3"], (3, soak.SERVED))

    def test_agents6_runs_six_agent_sessions_with_the_agents_settings(self):
        # the agents sessions, six at once; only the slot count of the default layout differs.
        count, layout = soak.AGENT_SCENARIOS["agents6"]
        self.assertEqual(count, 6)
        self.assertEqual(layout.replace("--streams 6 ", "--streams 4 "), soak.AGENT_SCENARIOS["agents"][1])
        seen = []

        def fake_session(name, kind, args, model, rows, lock):
            seen.append((name, kind, args.thinking, args.reasoning_history, args.max_tokens, model))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a6.json"
            argv = ["agent_soak.py", "agents6", str(out), "--url", "http://127.0.0.1:9", "--model", "m"]
            with mock.patch.object(soak, "run_session", fake_session), mock.patch.object(soak.time, "sleep"), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(soak.main(), 0)
            self.assertEqual(json.loads(out.read_text())["scenario"], "agents6")
        self.assertEqual(sorted(seen), [(f"agent-{i}", "agent", "on", "drop", 768, "m") for i in range(6)])

    def test_parse_stream(self):
        chunks = [dict(choices=[dict(delta=dict(reasoning_content="think "))]),
                  dict(choices=[dict(delta=dict(reasoning_content="more"))]),
                  dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id="c1", type="function",
                                                                 function=dict(name="read", arguments='{"pa'))]))]),
                  dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, function=dict(arguments='th":"a"}'))]))]),
                  dict(choices=[dict(delta={}, finish_reason="tool_calls")]),
                  dict(choices=[], usage=dict(prompt_tokens=100, completion_tokens=9,
                                              prompt_tokens_details=dict(cached_tokens=64)))]
        lines = [b"data: " + json.dumps(chunk).encode() for chunk in chunks] + [b"", b"data: [DONE]"]
        reply = soak.parse_stream(iter(lines))
        self.assertEqual(reply["reasoning"], "think more")
        self.assertEqual(reply["tool_calls"], [dict(id="c1", type="function",
                                                     function=dict(name="read", arguments='{"path":"a"}'))])
        self.assertEqual((reply["prompt_tokens"], reply["cached_tokens"], reply["completion_tokens"]), (100, 64, 9))
        self.assertEqual(reply["finish"], "tool_calls")

    def test_error_chunk(self):
        # Crucible sends an error chunk after HTTP 200 when the reply fails late (soak run Oct 5, turn 4).
        lines = [b'data: {"choices":[{"delta":{"content":"x"}}]}', b'',
                 b'data: {"error":{"message":"Malformed model tool call: x","type":"server_error"}}', b'',
                 b"data: [DONE]"]
        reply = soak.parse_stream(iter(lines))
        self.assertEqual(reply["error"], "Malformed model tool call: x")
        self.assertIsNone(reply["prompt_tokens"])

    def test_request_body(self):
        args = SimpleNamespace(max_tokens=384, thinking="off", reasoning_effort="high", thinking_budget=None)
        body = soak.request_body("m", [], [], args)
        self.assertEqual(body["chat_template_kwargs"], dict(enable_thinking=False))
        self.assertTrue(body["stream"] and body["stream_options"]["include_usage"])
        self.assertNotIn("reasoning_effort", body)
        body = soak.request_body("m", [], [], SimpleNamespace(max_tokens=None, thinking="on", reasoning_effort="high",
                                                               thinking_budget=128))
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("chat_template_kwargs", body)
        # Pi's max_tokens: a small one makes the answer room close the think block after one token.
        self.assertEqual((body["max_tokens"], body["max_thinking_tokens"]), (32768, 128))

    def test_next_messages(self):
        reply = dict(content="", reasoning="r", tool_calls=[dict(id="c1", type="function",
                                                                  function=dict(name="read", arguments="{}"))])
        added = soak.next_messages(reply, "RESULT")
        self.assertEqual(added[0]["role"], "assistant")
        self.assertEqual(added[0]["reasoning_content"], "r")
        self.assertEqual(added[1], dict(role="tool", tool_call_id="c1", content="RESULT"))
        # agents/served3: the client drops the reasoning, so a think turn's tail never repeats (Oct 6 failure).
        self.assertNotIn("reasoning_content", soak.next_messages(reply, "RESULT", keep_reasoning=False)[0])
        plain = soak.next_messages(dict(content="done", reasoning="", tool_calls=[]), "RESULT")
        self.assertEqual([m["role"] for m in plain], ["assistant", "user"])
        self.assertIn("RESULT", plain[1]["content"])


class EnvironmentTest(unittest.TestCase):
    def test_the_report_records_the_switch_environment(self):
        # Train 19: the gate checks that the short soaks ran with CRUCIBLE_PF_LAUNCH_ROWS=4096.
        env = {"CRUCIBLE_PF_LAUNCH_ROWS": "4096", "CRUCIBLE_MTP_SIDECAR": "/x", "PATH": "/bin"}
        self.assertEqual(soak.switch_environment(env),
                         {"CRUCIBLE_PF_LAUNCH_ROWS": "4096", "CRUCIBLE_MTP_SIDECAR": "/x"})
        source = Path(soak.__file__).read_text()
        self.assertIn("environment=switch_environment(os.environ)", source)

    def test_the_report_records_the_sha256_of_the_engine_it_started(self):
        # Review t19s1b: a report from another build at the same path must not pass the gate.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "crucible-t19"
            engine.write_bytes(b"binary")
            self.assertEqual(soak.file_sha256(engine), hashlib.sha256(b"binary").hexdigest())
            self.assertIsNone(soak.file_sha256(None))
            self.assertIsNone(soak.file_sha256(Path(tmp) / "missing"))
        source = Path(soak.__file__).read_text()
        self.assertIn("engine_sha256=file_sha256(args.engine)", source)

    def test_the_report_records_the_checkpoint_it_served(self):
        # gating_check.py --candidate compares the pack, streams and cache of the soak with the env file.
        source = Path(soak.__file__).read_text()
        self.assertIn('checkpoint=str(args.checkpoint or "") if args.tree else None', source)


class R2Test(unittest.TestCase):
    """the image turn before the sessions and the MemAvailable floor."""

    def meminfo(self, tmp, kb):
        path = Path(tmp) / "meminfo"
        path.write_text(f"MemTotal: 131072000 kB\nMemAvailable: {kb} kB\n" if kb is not None else "MemTotal: 1 kB\n")
        return path

    def test_memavailable_is_read_in_kb_or_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(soak.memavailable_kb(self.meminfo(tmp, 12345678)), 12345678)
            self.assertIsNone(soak.memavailable_kb(self.meminfo(tmp, None)))
            self.assertIsNone(soak.memavailable_kb(Path(tmp) / "missing"))

    def test_the_sampler_keeps_the_lowest_value(self):
        # no timing windows. run() reads the given values in order; the read after the last stops it.
        values = iter([20000000, None, 15000000, 18000000])
        sampler = None

        def read(path):
            value = next(values, "end")
            if value == "end":
                sampler.stopped.set()
                return None
            return value

        sampler = soak.MemorySampler(interval=0, read=read)
        sampler.run()
        self.assertEqual((sampler.lowest_kb, sampler.samples), (15000000, 3))
        started = soak.MemorySampler(interval=60, read=lambda path: 1000)
        started.start()
        started.stop()  # the first read comes before the first wait, so a started sampler has one sample
        self.assertEqual((started.samples, started.is_alive()), (1, False))
        soak.MemorySampler(read=read).stop()  # never started (the server did not start): no error

    LOG = ("conversation cache: 26843545600 bytes budget, host reserve 8192 MiB\n"
           "slot checkpoints: 3 per slot, 152055808 bytes each, 1824669696 bytes from the cache budget\n"
           'crucible_turn {"request_id": 1, "cache_bytes": 6357545336, "host_free_mib": 30000}\n'
           'crucible_turn {"request_id": 2, "cache_bytes": 20610000000, "host_free_mib": 18000}\n'
           'crucible_turn {"request_id": 3, "cache_bytes": 13000000000}\n')

    def test_the_cache_record_comes_from_the_server_log(self):
        self.assertEqual(soak.cache_record(self.LOG), dict(budget_bytes=26843545600, checkpoint_bytes=1824669696,
                                                           high_water_bytes=20610000000))
        self.assertIsNone(soak.cache_record(self.LOG.split("crucible_turn")[0]))  # no turn: no high-water
        self.assertIsNone(soak.cache_record("crucible_turn {\"cache_bytes\": 1}\n"))  # no budget line
        no_checkpoints = "\n".join(line for line in self.LOG.splitlines() if not line.startswith("slot"))
        self.assertEqual(soak.cache_record(no_checkpoints)["checkpoint_bytes"], 0)

    def test_the_floor_judges_the_lowest_value_projected_to_a_full_cache(self):
        # deep305 had 18,108 MiB at a 20.61 GB high-water of a 25,600 MiB budget (1.82 GB checkpoints).
        passed = soak.check(turns([(20000, 0), (21000, 20000)]))
        cache = soak.cache_record(self.LOG)
        unfilled = (26843545600 - 1824669696 - 20610000000) >> 20
        self.assertEqual(unfilled, 4204)
        ok = soak.apply_memory_floor(passed, 18108 * 1024, 10, 12288, cache)
        self.assertEqual((ok["verdict"], ok["memory"]["projected_memavailable_mib"]), ("PASS", 18108 - 4204))
        self.assertEqual(ok["memory"]["high_water_bytes"], 20610000000)
        # Above the floor as measured, under it at a full cache: FAIL.
        low = soak.apply_memory_floor(passed, 15000 * 1024, 10, 12288, cache)
        self.assertEqual(low["verdict"], "FAIL")
        self.assertEqual(low["failures"][-1], dict(rule="memory", lowest_memavailable_mib=15000, unfilled_mib=4204,
                                                   projected_memavailable_mib=15000 - 4204, floor_mib=12288))
        full = dict(cache, high_water_bytes=cache["budget_bytes"])  # over the budget: nothing to project
        self.assertEqual(soak.apply_memory_floor(passed, 12288 * 1024, 10, 12288, full)["verdict"], "PASS")
        self.assertEqual(soak.apply_memory_floor(passed, 12287 * 1024, 10, 12288, full)["verdict"], "FAIL")
        self.assertEqual(soak.apply_memory_floor(passed, None, 0, 12288, cache)["verdict"], "ERROR")
        none = soak.apply_memory_floor(passed, 20000 * 1024, 10, 12288, None)
        self.assertEqual((none["verdict"], none["error"]), ("ERROR", "no cache record in the server log"))
        self.assertEqual(soak.apply_memory_floor(passed, 1024, 5, None)["verdict"], "PASS")  # a record only
        failed = dict(passed, verdict="FAIL")
        self.assertEqual(soak.apply_memory_floor(failed, 1024, 5, 12288, cache)["failures"], [])  # the first cause stays

    def test_the_pinned_pool_bytes_replace_the_entry_high_water(self):
        # frontier-copy: the engine keeps freed and expected-tail chunks
        # pinned inside the budget, and they already lower MemAvailable. cache_pinned_bytes counts every pinned chunk
        # once, so the rule uses its maximum instead of the entry bytes.
        log = (self.LOG.replace('"cache_bytes": 20610000000,', '"cache_bytes": 20610000000, "cache_pinned_bytes": 22000000000,')
               .replace('"cache_bytes": 13000000000}', '"cache_bytes": 13000000000, "cache_pinned_bytes": 24000000000}'))
        record = soak.cache_record(log)
        self.assertEqual(record, dict(budget_bytes=26843545600, checkpoint_bytes=1824669696,
                                      high_water_bytes=24000000000, pinned_counted=True))
        self.assertNotIn("pinned_counted", soak.cache_record(self.LOG))  # an old log keeps today's rule and record
        passed = soak.check(turns([(20000, 0), (21000, 20000)]))
        memory = soak.apply_memory_floor(passed, 18108 * 1024, 10, 12288, record)["memory"]
        self.assertEqual(memory["unfilled_mib"], (26843545600 - 1824669696 - 24000000000) >> 20)
        self.assertIs(memory["pinned_counted"], True)
        # entries that share chunks count each chunk at full size, so cache_bytes can be more
        # than the pinned memory. The pinned bytes decide, so the shared chunks do not raise the projection.
        shared = log.replace('"cache_bytes": 13000000000,', '"cache_bytes": 25000000000,')
        self.assertEqual(soak.cache_record(shared)["high_water_bytes"], 24000000000)

    def test_the_a4v_budgets_count_the_pinned_pool_once(self):
        # a4v reruns: one fill at budgets of 11,100 and 10,400 MiB. The pool keeps chunks pinned up to the budget, so
        # the lowest MemAvailable already holds the 700 MiB difference. With the pinned bytes the projections differ
        # by that 700 MiB; the old rule (entry bytes) takes the unfilled budget off again, so they differ by 1,400.
        mib = 1 << 20
        checkpoints, live, other = 1824669696, 6000 * mib, 30000

        def run(budget_mib, pinned):
            fill = budget_mib * mib - checkpoints
            lines = (f"conversation cache: {budget_mib * mib} bytes budget, host reserve 8192 MiB\n"
                     f"slot checkpoints: 3 per slot, 152055808 bytes each, {checkpoints} bytes from the cache budget\n"
                     f'crucible_turn {{"request_id": 1, "cache_bytes": {live}' +
                     (f', "cache_pinned_bytes": {fill}' if pinned else "") + "}\n")
            report = soak.apply_memory_floor(soak.check(turns([(20000, 0)])), (other - fill // mib) * 1024, 10, 12288,
                                             soak.cache_record(lines))
            return report["memory"]["projected_memavailable_mib"]

        self.assertLessEqual(abs(run(10400, True) - run(11100, True) - 700), 1)
        self.assertLessEqual(abs(run(10400, False) - run(11100, False) - 1400), 1)

    def test_image_size_needs_width_x_height(self):
        self.assertEqual(soak.image_size("2560x1440"), "2560x1440")
        for bad in ("2560", "0x10", "ax1", "10x"):
            with self.assertRaises(Exception):
                soak.image_size(bad)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is needed for the image request")
    def test_image_first_sends_one_cold_image_request(self):
        import http.server
        import threading
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps(dict(usage=dict(prompt_tokens=3630))).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            record = soak.image_first(f"http://127.0.0.1:{server.server_address[1]}", "crucible", "256x256", 30)
        finally:
            server.shutdown()
        self.assertEqual((record["status"], record["prompt_tokens"], record["size"]), (200, 3630, "256x256"))
        ((path, body),) = seen
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((body["cache_control"], body["max_tokens"], body["stream"]), ("bypass", 1, False))
        parts = body["messages"][0]["content"]
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_main_wires_the_options_into_the_report(self):
        source = Path(soak.__file__).read_text()
        self.assertIn("first = image_first(args.url, model, args.image_first, args.request_timeout)", source)
        self.assertIn("report = apply_memory_floor(report, sampler.lowest_kb, sampler.samples, args.memory_floor_mib, cache)",
                      source)
        self.assertIn('cache = cache_record(args.output.with_suffix(".server.log").read_text(', source)


if __name__ == "__main__":
    unittest.main()
