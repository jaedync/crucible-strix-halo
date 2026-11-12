"""Per-request ablation: one engine process serves stock and ablated requests.

The server selects the mode with the request's model name (`<served id>-abliterated`, or
`<model name>-abliterated` when serve --model-name names one) or with the body field "crucible_ablate": true.
It sends "ablate": 1 in every engine request object of that turn (open, stage, the serial line) and leaves a stock
request byte-identical, so every stock cache and disk key stays as it is today. On a stock engine the ablated forms
are a clear 400, never a silent stock answer. The engine side is the per-request abliteration design.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from server import ablation
from server.api import ApiError, validate_request
from server.backend import EngineBackend, engine_environment
from server.batch_backend import BatchedEngineBackend
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, invoke, token_events

TESTS = Path(__file__).parent
SERVED = "crucible"
ALIAS = "qwen3.8-flash-next"
NAMES = ablation.ablated_names(SERVED, ALIAS)
# The think-block tokens of the fixture below: END closes the block, SENT is the answer's first text.
END, SENT = 100007, 100006


class BlockTokenizer(PiecesTokenizer):
    """A tokenizer whose END token closes the think block, so the server resumes the turn on a second job."""

    def __init__(self):
        super().__init__([])

    def encode(self, request):
        # The fake engine counts up from the prompt's last ID: these IDs never reach END, so the block never closes by
        # itself and the server closes it at the think cap, which starts the continuation job.
        return [10]

    def stream(self):
        return None

    def step(self, state, token):
        return "</think>" if token == END else "\n\nTime.\n" if token == SENT else "r "

    def decode(self, ids):
        return "".join(self.step(None, token) for token in ids)

    def think_close_ids(self):
        return (SENT, END, 100008)


def ablated(body, available=True, names=NAMES):
    return validate_request(body, "completion", SERVED, ablated_names=names, ablate_available=available)


class SelectionTests(unittest.TestCase):
    """The two accepted forms, and the refusal of everything else."""

    def test_the_served_ablated_name_selects_the_ablated_mode(self):
        request = ablated(dict(model=f"{SERVED}-abliterated", prompt="x"))
        self.assertTrue(request.ablate)
        self.assertEqual(request.model, f"{SERVED}-abliterated")

    def test_the_model_name_abhiterated_form_selects_the_ablated_mode(self):
        request = ablated(dict(model=f"{ALIAS}-abliterated", prompt="x"))
        self.assertTrue(request.ablate)

    def test_the_body_field_selects_the_ablated_mode(self):
        self.assertTrue(ablated(dict(prompt="x", crucible_ablate=True)).ablate)

    def test_both_forms_together_select_the_ablated_mode(self):
        self.assertTrue(ablated(dict(model=f"{SERVED}-abliterated", prompt="x", crucible_ablate=True)).ablate)

    def test_the_served_name_and_no_model_are_stock(self):
        self.assertFalse(ablated(dict(model=SERVED, prompt="x")).ablate)
        self.assertFalse(ablated(dict(prompt="x")).ablate)
        self.assertFalse(ablated(dict(prompt="x", crucible_ablate=False)).ablate)

    def test_a_false_field_with_an_ablated_name_is_an_error(self):
        for name in NAMES:
            with self.subTest(name=name), self.assertRaises(ApiError) as caught:
                ablated(dict(model=name, prompt="x", crucible_ablate=False))
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(caught.exception.param, "crucible_ablate")

    def test_the_field_must_be_a_boolean(self):
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ApiError) as caught:
                ablated(dict(prompt="x", crucible_ablate=value))
            self.assertEqual(caught.exception.param, "crucible_ablate")

    def test_an_unknown_model_name_is_still_a_404(self):
        for name in ("other", "qwen3.8-flash-next", f"{SERVED}-ablated"):
            with self.subTest(name=name), self.assertRaises(ApiError) as caught:
                ablated(dict(model=name, prompt="x"))
            self.assertEqual(caught.exception.status, 404)

    def test_an_ablated_request_on_a_stock_engine_is_a_clear_400(self):
        for body in (dict(model=f"{SERVED}-abliterated", prompt="x"),
                     dict(model=f"{ALIAS}-abliterated", prompt="x"),
                     dict(prompt="x", crucible_ablate=True)):
            with self.subTest(body=body), self.assertRaises(ApiError) as caught:
                ablated(body, available=False)
            self.assertEqual(caught.exception.status, 400)
            self.assertIn("ablated", str(caught.exception))

    def test_the_names_hold_the_served_id_and_the_model_name(self):
        self.assertEqual(NAMES, (f"{SERVED}-abliterated", f"{ALIAS}-abliterated"))
        # No --model-name, or a model name equal to the served id: one name only.
        self.assertEqual(ablation.ablated_names(SERVED), (f"{SERVED}-abliterated",))
        self.assertEqual(ablation.ablated_names(SERVED, SERVED), (f"{SERVED}-abliterated",))


class EnvironmentTests(unittest.TestCase):
    """The daily driver sets the engine's switches, and serve passes only the allow-listed names on."""

    def test_the_request_mode_switches_reach_the_engine(self):
        kept, removed = engine_environment({"CRUCIBLE_ABLATE": "/models/ablate.bin", "CRUCIBLE_ABLATE_MODE": "request",
                                            "PATH": "/bin"})
        self.assertEqual(kept.get("CRUCIBLE_ABLATE"), "/models/ablate.bin")
        self.assertEqual(kept.get("CRUCIBLE_ABLATE_MODE"), "request")
        self.assertEqual(removed, [])


class ModelListTests(unittest.TestCase):
    """/v1/models names the ablated model only when the engine can serve it."""

    def list_models(self, available):
        backend = ScriptBackend(token_events(["a"]))
        backend.ablate_request = available
        response = invoke(backend, PiecesTokenizer(["a"]), path="/v1/models", method="GET", raw=b"",
                          service_options=dict(model_name=ALIAS))
        self.assertEqual(response.statuses, [200])
        return [entry["id"] for entry in json.loads(response.body)["data"]]

    def test_the_ablated_names_are_listed_when_the_mode_is_available(self):
        self.assertEqual(self.list_models(True), [SERVED, f"{SERVED}-abliterated", f"{ALIAS}-abliterated"])

    def test_the_ablated_names_are_absent_when_the_mode_is_off(self):
        self.assertEqual(self.list_models(False), [SERVED])


class ServeTests(unittest.TestCase):
    """The HTTP path: the field selects the mode, and a stock engine refuses it with a 4xx."""

    def ask(self, available, body):
        backend = ScriptBackend(token_events(["a"]))
        backend.ablate_request = available
        return invoke(backend, PiecesTokenizer(["a"]), chat(**body)), backend

    def test_the_field_selects_the_mode_on_the_chat_path(self):
        response, backend = self.ask(True, dict(crucible_ablate=True))
        self.assertEqual(response.statuses, [200])
        self.assertTrue(backend.submissions[0][1].ablate)

    def test_a_stock_request_stays_stock(self):
        response, backend = self.ask(True, {})
        self.assertEqual(response.statuses, [200])
        self.assertFalse(backend.submissions[0][1].ablate)

    def test_a_stock_engine_refuses_the_field(self):
        response, backend = self.ask(False, dict(crucible_ablate=True))
        self.assertEqual(response.statuses, [400])
        self.assertEqual(backend.submissions, [])

    def test_the_completions_route_selects_the_mode_too(self):
        for body, expected in ((dict(model=f"{SERVED}-abliterated", prompt="x"), True),
                               (dict(prompt="x", crucible_ablate=True), True),
                               (dict(model=SERVED, prompt="x"), False)):
            with self.subTest(body=body):
                backend = ScriptBackend(token_events(["a"]))
                backend.ablate_request = True
                response = invoke(backend, PiecesTokenizer(["a"]), body, path="/v1/completions")
                self.assertEqual(response.statuses, [200], response.body)
                self.assertIs(backend.submissions[0][1].ablate, expected)

    def test_the_completions_route_refuses_the_mode_on_a_stock_engine(self):
        for body in (dict(model=f"{SERVED}-abliterated", prompt="x"), dict(prompt="x", crucible_ablate=True)):
            with self.subTest(body=body):
                backend = ScriptBackend(token_events(["a"]))
                backend.ablate_request = False
                response = invoke(backend, PiecesTokenizer(["a"]), body, path="/v1/completions")
                self.assertEqual(response.statuses, [400])
                self.assertEqual(backend.submissions, [])


class PayloadTests(unittest.TestCase):
    """The engine keys: a stock payload keeps today's bytes, an ablated one adds the mode only."""

    # Today's stock payload for this request (probed before the change): the golden key of the serial path.
    GOLDEN = {"ids": [1], "max_tokens": 4, "spec": 0, "stop_ids": [248046, 248044], "temperature": 0.0, "top_k": 0,
              "top_p": 1.0, "min_p": 0.0, "seed": 1, "cache_control": "auto", "cache_prefix_tokens": 0,
              "draft_q": 0.3, "draft_head": "64k", "mtp_prefix": 256}

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "requests.jsonl"
        popen = subprocess.Popen
        with patch.dict(os.environ, dict(FAKE_REQUEST_LOG=str(self.log), FAKE_DELAY="0", FAKE_ABLATE_REQUEST="1")), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(TESTS / "fake_engine.py")], **kwargs)):
            self.backend = EngineBackend("unused", "unused", context=32)

    def tearDown(self):
        self.backend.close()
        self.dir.cleanup()

    def payload(self, body):
        request = validate_request(dict(prompt="x", max_tokens=4, seed=1, spec=0, **body), "completion", SERVED,
                                   ablated_names=NAMES, ablate_available=True)
        job = self.backend.submit([1], request)
        while job.events.get(timeout=5)["event"] != "done":
            pass
        return json.loads(self.log.read_text().splitlines()[-1])

    def test_the_stock_payload_is_the_golden_key(self):
        self.assertEqual(self.payload({}), self.GOLDEN)

    def test_the_ablated_payload_adds_only_the_mode(self):
        self.assertEqual(self.payload(dict(crucible_ablate=True)), self.GOLDEN | {"ablate": 1})

    def test_the_stock_payload_has_no_mode_or_tenant_key(self):
        payload = self.payload({})
        self.assertNotIn("ablate", payload)
        self.assertNotIn("tenant", payload)


class BatchFixture(unittest.TestCase):
    """The batch fixture: a fake engine, its command log, and the backend that talks to it."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "commands.jsonl"

    def tearDown(self):
        self.dir.cleanup()
        for backend in getattr(self, "backends", []):
            backend.close()

    def start(self, **env):
        self.backends = getattr(self, "backends", [])
        popen = subprocess.Popen
        values = dict(os.environ, FAKE_DELAY="0", FAKE_COMMAND_LOG=str(self.log), FAKE_ABLATE_REQUEST="1",
                      **{"FAKE_SLOT_CONTEXTS": "40000,40000", **env})
        with patch.dict(os.environ, values), \
                patch("server.backend.subprocess.Popen",
                      side_effect=lambda *args, **kwargs: popen([sys.executable, str(TESTS / "fake_batch_engine.py")],
                                                                **kwargs)):
            return BatchedEngineBackend("unused", "unused", **self.options)

    @property
    def options(self):
        return dict(context=40000, streams=2)

    def run_job(self, backend, ids, body):
        request = validate_request(dict(prompt="x", max_tokens=4, spec=0, **body), "completion", SERVED,
                                   ablated_names=NAMES, ablate_available=True)
        job = backend.submit(ids, request)
        while True:
            event = job.events.get(timeout=20)
            self.assertNotEqual(event["event"], "error", event)
            if event["event"] == "done":
                return job

    def requests(self, op):
        return [line["request"] for line in map(json.loads, self.log.read_text().splitlines())
                if line["op"] == op and "request" in line]


class BatchKeyTests(BatchFixture):
    """The open of a turn, and the second job of a capped thinking turn, carry the mode."""

    def test_the_open_carries_the_mode(self):
        backend = self.start()
        self.run_job(backend, [1, 2, 3], dict(crucible_ablate=True))
        opens = self.requests("open")
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0]["ablate"], 1)

    def test_a_stock_open_carries_no_mode_key(self):
        backend = self.start()
        self.run_job(backend, [1, 2, 3], {})
        self.assertNotIn("ablate", self.requests("open")[0])

    def test_a_think_cap_continuation_carries_the_mode(self):
        # The server closes the block at the cap and resumes the turn on a second job (http.close_thinking).
        backend = self.start()
        body = dict(messages=[dict(role="user", content="hi")], enable_thinking=True, max_tokens=1200,
                    spec=0, crucible_ablate=True)
        response = invoke(backend, BlockTokenizer(), body)
        self.assertEqual(response.statuses, [200], response.body)
        opens = self.requests("open")
        self.assertGreaterEqual(len(opens), 2)
        self.assertEqual([payload.get("ablate") for payload in opens], [1] * len(opens))


class StateDiskKeyTests(BatchFixture):
    """The disk tier reads a state before an open: the stage of an ablated turn must use the ablated key too."""

    @property
    def options(self):
        return dict(context=40000, streams=2, conversation_cache_mib=64, session_replay=True,
                    state_dir=str(Path(self.dir.name) / "states"))

    def long_ids(self):
        return list(range(1000, 1000 + 20000))

    def test_the_stage_carries_the_mode(self):
        backend = self.start(FAKE_STATE_DISK="1", FAKE_DISK_FILE=str(Path(self.dir.name) / "disk.json"),
                             FAKE_STAGE_READS="1")
        self.run_job(backend, self.long_ids(), dict(crucible_ablate=True))
        stages = self.requests("stage")
        self.assertTrue(stages)
        self.assertEqual([request.get("ablate") for request in stages], [1] * len(stages))

    def test_a_stock_stage_carries_no_mode_key(self):
        backend = self.start(FAKE_STATE_DISK="1", FAKE_DISK_FILE=str(Path(self.dir.name) / "disk.json"),
                             FAKE_STAGE_READS="1")
        self.run_job(backend, self.long_ids(), {})
        stages = self.requests("stage")
        self.assertTrue(stages)
        self.assertNotIn("ablate", stages[0])


if __name__ == "__main__":
    unittest.main()
