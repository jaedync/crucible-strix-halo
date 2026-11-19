"""Dependency-free API validation tests. The tokenizer has its own integration tests."""
import unittest
from server.api import validate_request, ApiError

class ApiTests(unittest.TestCase):
    def valid(self, **options):
        return dict(model="crucible", messages=[dict(role="user", content="hello")],
                    max_tokens=8, temperature=0) | options

    def test_greedy_and_defaults(self):
        request = validate_request(self.valid(), "chat", "crucible")
        self.assertEqual(request.max_tokens, 8)
        self.assertFalse(request.enable_thinking)
        self.assertEqual(request.spec, 7)
        self.assertEqual((request.draft_q, request.draft_head, request.mtp_prefix, request.draft_adapt),
                         (0.3, "64k", 256, None))

    def test_server_mtp_window_is_the_default_and_a_client_value_wins(self):
        # serve --mtp-window: 0 is the full MTP window. A request without mtp_prefix gets the server value.
        for window in (0, 256, 4096):
            request = validate_request(self.valid(), "chat", "crucible", default_mtp_prefix=window)
            self.assertEqual(request.mtp_prefix, window)
        request = validate_request(self.valid(mtp_prefix=512), "chat", "crucible", default_mtp_prefix=0)
        self.assertEqual(request.mtp_prefix, 512)
        request = validate_request(dict(prompt="x"), "completion", "crucible", default_mtp_prefix=0)
        self.assertEqual(request.mtp_prefix, 0)

    def test_decode_off_switches(self):
        request = validate_request(self.valid(spec=0, draft_head="full", draft_q=0, mtp_prefix=0, draft_adapt=0),
                                   "chat", "crucible")
        self.assertEqual((request.spec, request.draft_head, request.draft_q, request.mtp_prefix, request.draft_adapt),
                         (0, "full", 0, 0, 0))

    def test_tool_fields_and_decode_controls_coexist(self):
        tool = dict(type="function", function=dict(name="lookup", parameters=dict(type="object")))
        request = validate_request(self.valid(tools=[tool], tool_choice="required", parallel_tool_calls=False,
                                             temperature=0.7, seed=97, spec=3, draft_head="full", draft_q=0,
                                             mtp_prefix=0, draft_adapt=0, cache_control="refresh", stop="END"),
                                   "chat", "crucible")
        self.assertEqual(request.tools[0]["function"]["name"], "lookup")
        self.assertEqual((request.tool_choice, request.parallel_tool_calls, request.seed), ("required", False, 97))
        self.assertEqual((request.cache_control, request.stop), ("refresh", ("END",)))
        self.assertEqual((request.spec, request.draft_head, request.draft_q, request.mtp_prefix, request.draft_adapt),
                         (3, "full", 0, 0, 0))

    def test_decode_policy_limits(self):
        for options in (dict(draft_q=float("nan")), dict(draft_q=1.00000001), dict(draft_q=-1),
                        dict(draft_head="bad"), dict(mtp_prefix=-1), dict(mtp_prefix=262145), dict(mtp_prefix=True),
                        dict(draft_adapt=-1), dict(draft_adapt=True), dict(draft_adapt=None),
                        dict(draft_adapt=1, draft_head="full")):
            with self.assertRaises(ApiError):
                validate_request(self.valid(**options), "chat", "crucible")

    def test_engine_only_draft_fields_are_not_public(self):
        # Prompt lookup and value-aware draft length stay engine experiments.
        for field, value in (("lookup", 7), ("lookup_min", 3), ("draft_value", 1)):
            with self.assertRaises(ApiError) as caught:
                validate_request(self.valid(**{field: value}), "chat", "crucible")
            self.assertEqual(caught.exception.param, field)

    def test_unsupported_sampling_rejected(self):
        for option in (dict(logit_bias={"1": 2}), dict(repetition_penalty=2.5), dict(n=2)):
            with self.assertRaises(ApiError):
                validate_request(self.valid(**option), "chat", "crucible")

    # The penalty fields: server/tests/test_penalties.py.

    def test_bad_content(self):
        for messages in ([], [dict(role="tool", content="x")],
                         [dict(role="user", content=[dict(type="image_url", image_url="x")])]):
            with self.assertRaises(ApiError):
                validate_request(self.valid(messages=messages), "chat", "crucible")

    def test_bad_scalars(self):
        for option in (dict(max_tokens=0), dict(max_tokens=True), dict(stream="true"),
                       dict(spec=8), dict(enable_thinking="false"), dict(temperature=float("nan"))):
            with self.assertRaises(ApiError):
                validate_request(self.valid(**option), "chat", "crucible")

    def test_wrong_model(self):
        with self.assertRaises(ApiError):
            validate_request(self.valid(model="other"), "chat", "crucible")

    def test_completion(self):
        request = validate_request(dict(model="crucible", prompt="hello", max_tokens=8),
                                   "completion", "crucible")
        self.assertEqual(request.prompt, "hello")

if __name__ == "__main__":
    unittest.main()
