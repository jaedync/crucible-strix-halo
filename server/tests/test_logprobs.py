"""The opt-in scoring position: API validation, the engine payload, and the response field."""
import unittest
from types import SimpleNamespace
from server.api import validate_request, ApiError
from server.response import scoring_field

class LogprobsApiTests(unittest.TestCase):
    def body(self, **options):
        return dict(model="crucible", prompt=[1, 2, 3], max_tokens=1, temperature=0, logprobs=True,
                    top_logprobs=5) | options

    def test_default_is_off(self):
        request = validate_request(self.body(logprobs=None, top_logprobs=None), "completion", "crucible")
        self.assertEqual(request.top_logprobs, 0)

    def test_count_and_limits(self):
        self.assertEqual(validate_request(self.body(), "completion", "crucible").top_logprobs, 5)
        self.assertEqual(validate_request(self.body(top_logprobs=0), "completion", "crucible").top_logprobs, 1)
        for bad in (dict(top_logprobs=21), dict(top_logprobs=-1), dict(max_tokens=2), dict(stream=True),
                    dict(temperature=0.5), dict(top_logprobs=True)):
            with self.assertRaises(ApiError, msg=str(bad)):
                validate_request(self.body(**bad), "completion", "crucible")
        with self.assertRaises(ApiError):
            validate_request(dict(model="crucible", messages=[dict(role="user", content="x")], max_tokens=1,
                                  logprobs=True), "chat", "crucible")

    def test_response_field(self):
        engine = [[7, -0.25], [9, -2.5]]
        field = scoring_field(SimpleNamespace(top_logprobs=2), 7, engine)
        self.assertEqual(field["content"][0]["token_id"], 7)
        self.assertEqual(field["content"][0]["logprob"], -0.25)
        self.assertEqual(field["content"][0]["top_logprobs"], [dict(token_id=7, logprob=-0.25),
                                                               dict(token_id=9, logprob=-2.5)])
        for bad in (None, [], [[7]], [[7, float("nan")]], [["a", -1.0]], [[7, -1.0]] * 21):
            with self.assertRaises(ApiError):
                scoring_field(SimpleNamespace(top_logprobs=2), 7, bad)

if __name__ == "__main__":
    unittest.main()
