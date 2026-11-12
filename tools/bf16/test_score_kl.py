"""CPU tests for tools/bf16/score_kl.py: the 21-bucket KL, the self-score zero and the lower bound (task 114)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_kl  # noqa: E402


def log_softmax(logits):
    z = np.asarray(logits, np.float64)
    return z - (z.max() + np.log(np.exp(z - z.max()).sum()))


def test_kl_matches_a_hand_computed_value():
    # P over 5 ids, Q gives 2 of them: the 21-bucket KL is written out by hand.
    p_row = np.log(np.array([0.5, 0.25, 0.125, 0.0625, 0.0625]))
    q_ids = np.array([0, 2])
    q_logp = np.log(np.array([0.6, 0.1]))
    got = score_kl.row_kl(p_row, q_ids, q_logp)
    want = (0.5 * math.log(0.5 / 0.6) + 0.125 * math.log(0.125 / 0.1) + 0.375 * math.log(0.375 / 0.3))
    assert got["kl"] == pytest.approx(want, abs=1e-12)
    assert got["q_rest"] == pytest.approx(0.3, abs=1e-12)
    assert got["p_rest"] == pytest.approx(0.375, abs=1e-12)
    assert not got["q_rest_clamped"]


def test_reference_against_its_own_top20_is_zero():
    p_row = log_softmax(np.linspace(-3.0, 3.0, 40) ** 2)
    top = np.argsort(-p_row)[:20]
    got = score_kl.row_kl(p_row, top, p_row[top])
    assert got["kl"] == 0.0


def test_top20_score_is_a_lower_bound_of_the_full_kl():
    p = np.exp(log_softmax(np.linspace(-2.5, 2.5, 200) ** 2))
    q = np.exp(log_softmax(np.linspace(2.0, -2.0, 200)))
    full = float((p * np.log(p / q)).sum())
    top = np.argsort(-q)[:20]  # the engine reports its own 20 best ids
    got = score_kl.row_kl(np.log(p), top, np.log(q[top]))
    assert got["kl"] > 0.0
    assert got["kl"] <= full + 1e-12


def test_a_rest_bucket_at_or_below_zero_is_clamped_and_counted():
    p_row = np.log(np.array([0.5, 0.5]))
    q_logp = np.log(np.array([0.6, 0.5]))  # the 20 log-probabilities sum to 1.1, so the rest is negative
    got = score_kl.row_kl(p_row, np.array([0, 1]), q_logp)
    assert got["q_rest_clamped"]
    assert got["q_rest"] == score_kl.REST_FLOOR
    assert math.isfinite(got["kl"])


def test_a_probability_row_cannot_pass_as_a_log_row():
    score_kl.check_log_softmax(np.log(np.array([0.5, 0.25, 0.25])))
    with pytest.raises(ValueError, match="above 0"):
        score_kl.check_log_softmax(np.array([0.5, 0.25, 0.25]))
    with pytest.raises(ValueError, match="log-sum-exp"):
        score_kl.check_log_softmax(np.log(np.array([0.5, 0.25, 0.1])))


def test_score_of_the_reference_top20_is_zero_end_to_end(tmp_path):
    # A tiny synthetic reference: the same file shapes as the real one, so the block loop and the checks run.
    vocab, rows = 128, 4
    row = log_softmax(np.linspace(-2.0, 2.0, vocab)).astype(np.float32)
    np.save(tmp_path / "logprobs.f32.npy", np.stack([row] * rows))
    order = np.argsort(-row)
    p_cols = np.arange(rows, dtype=np.int32) * 8 + 512
    ref = dict(logprobs=str(tmp_path / "logprobs.f32.npy"), vocab=vocab,
               chunk=np.zeros(rows, np.int32), p=p_cols,
               real_id=order[:rows].astype(np.int64), real_logprob=row[order[:rows]],
               argmax_id=np.full(rows, order[0], np.int32),
               top_ids=np.tile(order, (rows, 1))[:, :64], top_logprob=np.tile(row[order], (rows, 1))[:, :64],
               keys=[(0, int(p)) for p in p_cols])
    table = {(0, int(p_cols[i])): (order[:20].astype(np.int64), row[order[:20]].astype(np.float64))
             for i in range(rows)}
    got = score_kl.score(ref, {"self": table})
    assert got["self"]["mean"] == 0.0
    assert got["self"]["top1_agreement"] == 1.0
    assert got["checks"]["argmax_mismatch"] == 0
    assert got["self"]["nll_coverage"] == 1.0
    assert score_kl.self_check(ref, ("self",))["mean_kl"] == 0.0


def test_matched_nll_uses_only_the_positions_every_engine_covers(tmp_path):
    # Two engines with different real-token coverage: the NLL must use the shared positions only.
    vocab, rows = 64, 4
    row = log_softmax(np.linspace(-2.0, 2.0, vocab)).astype(np.float32)
    np.save(tmp_path / "logprobs.f32.npy", np.stack([row] * rows))
    order = np.argsort(-row)
    real = order[:rows]  # the real token of row i is the i-th best id
    p_cols = np.arange(rows, dtype=np.int32) * 8 + 512
    ref = dict(logprobs=str(tmp_path / "logprobs.f32.npy"), vocab=vocab,
               chunk=np.zeros(rows, np.int32), p=p_cols, real_id=real.astype(np.int64),
               real_logprob=row[real], argmax_id=np.full(rows, order[0], np.int32),
               top_ids=np.tile(order, (rows, 1))[:, :64], top_logprob=np.tile(row[order], (rows, 1))[:, :64],
               keys=[(0, int(p)) for p in p_cols])

    def table(cover):
        """The real id in first place where the engine covers it, else 20 ids that do not hold it."""
        out = {}
        for i in range(rows):
            ids = np.concatenate([[real[i]], order[5:24]]).astype(np.int64) if i in cover else order[5:25]
            out[(0, int(p_cols[i]))] = (ids.astype(np.int64), row[ids].astype(np.float64))
        return out

    got = score_kl.score(ref, {"A": table({0, 1, 2}), "B": table({1, 2, 3})})
    shared = (1, 2)
    want = float(np.mean([-row[real[i]] for i in shared]))
    assert got["checks"]["rows_matched"] == len(shared)
    assert got["checks"]["nll_reference_matched"] == pytest.approx(want)
    for name in ("A", "B"):
        assert got[name]["nll_matched"] == pytest.approx(want)
        assert got[name]["nll_coverage"] == pytest.approx(3 / rows)  # per-engine coverage stays in the JSON
    solo = {0: (0, 1, 2), 1: (1, 2, 3)}
    for k, name in enumerate(("A", "B")):
        assert got[name]["nll"] == pytest.approx(float(np.mean([-row[real[i]] for i in solo[k]])))


def test_summary_statistics():
    s = score_kl.summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["rows"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["max"] == pytest.approx(4.0)
    assert s["p99"] == pytest.approx(float(np.percentile([1.0, 2.0, 3.0, 4.0], 99)))
