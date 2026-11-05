#!/usr/bin/env python3
"""Cluster bootstrap for the KL table (numbers only).

The positions of one text chunk are not independent, so the resampling unit is the chunk: draw chunks with replacement, take the
mean KL and the top-1 rate of the drawn positions. A difference between two engines uses the same draw for both (paired).
`Recorder` captures the per-position KL and top-1 flag from the scoring loop of score_kl.score without a change to that file.
"""
import numpy as np


def _per_chunk(values, inverse, count):
    return np.bincount(inverse, weights=np.asarray(values, dtype=float), minlength=count)


def _interval(draws, point, alpha):
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(point), lo=float(lo), hi=float(hi))


def bootstrap(per_engine, chunks, resamples=2000, seed=20261103, alpha=0.05):
    """per_engine: label -> (kl per position, top-1 flag per position), all in the same position order. chunks: chunk id per position."""
    ids, inverse = np.unique(np.asarray(chunks), return_inverse=True)
    count = np.bincount(inverse, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(resamples, len(ids)))
    n_drawn = count[draw].sum(axis=1)
    sums = {}
    for label, (kl, top1) in per_engine.items():
        if len(kl) != len(inverse) or len(top1) != len(inverse):
            raise ValueError(f"{label}: {len(kl)} positions for {len(inverse)} chunk ids")
        sums[label] = dict(kl=_per_chunk(kl, inverse, len(ids)), top1=_per_chunk(top1, inverse, len(ids)))
    total = float(count.sum())

    def means(label, kind):
        return sums[label][kind][draw].sum(axis=1) / n_drawn

    out = dict(resamples=resamples, seed=seed, chunks=int(len(ids)), positions=int(total), alpha=alpha, engines={}, diff_kl={}, diff_top1={})
    for label in per_engine:
        out["engines"][label] = {kind: _interval(means(label, kind), sums[label][kind].sum() / total, alpha) for kind in ("kl", "top1")}
    labels = list(per_engine)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            for kind, name in (("kl", "diff_kl"), ("top1", "diff_top1")):
                point = (sums[b][kind].sum() - sums[a][kind].sum()) / total
                out[name][f"{b}-{a}"] = _interval(means(b, kind) - means(a, kind), point, alpha)
    return out


class Recorder:
    """Wraps row_kl of score_kl. score() calls row_kl once per position (sorted keys) and engine (dict order), skipping missing positions;
    with common positions only, nothing is skipped, so the call number gives the key and the engine."""

    def __init__(self):
        self.kl, self.first = [], []

    def wrap(self, row_kl):
        def wrapped(p_row, q_ids, q_logp, *args, **kwargs):
            m = row_kl(p_row, q_ids, q_logp, *args, **kwargs)
            self.kl.append(float(m["kl"]))
            self.first.append(int(q_ids[0]))
            return m
        return wrapped

    def arrays(self, keys, engines, argmax_of):
        if len(self.kl) != len(keys) * len(engines):
            raise ValueError(f"{len(self.kl)} calls for {len(keys)} positions and {len(engines)} engines")
        want = np.array([argmax_of(k) for k in keys])
        out = {}
        for j, name in enumerate(engines):
            kl = np.array(self.kl[j::len(engines)])
            first = np.array(self.first[j::len(engines)])
            out[name] = (kl, (first == want).astype(float))
        return out
