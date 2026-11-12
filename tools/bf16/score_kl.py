"""Score the engines' top-20 log-probabilities against the BF16 reference (task 114).

Method for every engine, the same: KL(P_bf16 || Q_engine) over 21 buckets per position. The buckets are
the engine's 20 best token ids plus one rest bucket. P bucket = the reference probability of each of those
20 ids, taken from the reference row; P rest = 1 - their sum. Q bucket = exp(engine log-probability);
Q rest = 1 - the sum of the 20. A rest at or below zero is clamped to REST_FLOOR and counted. By the log-sum
inequality the 21-bucket value is a lower bound of the full-vocabulary KL.

The engine log-probabilities must be a log-softmax over the full vocabulary. The check is the top-20 mass:
every row must be at or below 1 + MASS_TOLERANCE, else the run stops.

The join of the engine rows to the reference rows uses ref/bf16/tokens.py, the code that proved the join on
2026-10-19: hex token bytes to ids, real bytes equal ids[p], prompt_tokens equal p. This script adds no
second join.

Privacy: the script reads token ids and log-probabilities only. It prints and writes numbers, never token
text, prompt text or a token id list.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parents[2] / "ref"
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF))
from bf16 import tokens  # noqa: E402  (the shared join code of task 112, not a copy)

REST_FLOOR = 1e-10  # a rest bucket at or below zero (a rounding artifact) takes this value and is counted
MASS_TOLERANCE = 1e-4  # the top-20 mass of a normalized log-softmax must be at or below 1 + this
BLOCK = 16  # reference rows held in memory at once: 16 * vocab * 8 bytes, about 32 MB


def row_kl(p_row, q_ids, q_logp, floor=REST_FLOOR):
    """The 21-bucket KL of one position: P from a reference log-probability row, Q from the engine's top 20.

    p_row: the reference log-softmax row (float32 or float64). q_ids: the engine's token ids, best first.
    q_logp: the engine's log-probabilities for those ids, same order.
    """
    q_ids = np.asarray(q_ids, np.int64)
    p = np.exp(np.asarray(p_row[q_ids], np.float64))
    q = np.exp(np.asarray(q_logp, np.float64))
    p_sum = float(p.sum())
    q_mass = float(q.sum())
    p_rest, q_rest = 1.0 - p_sum, 1.0 - q_mass
    p_clamped, q_clamped = p_rest <= 0.0, q_rest <= 0.0
    if p_clamped:
        p_rest = floor
    if q_clamped:
        q_rest = floor
    q_zeros = int(np.count_nonzero(q <= 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p / np.maximum(q, floor)), 0.0)
    kl = float(terms.sum()) + p_rest * math.log(p_rest / q_rest)
    return dict(kl=kl, q_mass=q_mass, p_rest=p_rest, q_rest=q_rest, p_rest_clamped=p_clamped,
                q_rest_clamped=q_clamped, q_zeros=q_zeros)


def check_log_softmax(row, tol=1e-4):
    """A reference row must be a normalized log-probability row: no value above 0 and a log-sum-exp of 0.

    It catches a probability row that reached the log-probability path by mistake (all values above 0).
    """
    a = np.asarray(row, np.float64)
    if a.max() > 0.0:
        raise ValueError(f"row has a value above 0 ({a.max()!r}): it is not a log-probability row")
    lse = a.max() + math.log(float(np.exp(a - a.max()).sum()))
    if abs(lse) > tol:
        raise ValueError(f"row log-sum-exp is {lse!r}, not 0: the row is not normalized")


def summarize(values):
    """Mean, median, p99, max and count of a list of position values (linear interpolation for the p99)."""
    a = np.asarray(values, np.float64)
    return dict(rows=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p99=float(np.percentile(a, 99)) if a.size else 0.0, max=float(a.max()) if a.size else 0.0)


def read_rows(path):
    """The rows of one engine file; a last line cut by a lease kill is skipped (that position is not used)."""
    out, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


def load_top20(path, hexmap):
    """{(chunk, p): (ids int64, logp float64)} for an engine that returns top-20 log-probabilities.

    A row is used only when its prompt token count equals p (the rule of tools/quality_ab/analyze.py).
    """
    rows, bad = read_rows(path)
    out = dict(rows=len(rows), bad_json=bad, prompt_mismatch=0, no_top=0, duplicate_ids=0)
    table, dup = {}, 0
    for r in rows:
        if r.get("usage", {}).get("prompt_tokens") != r["p"]:
            out["prompt_mismatch"] += 1
            continue
        top = r.get("top") or []
        if len(top) != 20:
            out["no_top"] += 1
            continue
        ids = np.array([hexmap[h] for h, _ in top], np.int64)
        if len(np.unique(ids)) != len(ids):
            out["duplicate_ids"] += 1
            dup += 1
            continue
        table[(r["chunk"], r["p"])] = (ids, np.array([lp for _, lp in top], np.float64))
    out["used"] = len(table)
    out["dropped_duplicate"] = dup
    return table, out


def load_greedy(path):
    """{(chunk, p): greedy token bytes} for an engine that returns no log-probabilities (Gufo 0.9.0)."""
    rows, bad = read_rows(path)
    counts = dict(rows=len(rows), bad_json=bad, prompt_mismatch=0, no_greedy=0)
    table = {}
    for r in rows:
        if r.get("usage", {}).get("prompt_tokens") != r["p"]:
            counts["prompt_mismatch"] += 1
            continue
        text = (r.get("greedy") or {}).get("text")
        if not text:
            counts["no_greedy"] += 1
            continue
        table[(r["chunk"], r["p"])] = bytes.fromhex(text)
    counts["used"] = len(table)
    return table, counts


def plain_utf8(data):
    """True when the bytes are valid UTF-8, which is what a text-only engine can return undamaged."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def open_logprobs(path):
    """(file, rows, vocab, data offset) of a .npy log-probability array; the offset skips the .npy header."""
    f = open(path, "rb")
    major = np.lib.format.read_magic(f)
    reader = np.lib.format.read_array_header_1_0 if major[0] == 1 else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(f)
    if fortran or dtype != np.dtype("<f4") or len(shape) != 2:
        raise SystemExit(f"{path}: expected a C-order float32 [rows, vocab] array, got {dtype} {shape} FORTRAN={fortran}")
    return f, int(shape[0]), int(shape[1]), f.tell()


def score(ref, engines, gufo=None, block=BLOCK):
    """Score every engine against the reference in one pass over the reference file, in row blocks.

    The reference file is read block by block with seeks, not memory mapped: a map of the 7.1 GB file holds
    every page it touches, and that peak (7.3 GB measured) is above the 4 GB budget of this task.
    """
    path, vocab = ref["logprobs"], int(ref["vocab"])
    chunk, p = ref["chunk"], ref["p"]
    n = len(chunk)
    f, rows_file, vocab_file, offset = open_logprobs(path)
    if (rows_file, vocab_file) != (n, vocab):
        raise SystemExit(f"{path}: shape ({rows_file}, {vocab_file}) does not fit {n} rows of vocab {vocab}")
    keys = list(zip(chunk.tolist(), p.tolist()))
    if keys != sorted(keys):
        raise SystemExit("reference rows are not in (chunk, p) ascending order")
    index = {k: i for i, k in enumerate(keys)}
    acc = {name: dict(kl=[], kl_ok=[], nll=[], nll_by_key={}, top1=[], clamped=0, p_clamped=0, q_zero=0,
                      mass=[], missing=0)
           for name in engines}
    argmax_bad = 0
    real_delta = 0.0
    nll_ref = []
    gufo_hit, gufo_ok = [], []
    for i0 in range(0, n, block):
        f.seek(offset + i0 * vocab * 4)
        rows = min(block, n - i0)
        prob = np.fromfile(f, dtype=np.float32, count=rows * vocab).reshape(rows, vocab).astype(np.float64)
        check_log_softmax(prob[0])  # a probability row would give a wrong KL for every row of the block
        real_delta = max(real_delta, float(np.max(np.abs(
            prob[np.arange(prob.shape[0]), ref["real_id"][i0:i0 + block]] - ref["real_logprob"][i0:i0 + block]))))
        argmax_bad += int(np.count_nonzero(np.argmax(prob, axis=1) != ref["argmax_id"][i0:i0 + block]))
        for j in range(prob.shape[0]):
            i = i0 + j
            key = keys[i]
            row = prob[j]
            nll_ref.append(-float(ref["real_logprob"][i]))
            for name, table in engines.items():
                got = table.get(key)
                if got is None:
                    acc[name]["missing"] += 1
                    continue
                ids, logp = got
                m = row_kl(row, ids, logp)
                a = acc[name]
                a["kl"].append(m["kl"])
                if not m["q_rest_clamped"]:
                    a["kl_ok"].append(m["kl"])
                a["top1"].append(float(ids[0] == ref["argmax_id"][i]))
                a["mass"].append(m["q_mass"])
                a["clamped"] += int(m["q_rest_clamped"])
                a["p_clamped"] += int(m["p_rest_clamped"])
                a["q_zero"] += m["q_zeros"]
                hit = np.flatnonzero(ids == ref["real_id"][i])
                if hit.size:
                    nll = -float(logp[hit[0]])
                    a["nll"].append(nll)
                    a["nll_by_key"][key] = nll
            if gufo is not None:
                text = gufo.get(key)
                if text is not None:
                    want = tokens.token_bytes(TOKENIZER[0], TOKENIZER[1], int(ref["argmax_id"][i]))
                    if plain_utf8(want):
                        gufo_ok.append(1.0)
                        gufo_hit.append(float(text == want))
    f.close()
    # The NLL comparison uses one position set for every engine: the positions where all of them
    # report the real token's log-probability. A per-engine set makes the engines incomparable.
    matched = set.intersection(*(set(a["nll_by_key"]) for a in acc.values())) if acc else set()
    ref_nll = {k: -float(ref["real_logprob"][index[k]]) for k in keys}
    nll_ref_matched = float(np.mean([ref_nll[k] for k in matched])) if matched else None
    out = dict(checks=dict(argmax_mismatch=argmax_bad, real_logprob_max_delta=real_delta,
                           nll_reference=float(np.mean(nll_ref)), rows=n,
                           rows_matched=len(matched), nll_reference_matched=nll_ref_matched))
    for name, a in acc.items():
        s = summarize(a["kl"])
        used = len(a["kl"])
        cover = len(a["nll"]) / used if used else 0.0
        s |= dict(top1_agreement=float(np.mean(a["top1"])) if a["top1"] else 0.0,
                  nll=float(np.mean(a["nll"])) if a["nll"] else None, nll_coverage=cover,
                  nll_matched=float(np.mean([a["nll_by_key"][k] for k in matched])) if matched else None,
                  top20_mass_median=float(np.median(a["mass"])) if a["mass"] else 0.0,
                  top20_mass_max=float(np.max(a["mass"])) if a["mass"] else 0.0,
                  rows_clamped_rest=a["clamped"], rows_clamped_p_rest=a["p_clamped"],
                  q_zero_count=a["q_zero"], missing_rows=a["missing"],
                  mean_kl_without_clamped=float(np.mean(a["kl_ok"])) if a["kl_ok"] else None)
        out[name] = s
    if gufo is not None:
        out["gufo_agreement"] = dict(rows=len(gufo_hit),
                                     top1_agreement=float(np.mean(gufo_hit)) if gufo_hit else 0.0,
                                     positions=int(len(gufo_ok)))
    return out


TOKENIZER = [None, None]  # (tokenizers.Tokenizer, byte table), set by main


def compare_tables(main, rep):
    """Do two runs of one engine give the same top-20? Same key set, same ids, and the largest log-probability gap."""
    shared = set(main) & set(rep)
    out = dict(rows_main=len(main), rows_repeat=len(rep), keys_equal=set(main) == set(rep), rows_shared=len(shared))
    if not shared:
        return out | dict(max_logprob_delta=None, rows_same_ids=0)
    out |= dict(max_logprob_delta=max(float(np.max(np.abs(main[k][1] - rep[k][1]))) for k in shared),
                rows_same_ids=sum(1 for k in shared if np.array_equal(main[k][0], rep[k][0])))
    return out


def self_check(ref, names=("Crucible 4.05",)):
    """Score the reference against its own top-20: the join and the method must give exactly KL 0 and top-1 100%."""
    npz_top, npz_lp = ref["top_ids"][:, :20], ref["top_logprob"][:, :20]
    table = {k: (npz_top[i].astype(np.int64), npz_lp[i].astype(np.float64)) for i, k in enumerate(ref["keys"])}
    got = score(ref, {names[0]: table})[names[0]]
    if got["max"] > 1e-9 or got["top1_agreement"] != 1.0:
        raise SystemExit(f"self check failed: mean KL {got['mean']!r}, top-1 {got['top1_agreement']!r}")
    return dict(mean_kl=got["mean"], max_kl=got["max"], top1_agreement=got["top1_agreement"], rows=got["rows"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default=str(Path.home() / ".cache/bf16-ref/run/out"),
                    help="folder with logprobs.f32.npy, reference.npz and meta.json")
    ap.add_argument("--texts", default=str(Path.home() / ".cache/bf16-ref/harness/all/texts.json"),
                    help="the harness text set (join proof, and the ids of the reference rows)")
    ap.add_argument("--tokenizer", default=str(Path.home() / ".cache/bf16-ref/tok/tokenizer.json"))
    ap.add_argument("--engine", action="append", default=[], metavar="LABEL=PATH",
                    help="an engine file with top-20 log-probabilities (repeatable)")
    ap.add_argument("--greedy", default=None, metavar="LABEL=PATH",
                    help="an engine file with greedy tokens only, for top-1 agreement (Gufo)")
    ap.add_argument("--repeat", action="append", default=[], metavar="LABEL=PATH",
                    help="a second run of an engine, for the repeat check (repeatable)")
    ap.add_argument("--out-json", default=str(Path.home() / ".cache/kl-score/kl_table.json"))
    ap.add_argument("--out-md", default=str(Path.home() / ".cache/kl-score/kl_table.md"))
    ap.add_argument("--self-check", action=argparse.BooleanOptionalAction, default=True,
                    help="score the reference against its own top-20 first (must be KL 0)")
    args = ap.parse_args(argv)

    import tokenizers  # only the hex-bytes map needs it
    tok = tokenizers.Tokenizer.from_file(args.tokenizer)
    TOKENIZER[0], TOKENIZER[1] = tok, tokens.byte_table()
    hexmap, clash = tokens.hex_to_id(tok)
    if clash:
        raise SystemExit(f"{len(clash)} hex token strings map to more than one id: the join is ambiguous")

    meta = json.loads((Path(args.ref) / "meta.json").read_text())
    npz = np.load(Path(args.ref) / "reference.npz")
    ref = dict(logprobs=str(Path(args.ref) / "logprobs.f32.npy"), vocab=meta["vocab"],
               chunk=npz["chunk"], p=npz["p"], real_id=npz["real_id"],
               real_logprob=npz["real_logprob"], argmax_id=npz["argmax_id"],
               top_ids=npz["top_ids"], top_logprob=npz["top_logprob"])
    ref["keys"] = list(zip(ref["chunk"].tolist(), ref["p"].tolist()))

    texts = tokens.load_texts(args.texts, meta["texts_sha256"])  # proves the text set is the reference text set
    plan = tokens.positions(texts)
    engines, joins = {}, {}
    for spec in args.engine:
        label, _, path = spec.partition("=")
        table, counts = load_top20(path, hexmap)
        rows, _ = read_rows(path)
        counts["join"] = tokens.check_rows(texts, rows, tok)  # real bytes equal ids[p], prompt_tokens equal p
        counts["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if counts["join"]["distinct"] != len(plan) or counts["join"]["missing"]:
            raise SystemExit(f"{label}: {counts['join']} does not cover the {len(plan)} scored positions")
        engines[label], joins[label] = table, counts
    gufo, gufo_counts = (None, None)
    if args.greedy:
        label, _, path = args.greedy.partition("=")
        gufo, gufo_counts = load_greedy(path)
        gufo_counts["join"] = tokens.check_rows(texts, read_rows(path)[0], tok)
        gufo_counts["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        gufo_counts["label"] = label

    result = dict(method=dict(floor=REST_FLOOR, buckets=21, reference="BF16 weights, fp32 activations, fp64 log-sum-exp",
                              positions=len(plan), vocab=meta["vocab"], texts_sha256=meta["texts_sha256"]),
                  engines=joins, gufo=gufo_counts)
    result |= score(ref, engines, gufo)
    if gufo_counts is not None:  # keep the join proof next to the agreement count
        result["gufo"] |= result.pop("gufo_agreement")

    for name in engines:
        mass_max = result[name]["top20_mass_max"]
        if mass_max > 1.0 + MASS_TOLERANCE:
            raise SystemExit(f"{name}: top-20 mass {mass_max!r} is above 1 + {MASS_TOLERANCE}: the engine "
                             "log-probabilities are not a log-softmax over the full vocabulary")
    result["row_checks"] = dict(logprobs_rows=int(ref["chunk"].size), texts_positions=len(plan))
    if args.self_check:
        result["self_check"] = self_check(ref, tuple(sorted(engines))[:1] or ("reference",))

    if args.repeat:
        rpt, rjoins = {}, {}
        for spec in args.repeat:
            label, _, path = spec.partition("=")
            table, counts = load_top20(path, hexmap)
            counts["join"] = tokens.check_rows(texts, read_rows(path)[0], tok)
            counts["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if counts["join"]["missing"]:
                raise SystemExit(f"{label} (repeat): {counts['join']} does not cover every scored position")
            rpt[label], rjoins[label] = table, counts
        rep = score(ref, rpt)
        result["repeat"] = dict(engines=rjoins, checks=rep["checks"])
        for label in rpt:
            result["repeat"][label] = rep[label]
            result["repeat"][label]["same_bytes"] = rjoins[label]["sha256"] == joins.get(label, {}).get("sha256")
            if label in engines:
                result["repeat"][label]["against_run1"] = compare_tables(engines[label], rpt[label])

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out_md).write_text(markdown(result) + "\n")
    print(f"wrote {out} and {args.out_md}")
    return result


def fmt(v, digits=5):
    return "-" if v is None else f"{v:.{digits}f}"


def markdown(r):
    """The public table: engine names and numbers only. No host, path, worker or product name."""
    names = [n for n in r["engines"] if isinstance(r[n], dict) and "mean" in r[n]]
    names.sort(key=lambda n: r[n]["mean"])
    lines = ["# Engine quality against the BF16 reference", "",
             "| Engine | Mean KLD | Median KLD | p99 KLD | Top-1 agreement | NLL (matched positions) | Rows |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        s = r[n]
        lines.append(f"| {n} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['p99'])} | "
                     f"{100 * s['top1_agreement']:.2f}% | {fmt(s['nll_matched'], 4)} | {s['rows']:,} |")
    g = r.get("gufo")
    if g and g.get("top1_agreement"):
        lines.append(f"| {g['label']} | - | - | - | {100 * g['top1_agreement']:.2f}% | - | {g['positions']:,} |")
    lines += ["", f"The NLL column uses the {r['checks']['rows_matched']:,} positions where all three "
              "log-probability engines give the real token's log-probability. On the same positions the "
              f"BF16 reference gives {fmt(r['checks']['nll_reference_matched'], 4)} nats.",
              "", "Method:", "",
              f"- Reference: {r['method']['reference']}, {r['method']['positions']:,} positions of our own text.",
              "- KLD: KL(reference || engine) over 21 buckets, the engine's 20 best token ids plus one rest "
              "bucket. A rest at or below zero takes 1e-10 and its row is counted.",
              "- The 21-bucket value is a lower bound of the full-vocabulary KL (log-sum inequality).",
              "- Top-1 agreement: the engine's best id equals the reference best id.",
              f"- NLL: mean negative log-probability of the real next token, on the matched positions of the "
              "single position set above, so the engines are comparable and the reference is measured on the "
              "same text. That set holds the positions every engine knows, so all rows sit near the "
              "reference value.",
              "- This table is measured against our own BF16 reference on 7,168 positions of our own text. "
              "It is a lower bound from the top 20 tokens. It is NOT comparable with published KLD values, "
              "which use other text and full-vocabulary logits.", ""]
    c = r["checks"]
    extra = [f"Largest single-position KLD: " + ", ".join(f"{n} {fmt(r[n]['max'])}" for n in names) + ".",
             f"Rows with a clamped rest bucket: " +
             ", ".join(f"{n} {r[n]['rows_clamped_rest']}" for n in names) + ".",
             f"Median top-20 mass: " + ", ".join(f"{n} {r[n]['top20_mass_median']:.12g}" for n in names) + ".",
             f"Largest top-20 mass: " + ", ".join(f"{n} {r[n]['top20_mass_max']:.12g}" for n in names) + "."]
    rep = r.get("repeat")
    if rep:
        same = [n for n in names if isinstance(rep.get(n), dict) and rep[n].get("against_run1")
                and rep[n]["against_run1"]["keys_equal"]]
        if same:
            worst = max(rep[n]["against_run1"]["max_logprob_delta"] for n in same)
            rows_same = min(rep[n]["against_run1"]["rows_same_ids"] for n in same)
            extra += ["", f"Repeat check: a second run of {', '.join(same)} gave the same top-20 token ids on "
                      f"{rows_same:,} positions and a largest log-probability gap of {worst:.3g}."]
    return "\n".join(lines + extra)


if __name__ == "__main__":
    main()
