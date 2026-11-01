#!/usr/bin/env python3
"""Exactness of a cache-extended prefill: does restoring the first s tokens of a prompt P and prefilling the rest give
the same arithmetic as a cold prefill of all of P?

  exact_extend.py prompts OUT.json [--pack 4.05]       build the synthetic base prompt and the case markers (needs `tokenizers`)
  exact_extend.py run --base URL --prompts F --out F.jsonl --group short|long [--arm A --pack P] [--deadline-s 540]
                                                        the driver. Runs inside one lease, against a running `serve`.
  exact_extend.py report DIR_OR_JSONL...                one PASS/FAIL/INVALID table: arm, pack, N, s, level
  (group load, does a cold prompt's output depend on server load? see the section "the load group")
  (group sp, save points: resumes from the end-64 Pre point and from the shared-header point; section "the sp group")

Two levels per case, reported apart:
  lp    max_tokens 1, top_logprobs 20: the ids and the float values must be bitwise equal (needs no near tie)
  g0/g7 32 greedy tokens, spec 0 or spec 7: the ids must be equal
For every case the driver sends, to ONE server and in this order:
  seed  P[:s]   cache_control auto     fills the prefix cache with exactly P[:s]; cached_tokens must be 0
  ext   P       cache_control auto     restores the seed and prefills the rest; cached_tokens says what it restored
  cold  P       cache_control bypass   "performs neither cache read nor write": a full prefill
Each case has its own first token (a marker), so no case shares a prefix with another one and the cache of the
other cases cannot be hit. This replaces a fresh server per split. The control case (ctl) checks the one assumption:
bypass-cold equals auto-cold with an empty cache, and cold twice gives equal answers.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

LP_TOP = 20
TOKENS = 32
LEVELS = {"lp": dict(spec=0), "g0": dict(spec=0), "g7": dict(spec=7)}
# (N, [s...]) of each shape. N = 34816 is the P1c shape (2K new tokens after a cached 32K).
SPLITS = [(8900, [512, 4096, 8192, 8300, 8899, 5001]), (4100, [2048]), (34816, [32768])]
# group -> (levels run for N <= 10000, levels run for the long shape). short: about 5 minutes of GPU on the test machine; long: the
# rest, about 6 minutes (every lease is at most 10 minutes, the model load included).
GROUPS = {"short": (("lp", "g0"), ()), "long": (("g7",), ("lp", "g0", "g7")),
          # the resumes on a 2,048-row cut, all three levels, one lease (about 8 minutes).
          "fix1": (("lp", "g0", "g7"), ("lp", "g0", "g7")),
          # the off-grid splits of B_ONLY plus the chat turn, levels lp and g0 (about 8 minutes).
          "b": (("lp", "g0"), ("lp", "g0")),
          # cold bits against server load (the plan is LOAD_CASES; the levels here are not used).
          "load": ((), ()),
          # Save points: the cases are SP_CASES (the levels here are not used).
          "sp": ((), ())}
FIX1_ONLY = {(8900, 4096), (8900, 8192), (4100, 2048), (34816, 32768)}
# the off-grid End points. The resume point is the seed's Pre point, the last
# multiple of 4 rows that leaves more than 8 rows, or the End point itself when it is a multiple of 4 with more than 8
# new rows (5001 -> 4992, 8899 -> 8888, 8300 and 33768 -> s). (35816, [33768]) is a chat turn: a cached
# off-grid 33,768 and 2,048 new rows. Extra splits take marker indexes after the control; a split that is added later
# gets FIXED_SPLITS indexes, so that no existing case moves.
# (8904, 8896) is the short tail: s is a multiple of 4 and only 8 rows are new, so the End point is not used (b-fallback:
# 8896 -> the seed's Pre point 8884).
B_ONLY = {(8900, 5001), (8900, 8300), (8900, 8899), (35816, 33768), (8904, 8896)}
EXTRA_SPLITS = [(35816, [33768]), (8904, [8896])]
# (8196, 6144) is an End point 2,048 rows off the 4,096 grid and a cold small-row tail of 4 rows.
# It has fixed marker indexes, so no other split of group b moves: 29 is free (the extras use 25 to 28), and 36 is the
# last marker (the load group uses 30 to 35). It is not in B_ONLY: run it with EE_ONLY=8196:6144 (group b).
FIXED_SPLITS = {(8196, 6144): {"lp": 29, "g0": 36}}
CONTROL_N = 8900
# on trunk a
# cold prompt of N = 4096 k + t rows, t in 1..8 and k odd, ran its tail as bulk rows on an idle server (PrefillLaunchRows
# merged the tail into the launch before it) and as small-row kernels while a decoder was ready (decoders blocked the
# merge; a tail of 8 rows or fewer never joined). N = 8192 + t was small-row in both states and t = 9 bulk in both.
# F3: a tail of 8 rows or fewer never joins a launch, so the contract is "same"
# for all of them; a DIFF is a SURPRISE (trunk shows it at 4,100, 4,104, 12,292 and 12,296).
LOAD_CASES = [(4100, "same"), (4104, "same"), (12292, "same"), (12296, "same"), (8196, "same"), (4105, "same")]
LOAD_INDEX = 30  # first marker index of the load group (the other groups use indexes below 30)
LOAD_SESSIONS = 3  # decoding sessions during the loaded probe (the server has 4 streams: 3 + the probe)
LOAD_PROMPT_TOKENS = 512
LOAD_MAX_TOKENS = 6000  # long enough that no session ends (or stops at its end token) before the probe returns
LOAD_READY_EVENTS = 5  # token events of a session before it counts as decoding
LOAD_READY_S = 90
LOAD_SETTLE_S = 5  # after the sessions stop: the server cancels them and frees their streams
LOAD_CASE_SETUP_S = 40
EST_PREFILL_TOK_S = 1000  # a slow estimate, only to decide that a case no longer fits in the lease
CASE_SETUP_S = 8


# ---- the plan and the prompts (pure) -----------------------------------------------------------------------------

def all_splits():
    return [(n, s) for n, ss in SPLITS for s in ss]


def parse_only(text):
    """`--only 8900:4096,8900:8192` -> {(8900, 4096), (8900, 8192)}; empty: no filter."""
    return {tuple(int(x) for x in item.split(":")) for item in text.split(",") if item} if text else set()


def index_of(li, si, base, old_count, total):
    """Marker index of split `si` at level `li`: an old split keeps its index; an extra split goes after the control."""
    return li * old_count + si if si < old_count else base + li * (total - old_count) + (si - old_count)


def plan(group, only=frozenset()):
    """Cases of one group (`only`: keep the splits (N, s) of that set; the control stays). the marker index
    is made before the filter, so a case is the same prompt with and without it. Levels run in the outer loop, so a lease that ends early keeps the lp level of every split.
    `index` is the marker index: one for each (level, split), and the same in every group and every run, so a case is
    the same prompt on all arms."""
    if group == "load":
        return load_plan()
    if group == "sp":
        return sp_plan(only)
    every, (short_levels, long_levels) = all_splits(), GROUPS[group]
    cases = [dict(kind="split", level=level, n=n, s=s, index=li * len(every) + si)
             for li, level in enumerate(LEVELS) for si, (n, s) in enumerate(every)
             if level in (short_levels if n < 10000 else long_levels)]
    if group == "b":
        every = all_splits() + [(n, s) for n, ss in EXTRA_SPLITS for s in ss]
        base = len(LEVELS) * len(all_splits()) + 1
        cases = [dict(kind="split", level=level, n=n, s=s,
                      index=index_of(li, si, base, len(all_splits()), len(every)))
                 for li, level in enumerate(LEVELS) for si, (n, s) in enumerate(every)
                 if level in (short_levels if n < 10000 else long_levels)]
        for (n, s), indexes in FIXED_SPLITS.items():
            cases += [dict(kind="split", level=level, n=n, s=s, index=index) for level, index in indexes.items()]
        cases.sort(key=lambda c: list(LEVELS).index(c["level"]))  # levels in the outer loop (stable)
        only = only or B_ONLY
    if group == "fix1":
        only = only or FIX1_ONLY
    if only:
        cases = [c for c in cases if (c["n"], c["s"]) in only]
    if group == "short":
        cases.insert(0, dict(kind="ctl", level="lp", n=CONTROL_N, s=0, index=len(LEVELS) * len(every)))
    return cases


def make_prompt(base, markers, index, n):
    """N token IDs: one marker token, then base[:N-1]. A different marker gives a different first token, so two
    prompts never share a prefix."""
    if n - 1 > len(base):
        raise ValueError(f"base prompt has {len(base)} tokens, N={n} needs {n - 1}")
    if index >= len(markers):
        raise ValueError(f"only {len(markers)} markers, case index {index}")
    return [markers[index]] + base[:n - 1]


def load_plan():
    """one case per length. Each case sends the probe idle, idle again, and with the sessions decoding."""
    return [dict(kind="load", level="load", n=n, s=0, index=LOAD_INDEX + i, expect=expect)
            for i, (n, expect) in enumerate(LOAD_CASES)]


def estimate_s(case):
    """Seconds of GPU time that a case needs at a slow prefill rate: seed + ext + cold (ctl: three full prompts)."""
    if case["kind"] == "load":
        return 3 * case["n"] / EST_PREFILL_TOK_S + LOAD_CASE_SETUP_S  # idle, idle again, loaded, plus the session setup
    if case["kind"] == "div":
        tokens = case["seed_len"] + 2 * case["n"]  # the seed, then the extension and the cold run
    elif case["kind"] == "hdr":
        tokens = 3 * 1500  # three chat prompts of about 600 rows each, and the template
    else:
        tokens = (3 if case["kind"] == "ctl" else 2) * case["n"]  # seed + ext = N tokens, cold = N
    return tokens / EST_PREFILL_TOK_S + CASE_SETUP_S


def body_for(level, prompt, cache_control="auto"):
    body = dict(prompt=prompt, temperature=0, spec=LEVELS[level]["spec"], return_token_ids=True,
                cache_control=cache_control)
    if level == "lp":
        return body | dict(max_tokens=1, top_logprobs=LP_TOP)
    return body | dict(max_tokens=TOKENS)


# ---- comparison (pure) -------------------------------------------------------------------------------------------

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_lp(a, b):
    """Bitwise comparison of two top-N lists [[id, logprob], ...]: same ids in the same order, equal float values.
    Returns a dict with `equal`, the first differing rank and the largest value difference on shared ids."""
    if a is None or b is None:
        return dict(equal=False, note="missing log-probabilities")
    rank = first_diff([p[0] for p in a], [p[0] for p in b])
    values = {p[0]: p[1] for p in a}
    diffs = [abs(values[p[0]] - p[1]) for p in b if p[0] in values]
    value_rank = next((i for i, (x, y) in enumerate(zip(a, b)) if x[0] == y[0] and x[1] != y[1]), None)
    equal = rank is None and value_rank is None and len(a) == len(b)
    return dict(equal=equal, id_diff_rank=rank, value_diff_rank=value_rank, max_abs_diff=max(diffs, default=None),
                top1=[a[0][0], b[0][0]] if a and b else None)


def compare_tokens(a, b):
    d = first_diff(a, b)
    return dict(equal=d is None, first_diff=d, tokens=[len(a), len(b)],
                a=a[d] if d is not None and d < len(a) else None, b=b[d] if d is not None and d < len(b) else None)


def cache_status(case, seed, ext, cold):
    """(status, restored, note). EXACT: ext restored exactly s. PARTIAL: it restored 0 < c < s (the cache holds only
    chunk-aligned or checkpointed lengths; the split that was tested is c). INVALID: the test did not test an extend."""
    c = lambda r: (r or {}).get("cached") or 0  # noqa: E731
    problems = []
    if c(cold):
        problems.append(f"cold request restored {c(cold)} tokens")
    if c(seed):
        problems.append(f"seed restored {c(seed)} tokens (a marker prompt must miss)")
    restored = c(ext)
    if restored == 0:
        problems.append("ext restored nothing (no extend happened)")
    elif restored > case["s"]:
        problems.append(f"ext restored {restored} > s {case['s']}")
    if problems:
        return "INVALID", restored, "; ".join(problems)
    return ("EXACT" if restored == case["s"] else "PARTIAL"), restored, ""


def answer_of(level, response):
    return response.get("scores") if level == "lp" else response.get("ids")


def compare_answers(level, a, b):
    if level == "lp":
        return compare_lp(answer_of(level, a), answer_of(level, b))
    return compare_tokens(answer_of(level, a) or [], answer_of(level, b) or [])


def judge(record):
    """Verdict of one stored case record: dict(verdict, restored, status, level, detail)."""
    case, level = record["case"], record["case"]["level"]
    if record.get("error"):
        return dict(verdict="ERROR", level=level, restored=None, status="", detail=record["error"])
    if case["kind"] == "load":
        return judge_load(record)
    if case["kind"] == "ctl":
        auto, cold, cold2 = record["auto"], record["cold"], record["cold2"]
        bad = [n for n, r in (("auto", auto), ("cold", cold), ("cold2", cold2)) if r.get("status") != 200]
        if bad:
            return dict(verdict="ERROR", level=level, restored=None, status="", detail=f"HTTP error in {bad}")
        a, b = compare_answers(level, auto, cold), compare_answers(level, cold, cold2)
        note = "" if not (auto.get("cached") or 0) else f"auto restored {auto['cached']}"
        ok = a["equal"] and b["equal"] and not note
        return dict(verdict="PASS" if ok else "FAIL", level=level, restored=0, status="CTL",
                    detail=f"auto=bypass {a['equal']}, bypass=bypass {b['equal']} {note}".strip())
    seed, ext, cold = record["seed"], record["ext"], record["cold"]
    if case["kind"] == "hdr" and not record.get("header") and seed.get("status") == 200:
        return dict(verdict="INVALID", level=level, restored=(ext or {}).get("cached"), status="",
                    detail="the seed saved no header point (no log line)")
    bad = [n for n, r in (("seed", seed), ("ext", ext), ("cold", cold)) if r.get("status") != 200]
    if bad:
        errors = "; ".join(f"{n}: {record[n].get('status')} {record[n].get('error', '')}" for n in bad)
        return dict(verdict="ERROR", level=level, restored=None, status="", detail=errors)
    status, restored, note = cache_status(case, seed, ext, cold)
    cmp_ = compare_answers(level, ext, cold)
    detail = note or ("" if cmp_["equal"] else describe_diff(level, cmp_))
    verdict = "INVALID" if status == "INVALID" else ("PASS" if cmp_["equal"] else "FAIL")
    return dict(verdict=verdict, level=level, restored=restored, status=status, detail=detail, compare=cmp_)


def first_diff_rank(cmp_):
    """The first differing position of two top-N lists (an id or a value), or None."""
    ranks = [r for r in (cmp_.get("id_diff_rank"), cmp_.get("value_diff_rank")) if r is not None]
    return min(ranks, default=None)


def load_held(load):
    """True when every session was still decoding after the probe returned (so each one was a ready decoder during it)."""
    alive = (load or {}).get("alive") or []
    return len(alive) >= LOAD_SESSIONS and all(alive)


def judge_load(record):
    """the idle and the loaded answer are bitwise equal (ids and top-N log-probabilities). DIFF: they
    differ. INVALID: the idle answer is not repeatable, or a session ended before the probe returned. `status` says
    whether the verdict is the one that the case expects (EXPECTED or SURPRISE)."""
    case, runs = record["case"], {k: record.get(k) or {} for k in ("idle", "idle2", "loaded")}
    bad = [k for k, r in runs.items() if r.get("status") != 200]
    if bad:
        errors = "; ".join(f"{k}: {runs[k].get('status')} {runs[k].get('error', '')}" for k in bad)
        return dict(verdict="ERROR", level="load", restored=None, status="", detail=errors)
    plans = f"chunks idle {runs['idle'].get('chunks')} loaded {runs['loaded'].get('chunks')}"
    steady = compare_answers("lp", runs["idle"], runs["idle2"])
    if not (steady["equal"] and runs["idle"].get("ids") == runs["idle2"].get("ids")):
        return dict(verdict="INVALID", level="load", restored=None, status="", detail=f"the idle answer is not repeatable; {plans}")
    if not load_held(record.get("load")):
        return dict(verdict="INVALID", level="load", restored=None, status="",
                    detail=f"a session ended before the probe returned (alive {(record.get('load') or {}).get('alive')}); {plans}")
    cmp_ = compare_lp(runs["idle"].get("scores"), runs["loaded"].get("scores"))
    ids_equal = runs["idle"].get("ids") == runs["loaded"].get("ids")
    equal = cmp_["equal"] and ids_equal
    detail = f"ids equal {ids_equal}, " + ("all values equal" if cmp_["equal"] else
             f"first diff rank {first_diff_rank(cmp_)}, max |diff| {cmp_['max_abs_diff']}, top1 {cmp_['top1']}")
    expected = case.get("expect")
    status = "" if expected is None else ("EXPECTED" if (expected == "same") == equal else "SURPRISE")
    return dict(verdict="SAME" if equal else "DIFF", level="load", restored=None, status=status,
                detail=f"{detail}; {plans}", compare=cmp_)


def describe_diff(level, cmp_):
    if level == "lp":
        return (f"id rank {cmp_['id_diff_rank']}, value rank {cmp_['value_diff_rank']}, "
                f"max |diff| {cmp_['max_abs_diff']}, top1 {cmp_['top1']}")
    return f"token {cmp_['first_diff']}: ext {cmp_['a']} cold {cmp_['b']}"


# ---- the driver (inside the lease) -------------------------------------------------------------------------------

def post(base, body, timeout=420):
    """One non-streaming completion. An HTTP error or a lost connection is a result, not an exception. `_path` in the
    body (not sent) names another endpoint, for example /chat/completions."""
    body = dict(body)
    path = body.pop("_path", "/completions")
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return parse_response(json.load(response), time.monotonic() - start)
    except urllib.error.HTTPError as error:
        try:
            detail = str(json.load(error).get("error", ""))[:200]
        except ValueError:
            detail = ""
        return dict(status=error.code, error=detail, seconds=round(time.monotonic() - start, 3))
    except (urllib.error.URLError, OSError, ValueError) as error:
        return dict(status=0, error=type(error).__name__, seconds=round(time.monotonic() - start, 3))


def parse_response(payload, seconds):
    choice = (payload.get("choices") or [{}])[0]
    content = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
    scores = [[p["token_id"], p["logprob"]] for p in content.get("top_logprobs") or []] or None
    usage = payload.get("usage") or {}
    return dict(status=200, ids=choice.get("token_ids") or [], scores=scores, finish=choice.get("finish_reason"),
                cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                prompt_tokens=usage.get("prompt_tokens"), seconds=round(seconds, 3))


def run_split(base, case, prompt, send=post):
    level = case["level"]
    seed = send(base, body_for(level, prompt[:case["s"]]))
    ext = send(base, body_for(level, prompt)) if seed["status"] == 200 else dict(status=0, error="seed failed")
    cold = send(base, body_for(level, prompt, "bypass")) if ext["status"] == 200 else dict(status=0, error="ext failed")
    return dict(case=case, seed=seed, ext=ext, cold=cold)


def run_control(base, case, prompt, send=post):
    level = case["level"]
    return dict(case=case, auto=send(base, body_for(level, prompt)), cold=send(base, body_for(level, prompt, "bypass")),
                cold2=send(base, body_for(level, prompt, "bypass")))



# ---- the sp group (save points) ----------------------------------------------------

# CRUCIBLE_EXACT_RESUME=1 saves a Pre point SP_MARGIN rows before the end of a prompt (the next turn of a chat diverges
# about SP_TURN_GAP rows before that end, where the template closes the user turn) and one point at the end of the
# shared system block. (kind, N, length of the seed prompt, levels). N = 132,050 is the P2c shape.
SP_MARGIN = 64
SP_TURN_GAP = 23
# The header case is a chat request, and the chat endpoint rejects a scoring request (server/api.py scoring_count:
# logprobs needs a completion with max_tokens 1, temperature 0 and no stream), so it runs level g0 only: the token ids
# are the comparison. The Oct 26 night sent it at lp and every case reported 400.
SP_CASES = [("hdr", 0, 0, ("g0",)), ("div", 8900, 4100, ("lp", "g0")), ("div", 36900, 34816, ("lp", "g0")),
            ("div", 132050, 130003, ("lp",)),  # the cheap cases first: a lease that ends early keeps them
            # t29 review M1: a resume whose last part has 68 rows, where the cold plan runs the last 4 rows (10,240 to
            # 10,243) as a 4-row QSA part of their own (prefill_qsa_select.h:124); the second also crosses 32,768
            # positions, which takes another top-k kernel (qsa_long.h:371). They go last: they do not move the others.
            ("div", 10244, 10241, ("lp", "g0")), ("div", 34820, 34817, ("lp", "g0"))]
HDR_WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nectar orchid "
             "pebble quartz ridge saffron thistle umber velvet willow xenon yarrow zephyr").split()
HDR_SYSTEM_WORDS, HDR_USER_WORDS = 170, 420


def sp_point(seed_len):
    """The Pre point that the engine saves for a prompt of `seed_len` rows: the margin point, a multiple of 4."""
    return (seed_len - SP_MARGIN) // 4 * 4


def sp_plan(only=frozenset()):
    """One case for each (kind, level). `s` is the row where the extension must restore: the margin point (div), or
    the header point that the server log gives (hdr: filled in by the driver). `d` is where the extension leaves the seed.
    `only` keeps the cases of these (N, s) pairs, as the other groups do (the header case is (0, 0))."""
    cases = []
    for kind, n, seed_len, levels in SP_CASES:
        for level in levels:
            case = dict(kind=kind, level=level, n=n, s=sp_point(seed_len) if kind == "div" else 0, index=len(cases))
            if kind == "div":
                case |= dict(seed_len=seed_len, d=seed_len - SP_TURN_GAP)
            cases.append(case)
    return [c for c in cases if not only or (c["n"], c["s"]) in only]


def hdr_messages(index, which):
    """Chat messages of the header case: the same system turn for every `which` of one case, another user turn for each."""
    words = HDR_WORDS
    system = f"Session {index}. " + " ".join(words[(index * 7 + i * 3) % len(words)] for i in range(HDR_SYSTEM_WORDS))
    stride = 5 if which == "A" else 11
    user = f"Question {which}. " + " ".join(words[(i * stride + 1) % len(words)] for i in range(HDR_USER_WORDS))
    return [dict(role="system", content=system), dict(role="user", content=user)]


def chat_body(level, messages, cache_control="auto"):
    body = body_for(level, [], cache_control)
    body.pop("prompt")
    return body | dict(messages=messages, _path="/chat/completions")


HEADER_LINE = "slot_checkpoint header tokens="


def header_tokens(path, offset):
    """The row count of the last `slot_checkpoint header` log line after byte `offset` of the server log, or None."""
    if not path:
        return None
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    found = None
    for line in text.splitlines():
        if line.startswith(HEADER_LINE):
            try:
                found = int(line[len(HEADER_LINE):].split()[0])
            except (ValueError, IndexError):
                continue
    return found


def run_div(base, case, prompts, send=post):
    """seed = P[:seed_len]; ext leaves the seed `d` rows in (inside its last SP_TURN_GAP rows) and adds new rows from
    another part of the base text; cold = ext with cache bypass. The restore must be the seed's end-64 Pre point."""
    level, text = case["level"], prompts["base"]
    seed_prompt = make_prompt(text, prompts["markers"], case["index"], case["seed_len"])
    ext_prompt = seed_prompt[:case["d"]] + text[len(text) - (case["n"] - case["d"]):]
    seed = send(base, body_for(level, seed_prompt))
    ext = send(base, body_for(level, ext_prompt)) if seed["status"] == 200 else dict(status=0, error="seed failed")
    cold = send(base, body_for(level, ext_prompt, "bypass")) if ext["status"] == 200 else dict(status=0, error="ext failed")
    return dict(case=case, seed=seed, ext=ext, cold=cold)


def run_hdr(base, case, log, send=post, sleep=time.sleep):
    """Two chat requests with one system turn and different user turns. The first saves the header point (the server
    log says where); the second must restore exactly there; cold = the second with cache bypass."""
    level, index = case["level"], case["index"]
    offset = Path(log).stat().st_size if log else 0
    seed = send(base, chat_body(level, hdr_messages(index, "A")))
    if log:
        sleep(0.5)  # the log line is written before the answer; this only covers a slow pipe
    point = header_tokens(log, offset)
    ext = send(base, chat_body(level, hdr_messages(index, "B"))) if seed["status"] == 200 else dict(status=0, error="seed failed")
    cold = send(base, chat_body(level, hdr_messages(index, "B"), "bypass")) if ext["status"] == 200 else dict(status=0, error="ext failed")
    return dict(case=case | dict(s=point or 0, n=seed.get("prompt_tokens") or 0), seed=seed, ext=ext, cold=cold, header=point)


# ---- the load group -----------------------------------------------------------------------------------

def load_bodies(prompts):
    """The requests of the decoding sessions: 512-token synthetic prompts (not the probe's), a long answer, streamed
    with one event for each token, cache bypass (no entry, no shared prefix with the probe)."""
    base, markers = prompts["base"], prompts["markers"]
    return [dict(prompt=[markers[-1 - j]] + base[1000 * (j + 1):1000 * (j + 1) + LOAD_PROMPT_TOKENS - 1], temperature=0,
                 spec=0, max_tokens=LOAD_MAX_TOKENS, stream=True, return_token_ids=True, cache_control="bypass")
            for j in range(LOAD_SESSIONS)]


def read_stream(base, body, on_event, stop, timeout=900):
    """Read one streamed completion until it ends or `stop` is set; on_event() runs for every data event."""
    request = urllib.request.Request(base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if stop.is_set():
                return
            if line.startswith(b"data:") and b"[DONE]" not in line:
                on_event()


class StreamLoad:
    """Decoding sessions kept running while a probe is sent. One thread for each session counts its token events."""

    def __init__(self, base, bodies, reader=read_stream, clock=time.monotonic, sleep=time.sleep):
        self.stop_flag, self.counts_, self.errors = threading.Event(), [0] * len(bodies), [None] * len(bodies)
        self.clock, self.sleep = clock, sleep
        self.threads = [threading.Thread(target=self._run, args=(i, base, body, reader), daemon=True)
                        for i, body in enumerate(bodies)]
        for thread in self.threads:
            thread.start()

    def _run(self, i, base, body, reader):
        def tick():
            self.counts_[i] += 1
        try:
            reader(base, body, tick, self.stop_flag)
        except Exception as error:  # a lost session shows in alive(); it must not raise in the thread
            self.errors[i] = type(error).__name__

    def counts(self):
        return list(self.counts_)

    def alive(self):
        return [t.is_alive() and e is None for t, e in zip(self.threads, self.errors)]

    def wait_ready(self, events=LOAD_READY_EVENTS, timeout=LOAD_READY_S):
        """True when every session has `events` token events and is still running."""
        end = self.clock() + timeout
        while self.clock() < end:
            if all(self.alive()) and all(c >= events for c in self.counts_):
                return True
            self.sleep(0.25)
        return False

    def stop(self, timeout=15):
        self.stop_flag.set()
        for thread in self.threads:
            thread.join(timeout)


def log_chunks(path, offset, total):
    """The `done` values of the crucible_prefill log lines with `total` rows after byte `offset` of the server log:
    the launch plan of the probe (counts only, no content). None when there is no log."""
    if not path:
        return None
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    chunks = []
    for line in text.splitlines():
        if line.startswith("crucible_prefill "):
            try:
                fields = json.loads(line[len("crucible_prefill "):])
            except ValueError:
                continue
            if fields.get("total") == total and isinstance(fields.get("done"), int):
                chunks.append(fields["done"])
    return chunks


def probe_once(base, body, send, log=None, sleep=time.sleep):
    offset = Path(log).stat().st_size if log else 0
    result = send(base, body)
    if log:
        sleep(0.5)  # the log line of the last chunk is written before the answer, this only covers a slow pipe
        result["chunks"] = log_chunks(log, offset, len(body["prompt"]))
    return result


def run_load(base, case, prompt, start_load, send=post, log=None, sleep=time.sleep):
    """the same cold prompt (bypass, temperature 0, spec 0, one token with the top-20 log-probabilities)
    idle, idle again (the answer must repeat), and while LOAD_SESSIONS sessions decode."""
    body = body_for("lp", prompt, "bypass")
    idle = probe_once(base, body, send, log, sleep)
    idle2 = probe_once(base, body, send, log, sleep) if idle["status"] == 200 else dict(status=0, error="idle failed")
    if idle2["status"] != 200:
        return dict(case=case, idle=idle, idle2=idle2, loaded=dict(status=0, error="idle failed"))
    load = start_load()
    try:
        if not load.wait_ready():
            return dict(case=case, error=f"the sessions were not decoding after {LOAD_READY_S} s (events {load.counts()})",
                        idle=idle, idle2=idle2)
        before = load.counts()
        loaded = probe_once(base, body, send, log, sleep)
        return dict(case=case, idle=idle, idle2=idle2, loaded=loaded,
                    load=dict(before=before, after=load.counts(), alive=load.alive()))
    finally:
        load.stop()
        sleep(LOAD_SETTLE_S)


def drive(base, prompts, group, out, deadline_s, meta, send=post, clock=time.monotonic, only=frozenset(), log=None,
          start_load=None, sleep=time.sleep):
    """Run the cases of `group`, one JSON line per case into `out`. A case that no longer fits before `deadline_s`
    is written as skipped (it stays in the table). Returns the number of finished cases."""
    start, finished = clock(), 0
    cases = plan(group, only)
    expect = [[c["n"], c["s"], c["level"]] for c in cases]  # the report checks every planned case (summary)
    with open(out, "a") as f:
        f.write(json.dumps(dict(meta=meta | dict(group=group, started=time.time(), expect=expect))) + "\n")
        send(base, dict(prompt=prompts["base"][:512], max_tokens=4, temperature=0, cache_control="bypass"))  # warm-up
        for case in cases:
            elapsed = clock() - start
            if elapsed + estimate_s(case) > deadline_s:
                record = dict(case=case, skipped=f"no time left ({elapsed:.0f} s used of {deadline_s})")
            else:
                prompt = (make_prompt(prompts["base"], prompts["markers"], case["index"], case["n"])
                          if case["kind"] in ("split", "ctl", "load") else None)
                try:
                    if case["kind"] == "load":
                        record = run_load(base, case, prompt, start_load or (lambda: StreamLoad(base, load_bodies(prompts))),
                                          send, log, sleep)
                    elif case["kind"] == "div":
                        record = run_div(base, case, prompts, send)
                    elif case["kind"] == "hdr":
                        record = run_hdr(base, case, log, send, sleep)
                    else:
                        record = (run_control if case["kind"] == "ctl" else run_split)(base, case, prompt, send)
                except Exception as error:  # a broken case must not cost the rest of the lease
                    record = dict(case=case, error=f"{type(error).__name__}: {error}")
                finished += "skipped" not in record
            f.write(json.dumps(record) + "\n")
            f.flush()
    return finished


# ---- the report (pure) -------------------------------------------------------------------------------------------

def load_runs(paths):
    """[(meta, [records])] from result files or folders (every *.jsonl inside)."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("**/*.jsonl")) if p.is_dir() else [p]
    runs = []
    for file in files:
        meta, records = {}, []
        for line in file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "meta" in row:
                    meta = row["meta"] | dict(file=str(file))
                else:
                    records.append(row)
        if meta or records:
            runs.append((meta, records))
    return runs


def table_rows(runs):
    """One row per (arm, pack, N, s, level). Several runs of one key: the last one in file order wins."""
    rows = {}
    for meta, records in runs:
        for record in records:
            case = record["case"]
            key = (meta.get("arm", "?"), meta.get("pack", "?"), case["n"], case["s"], case["level"])
            if "skipped" in record:
                rows.setdefault(key, dict(verdict="SKIPPED", level=case["level"], restored=None, status="",
                                          detail=record["skipped"]))
            else:
                rows[key] = judge(record)
    return rows


ARM_ORDER = ["t26", "t25", "trunk", "fast", "resume", "sp"]
LEVEL_ORDER = {"lp": 0, "g0": 1, "g7": 2}


def sort_key(key):
    arm, pack, n, s, level = key
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else 99, pack, LEVEL_ORDER.get(level, 9), n, s)


def expected_cases(runs):
    """{(arm, pack, level): cases the run planned}: the `expect` list that the driver wrote into the meta, and for an
    older sp run (no `expect`) the sp plan. A run of another group without `expect` is left out: the caller then uses
    the splits and the control. A later run of the same arm and pack replaces an earlier one, as table_rows does."""
    want = {}
    for meta, _records in runs:
        arm, pack = meta.get("arm", "?"), meta.get("pack", "?")
        keys = meta.get("expect")
        if keys is None:
            if meta.get("group") != "sp":
                continue
            keys = [[c["n"], c["s"], c["level"]] for c in sp_plan()]
        for old in [k for k in want if k[:2] == (arm, pack)]:
            del want[old]
        for _n, _s, level in keys:
            want[(arm, pack, level)] = want.get((arm, pack, level), 0) + 1
    return want


def format_table(rows):
    lines = [f"{'arm':6} {'pack':5} {'level':5} {'N':>6} {'s':>6} {'restored':>8}  {'verdict':8} {'cache':8} detail"]
    for key in sorted(rows, key=sort_key):
        arm, pack, n, s, level = key
        r = rows[key]
        lines.append(f"{arm:6} {pack:5} {level:5} {n:6} {s:6} {str(r['restored'] if r['restored'] is not None else '-'):>8}  "
                     f"{r['verdict']:8} {r['status']:8} {r['detail']}")
    return "\n".join(lines)


def summary(rows, want=None):
    """Per arm, pack and level: the counts of each verdict. PASS needs every case that the run planned: `want` gives
    those counts (expected_cases), else the splits and the control are assumed, and an sp arm is expected to hold every
    case of the sp plan for that level. A level with a missing case is INCOMPLETE, one FAIL makes it FAIL."""
    groups = {}
    for (arm, pack, _n, _s, level), r in rows.items():
        counts = groups.setdefault((arm, pack, level), {})
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        if r.get("status") == "SURPRISE":
            counts["SURPRISE"] = counts.get("SURPRISE", 0) + 1
    out = []
    for key in sorted(set(groups) | set(want or {}), key=lambda k: sort_key((k[0], k[1], 0, 0, k[2]))):
        counts = groups.get(key, {})
        expected = (want or {}).get(key)
        if expected is None:
            expected = len(all_splits()) + (1 if key[2] == "lp" else 0)
            if key[0] == "sp":  # an sp arm without a plan: every case of the sp plan at that level
                expected = sum(1 for c in sp_plan() if c["level"] == key[2])
        if key[2] == "load":  # SAME and DIFF are both answers; SURPRISE counts the lengths that differ from the expectation
            done = counts.get("SAME", 0) + counts.get("DIFF", 0)
            verdict = "COMPLETE" if done == len(LOAD_CASES) else "INCOMPLETE"
            out.append(f"{key[0]:6} {key[1]:5} {key[2]:5} {verdict:10} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            continue
        verdict = "FAIL" if counts.get("FAIL") else ("PASS" if counts.get("PASS") == expected else "INCOMPLETE")
        out.append(f"{key[0]:6} {key[1]:5} {key[2]:5} {verdict:10} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


# ---- prompts (needs `tokenizers`) --------------------------------------------------------------------------------

def build_prompts(pack, chars=200_000):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fastgate_lib as lib
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(lib.PACKS[pack] / "tokenizer.json"))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731
    base = encode(lib.synth_code(chars, 11))
    markers = []
    for word in lib.WORDS:
        ids = encode(" " + word)
        if len(ids) == 1 and ids[0] not in markers:
            markers.append(ids[0])
    return dict(base=base, markers=markers, pack=pack, source=f"fastgate_lib.synth_code({chars}, 11)")


# ---- command line ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("out")
    p.add_argument("--pack", default="4.05")
    p.add_argument("--chars", type=int, default=200_000, help="size of the base text (group sp needs 520000: 133K+ tokens)")
    r = sub.add_parser("run")
    r.add_argument("--base", required=True, help="server base URL, for example http://127.0.0.1:18811/v1")
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--group", choices=sorted(GROUPS), required=True)
    r.add_argument("--arm", default="?")
    r.add_argument("--pack", default="?")
    r.add_argument("--engine-sha", default="")
    r.add_argument("--deadline-s", type=float, default=540)
    r.add_argument("--log", default="", help="the server log (group load: the launch plan of each probe)")
    r.add_argument("--only", default="", help="N:s,N:s: run only these splits (the on-grid cases of one fix)")
    t = sub.add_parser("report")
    t.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    if args.command == "prompts":
        data = build_prompts(args.pack, args.chars)
        Path(args.out).write_text(json.dumps(data))
        print(f"base {len(data['base'])} tokens, {len(data['markers'])} markers -> {args.out}")
        return 0
    if args.command == "run":
        prompts = json.loads(Path(args.prompts).read_text())
        meta = dict(arm=args.arm, pack=args.pack, engine_sha=args.engine_sha)
        done = drive(args.base, prompts, args.group, args.out, args.deadline_s, meta, only=parse_only(args.only),
                    log=args.log or None)
        print(f"exact_extend: {done} cases finished -> {args.out}")
        return 0
    runs = load_runs(args.paths)
    rows = table_rows(runs)
    print(format_table(rows))
    print()
    print(summary(rows, expected_cases(runs)))
    return 1 if any(r["verdict"] == "FAIL" for r in rows.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
