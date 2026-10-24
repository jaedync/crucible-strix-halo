#!/usr/bin/env python3
"""an identical re-send restores its last grid frontier inside the MTP window and gives the same IDs.

  resend_proof.py capture ENGINE CHECKPOINT OUT.json      (one model load; run it under gpu-run)
  resend_proof.py check OUT.json ENGINE_LOG [--summary S]  (CPU only; exit 0 PASS, 1 FAIL)

Synthetic token prompts (frozen code-like text, no user data), sent as completion prompts, so the stable boundary is
0 and the own frontier is the last 4,096-row grid point. The MTP window is the request default (256).
  A 32,769 tokens: frontier 32,768, a 1-row suffix (MTP warm-up pairs 0 after the restore)
  B 32,776 tokens: an 8-row suffix
  C 32,843 and D 32,849 tokens: the head-to-head sizes (75 and 81 rows)
  E 33,100 tokens: control, frontier below len - 256
Steps: each prompt cold (greedy, spec 7, cache auto: the reference, and it saves the frontier); A to D again at the
same time; C again at spec 0; E again; D sampled cold (cache bypass) and sampled again (cache auto, same seed).
Pass: every cold send restored 0 rows; every re-send restored 32,768 rows (or, with its `slot_checkpoint end_resume`
line, the whole prompt: re-send fix A) and gave the IDs of its cold send; the spec 0
re-send gave the spec 7 IDs; single re-sends have TTFT under 2 s and the 4 concurrent ones under 5 s; the engine log
(CRUCIBLE_SPEC_STATS=1) shows mtp_origin = max(restored, len - 256) for each spec 7 send (BatchSpecDecoder::Open):
32,768 for the re-sends of A to D, 32,844 for the re-send of E, len - 256 for each cold send, len for an End resume. The verdict also counts
the cache_capture lines (saves and evictions).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.api import validate_request
from tools.serve_replay_gate import checkpoint_encoder, collect, file_digest, save, synthetic_code

GRID, WINDOW, FRONTIER = 4096, 256, 32768
PROMPTS = dict(A=32769, B=32776, C=32843, D=32849, E=33100)
MAX_TOKENS = 160
SAMPLED = dict(temperature=0.6, top_p=0.95, seed=7)
TTFT_ONE_S, TTFT_FOUR_S = 2.0, 5.0
SETTINGS = dict(context=262144, streams=4, prefill=4096, cache_mib=26624, prefill_interleave=10)
CAPTURE = re.compile(r"^cache_capture tokens=(\d+) class=\w+ saved=(\d) evicted=(\d+) superseded_evicted=(\d+) ")
END_EVENT = re.compile(r"^slot_checkpoint end_resume tokens=(\d+) slot=(\d+) request=(\d+)$")
SPEC_STATS = re.compile(r"^spec_stats id=(\d+) cached=(\d+) proposed=(\d+) accepted=(\d+) steps=(\d+) .*"
                        r"mtp_origin=(\d+) mtp_continued=(\d+) mtp_restarts=(\d+)$")


def prompt_ids(encode, label, length):
    """A fixed prompt of exactly `length` tokens; each label has its own text from the first line on."""
    head = encode(f"<|im_start|>system\nYou review code. Prompt {label}.<|im_end|>\n<|im_start|>user\n"
                  "Read this code, then name one risk in it.\n")
    tail = encode("<|im_end|>\n<|im_start|>assistant\n")
    offset = (ord(label) - ord("A") + 1) * 100000
    body = encode(synthetic_code(offset, length // 5))
    need = length - len(head) - len(tail)
    if len(body) < need:
        raise ValueError("not enough synthetic text for the prompt")
    return head + body[:need] + tail


def request(ids, spec=7, cache="auto", sampled=False, max_tokens=MAX_TOKENS):
    body = dict(prompt=list(ids), max_tokens=max_tokens, spec=spec, temperature=0.0, seed=1, cache_control=cache)
    if sampled:
        body |= SAMPLED
    return validate_request(body, "completion", "crucible")


def row(step, label, ids, result, spec, cache, sampled):
    done = result["done"]
    return dict(step=step, prompt=label, prompt_tokens=len(ids), spec=spec, cache=cache, sampled=sampled,
                output_ids=result["output_ids"], cached_tokens=done.get("cached_tokens"),
                request_id=done.get("request_id"), ttft=result["ttft"], seconds=result["seconds"],
                end_resume=done.get("end_resume") is True)


def run_steps(backend, prompts, timeout):
    rows = []
    def one(step, label, spec=7, cache="auto", sampled=False):
        ids = prompts[label]
        return row(step, label, ids, collect(backend, ids, request(ids, spec, cache, sampled), timeout), spec, cache,
                   sampled)
    for label in PROMPTS:
        rows.append(one("cold", label))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows += [future.result(timeout=2 * timeout) for future in
                 [pool.submit(one, "resend4", label) for label in "ABCD"]]
    rows.append(one("resend-spec0", "C", spec=0))
    rows.append(one("control", "E"))
    rows.append(one("sampled-cold", "D", cache="bypass", sampled=True))
    rows.append(one("sampled-resend", "D", sampled=True))
    return rows


def capture(engine, checkpoint, timeout):
    from server.batch_backend import BatchedEngineBackend
    encode = checkpoint_encoder(checkpoint)
    prompts = {label: prompt_ids(encode, label, length) for label, length in PROMPTS.items()}
    environment = {name: value for name, value in sorted(os.environ.items()) if name.startswith("CRUCIBLE_")}
    report = dict(status="RUNNING", settings=SETTINGS, environment=environment, engine_sha256=file_digest(engine),
                  prompt_tokens={label: len(ids) for label, ids in prompts.items()}, rows=[])
    backend = None
    try:
        backend = BatchedEngineBackend(engine, checkpoint, SETTINGS["context"], streams=SETTINGS["streams"],
                                       prefill=SETTINGS["prefill"], conversation_cache_mib=SETTINGS["cache_mib"],
                                       session_replay=True, slot_resume=1, slot_checkpoints=1,
                                       prefill_interleave=SETTINGS["prefill_interleave"])
        report["launch"] = list(backend.process.args)
        warm = encode("<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n")
        collect(backend, warm, validate_request(dict(prompt=warm, max_tokens=1, spec=0, temperature=0.0,
                                                     cache_control="bypass"), "completion", "crucible"), timeout)
        started = time.monotonic()
        report["rows"] = run_steps(backend, prompts, timeout)
        report["seconds"] = time.monotonic() - started
        if backend.fatal or backend.process.poll() is not None:
            raise RuntimeError("the native serve process stopped before the capture finished")
        report["status"] = "CAPTURED"
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        report |= dict(status="FAIL", error=str(error) or type(error).__name__)
    finally:
        if backend is not None:
            backend.close()
    return report


def spec_stats(lines):
    """Engine request id -> (cached, proposed, accepted, steps, mtp_origin) from the CRUCIBLE_SPEC_STATS lines."""
    stats = {}
    for line in lines:
        match = SPEC_STATS.match(line.strip())
        if match:
            rid, cached, proposed, accepted, steps, origin, _, _ = map(int, match.groups())
            stats[rid] = dict(cached=cached, proposed=proposed, accepted=accepted, steps=steps, mtp_origin=origin)
    return stats


def capture_counts(lines):
    """The cache_capture lines of the engine log: lines, saves, evictions and superseded evictions."""
    counts = dict(lines=0, saved=0, evicted=0, superseded_evicted=0)
    for line in lines:
        match = CAPTURE.match(line.strip())
        if match:
            _, saved, evicted, superseded = map(int, match.groups())
            counts["lines"] += 1
            counts["saved"] += saved
            counts["evicted"] += evicted
            counts["superseded_evicted"] += superseded
    return counts


def check(report, log_lines):
    """Fail-closed checks; returns (problems, summary rows)."""
    if report.get("status") != "CAPTURED":
        return [f"capture status {report.get('status')}: {report.get('error')}"], []
    rows, problems, summary = report["rows"], [], []
    if [r["step"] for r in rows] != ["cold"] * 5 + ["resend4"] * 4 + ["resend-spec0", "control", "sampled-cold",
                                                                       "sampled-resend"]:
        return ["missing or reordered steps"], []
    stats = spec_stats(log_lines)
    # Re-send fix A (always on with slot checkpoints): a re-send on the slot that holds
    # its End point restores the whole prompt and logs `slot_checkpoint end_resume tokens=<len> request=<id>`. Such a
    # request must restore its whole prompt; a request without the line keeps the older rule (the 32,768 frontier).
    ends = {(int(m.group(1)), int(m.group(3))) for m in map(END_EVENT.match, map(str.strip, log_lines)) if m}
    cold = {r["prompt"]: r for r in rows if r["step"] == "cold"}
    sampled_cold = next(r for r in rows if r["step"] == "sampled-cold")
    for r in rows:
        where = f"{r['step']} {r['prompt']}"
        length = PROMPTS[r["prompt"]]
        if r["prompt_tokens"] != length:
            problems.append(f"{where}: prompt has {r['prompt_tokens']} tokens, expected {length}")
        resend = r["step"] not in ("cold", "sampled-cold")
        resumed = (length, r["request_id"]) in ends
        if resumed and not resend:
            problems.append(f"{where}: end_resume line for a cold send")
        want = (length if resumed else FRONTIER) if resend else 0
        if r["cached_tokens"] != want:
            problems.append(f"{where}: cached_tokens {r['cached_tokens']}, expected {want}")
        if resend:
            reference = sampled_cold if r["sampled"] else cold[r["prompt"]]
            if r["output_ids"] != reference["output_ids"]:
                problems.append(f"{where}: output IDs differ from {reference['step']} {reference['prompt']}")
            limit = TTFT_FOUR_S if r["step"] == "resend4" else TTFT_ONE_S
            if r["ttft"] is None or r["ttft"] > limit:
                problems.append(f"{where}: TTFT {r['ttft']} s above {limit} s")
        stat = stats.get(r["request_id"]) if r["spec"] else None
        if r["spec"]:
            origin = max(want, length - WINDOW)  # BatchSpecDecoder::Open: origin = max(cached, mtp_start)
            if not stat:
                problems.append(f"{where}: no spec_stats line (run with CRUCIBLE_SPEC_STATS=1)")
            elif stat["mtp_origin"] != origin or stat["cached"] != want:
                problems.append(f"{where}: spec_stats cached {stat['cached']} mtp_origin {stat['mtp_origin']}, "
                                f"expected {want} and {origin}")
        summary.append(dict(step=r["step"], prompt=r["prompt"], prompt_tokens=r["prompt_tokens"], spec=r["spec"],
                            sampled=r["sampled"], cached_tokens=r["cached_tokens"], ttft=r["ttft"],
                            end_resume=resumed, tokens=len(r["output_ids"]), **({k: stat[k] for k in ("proposed", "accepted", "mtp_origin")}
                                                          if stat else {})))
    return problems, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("capture")
    one.add_argument("engine", type=Path)
    one.add_argument("checkpoint", type=Path)
    one.add_argument("output", type=Path)
    one.add_argument("--request-timeout", type=float, default=120)
    verdict = commands.add_parser("check")
    verdict.add_argument("capture", type=Path)
    verdict.add_argument("engine_log", type=Path)
    verdict.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = capture(args.engine, args.checkpoint, args.request_timeout)
        save(args.output, report)
        print(json.dumps(dict(status=report["status"], error=report.get("error"))))
        return 0 if report["status"] == "CAPTURED" else 1
    report = json.loads(args.capture.read_text())
    lines = args.engine_log.read_text(errors="replace").splitlines()
    problems, summary = check(report, lines)
    result = dict(status="FAIL" if problems else "PASS", problems=problems, engine_sha256=report.get("engine_sha256"),
                  cache_capture=capture_counts(lines), rows=summary)
    if args.summary:
        save(args.summary, result)
    print(json.dumps(result, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
