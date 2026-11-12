#!/usr/bin/env python3
"""Judge of the end-logits unit (tools/end_logits_probe.py). The verdict is about the answer; the rows are data.

  PASS  every arm gave rows. The reason names the outcome of each arm:
          scoring: OK (the End-resumed scoring list R5 equals the cold one R8), STALE (it differs), NO-END (R5
                   was no End resume, so the check is empty)
          sampled: OK (R2 = R1, R7 = R3, R6 = R9, and no reply is empty), DIFF (a sampled regenerate differs or
                   is empty), NO-END (all equal, but R6 was no End resume, so the End check is empty)
Row keys (tools/end_logits_probe.py one_prompt): r1, r2, r3, r7 (chat token IDs), auto = R5 and cold = R8 (top-5
pairs), s_auto = R6 and s_cold = R9 (sampled completion IDs), cached = the cached tokens of R2, R3, R5, R6.
  KILL  no arm gave a row (a setup fault).
  ERROR no summary.
Reads NU_DIR/summary.json. Prints VERDICT and NOTE lines only.
"""
import json
import os
import sys


def outcome(arm):
    rows = arm.get("rows") or []
    scoring, sampled = [], []
    for row in rows:
        resumed = row["cached"][2] == row["prompt"]
        same = row["auto"] == row["cold"] and bool(row["auto"])
        scoring.append("NO-END" if not resumed else "OK" if same else "STALE")
        regen_ok = row["r1"] == row["r2"] and row["r3"] == row["r7"] and all(row[k] for k in ("r1", "r2", "r3", "r7"))
        if "s_auto" in row:  # R6 against R9: the sampled completion on the End point (needs an End resume to count)
            regen_ok = regen_ok and row["s_auto"] == row["s_cold"] and bool(row["s_auto"])
            if regen_ok and row["cached"][3] != row["prompt"]:
                sampled.append("NO-END")
                continue
        sampled.append("OK" if regen_ok else "DIFF")
    pick = lambda marks: next((m for m in ("STALE", "DIFF", "NO-END") if m in marks), "OK") if marks else "NONE"
    return len(rows), pick(scoring), pick(sampled)


def main():
    path = os.path.join(os.environ.get("NU_DIR", ""), "summary.json")
    try:
        with open(path, encoding="utf-8") as handle:
            arms = json.load(handle).get("arms") or []
    except (OSError, ValueError) as error:
        print(f"VERDICT ERROR cannot read {path}: {error}")
        return 1
    if not arms:
        print("VERDICT ERROR the summary holds no arm")
        return 1
    parts = []
    for arm in arms:
        count, scoring, sampled = outcome(arm)
        print(f"NOTE arm {arm.get('arm')}: prompts {count}, scoring {scoring}, sampled {sampled}, error {arm.get('error')}")
        parts.append(f"{arm.get('arm')}: scoring {scoring}, sampled {sampled}")
    if all(not arm.get("rows") for arm in arms):
        print("VERDICT KILL no arm gave a row: the setup, not the claim, is on trial")
        return 0
    print("VERDICT PASS " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
