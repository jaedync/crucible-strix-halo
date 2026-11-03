#!/usr/bin/env python3
"""GPU exactness checks for the vision tower of third-party checkpoints (issue 3).

  python3 tools/vision_bf16_check.py prep CKPT WORK        synthetic cases: request lines and raw pixel files (CPU)
  python3 tools/vision_bf16_check.py emb BENCH WORK A B [STOCK]
  python3 tools/vision_bf16_check.py tok ENGINE_NEW ENGINE_REF WORK A B
  python3 tools/vision_bf16_check.py refuse ENGINE CKPT    (no model load)

Test folders: tools/vision_bf16_ckpt.py (A holds the touched vision tensors as F16 words, B the same values as
BF16 words) and tools/vision_hf_ckpt.py (H: the stock pack plus an unused HF-format tower shard in the index).
Every pass must give equal bytes:
  emb     vision_bench (the new build) writes the bf16 embeddings of each image from A and from B: they must be byte equal.
          The pair (stock pack, H) works the same. With STOCK it also prints how many files differ from A (information).
  tok     greedy tokens of the image requests: the new engine on A, the new engine on B and the reference (trunk)
          engine on A. The three must be equal. The pair (stock pack, H) works the same.
  refuse  `serve --vision 1` on a folder with the HF tower and no vision_k6.safetensors exits 2 with the message
          that names the file and the download command, before the model loads.
Each command prints one verdict line that starts with PASS or FAIL and exits 1 on FAIL.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "bench", "vision_graded", "set", "cases.json")


def prep(ckpt, work):
    os.makedirs(work, exist_ok=True)
    seen, picked = set(), []
    for case in json.load(open(CASES)):  # one case per image
        if case["image"] not in seen:
            seen.add(case["image"])
            picked.append(case)
    small = os.path.join(work, "cases", "cases.json")
    os.makedirs(os.path.dirname(small), exist_ok=True)
    for case in picked:
        link = os.path.join(os.path.dirname(small), case["image"])
        if not os.path.exists(link):
            os.symlink(os.path.join(os.path.dirname(CASES), case["image"]), link)
    json.dump(picked, open(small, "w"))
    lines = os.path.join(work, "lines.jsonl")
    subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "vision_probe.py"), "lines", ckpt, small, lines,
                    "--max-tokens", "24", "--spec", "0"], check=True)
    inputs = []
    for raw in open(lines):
        for image in json.loads(raw).get("images", []):
            inputs.append(f"{image['path']}:{image['height']}:{image['width']}")
    json.dump(inputs, open(os.path.join(work, "bench_inputs.json"), "w"))
    print(f"prep: {len(picked)} requests, {len(inputs)} pixel files")


def read_dir(path):
    return {name: open(os.path.join(path, name), "rb").read() for name in sorted(os.listdir(path))}


def emb(bench, work, a, b, stock=None):
    inputs = json.load(open(os.path.join(work, "bench_inputs.json")))
    outs = {}
    for label, ckpt in (("A", a), ("B", b)) + ((("S", stock),) if stock else ()):
        out = os.path.join(work, "emb-" + os.path.basename(os.path.normpath(ckpt)))
        if os.path.isdir(out):
            for old in os.listdir(out):
                os.remove(os.path.join(out, old))
        os.makedirs(out, exist_ok=True)
        run = subprocess.run([bench, ckpt, out, *inputs, "--repeat", "1"], capture_output=True, text=True, timeout=900)
        if run.returncode:
            print("FAIL vision_bench on " + label + " exited " + str(run.returncode) + ": " + run.stderr[-300:])
            return 1
        outs[label] = read_dir(out)
    if len(outs["A"]) != len(inputs) or set(outs["A"]) != set(outs["B"]):
        print("FAIL embedding files differ between A and B: " + str(sorted(set(outs["A"]) ^ set(outs["B"]))))
        return 1
    bad = [n for n in outs["A"] if outs["A"][n] != outs["B"][n]]
    total = sum(len(v) for v in outs["A"].values())
    if bad or total == 0:
        print(f"FAIL embeddings from B differ from A in {len(bad)} of {len(outs['A'])} files: {bad[:3]}")
        return 1
    note = ""
    if "S" in outs:
        note = f"; stock differs from A in {sum(outs['S'][n] != outs['A'][n] for n in outs['A'])} files (information)"
    print(f"PASS embeddings A equal B: {len(outs['A'])} files, {total} bytes{note}")
    return 0


def serve(engine, ckpt, lines, out, timeout=1500):
    with open(lines) as stdin, open(out, "w") as stdout:
        run = subprocess.run([engine, "serve", ckpt, "--vision", "1", "--ctx", "8192"], stdin=stdin, stdout=stdout,
                             stderr=subprocess.PIPE, text=True, timeout=timeout)
    tokens, current = [], []
    for raw in open(out):
        event = json.loads(raw)
        if event.get("event") == "token":
            current.append(event["id"])
        elif event.get("event") in ("done", "error"):
            tokens.append(dict(ids=current, error=event.get("error")))
            current = []
    return run.returncode, tokens, run.stderr


def tok(new, ref, work, a, b):
    lines = os.path.join(work, "lines.jsonl")
    n = sum(1 for _ in open(lines))
    results = {}
    for label, engine, ckpt in (("new-A", new, a), ("new-B", new, b), ("ref-A", ref, a)):
        rc, tokens, err = serve(engine, ckpt, lines, os.path.join(work, f"events-{label}.jsonl"))
        print(f"{label}: exit {rc}, {len(tokens)} replies, {sum(len(t['ids']) for t in tokens)} tokens", flush=True)
        if rc or len(tokens) != n or any(t["error"] for t in tokens):
            print(f"FAIL {label} did not finish: exit {rc}, {err[-300:]}")
            return 1
        results[label] = [t["ids"] for t in tokens]
    if any(len(i) < 8 for i in results["new-A"]):
        print("FAIL a reply is too short to compare")
        return 1
    for label in ("new-B", "ref-A"):
        if results[label] != results["new-A"]:
            print(f"FAIL tokens of {label} differ from new-A")
            return 1
    print(f"PASS tokens equal for new-A, new-B and ref-A: {n} requests, {sum(map(len, results['new-A']))} tokens")
    return 0


def refuse(engine, ckpt):
    run = subprocess.run([engine, "serve", ckpt, "--vision", "1"], stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=300)
    need = ("vision_k6.safetensors", "hf download turboderp/Qwen3.8-Flash-Next-exl3 vision_k6.safetensors --revision 4.05bpw_h6_ng6")
    if run.returncode != 2 or not all(part in run.stderr for part in need) or '"event":"ready"' in run.stdout:
        print(f"FAIL exit {run.returncode}, expected 2 with the missing-file message: {run.stderr[-300:]}")
        return 1
    print("PASS refused with exit 2 and the vision_k6.safetensors message, before the model loaded")
    return 0


def main():
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "prep":
        return prep(*args) or 0
    if cmd == "emb":
        return emb(*args)
    if cmd == "tok":
        return tok(*args)
    if cmd == "refuse":
        return refuse(*args)
    print(__doc__)
    return 0 if cmd in ("-h", "--help") else 2


if __name__ == "__main__":
    sys.exit(main())
