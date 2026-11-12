"""Capture the top-20 log-probabilities of one engine at the scored positions of kl/texts.json (the quality table of the README).

One scoring position per request, the same method for Crucible, Halogen and Gufo. Standard library plus tokenizers.

Usage: kl_capture.py --engine crucible|halogen|gufo --url http://127.0.0.1:PORT --texts kl/texts.json --tokenizer tokenizer.json
                --stride 8 --passes 4 --out rows.jsonl [--limit N] [--budget SECONDS] [--gufo-logprobs]
Each row: chunk, p (scored position), real (hex bytes of the real next token), top (hex bytes, logprob), usage.
Rows resume: a second run continues the same file. Gufo before v0.27.0 rejects logprobs: its rows have no top (greedy text only).
Score the rows with kl_score.py.
"""
import argparse, json, sys, time, urllib.error, urllib.request
from tokenizers import Tokenizer

def byte_table():
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table = {chr(b): b for b in keep}
    extra = 0
    for b in range(256):
        if b not in keep:
            table[chr(256 + extra)] = b
            extra += 1
    return table

def token_bytes(tok, table, token_id):
    text = tok.id_to_token(token_id)
    return bytes(table[c] for c in text) if all(c in table for c in text) else text.encode()

MAX_ERRORS = 20


def gufo_top(reply):
    """(top, partial) from a Gufo raw Completions reply made with `logprobs: 20` (Gufo v0.27.0 and later). `top` is a list of
    (hex bytes of the token text, log-probability), best first. The reply gives a dict from token text to log-probability, so
    a token with a partial UTF-8 sequence has the text "" and cannot be told apart: it is counted in `partial`, not scored,
    and its mass falls into the rest bucket of the KL (a lower bound stays a lower bound)."""
    logprobs = reply["choices"][0].get("logprobs")
    if not logprobs or not logprobs.get("top_logprobs"):
        raise ValueError("the reply has no logprobs object: the server does not support logprobs, or they were not asked for")
    entries = logprobs["top_logprobs"][0]
    if len(entries) > 21:
        raise ValueError(f"{len(entries)} entries in top_logprobs, the most is 21 (the sampled token and 20 more)")
    top = sorted(((text.encode().hex(), lp) for text, lp in entries.items() if text), key=lambda t: -t[1])
    return top, sum(1 for text in entries if not text)


def post(url, body, timeout=1800):
    request = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)

def ask(args, model, tok, table, ids, check=False):
    if args.engine == "crucible":
        body = dict(model=model, prompt=ids, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20, spec=0)
        reply = post(args.url, body)
        entry = reply["choices"][0]["logprobs"]["content"][0]
        top = [(token_bytes(tok, table, t["token_id"]).hex(), t["logprob"]) for t in entry["top_logprobs"]]
        if check:  # the option must not change the token: same prompt without it, compare IDs
            plain = post(args.url, dict(model=model, prompt=ids, max_tokens=1, temperature=0, spec=0, return_token_ids=True))
            reply["plain_match"] = plain["choices"][0]["token_ids"][0] == entry["token_id"] == reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["token_id"]
    elif args.engine == "gufo" and args.gufo_logprobs:  # Gufo v0.27.0 and later: raw Completions logprobs 20, prompt as text
        reply = post(args.url, dict(model=model, prompt=tok.decode(ids), max_tokens=1, temperature=0, logprobs=20))
        top, partial = gufo_top(reply)
        reply["greedy"] = dict(text=reply["choices"][0]["text"].encode().hex(), finish=reply["choices"][0].get("finish_reason"),
                               partial=partial)
    elif args.engine == "gufo":  # greedy text only; the server starts with --speculative off
        reply = post(args.url, dict(model=model, prompt=tok.decode(ids), max_tokens=1, temperature=0))
        choice = reply["choices"][0]
        reply["greedy"] = dict(text=choice["text"].encode().hex(), finish=choice.get("finish_reason"))
        top = []
    else:
        body = dict(model=model, prompt=tok.decode(ids), max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
                    drafter="serial")
        reply = post(args.url, body)
        entry = reply["choices"][0]["logprobs"]["content"][0]
        top = [(bytes(t["bytes"]).hex(), t["logprob"]) for t in entry["top_logprobs"]]
    return top, reply.get("usage", {}), reply.get("timings"), reply.get("plain_match"), reply.get("greedy")


def read_rows(path):
    """The rows of a file; a last line cut by a lease kill is skipped (the position is asked again)."""
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def pass_order(passes):
    """Bit-reversed offsets: after each whole pass, the scored positions are an even stride over every chunk."""
    if passes not in (1, 2, 4, 8):
        raise ValueError("passes must be 1, 2, 4 or 8")
    bits = passes.bit_length() - 1
    return [int(format(k, f"0{bits}b")[::-1], 2) if bits else 0 for k in range(passes)]


def schedule(chunks, stride, passes, probe=0):
    """(pass, chunk index, position) in scoring order. Pass k takes the positions with (p // stride) % passes equal to
    its offset; inside a pass, chunk by chunk and position by position, so each prompt extends the one before."""
    out = []
    for k, offset in enumerate(pass_order(passes)):
        for index, chunk in enumerate(chunks):
            if probe and index:
                break
            for p in (chunk["positions"][-probe:] if probe else chunk["positions"]):
                if p % stride == 0 and (p // stride) % passes == offset:
                    out.append((k, index, p))
    return out

def main():
    parser = argparse.ArgumentParser()
    for name in ("engine", "url", "texts", "tokenizer", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stride", type=int, default=8, choices=(8, 16))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check", type=int, default=0)
    parser.add_argument("--probe", type=int, default=0, help="score only the last N stride-8 positions of chunk 0")
    parser.add_argument("--model", default="")
    parser.add_argument("--gufo-logprobs", action="store_true", help="gufo: ask raw Completions logprobs 20 (Gufo v0.27.0 and later)")
    parser.add_argument("--passes", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument("--budget", type=float, default=0, help="stop before a new request after this many seconds")
    args = parser.parse_args()
    if args.engine not in ("crucible", "halogen", "gufo"):
        parser.error("--engine must be crucible, halogen or gufo")
    tok, table = Tokenizer.from_file(args.tokenizer), byte_table()
    model = args.model or json.load(urllib.request.urlopen(args.url + "/v1/models"))["data"][0]["id"]
    data = json.load(open(args.texts))
    done = 0
    try:  # resume: a lease that ended early leaves its rows, and the next lease skips them
        # A refused position (row with greedy.error) is asked again by the next lease.
        seen = {(r["chunk"], r["p"]) for r in read_rows(args.out) if "error" not in r.get("greedy", {})}
    except FileNotFoundError:
        seen = set()
    began, errors = time.monotonic(), 0
    with open(args.out, "a+b") as tail:  # end a cut last line, so the next row starts on its own line
        if tail.tell() and (tail.seek(-1, 2), tail.read(1))[1] != b"\n":
            tail.write(b"\n")
    with open(args.out, "a") as out:
        for k, index, p in schedule(data["chunks"], args.stride, args.passes, args.probe):
            chunk = data["chunks"][index]
            if (index, p) in seen:
                continue
            if args.limit and done >= args.limit:
                return
            if args.budget and time.monotonic() - began >= args.budget:
                print(f"budget {args.budget:.0f} s reached after {done} rows", file=sys.stderr)
                return
            start = time.perf_counter()
            try:
                top, usage, timings, match, greedy = ask(args, model, tok, table, chunk["ids"][:p], done < args.check)
            except urllib.error.HTTPError as error:  # one refused position must not stop the run; analyze.py drops it
                errors += 1
                if errors >= MAX_ERRORS:  # a broken server, not one position: stop, and the next lease retries
                    sys.exit(f"{errors} refused requests in a row, last HTTP {error.code}")
                top, usage, timings, match, greedy = [], {}, None, None, dict(error=error.code)
            else:
                errors = 0
            row = dict(chunk=index, kind=chunk["kind"], p=p, real=token_bytes(tok, table, chunk["ids"][p]).hex(),
                       top=top, usage=usage, seconds=round(time.perf_counter() - start, 4))
            if args.passes > 1:
                row["pass"] = k
            if greedy is not None:
                row["greedy"] = greedy
            if match is not None:
                row["plain_match"] = match
            if timings:
                row["timings"] = timings
            out.write(json.dumps(row) + "\n"); out.flush()
            done += 1

if __name__ == "__main__":
    main()
