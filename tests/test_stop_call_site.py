"""The group-signal path of a tier engine at its call site (engine/serve/batched_serve.h BatchedServe, review stop-fit L4).

tests/state_stop_test.cpp runs the pieces in real processes: KeepRunningOnTerm keeps the engine alive on SIGINT and
SIGTERM, and a ServeLineReader then ends an open input after the stop wait, so the flush runs. The server tests use the
fake engine (server/tests/test_stop_signals.py). The call site is GPU code that no CPU test can build, so this test
checks that BatchedServe uses those pieces in the order that the C++ test runs them:
1. the rule goes on only with the disk tier, before the ready event (a group signal after "ready" does not kill it);
2. the command loop reads a ServeLineReader (the reader that asks the stop hook), not std::cin;
3. after the loop, the flush runs with kStateFlushSeconds, and no return comes between the loop end and the flush.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine/serve/batched_serve.h"


def batched_serve_body(text):
    start = text.index("inline int BatchedServe(")
    return text[start:text.index("inline std::optional<int> TryBatchedServeCommand(", start)]


def stop_path_faults(body):
    """The faults of the stop path in the body of BatchedServe (an empty list when it is correct)."""
    faults = []
    disk = re.search(r"const bool disk = cache && cache->DiskOn\(\);", body)
    rule = re.search(r"^\s*if \(disk\) KeepRunningOnTerm\(\);$", body, re.M)
    ready = body.find('"{\\"event\\":\\"ready\\"')
    reader = re.search(r"crucible::ServeLineReader reader\(0, ", body)
    loop = re.search(r"^\s*while \(reader\.Next\(line, oversized\)\) \{$", body, re.M)
    flush = re.search(r"^  if \(cache\) cache->FlushToDisk\(kStateFlushSeconds, ", body, re.M)
    if body.count("KeepRunningOnTerm()") != 1: faults.append("KeepRunningOnTerm is not called once")
    if not (disk and rule): faults.append("the rule is not on with the disk tier only")
    if ready < 0: faults.append("no ready event")
    if rule and disk and not disk.end() <= rule.start(): faults.append("the rule comes before the disk test")
    if rule and ready >= 0 and not rule.end() < ready: faults.append("the rule comes after the ready event")
    if not reader: faults.append("the loop does not read a ServeLineReader")
    if "std::cin" in body: faults.append("the loop reads std::cin")
    if not loop: faults.append("the command loop does not read the ServeLineReader")
    if rule and loop and not rule.end() < loop.start(): faults.append("the rule comes after the command loop")
    if not flush: faults.append("no flush with kStateFlushSeconds")
    if loop and flush:
        # The loop ends at its closing brace at two spaces; only that brace and comments may come before the flush.
        end = body.rindex("\n  }\n", loop.end(), flush.start())
        between = [line.strip() for line in body[end + len("\n  }\n"):flush.start()].splitlines()]
        if any(line and not line.startswith("//") for line in between):
            faults.append("a statement comes between the command loop and the flush")
        if not loop.start() < flush.start(): faults.append("the flush comes before the command loop")
    return faults


def test_the_call_site_runs_the_stop_path_of_the_cpu_test():
    assert stop_path_faults(batched_serve_body(SOURCE.read_text())) == []


def test_the_check_finds_each_fault():
    body = batched_serve_body(SOURCE.read_text())
    rule = "  if (disk) KeepRunningOnTerm();\n"
    flush = "  if (cache) cache->FlushToDisk(kStateFlushSeconds, "
    assert rule in body and flush in body
    mutants = {
        "no rule": body.replace(rule, ""),
        "rule after the loop": body.replace(rule, "").replace(flush, rule + flush),
        "rule without the disk tier": body.replace(rule, "  KeepRunningOnTerm();\n"),
        "return before the flush": body.replace(flush, "  return 0;\n" + flush),
        "std::cin loop": body.replace("while (reader.Next(line, oversized)) {", "while (std::getline(std::cin, line)) {"),
    }
    for name, mutant in mutants.items():
        assert mutant != body and stop_path_faults(mutant), name
