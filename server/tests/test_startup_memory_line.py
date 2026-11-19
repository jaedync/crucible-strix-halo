"""Batch serve reports host memory again once its slot states and checkpoints exist.

`memory ready` prints before batch serve allocates and warms its slots and checkpoints (about 13.8 GiB on the
4 x 106K profile), so it overstates the headroom of the cache. The `memory serve` lines come after them, before the
ready event. The existing lines stay as they are: other tools parse `memory ready`.
"""
from pathlib import Path

ROOT = Path(__file__).parents[2]
ENGINE = ROOT / "engine"


def test_batch_serve_reports_memory_after_slots_and_checkpoints():
    source = (ROOT / "engine/serve/batched_serve.h").read_text()
    body = source[source.index("inline int BatchedServe("):]
    report = body.index('Engine::ReportMemory("serve")')
    assert body.index("crucible::BatchSession<Engine> session(") < report
    assert body.index("std::make_unique<BatchPrefixCache>(") < report
    assert body.index("engine.WarmPlans();") < report
    assert report < body.index('"{\\"event\\":\\"ready\\"')


def test_the_ready_lines_keep_their_format():
    source = (ENGINE / "engine.hip").read_text()
    assert 'ReportMemory("ready");' in source
    assert '"memory %s: device free %zu / %zu bytes\\n"' in source
    assert '"memory %s: %s\\n"' in source
