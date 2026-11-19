"""Every fixed structured output text of the engine is a known engine message (server/engine_messages.py), so its
crucible_engine_error line names it instead of "other"; and a refusal that is no client fault is not a 400."""
import pathlib
import re

from server import engine_messages, structured

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = ("engine/runtime/constraint_helper.h", "engine/runtime/constraint_xgr.cc", "engine/serve/batch_session.h",
           "engine/serve/serve.h")


def engine_texts():
    texts = set()
    for name in SOURCES:
        texts |= set(re.findall(r'"(constraint_[a-z]+: [^"\\]*)"', (ROOT / name).read_text()))
    return texts


def test_the_scan_finds_the_helper_texts():
    texts = engine_texts()
    assert "constraint_busy: the grammar compile queue is full (kCompileQueue = 8)" in texts
    assert "constraint_spec: the spec exceeds kMaxSpecBytes = 1 MiB" in texts
    assert len(texts) >= 25


def test_every_engine_constraint_text_is_known():
    missing = sorted(text for text in engine_texts() if text not in engine_messages.KNOWN)
    assert missing == []


def test_only_a_full_compile_queue_is_a_503():
    assert structured.refusal_status("constraint_busy: the grammar compile queue is full (kCompileQueue = 8)") == 503
    for text in engine_texts() - {"constraint_busy: the grammar compile queue is full (kCompileQueue = 8)"}:
        assert structured.refusal_status(text) == 400, text
