"""a device write to a slot ends the lazy disk reads of the slot's rows BEFORE the write is enqueued.

The reader uses its own stream, the write the null stream: a write that is enqueued first can reach rows that a
lazy read has not copied yet. The CPU tests cannot run the hooks of batched_serve.h, so this test reads the order of
the calls in their source."""
import re
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parent.parent / "engine/serve/batched_serve.h").read_text()


def body(start):
    """The text from `start` to the end of its brace block."""
    begin = SOURCE.index(start)
    depth, i = 0, SOURCE.index("{", begin)
    first = i
    while True:
        depth += {"{": 1, "}": -1}.get(SOURCE[i], 0)
        if depth == 0:
            return SOURCE[first:i]
        i += 1


def call_argument(text, name):
    """The argument text of the first call `name(...)` in `text` (parentheses balanced)."""
    begin = text.index(name + "(") + len(name) + 1
    depth, i = 1, begin
    while depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    return " ".join(text[begin:i - 1].split())


class LazyOrderTest(unittest.TestCase):
    def test_checkpoint_load_ends_reads_before_the_restore(self):
        text = body("hooks.checkpoint_load = ")
        self.assertLess(text.index("LazyLimit("), text.index("SlotCheckpointIO::Restore("))

    def test_checkpoint_load_limit_is_the_settled_length(self):
        # review 2 MEDIUM-3: the restore writes the rows from the point's length less its replay rows; a limit of the
        # whole length would keep a lazy source alive over rows that the restore rewrites.
        limit = call_argument(body("hooks.checkpoint_load = "), "LazyLimit")
        self.assertRegex(limit, r"^slot, .*point\.hist\.size\(\) > size_t\(std::max\(point\.replay, 0\)\) \? "
                                r"point\.hist\.size\(\) - size_t\(std::max\(point\.replay, 0\)\) : 0$")

    def test_overwrite_hook_ends_the_slot_reads(self):
        # review 2 MEDIUM-3: a new session that overwrites a slot from row 0 ends every lazy read of it first.
        line = next(l for l in SOURCE.splitlines() if "hooks.overwrite = " in l)
        self.assertIn("LazyEnd(slot)", line.split("//")[0])

    def test_limit_drops_the_lost_jobs_when_a_source_ended(self):
        # round 3 item 4: LazyLimit is in a GPU-only class, so the CPU test reads its source. A source that ended
        # removes the jobs that wait for the writer (finding 7); no ended source means no queue walk.
        text = body("void LazyLimit(int slot, size_t kept)")
        self.assertRegex(text, r"if \(ended && disk_\) disk_->Disk\(\)\.DropLost\(\);")
        self.assertLess(text.index("Invalidate()"), text.index("DropLost()"))

    def test_frontier_copy_ends_reads_before_the_copy(self):
        text = body("crucible::BatchRestore FrontierCopy(")
        self.assertLess(text.index("LazyEnd(target)"), text.index("SlotCheckpointIO::CopyTo("))

    def test_host_restore_ends_reads_before_any_write(self):
        text = body("crucible::BatchRestore Restore(const crucible::ServeRequest& r)")
        self.assertLess(text.index("LazyEnd(Slot())"), text.index("miss.tier"))

    def test_snapshot_write_ends_reads_first(self):
        text = body("void WriteSnapshot(")
        self.assertLess(text.index("LazyEnd(slot)"), text.index("ConversationStateIO"))


if __name__ == "__main__":
    unittest.main()
