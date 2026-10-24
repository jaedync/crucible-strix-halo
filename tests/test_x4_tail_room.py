"""x4-tail: CRUCIBLE_PF_LAUNCH_ROWS=8192 allocates the bytes it allocated before.

The join rows come only from PrefillJoinSetting, which gives 0 for a set value without "+T" and the default 128 when the
variable is unset (tests/prefill_join_test.cpp), and PrefillRoomRows(launch, 0) is the launch rows. So the value 8192 is
the old size when every row buffer that held pf_launch_rows_ rows now holds pf_room_rows_ rows and nothing else reads
the join rows for a size.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def body(text, start):
    """The text of the function that starts at `start` (from its first brace to the matching brace)."""
    index = text.index(start)
    open_at = text.index('{', index)
    depth = 0
    for position in range(open_at, len(text)):
        depth += {'{': 1, '}': -1}.get(text[position], 0)
        if depth == 0:
            return text[open_at:position + 1]
    raise AssertionError(f'no end for {start}')


class RoomRows(unittest.TestCase):
    def setUp(self):
        self.engine = (ROOT / 'engine/engine.hip').read_text()
        self.prefill = (ROOT / 'engine/runtime/prefill_engine.inc').read_text()

    def test_room_rows_come_from_the_launch_and_join_settings_only(self):
        # The member declaration's '= 0' is not an assignment from a setting.
        assigned = [a for a in re.findall(r'\bpf_room_rows_\s*=[^=;][^;]*;', self.engine + self.prefill)
                    if not a.endswith('= 0;')]
        self.assertEqual(assigned, ['pf_room_rows_ = PrefillRoomRows(pf_launch_rows_, pf_join_rows_);'])
        joins = [a for a in re.findall(r'\bpf_join_rows_\s*=[^=;][^;]*;', self.engine + self.prefill)
                 if not a.endswith('= 0;')]
        self.assertEqual(joins, ['pf_join_rows_ = PrefillJoinSetting(launch_value, prefill_rows_);'])

    def test_row_buffers_use_room_rows(self):
        self.assertIn('const int capacity = std::max(kPackedRows, pf_room_rows_);', self.engine)
        arena = body(self.engine, 'void AllocateStageArena(int capacity)')
        self.assertIn('(size_t)pf_room_rows_', arena)
        init = body(self.prefill, 'void Engine::InitPrefill()')
        self.assertGreaterEqual(init.count('pf_room_rows_'), 9)
        for text in (arena, init):
            self.assertNotIn('pf_launch_rows_', text)
            self.assertNotIn('pf_join_rows_', text)

    def test_bulk_mtp_keeps_the_launch_rows(self):
        # MtpBulkFits keeps the launch rows (design section 7): a join does not change the bulk MTP rule.
        warm = body(self.prefill, 'bool Engine::MtpWarm(')
        self.assertIn('MtpBulkFits(count,pf_launch_rows_)', warm)

    def test_bad_join_value_stops_the_engine(self):
        # A bad "+T" value is refused at startup (PrefillLaunchSetting gives -1), not read as join 0.
        start = self.engine.index('pf_launch_rows_ = PrefillLaunchSetting(launch_value, prefill_rows_);')
        block = self.engine[start:start + 900]
        self.assertIn('if (pf_launch_rows_ < 0)', block)
        self.assertIn('std::exit(2);', block)
        self.assertIn('\\"+T\\" suffix', block)


if __name__ == '__main__':
    unittest.main()
