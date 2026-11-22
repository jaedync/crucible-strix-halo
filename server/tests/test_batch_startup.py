import io
import sys
import unittest
from unittest.mock import MagicMock, patch
import server.__main__ as startup

class BatchStartupTest(unittest.TestCase):
    def test_explicit_prefill_is_allowed_and_passed(self):
        for rows in (0, 1024, 4096):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', '--prefill', str(rows)]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['prefill'], rows)
                self.assertEqual(backend.call_args.kwargs['streams'], 2)

    def test_prefill_interleave_defaults_to_ten_and_can_be_turned_off(self):
        for extra, expected in (([], 10), (['--prefill-interleave', '0'], 0), (['--prefill-interleave', '6'], 6)):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', *extra]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['prefill_interleave'], expected)

    def test_automatic_prefill_stays_automatic(self):
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2']), \
                patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                patch('server.batch_backend.BatchedEngineBackend') as backend:
            self.assertEqual(startup.main(), 0)
            self.assertIsNone(backend.call_args.kwargs['prefill'])

    def test_cache_budget_is_passed_to_the_batch_engine(self):
        for mib in (0, 4096):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', '--conversation-cache-mib', str(mib)]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['conversation_cache_mib'], mib)

    def test_cache_option_reaches_the_engine_command(self):
        from server.batch_backend import BatchedEngineBackend
        with patch('server.backend.EngineBackend.__init__', return_value=None) as base:
            BatchedEngineBackend('engine', 'checkpoint', 128, streams=2, conversation_cache_mib=4096)
            self.assertIn(('--conversation-cache-mib', '4096'),
                          [tuple(base.call_args.kwargs['engine_options'][i:i + 2]) for i in range(0, 6, 2)])
            BatchedEngineBackend('engine', 'checkpoint', 128, streams=4, conversation_cache_mib=28672)
            self.assertIn(('--conversation-cache-mib', '28672'),
                          [tuple(base.call_args.kwargs['engine_options'][i:i + 2]) for i in range(0, 6, 2)])
            BatchedEngineBackend('engine', 'checkpoint', 128, streams=2)
            self.assertNotIn('--conversation-cache-mib', base.call_args.kwargs['engine_options'])
        for bad in (-1, 131073, True):
            with self.assertRaises(ValueError): BatchedEngineBackend('engine', 'checkpoint', streams=2, conversation_cache_mib=bad)

    def test_host_reserve_reaches_the_engine_only_when_set(self):
        # The engine keeps 8 GiB of MemAvailable free at each cache save unless the server names another reserve.
        for extra, expected in (([], ()), (['--host-reserve-mib', '4096'], ('--host-reserve-mib', '4096'))):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', '--conversation-cache-mib', '64',
                                            *extra]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['engine_options'], expected)
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--serial', '--conversation-cache-mib', '64', *extra]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch.object(startup, 'EngineBackend') as serial:
                self.assertEqual(startup.main(), 0)
                options = serial.call_args.kwargs['engine_options']
                self.assertEqual(tuple(options[-2:]) == expected, bool(expected))
                self.assertEqual('--host-reserve-mib' in options, bool(expected))

    def test_host_reserve_is_checked(self):
        for extra in (['--conversation-cache-mib', '64', '--host-reserve-mib', '1023'],
                      ['--conversation-cache-mib', '64', '--host-reserve-mib', '131073'],
                      ['--host-reserve-mib', '4096']):
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2'] + extra):
                with self.assertRaises(SystemExit): startup.main()

    def test_admission_reserve_reaches_the_engine_only_when_set(self):
        # The start-time reserve stays the engine's 9216 MiB unless the server names another; the default command is unchanged.
        for extra, expected in (([], ()), (['--admission-reserve-mib', '4096'], ('--admission-reserve-mib', '4096')),
                                (['--admission-reserve-mib', '1024'], ('--admission-reserve-mib', '1024')),
                                (['--admission-reserve-mib', '1048576'], ('--admission-reserve-mib', '1048576'))):
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '2', *extra]), \
                    patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                    patch('server.batch_backend.BatchedEngineBackend') as backend:
                self.assertEqual(startup.main(), 0)
                self.assertEqual(backend.call_args.kwargs['engine_options'], expected)

    def test_admission_reserve_is_checked(self):
        # Out of range: the parser error, exit 2, before the engine loads. The serial engine has no start-time reserve and
        # refuses the option, so the server refuses it with --serial.
        for extra, needle in ((['--streams', '2', '--admission-reserve-mib', '1023'], 'must be in [1024,1048576]'),
                              (['--streams', '2', '--admission-reserve-mib', '0'], 'must be in [1024,1048576]'),
                              (['--streams', '2', '--admission-reserve-mib', '1048577'], 'must be in [1024,1048576]'),
                              (['--serial', '--admission-reserve-mib', '4096'], 'do not use it with --serial')):
            stderr = io.StringIO()
            with patch.object(sys, 'argv', ['crucible', 'checkpoint', *extra]), patch.object(sys, 'stderr', stderr), \
                    patch.object(startup, 'ModelTokenizer') as tokenizer:
                with self.assertRaises(SystemExit) as stop: startup.main()
            self.assertEqual(stop.exception.code, 2)
            self.assertIn('--admission-reserve-mib', stderr.getvalue())
            self.assertIn(needle, stderr.getvalue())
            tokenizer.assert_not_called()

    def test_slot_classes_are_passed_and_checked(self):
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--streams', '8', '--ctx', '131072',
                                        '--short-streams', '5', '--short-ctx', '32768']), \
                patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'ThreadingHTTPServer', return_value=listener), \
                patch('server.batch_backend.BatchedEngineBackend') as backend:
            self.assertEqual(startup.main(), 0)
            self.assertEqual((backend.call_args.kwargs['short_streams'], backend.call_args.kwargs['short_ctx']), (5, 32768))
        for extra in (['--short-streams', '2', '--short-ctx', '64'], ['--streams', '2', '--short-streams', '2', '--short-ctx', '64'],
                      ['--streams', '2', '--short-streams', '1'], ['--streams', '2', '--short-ctx', '64'],
                      ['--streams', '2', '--short-streams', '1', '--short-ctx', '32768']):
            with patch.object(sys, 'argv', ['crucible', 'checkpoint'] + extra):
                with self.assertRaises(SystemExit): startup.main()

if __name__ == '__main__': unittest.main()
