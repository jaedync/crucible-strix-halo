"""Engine death closes the HTTP listener in service and hand-start modes."""
import os
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest
from server import __main__ as startup


@pytest.mark.parametrize('serial', [False, True])
@pytest.mark.parametrize('systemd', [False, True])
def test_dead_engine_exits_and_releases_port(serial, systemd, tmp_path, capsys):
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    argv = ['crucible', 'checkpoint', '--port', str(port), '--no-update-check']
    if serial:
        argv += ['--serial']
    if systemd:
        argv += ['--systemd']
    backend = MagicMock()
    backend.fatal = 'The engine stopped. Check its stderr log.'
    with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'NOTIFY_SOCKET': str(tmp_path / 'notify')}), \
         patch.object(startup, 'ModelTokenizer'), patch.object(startup, 'EngineBackend', return_value=backend), \
         patch('server.batch_backend.BatchedEngineBackend', return_value=backend):
        # A bounded serve loop makes the old behavior fail instead of hanging the test.
        original = startup.ThreadingHTTPServer.serve_forever
        def bounded_serve(server):
            import threading
            timer = threading.Timer(1, server.shutdown)
            timer.start()
            try:
                original(server, poll_interval=.01)
            finally:
                timer.cancel()
                timer.join()
        with patch.object(startup.ThreadingHTTPServer, 'serve_forever', bounded_serve):
            assert startup.main() == 3
    assert 'The engine stopped. The server exits and frees its port.' in capsys.readouterr().err
    backend.close.assert_called_once()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
