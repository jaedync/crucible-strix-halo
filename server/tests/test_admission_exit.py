import sys
from unittest.mock import patch

import pytest
from server import __main__ as startup
from server.backend import EngineBackend
from server.engine_messages import AdmissionRefused


def test_admission_event_is_a_typed_start_error():
    backend = object.__new__(EngineBackend)
    backend.context = 128
    with pytest.raises(AdmissionRefused, match='batch memory does not fit'):
        backend._validate_ready({'event': 'admission_refused', 'error': 'batch memory does not fit: test'})


def test_admission_refusal_exits_4(capsys):
    with patch.object(sys, 'argv', ['crucible', 'checkpoint', '--no-update-check']), \
         patch.object(startup, 'ModelTokenizer'), \
         patch('server.batch_backend.BatchedEngineBackend', side_effect=AdmissionRefused('batch memory does not fit: test')):
        assert startup.main() == 4
    assert 'server error: batch memory does not fit: test' in capsys.readouterr().err
