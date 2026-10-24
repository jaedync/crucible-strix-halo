"""Use the central runtime allowlist. Do not store unknown variable values."""
import hashlib
import importlib.util
from pathlib import Path

CENTRAL = Path(__file__).resolve().parents[2]/'tools/gates.py'
_LOADED_POLICY = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__).resolve(), CENTRAL)}
_spec = importlib.util.spec_from_file_location('crucible_central_runtime_environment', CENTRAL)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


def policy_hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), CENTRAL)}


# Read by the server only (server/update_check.py). It cannot change a measurement, so it is no unknown switch.
SERVER_ONLY_VARIABLES = frozenset({'CRUCIBLE_NO_UPDATE_CHECK'})


def snapshot(env):
    hashes = policy_hashes()
    if hashes != _LOADED_POLICY:
        raise ValueError('environment policy changed after import')
    known = _shared.runtime_environment(env)
    unknown = sorted(key for key in env if key.startswith('CRUCIBLE_') and key not in known
                     and key not in SERVER_ONLY_VARIABLES)
    return {'environment': {**known, **dict.fromkeys(unknown, '[REDACTED]')},
            'unknown_crucible_keys': unknown, 'environment_policy_hashes': hashes}


def require_known(state):
    # An unknown name can be a new engine switch. Reject all such names conservatively.
    if state['unknown_crucible_keys']:
        names = ', '.join(state['unknown_crucible_keys'])
        raise ValueError(f'unknown Crucible environment keys: {names}; unset them before this run')
