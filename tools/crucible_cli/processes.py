"""Read-only process checks. Never signal a process found by a scan."""
from pathlib import Path


def own_processes(host):
    try:
        entries = host.proc.iterdir()
        for entry in entries:
            try:
                if entry.name.isdecimal() and entry.stat().st_uid == host.uid:
                    yield entry
            except OSError:
                continue  # A process can exit between directory and status reads.
    except OSError:
        return


def process_name(entry):
    try:
        name = (entry / 'comm').read_text().strip()
    except OSError:
        name = 'unknown'
    return ''.join(c for c in name if c.isprintable())[:80] or 'unknown'


def crucible_servers(host, repo):
    """Find this user's hand-started servers, including other Crucible installs."""
    found = []
    for entry in own_processes(host):
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            cwd = (entry / 'cwd').resolve(strict=True)
            is_module = any(argv[i:i + 2] == [b'-m', b'server'] for i in range(len(argv) - 1))
            is_install = cwd == Path(repo).resolve() or ((cwd / 'crucible').is_file() and (cwd / 'server/api.py').is_file())
            if is_module and is_install:
                found.append((int(entry.name), process_name(entry)))
        except OSError:
            continue
    return sorted(found)


def remaining_server(host, s):
    found = crucible_servers(host, s.repo)
    if found:
        names = ', '.join(f'PID {pid} ({name})' for pid, name in found)
        return (f'A Crucible server still runs: {names}. This tool did not stop it. '
                'Use the terminal, service or container that started it.')
    if host.port_in_use(s.bind, s.port):
        try:  # /health needs no key: never send it (the listener may not be this server)
            code, _ = host.http.get(f'{s.base_url}/health', timeout=2)
        except OSError:
            code = 0
        if code in (200, 503):
            return (f'A server still answers on port {s.port}. Its PID is not available to this user. '
                    'Use the terminal, service or container that started it.')
    return None
