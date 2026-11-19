"""Read-only process checks. Never signal a process found by a scan."""
import os
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
    """This user's hand-started servers of the install REPO (python -m server with REPO as its working folder)."""
    found = []
    for entry in own_processes(host):
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            cwd = (entry / 'cwd').resolve(strict=True)
            is_module = any(argv[i:i + 2] == [b'-m', b'server'] for i in range(len(argv) - 1))
            if is_module and cwd == Path(repo).resolve():
                found.append((int(entry.name), process_name(entry)))
        except OSError:
            continue
    return sorted(found)


def _listen_inodes(host, port):
    """The socket inodes that listen on PORT (TCP, IPv4 and IPv6), from /proc/net/tcp and tcp6."""
    inodes = set()
    for name in ('tcp', 'tcp6'):
        try:
            lines = (host.proc / 'net' / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            try:
                if len(fields) > 9 and fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                    inodes.add(fields[9])
            except (IndexError, ValueError):
                continue
    return inodes


def port_holders(host, port):
    """(pid, name) of this user's processes that listen on PORT. Other users' processes are not readable."""
    inodes = _listen_inodes(host, port)
    found = []
    if not inodes:
        return found
    for entry in own_processes(host):
        try:
            links = [os.readlink(fd) for fd in (entry / 'fd').iterdir()]
        except OSError:
            continue  # the process exited, or its fd folder is not readable
        if any(link.startswith('socket:[') and link[8:-1] in inodes for link in links):
            found.append((int(entry.name), process_name(entry)))
    return sorted(found)


def _names(found):
    return ', '.join(f'PID {pid} ({name})' for pid, name in found)


def remaining_server(host, s):
    """A message when a server that this tool did not stop still runs, else None. It names the servers of this install
    and a server that answers on the configured port. A server of another install on another port does not count."""
    found = crucible_servers(host, s.repo)
    if found:
        return (f'A Crucible server still runs: {_names(found)}. This tool did not stop it. '
                'Use the terminal, service or container that started it.')
    if host.port_in_use(s.bind, s.port):
        try:  # /health needs no key: never send it (the listener may not be this server)
            code, _ = host.http.get(f'{s.base_url}/health', timeout=2)
        except OSError:
            code = 0
        if code in (200, 503):
            holders = port_holders(host, s.port)
            if holders:
                return (f'A server still answers on port {s.port}: {_names(holders)}. This tool did not start it. '
                        'Use the terminal, service or container that started it.')
            return (f'A server still answers on port {s.port}. Its PID is not available to this user. '
                    'Use the terminal, service or container that started it.')
    return None
