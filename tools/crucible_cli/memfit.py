"""Stage 1 memory warnings only. This module never changes a layout or stops a process."""
from dataclasses import dataclass

from .processes import own_processes, process_name


@dataclass(frozen=True)
class MemoryHolder:
    pid: int
    name: str
    rss: int
    gtt: int


def _fields(path):
    try:
        return dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
    except (OSError, UnicodeError):
        return {}


def _size(text):
    parts = text.split()
    factors = {'B': 1, 'bytes': 1, 'kB': 1024, 'KiB': 1024, 'MiB': 1024 ** 2, 'GiB': 1024 ** 3}
    if len(parts) != 2 or parts[1] not in factors:
        return 0
    try:
        return max(0, int(parts[0])) * factors[parts[1]]
    except ValueError:
        return 0


def memory_holders(host, limit=5):
    holders = []
    for entry in own_processes(host):
        rss = _size(_fields(entry / 'status').get('VmRSS', ''))
        clients = {}
        try:
            for fd in (entry / 'fdinfo').iterdir():
                fields = _fields(fd)
                # Duplicated DRM fds describe the same client. Total and memory are alternative kernel field names.
                key = (fields.get('drm-pdev', '').strip(), fields.get('drm-client-id', 'unknown').strip())
                size = max(_size(fields.get('drm-memory-gtt', '')), _size(fields.get('drm-total-gtt', '')))
                clients[key] = max(clients.get(key, 0), size)
        except OSError:
            pass  # fdinfo can be inaccessible, or the process can exit during the scan.
        gtt = sum(clients.values())
        if rss or gtt:
            holders.append(MemoryHolder(int(entry.name), process_name(entry), rss, gtt))
    return sorted(holders, key=lambda p: (-max(p.rss, p.gtt), p.pid))[:limit]


def memory_advice(host):
    holders = memory_holders(host)
    names = '; '.join(f'PID {p.pid} {p.name}: {p.rss / 1e9:.1f} GB host, {p.gtt / 1e9:.1f} GB GPU' for p in holders)
    return ('Programs that hold the most memory: ' + (names or 'no readable process memory values') + '. '
            'Only processes owned by your user are listed. Host and GPU sizes can overlap. '
            'Crucible does not stop other programs. Choose the smaller 3.05 pack, a smaller KV cache (--ctx), '
            'a smaller conversation cache (--conversation-cache-mib), or fewer sessions (--streams). '
            'The engine checks the requested layout before it serves requests.')
