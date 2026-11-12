"""CPU checks of tools/state_digest.py: a digest of the state and block files of a disk-tier folder that leaves out the identity
. A state file and a block file name the identity of the engine that wrote them, so two engines with
another identity never write equal files. The digest drops the identity and keeps every other byte."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import state_digest as sd

ALIGN = 4096
STATE_MAGIC, BLOCK_MAGIC = 0x3254415453435243, 0x314B4F4C42435243


def round_up(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def fixed(magic, version, header_bytes, used, payload, index_used):
    return struct.pack('<QIIQQQQQ', magic, version, 0, header_bytes, used, payload, 1 << 20, index_used)


def state_file(identity, body=b'index-body', resume=b'resume-body', rest=b'R' * 100, payload=777):
    """A file laid out like BuildStateHeader: fixed(56), index = Str(identity) + body, sum(8), resume, sum(8), pad, rest."""
    ident = struct.pack('<Q', len(identity)) + identity.encode()
    index_used = 56 + len(ident) + len(body) + 8
    used = index_used + len(resume) + 8
    total = round_up(used)
    head = fixed(STATE_MAGIC, 2, total, used, payload, index_used) + ident + body + b'\x11' * 8 + resume + b'\x22' * 8
    return head + b'\0' * (total - len(head)) + rest + b'\0' * (round_up(len(rest)) - len(rest))


def block_file(identity, tenant=b'T' * 16, digest=b'D' * 32, payload=b'P' * 5000):
    body = struct.pack('<Q', len(identity)) + identity.encode() + tenant + digest + struct.pack('<Q', 1) + b'\x33' * 8
    used = 56 + len(body) + 8
    total = round_up(used)
    head = fixed(BLOCK_MAGIC, 1, total, used, len(payload), used) + body + b'\x44' * 8
    return head + b'\0' * (total - len(head)) + payload + b'\0' * (round_up(len(payload)) - len(payload))


def folder(root, name, identity, **kw):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / (hashlib.sha256(identity.encode()).hexdigest() + '.state')).write_bytes(state_file(identity, **{k: v for k, v in kw.items() if k != 'block'}))
    (d / (hashlib.sha256(b'b' + identity.encode()).hexdigest() + '.block')).write_bytes(block_file(identity, **kw.get('block', {})))
    (d / '.lock').write_bytes(b'')
    return d


class DigestTests(unittest.TestCase):
    def test_another_identity_gives_the_same_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = folder(tmp, 'a', 'a' * 64)
            b = folder(tmp, 'b', 'b' * 64)
            self.assertNotEqual(next(a.glob('*.state')).read_bytes(), next(b.glob('*.state')).read_bytes())
            self.assertEqual(sd.digest(a), sd.digest(b))
            self.assertEqual(sd.compare(a, b), [])

    def test_every_other_byte_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = folder(tmp, 'a', 'a' * 64)
            for i, kw in enumerate((dict(body=b'index-bodY'), dict(resume=b'resumE-body'), dict(rest=b'R' * 99 + b'S'),
                                    dict(payload=778), dict(block=dict(payload=b'P' * 4999 + b'Q')),
                                    dict(block=dict(digest=b'E' * 32)), dict(block=dict(tenant=b'U' * 16)))):
                other = folder(tmp, f'o{i}', 'c' * 64, **kw)
                self.assertTrue(sd.compare(a, other), kw)

    def test_a_missing_file_is_a_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = folder(tmp, 'a', 'a' * 64)
            b = folder(tmp, 'b', 'b' * 64)
            next(b.glob('*.block')).unlink()
            self.assertTrue(sd.compare(a, b))
            self.assertTrue(sd.compare(b, a))

    def test_identity_length_may_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = folder(tmp, 'a', 'a' * 64)
            b = folder(tmp, 'b', 'bb' * 40)
            self.assertEqual(sd.compare(a, b), [])

    def test_a_root_with_one_identity_folder_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'
            folder(root, '-'.join(['0123456789abcdef'] * 3), 'a' * 64)
            (root / 'write-ledger').write_bytes(b'x')
            self.assertEqual(len(sd.digest(root)), 2)
            folder(root, '-'.join(['fedcba9876543210'] + ['0123456789abcdef'] * 2), 'b' * 64)
            with self.assertRaises(ValueError):
                sd.digest(root)

    def test_damaged_files_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (('short', b'CRCSTAT2'), ('magic', b'\0' * 8192),
                               ('cut', state_file('a' * 64)[:5000])):
                d = Path(tmp) / name
                d.mkdir()
                (d / 'x.state').write_bytes(data)
                with self.assertRaises(ValueError, msg=name):
                    sd.digest(d)

    def test_an_empty_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sd.digest(Path(tmp))

    def test_the_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = folder(tmp, 'a', 'a' * 64)
            b = folder(tmp, 'b', 'b' * 64)
            c = folder(tmp, 'c', 'c' * 64, rest=b'zzz')
            run = lambda *args: subprocess.run([sys.executable, '-I', str(ROOT / 'tools/state_digest.py'), *map(str, args)],
                                               capture_output=True, text=True)
            self.assertEqual(run('compare', a, b).returncode, 0)
            self.assertIn('EQUAL', run('compare', a, b).stdout)
            r = run('compare', a, c)
            self.assertEqual(r.returncode, 1)
            self.assertIn('DIFFERENT', r.stdout)
            self.assertEqual(run('compare', a, Path(tmp) / 'none').returncode, 2)
            out = run('digest', a).stdout
            self.assertRegex(out, r'state [0-9a-f]{64} ')
            self.assertRegex(out, r'block [0-9a-f]{64} ')


if __name__ == '__main__':
    unittest.main()
