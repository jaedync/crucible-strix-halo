"""ISA gate (tools/isa_diff.py): the code AND the resources of every kernel.

gfx1151 puts a kernel's resources after .Lfunc_end (the .set NAME.field lines and the "; Name: value" kernel info),
so the gate must read them too (review M2). A change in a resource only is a DIFFERENT.
"""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tools"


def load():
    spec = importlib.util.spec_from_file_location('isa_diff', EVIDENCE / 'isa_diff.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


isa = load()

KERNEL = """\
\t.text
\t.protected\t{name} ; -- Begin function {name}
\t.globl\t{name}
\t.p2align\t8
\t.type\t{name},@function
{name}: ; @{name}
; %bb.0:
{body}\
\t.s_endpgm
\t.section\t.rodata,"a",@progbits
\t.p2align\t6, 0x0
\t.amdhsa_kernel {name}
\t\t.amdhsa_group_segment_fixed_size 0
\t.end_amdhsa_kernel
\t.text
.Lfunc_end0:
\t.size\t{name}, .Lfunc_end0-{name}
                                        ; -- End function
\t.set {name}.num_vgpr, {vgpr}
\t.set {name}.num_agpr, 0
\t.set {name}.numbered_sgpr, 32
\t.set {name}.private_seg_size, {scratch}
\t.section\t.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2400
; TotalNumSgprs: {sgpr}
; NumVgprs: {vgpr}
; ScratchSize: {scratch}
; FloatMode: 240
; LDSByteSize: {lds} bytes/workgroup (compile time only)
; Occupancy: {occupancy}
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
"""


def kernel(name='_Z1kv', body='\tv_add_f32 v0, v1, v2\n', vgpr=20, sgpr=34, scratch=0, lds=0, occupancy=16):
    return KERNEL.format(name=name, body=body, vgpr=vgpr, sgpr=sgpr, scratch=scratch, lds=lds, occupancy=occupancy)


def scratch():
    """A temporary directory that is not /tmp (the test machine /tmp is a RAM tmpfs that every worker shares)."""
    root = Path.home() / '.cache'
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='x4-test.', dir=root)


def verdict(base, new):
    with scratch() as tmp:
        first, second = Path(tmp) / 'engine-base.s', Path(tmp) / 'engine-new.s'
        first.write_text(base)
        second.write_text(new)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = isa.main(['asm', str(first), str(second)])
    return rc, out.getvalue()


class Isa(unittest.TestCase):
    def test_the_same_file_is_same(self):
        rc, out = verdict(kernel(), kernel())
        self.assertEqual(rc, 0, out)
        self.assertIn('SAME: 1 equal, 0 changed, 0 missing, 0 added', out)

    def test_a_resource_only_change_fails(self):
        # M2: every "; Name: value" line counts, also when the code is byte-equal.
        for name, changed in {'NumVgprs': dict(vgpr=99), 'LDSByteSize': dict(lds=65536), 'Occupancy': dict(occupancy=8),
                              'ScratchSize': dict(scratch=64), 'TotalNumSgprs': dict(sgpr=40)}.items():
            rc, out = verdict(kernel(), kernel(**changed))
            self.assertEqual(rc, 1, (name, out))
            self.assertIn('CHANGED RESOURCES _Z1kv', out, (name, out))
            self.assertIn('DIFFERENT: 0 equal, 1 changed', out, (name, out))

    def test_a_set_line_change_fails(self):
        rc, out = verdict(kernel(), kernel(vgpr=21))
        self.assertEqual(rc, 1, out)
        self.assertIn('CHANGED RESOURCES _Z1kv', out)
        self.assertIn('  - set num_vgpr = 20', out)
        self.assertIn('  + set num_vgpr = 21', out)

    def test_a_code_change_is_tagged_code(self):
        rc, out = verdict(kernel(), kernel(body='\tv_sub_f32 v0, v1, v2\n'))
        self.assertEqual(rc, 1, out)
        self.assertIn('CHANGED CODE _Z1kv', out)

    def test_a_missing_or_added_function(self):
        both = kernel() + kernel('_Z1kw')
        rc, out = verdict(both, kernel())
        self.assertEqual(rc, 1, out)
        self.assertIn('MISSING _Z1kw', out)
        rc, out = verdict(kernel(), both)
        self.assertEqual(rc, 0, out)
        self.assertIn('ADDED _Z1kw', out)

    def test_the_resources_go_to_their_own_function(self):
        # Two kernels with the same code and other resources: the gate must not mix them up.
        base = kernel('_Z1kv') + kernel('_Z1kw', vgpr=40)
        found, _names = isa.functions_text(base)
        self.assertIn('NumVgprs: 20', found['_Z1kv'][1])
        self.assertIn('set num_vgpr = 20', found['_Z1kv'][1])
        self.assertIn('NumVgprs: 40', found['_Z1kw'][1])
        self.assertIn('set num_vgpr = 40', found['_Z1kw'][1])
        self.assertNotIn('NumVgprs: 40', found['_Z1kv'][1])
        # The rest of the second kernel's differences come from its own name only.
        self.assertEqual(sorted(found['_Z1kv'][1]), ['COMPUTE_PGM_RSRC2: USER_SGPR: 2', 'FloatMode: 240',
                                                     'LDSByteSize: 0 bytes/workgroup (compile time only)',
                                                     'NumVgprs: 20', 'Occupancy: 16', 'ScratchSize: 0',
                                                     'TotalNumSgprs: 34', 'set num_agpr = 0',
                                                     'set num_vgpr = 20', 'set numbered_sgpr = 32',
                                                     'set private_seg_size = 0'])
        # Swapping the functions in the file must not change any function's lines.
        swapped, _names = isa.functions_text(kernel('_Z1kw', vgpr=40) + kernel('_Z1kv'))
        self.assertEqual(swapped, found)

    def test_bad_input_is_an_error(self):
        rc, out = verdict('no kernel here\n', kernel())
        self.assertEqual(rc, 2, out)
        self.assertIn('bad input', out)


if __name__ == '__main__':
    unittest.main()
