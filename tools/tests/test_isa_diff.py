"""CPU tests for tools/isa_diff.py on small disassembly fixtures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isa_diff  # noqa: E402

SYMS = isa_diff.symbol_table("""
Symbol table '.symtab' contains 3 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
     1: 0000000000001000    64 OBJECT  GLOBAL DEFAULT     4 table_a
     2: 0000000000002000    64 OBJECT  GLOBAL DEFAULT     4 table_b
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND undefined_fn
""")


def listing(base, literal, tail="s_endpgm"):
    """One kernel whose PC-relative load reaches table_a at `base` + `literal` (the literal moves with the code)."""
    return f"""
{base:016x} <_ZN8crucible3FooILi4EEEvPf>:
\ts_getpc_b64 s[2:3]                                         // {base:012X}: BE824700
\ts_add_u32 s2, s2, {literal:#x}                             // {base + 4:012X}: 8002FF02 {literal:08X}
\ts_addc_u32 s3, s3, 0                                       // {base + 12:012X}: 8203FF03 00000000
\ts_cbranch_scc1 3                                          // {base + 20:012X}: BFA20003 <_ZN8crucible3FooILi4EEEvPf+0x20>
\t{tail}                                                     // {base + 24:012X}: BFB00000
"""


def test_symbol_table_skips_undefined_and_sorts():
    assert [s[2] for s in SYMS] == ["table_a", "table_b"]


def test_pcrel_literal_is_masked_to_its_target():
    # pc after getpc = base + 4; target 0x1000 for base 0x100 needs literal 0x1000 - 0x104
    a = isa_diff.parse_disassembly(listing(0x100, 0x1000 - 0x104), SYMS)
    # the same kernel moved to 0x300: the literal changes, the target (table_a+0) does not
    b = isa_diff.parse_disassembly(listing(0x300, 0x1000 - 0x304), SYMS)
    assert a == b
    assert "<pcrel hi> ; table_a+0x0" in a["_ZN8crucible3FooILi4EEEvPf"]


def test_pcrel_to_a_different_symbol_differs():
    a = isa_diff.parse_disassembly(listing(0x100, 0x1000 - 0x104), SYMS)
    b = isa_diff.parse_disassembly(listing(0x100, 0x2000 - 0x104), SYMS)
    assert isa_diff.compare_units(a, b)["changed"] == ["_ZN8crucible3FooILi4EEEvPf"]


def test_changed_instruction_is_found_and_unchanged_is_not():
    a = isa_diff.parse_disassembly(listing(0x100, 0x100), SYMS)
    b = isa_diff.parse_disassembly(listing(0x100, 0x100, tail="s_nop 0"), SYMS)
    assert isa_diff.compare_units(a, a) == dict(changed=[], added=[], removed=[])
    assert isa_diff.compare_units(a, b)["changed"] == ["_ZN8crucible3FooILi4EEEvPf"]


def test_added_and_removed():
    d = isa_diff.compare_units({"x": "a", "y": "b"}, {"y": "b", "z": "c"})
    assert d == dict(changed=[], added=["z"], removed=["x"])


def test_base_name_groups_instantiations():
    assert isa_diff.base_name("void crucible::Foo<4, 8>(float*)") == "Foo"
    assert isa_diff.base_name("crucible::Bar(int)") == "Bar"
    counts = isa_diff.group_counts(["a", "b", "c"], {"a": "void crucible::Foo<4>(int)",
                                                       "b": "void crucible::Foo<8>(int)", "c": "crucible::Bar(int)"})
    assert counts == {"Bar": 1, "Foo": 2}


def test_strict_mode_ignores_fill_after_a_function():
    a = isa_diff.strip_fill({"f": "s_mov_b32 s0, 1\ns_endpgm\ns_nop 0\ns_nop 0", "g": "s_endpgm\ns_code_end"})
    b = isa_diff.strip_fill({"f": "s_mov_b32 s0, 1\ns_endpgm", "g": "s_endpgm\ns_code_end\ns_code_end"})
    assert a == b == {"f": "s_mov_b32 s0, 1\ns_endpgm", "g": "s_endpgm"}
    # an s_nop 0 inside the function is code, not fill
    assert isa_diff.strip_fill({"f": "s_nop 0\ns_endpgm"})["f"] == "s_nop 0\ns_endpgm"


def test_kernel_descriptor_hash_skips_the_kernel_offset(tmp_path, monkeypatch):
    readelf = "LOAD 0x000000 0x0000000000001000 0x0 0x000100 0x000100 R 0x1000"
    monkeypatch.setattr(isa_diff.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": readelf})())

    def digest(offset, last_byte):
        # 64-byte descriptor at address 0x1040; bytes 16 to 24 are the offset to the kernel
        kd = bytes(16) + offset.to_bytes(8, "little") + bytes(39) + bytes([last_byte])
        co = tmp_path / f"{offset}-{last_byte}.co"
        co.write_bytes(bytes(0x40) + kd + bytes(0x80))
        return isa_diff.kernel_descriptors(co, Path("/x"), [(0x1040, 64, "k.kd"), (0x1000, 8, "f")])
    assert list(digest(0x100, 1)) == ["k.kd"]  # only .kd symbols count
    assert digest(0x100, 1) == digest(0x900, 1)  # the link layout moved the kernel: same descriptor
    assert digest(0x100, 1) != digest(0x100, 2)  # a resource field changed: different descriptor
