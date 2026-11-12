"""WP part 1: a file made from capture data carries the canary. The rule for the training code of this repository.

Rule (taint): a derived file is tainted when any input is tainted. A tainted list shard starts with the line
`#<CANARY>` (one token, no space: mtp_dump skips a line without a space). A JSON document has the key
"canary". A safetensors file has the metadata key "crucible_capture_canary". A torch checkpoint dict has
the key "canary". A file from public text only carries nothing, so a public draft head stays shippable.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.leak_canary import CANARY  # noqa: E402
from training.mtp_tune import taint  # noqa: E402


def test_a_tainted_list_shard_starts_with_the_canary_line_and_keeps_its_rows(tmp_path):
    path = tmp_path / "train-000.list"
    taint.write_list(path, [("chat-0001", [1, 2, 3]), ("chat-0002", [4, 5])], tainted=True)
    lines = path.read_text().splitlines()
    assert lines[0] == "#" + CANARY and " " not in lines[0]
    assert lines[1:] == ["chat-0001 1,2,3", "chat-0002 4,5"]
    assert taint.list_tainted(path)
    assert list(taint.read_list(path)) == [("chat-0001", [1, 2, 3]), ("chat-0002", [4, 5])]


def test_an_untainted_list_shard_is_the_old_format(tmp_path):
    path = tmp_path / "t.list"
    taint.write_list(path, [("a-1", [1, 2])], tainted=False)
    assert path.read_text() == "a-1 1,2\n"
    assert not taint.list_tainted(path)


def test_the_list_reader_of_the_dump_tool_skips_the_header(tmp_path):
    # mtp_dump skips a line with no space: the header is one token.
    path = tmp_path / "t.list"
    taint.write_list(path, [("a-1", [1, 2])], tainted=True)
    rows = [line for line in path.read_text().splitlines() if line.find(" ") != -1 and line]
    assert rows == ["a-1 1,2"]


def test_an_ids_file_header_is_skipped_by_the_loader(tmp_path):
    path = tmp_path / "x.ids"
    taint.write_ids(path, [5, 6, 7], tainted=True)
    assert path.read_text().splitlines()[0] == "#" + CANARY
    assert taint.read_ids(path) == [5, 6, 7]
    taint.write_ids(path, [5, 6, 7], tainted=False)
    assert path.read_text().strip() == "5,6,7" and taint.read_ids(path) == [5, 6, 7]


def test_tainted_inputs_taint_the_output(tmp_path):
    a, b = tmp_path / "a.list", tmp_path / "b.list"
    taint.write_list(a, [("x-1", [1, 2])], tainted=False)
    taint.write_list(b, [("x-2", [3, 4])], tainted=True)
    assert taint.any_tainted([a]) is False
    assert taint.any_tainted([a, b]) is True


def test_a_dump_folder_is_tainted_through_its_ids_files(tmp_path):
    (tmp_path / "s1.stk").write_bytes(b"\0" * 8)
    taint.write_ids(tmp_path / "s1.ids", [1, 2], tainted=True)
    assert taint.dir_tainted(tmp_path)
    clean = tmp_path / "clean"
    clean.mkdir()
    taint.write_ids(clean / "s1.ids", [1, 2], tainted=False)
    assert not taint.dir_tainted(clean)


def test_json_documents_and_checkpoints_carry_the_key(tmp_path):
    assert taint.tag_doc({"a": 1}, True) == {"a": 1, "canary": CANARY}
    assert taint.tag_doc({"a": 1}, False) == {"a": 1}
    assert taint.tag_meta({"step": "3"}, True) == {"step": "3", "crucible_capture_canary": CANARY}
    assert taint.tag_meta({"step": "3"}, False) == {"step": "3", "crucible_provenance": "public"}
    assert taint.tag_checkpoint({"step": 3}, True)["canary"] == CANARY


def test_every_writer_of_the_training_code_goes_through_the_taint_helpers():
    """prep.py and gen_to_list.py write list shards and JSON; export.py writes the sidecar; train.py saves."""
    base = ROOT / "training/mtp_tune"
    prep = (base / "prep.py").read_text()
    assert "taint.write_list" in prep and 'open(f"{a.out}/{name}.list", "w")' not in prep
    assert "taint.tag_doc" in prep
    gen = (base / "gen_to_list.py").read_text()
    assert "taint.write_list" in gen and "taint.tag_doc" in gen
    export = (base / "export.py").read_text()
    assert "taint.tag_meta" in export
    train = (base / "train.py").read_text()
    assert "taint.tag_checkpoint" in train
    data = (base / "data.py").read_text()
    assert "taint.read_ids" in data


def test_one_helper_refuses_every_output_folder_inside_a_git_tree(tmp_path):
    from server.leak_canary import refuse_inside_git_tree

    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside every git work tree"):
        refuse_inside_git_tree(tmp_path / "repo" / "out" / "deeper")
    refuse_inside_git_tree(tmp_path / "plain" / "out")


def test_a_stk_trailer_is_the_canary_and_the_loader_accepts_it():
    from training.mtp_tune import taint as t

    assert t.STK_TRAILER == CANARY.encode() and t.STK_TRAILER_LEN == len(CANARY)
    src = (ROOT / "training/mtp_tune/data.py").read_text()
    assert "STK_TRAILER_LEN" in src


def _fake_dump(directory, names, check=False):
    """What mtp_dump writes for each sequence: NAME.stk (fp16 rows) and NAME.ids (one line), no canary."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.stk").write_bytes(b"\x01\x02" * 64)
        (directory / f"{name}.ids").write_text("1,2,3\n")
        if check:
            (directory / f"{name}.f32").write_bytes(b"\0" * 8)
            (directory / f"{name}.mtp").write_text("x\n")


def test_dump_of_a_tainted_list_is_tagged_and_the_run_sees_it(tmp_path):
    from training.mtp_tune import taint as t

    lst = tmp_path / "a.list"
    t.write_list(lst, [("s0", [1, 2, 3])], True)
    out = tmp_path / "stk-a"
    _fake_dump(out, ["s0", "s1"])
    assert not t.dir_tainted(out)                       # the raw dump carries no canary: the leak path
    assert t.tag_dump(lst, out) == 4                 # two sequences, one .stk and one .ids each
    assert t.dir_tainted(out)                           # train.py decides taint by this call
    for name in ("s0", "s1"):
        data = (out / f"{name}.stk").read_bytes()
        assert data == b"\x01\x02" * 64 + t.STK_TRAILER
        assert (out / f"{name}.ids").read_text() == t.HEADER + "\n1,2,3\n"
        assert t.read_ids(out / f"{name}.ids") == [1, 2, 3]
    assert t.tag_dump(lst, out) == 0                    # a second call changes nothing
    assert (out / "s0.stk").read_bytes().count(t.STK_TRAILER) == 1


def test_dump_of_a_public_list_gets_a_marker_and_stays_clean(tmp_path):
    from training.mtp_tune import taint as t

    lst = tmp_path / "a.list"
    t.write_list(lst, [("s0", [1, 2, 3])], False, public=True)
    out = tmp_path / "stk-a"
    _fake_dump(out, ["s0"])
    assert not t.dir_is_clean(out)                      # a raw dump has no marker: unknown provenance is not clean
    assert t.tag_dump(lst, out) == 0
    assert (out / "s0.stk").read_bytes() == b"\x01\x02" * 64 and not t.dir_tainted(out)
    assert t.dir_is_clean(out)


def test_a_dump_of_an_unknown_list_is_tagged_not_trusted(tmp_path):
    from training.mtp_tune import taint as t

    lst = tmp_path / "a.list"
    t.write_list(lst, [("s0", [1, 2, 3])], False)       # no public header, no canary: provenance unknown
    out = tmp_path / "stk-a"
    _fake_dump(out, ["s0"])
    assert t.tag_dump(lst, out) == 2 and t.dir_tainted(out) and not t.dir_is_clean(out)


def test_an_untagged_raw_dump_makes_a_tainted_run(tmp_path):
    """The fail-closed rule: a crash before the tagger, or a direct mtp_dump call, can only over-taint."""
    from training.mtp_tune import taint as t

    raw = tmp_path / "raw"
    _fake_dump(raw, ["s0"])
    assert t.run_tainted([raw]) is True
    marked = tmp_path / "marked"
    _fake_dump(marked, ["s0"])
    t.mark_dir(marked, "public")
    assert t.run_tainted([marked]) is False
    assert t.run_tainted([marked, raw]) is True
    (marked / "s1.ids").write_text("9,9\n")             # a file added after the marker breaks it
    assert t.run_tainted([marked]) is True


def test_a_tainted_init_state_or_sidecar_taints_the_run(tmp_path):
    from training.mtp_tune import taint as t

    marked = tmp_path / "m"
    _fake_dump(marked, ["s0"])
    t.mark_dir(marked, "public")
    assert t.run_tainted([marked], states=[{"canary": CANARY}]) is True
    assert t.run_tainted([marked], states=[{"step": 3, "provenance": "public"}, None]) is False
    # absence of the canary is not provenance: an unmarked state (old, or from any other tool) counts as tainted
    assert t.run_tainted([marked], states=[{"step": 3}]) is True
    side = tmp_path / "h.safetensors"
    header = json.dumps({"__metadata__": {t.META_KEY: CANARY}}).encode()
    side.write_bytes(len(header).to_bytes(8, "little") + header)
    assert t.safetensors_tainted(side) is True
    clean = tmp_path / "c.safetensors"
    header = json.dumps({"__metadata__": {"step": "1"}}).encode()
    clean.write_bytes(len(header).to_bytes(8, "little") + header)
    assert t.safetensors_tainted(clean) is True         # no positive public marker: unknown provenance
    pub = tmp_path / "p.safetensors"
    header = json.dumps({"__metadata__": {"crucible_provenance": "public"}}).encode()
    pub.write_bytes(len(header).to_bytes(8, "little") + header)
    assert t.safetensors_tainted(pub) is False
    assert t.run_tainted([marked], sidecar=pub) is False
    assert t.run_tainted([marked], sidecar=side) is True


def test_train_uses_the_fail_closed_rule():
    text = (ROOT / "training/mtp_tune/train.py").read_text()
    assert "taint.run_tainted(" in text and "dir_tainted(d)" not in text


def test_prep_marks_its_output_public_unless_tainted():
    text = (ROOT / "training/mtp_tune/prep.py").read_text()
    assert "mark_dir" in text and "public=" in text


def test_check_files_of_a_tainted_dump_are_refused(tmp_path):
    from training.mtp_tune import taint as t

    lst = tmp_path / "a.list"
    t.write_list(lst, [("s0", [1, 2, 3])], True)
    out = tmp_path / "stk-a"
    _fake_dump(out, ["s0"], check=True)                 # --check output (.f32, .mtp) cannot carry the canary
    with pytest.raises(ValueError, match="--check"):
        t.tag_dump(lst, out)


def test_every_dump_script_tags_after_the_dump():
    for name in ("dump.sh", "postgen.sh", "stage1.sh"):
        text = (ROOT / "training/mtp_tune" / name).read_text()
        assert "mtp_dump" in text and "taint tag-dump" in text, name


def test_a_stk_only_trailer_marks_the_folder_tainted(tmp_path):
    from training.mtp_tune import taint as t

    out = tmp_path / "stk"
    out.mkdir()
    (out / "s0.stk").write_bytes(b"\x01\x02" * 64 + t.STK_TRAILER)
    assert t.dir_tainted(out)


def test_every_writer_of_a_result_file_reads_the_taint_of_its_inputs():
    for name in ("calib.py", "requant.py", "train.py"):
        assert "taint.run_tainted(" in (ROOT / "training/mtp_tune" / name).read_text(), name


def test_the_training_environment_drops_the_engine_diagnostic_dump_switches():
    lines = (ROOT / "training/mtp_tune/env.sh").read_text().splitlines()
    unset = " ".join(line for line in lines if line.startswith("unset "))
    for name in ("CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR", "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY",
                 "CRUCIBLE_ROUTE_DUMP"):
        assert name in unset, name


def test_the_marker_binds_the_full_content_the_file_set_and_the_format(tmp_path):
    from training.mtp_tune import taint as t

    d = tmp_path / "m"
    _fake_dump(d, ["s0"])
    t.mark_dir(d, "public")
    assert t.dir_is_clean(d)
    (d / "s0.stk").write_bytes(b"\x09\x09" * 64)        # same size, other bytes (a copy from a capture)
    assert not t.dir_is_clean(d)
    _fake_dump(d, ["s0"])
    assert t.dir_is_clean(d)                            # the same bytes again: clean again
    (d / ".hidden").write_bytes(b"x")                   # a hidden file is part of the set
    assert not t.dir_is_clean(d)
    (d / ".hidden").unlink()
    (d / "sub").mkdir()                                 # a sub folder is not read: not clean
    assert not t.dir_is_clean(d)
    (d / "sub").rmdir()
    doc = json.loads((d / t.PROVENANCE).read_text())
    doc["v"] = 0
    (d / t.PROVENANCE).write_text(json.dumps(doc))      # an unknown format version is not clean
    assert not t.dir_is_clean(d)


# ---- outside check H7 and H8 ----------------------------------------------------------------------------------

def test_h7_a_list_with_a_canary_line_after_the_public_line_is_tainted(tmp_path):
    """cat public.list tainted.list > merged.list starts with the public line; mtp_dump skips the canary line."""
    from training.mtp_tune import taint as t

    pub, tnt, merged = tmp_path / "p.list", tmp_path / "t.list", tmp_path / "m.list"
    t.write_list(pub, [("a", [1, 2, 3])], False, public=True)
    t.write_list(tnt, [("b", [7, 8, 9])], True)
    merged.write_text(pub.read_text() + tnt.read_text())
    assert t.list_public(merged) is False
    out = tmp_path / "out"
    _fake_dump(out, ["a", "b"])
    t.tag_dump(merged, out)
    assert t.dir_tainted(out) and not t.dir_is_clean(out)
    out2 = tmp_path / "out2"
    _fake_dump(out2, ["a", "b"])
    t.tag_dump(merged, out2, public=True)       # even a person's statement cannot make a canary line public
    assert t.dir_tainted(out2) and not t.dir_is_clean(out2)


@pytest.mark.parametrize("extra", ["#anything else\n", "#note with a space 1,2\n", "x " + CANARY + "\n"])
def test_h7_any_other_hash_line_or_a_canary_anywhere_makes_a_list_not_public(tmp_path, extra):
    from training.mtp_tune import taint as t

    lst = tmp_path / "m.list"
    rows = "a 1,2,3\n" + extra + "b 4,5\n"
    lst.write_text(t.public_line(rows.encode()) + "\n" + rows)
    assert t.list_public(lst) is False


def test_h8_a_public_list_does_not_vouch_for_files_it_did_not_make(tmp_path):
    from training.mtp_tune import taint as t

    pub = tmp_path / "p.list"
    t.write_list(pub, [("a", [1, 2, 3])], False, public=True)
    out = tmp_path / "out"
    _fake_dump(out, ["b", "a"])                  # b.stk and b.ids: an earlier direct dump of another list
    with pytest.raises(ValueError, match="did not make"):
        t.tag_dump(pub, out)
    assert not t.dir_is_clean(out)


def test_h8_a_leftover_file_older_than_the_list_is_refused(tmp_path):
    import os
    from training.mtp_tune import taint as t

    pub = tmp_path / "p.list"
    t.write_list(pub, [("a", [1, 2, 3])], False, public=True)
    out = tmp_path / "out"
    _fake_dump(out, ["a"])
    old = pub.stat().st_mtime - 1000
    os.utime(out / "a.stk", (old, old))          # a.stk is from before this list: not made by this dump
    with pytest.raises(ValueError, match="did not make"):
        t.tag_dump(pub, out)
    assert not t.dir_is_clean(out)


def test_h8_a_clean_public_dump_still_passes_and_a_second_call_is_the_same(tmp_path):
    from training.mtp_tune import taint as t

    pub = tmp_path / "p.list"
    t.write_list(pub, [("a", [1, 2, 3]), ("b", [4, 5])], False, public=True)
    out = tmp_path / "out"
    _fake_dump(out, ["a", "b"])
    (out / "b.ids").write_text("4,5\n")                # the ids of sequence b in the list (H8: a differing file is stale)
    assert t.tag_dump(pub, out) == 0 and t.dir_is_clean(out)
    assert t.tag_dump(pub, out) == 0 and t.dir_is_clean(out)
