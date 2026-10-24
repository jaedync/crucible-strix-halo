import hashlib
import json
import os

import numpy as np
import pytest

import common
import convert
from common import read_header
from synth import PREFIX, write_shard_set

N, R = 5, 7
# Shards out of order across files, with other tensors in the same files.
LAYOUT = [[3, 0], [4], [1, 2]]


@pytest.fixture
def shard_set(tmp_path):
    rng = np.random.default_rng(3)
    shards = rng.integers(0, 65535, size=(N, R, 160), dtype=np.uint16)
    src = tmp_path / "src"
    write_shard_set(str(src), shards, LAYOUT)
    return src, shards


def run(src, out, *extra):
    return convert.main([str(src), str(out), "--revision", "abc123", "--chunk-mb", "0.0013",
                         "--rate-mb", "1e6", *extra])


def test_output_order_dtype_header(shard_set, tmp_path, capsys):
    src, shards = shard_set
    out = tmp_path / "out.safetensors"
    assert run(src, out) == 0
    header, base = read_header(str(out))
    meta = header.pop("__metadata__")
    assert list(header) == [f"{PREFIX}.weight"]
    info = header[f"{PREFIX}.weight"]
    assert info == {"dtype": "BF16", "shape": [N * R, 160], "data_offsets": [0, N * R * 320]}
    data = np.fromfile(out, dtype=np.uint16, offset=base).reshape(N * R, 160)
    assert np.array_equal(data, shards.reshape(N * R, 160))
    assert os.path.getsize(out) == base + N * R * 320
    assert meta["source_revision"] == "abc123"
    assert meta["format"] == "ngram_bf16_concat"
    assert meta["rows"] == str(N * R) and meta["shard_rows"] == str(R) and meta["num_shards"] == str(N)
    files = json.loads(meta["shard_files"])
    assert files == ["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors",
                     "model-00003-of-00003.safetensors", "model-00001-of-00003.safetensors",
                     "model-00002-of-00003.safetensors"]
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert f"sha256 {digest}" in capsys.readouterr().out
    assert (tmp_path / "out.safetensors.sha256").read_text().split()[0] == digest
    assert not (tmp_path / "out.safetensors.part").exists()


def test_refuses_to_overwrite(shard_set, tmp_path):
    src, _ = shard_set
    out = tmp_path / "out.safetensors"
    out.write_bytes(b"keep me")
    assert run(src, out) == 2
    assert out.read_bytes() == b"keep me"
    assert not (tmp_path / "out.safetensors.part").exists()


def test_missing_shard_is_an_error(tmp_path):
    rng = np.random.default_rng(4)
    shards = rng.integers(0, 65535, size=(3, R, 160), dtype=np.uint16)
    write_shard_set(str(tmp_path / "src"), shards, [[0], [2]])
    assert run(tmp_path / "src", tmp_path / "out.safetensors") == 2


def test_wrong_dtype_is_an_error(tmp_path):
    from synth import write_safetensors
    os.makedirs(tmp_path / "src")
    write_safetensors(str(tmp_path / "src" / "a.safetensors"),
                      {f"{PREFIX}.shard_0.weight": ("F16", np.zeros((2, 160), np.uint16))})
    assert run(tmp_path / "src", tmp_path / "out.safetensors") == 2


def test_resume_after_stop_gives_same_file(shard_set, tmp_path, monkeypatch):
    src, shards = shard_set
    out = tmp_path / "out.safetensors"
    calls = {"n": 0}
    real_tick = common.Guard.tick

    def stop_after_three(self, progress):
        calls["n"] += 1
        if calls["n"] > 3:
            raise common.Deadline("test stop")
        real_tick(self, progress)

    monkeypatch.setattr(common.Guard, "tick", stop_after_three)
    assert run(src, out) == 3
    assert not out.exists()
    part = tmp_path / "out.safetensors.part"
    assert part.exists()
    synced = int((tmp_path / "out.safetensors.part.progress").read_text())
    assert os.path.getsize(part) == synced
    monkeypatch.setattr(common.Guard, "tick", real_tick)
    assert run(src, out) == 0
    _, base = read_header(str(out))
    data = np.fromfile(out, dtype=np.uint16, offset=base).reshape(N * R, 160)
    assert np.array_equal(data, shards.reshape(N * R, 160))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    assert (tmp_path / "out.safetensors.sha256").read_text().split()[0] == digest


def test_deadline_in_the_past_stops_before_copy(shard_set, tmp_path):
    src, _ = shard_set
    out = tmp_path / "out.safetensors"
    assert run(src, out, "--stop-at", "2000-01-01 00:00") == 3
    assert not out.exists()
