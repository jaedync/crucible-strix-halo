"""Tests for fetch.py. No network: the index, the file sizes and the download are fakes."""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

import fetch
from synth import PREFIX, write_shard_set

N, R = 5, 7
LAYOUT = [[3, 0], [4], [1, 2]]
REVISION = "abc123"


@pytest.fixture
def remote(tmp_path):
    """A fake Hugging Face repo: shard files, the index and the tree listing."""
    rng = np.random.default_rng(3)
    shards = rng.integers(0, 65535, size=(N, R, 160), dtype=np.uint16)
    repo = tmp_path / "remote"
    write_shard_set(str(repo), shards, LAYOUT)
    other = repo / "model-00004-of-00004.safetensors"  # holds no n-gram tensor
    other.write_bytes(b"x" * 1000)
    weight_map = {"model.layers.0.mlp.weight": other.name}
    for f, idxs in enumerate(LAYOUT):
        for i in idxs:
            weight_map[f"{PREFIX}.shard_{i}.weight"] = f"model-{f + 1:05d}-of-{len(LAYOUT):05d}.safetensors"
    tree = [{"type": "file", "path": p.name, "size": p.stat().st_size} for p in sorted(repo.iterdir())]
    index = {"metadata": {"total_size": 1}, "weight_map": weight_map}
    return repo, shards, index, tree


class Fake:
    """Replaces the two network calls of fetch.py and records the downloads."""

    def __init__(self, repo, index, tree):
        self.repo, self.index, self.tree = repo, index, tree
        self.downloaded = []

    def get_json(self, url):
        if url.endswith("model.safetensors.index.json"):
            return self.index
        if "/api/models/" in url:
            return self.tree
        raise AssertionError(f"unexpected URL {url}")

    def download(self, repo_id, revision, files, local_dir):
        assert revision == REVISION
        os.makedirs(local_dir, exist_ok=True)
        for name in files:
            self.downloaded.append(name)
            shutil.copy(self.repo / name, os.path.join(local_dir, name))


def run(fake, tmp_path, *extra, expected=None, **kw):
    out = tmp_path / "work"
    args = ["--out", str(out), "--chunk-mb", "0.0013", "--rate-mb", "1e6", *extra]
    # A small output size keeps the disk check of the host out of the tests that do not test it.
    kwargs = dict(revision=REVISION, get_json=fake.get_json, download=fake.download, expected_bytes=10_000)
    if expected:
        kwargs.update(expected_bytes=expected[0], expected_sha=expected[1])
    kwargs.update(kw)
    return fetch.main(args, **kwargs), out


def learn(fake, tmp_path):
    """Run once with wrong expectations to learn the size and sha256 of the output."""
    rc, out = run(fake, tmp_path, expected=(1, "0" * 64))
    assert rc == 4
    final = out / fetch.OUT_NAME
    result = final.stat().st_size, hashlib.sha256(final.read_bytes()).hexdigest()
    shutil.rmtree(out)
    fake.downloaded.clear()
    return result


def test_pick_files_only_ngram_files(remote):
    _, _, index, _ = remote
    assert fetch.pick_files(index) == [
        "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors"]


def test_pick_files_without_ngram_tensors_is_an_error():
    with pytest.raises(ValueError, match="no .*ngram_embedding"):
        fetch.pick_files({"weight_map": {"a.weight": "f.safetensors"}})


def test_plan_sums_the_sizes(remote):
    repo, _, index, tree = remote
    files = fetch.pick_files(index)
    plan = fetch.plan_sizes(files, tree)
    assert [n for n, _ in plan] == files
    assert sum(s for _, s in plan) == sum((repo / n).stat().st_size for n in files)


def test_plan_missing_size_is_an_error(remote):
    _, _, index, tree = remote
    with pytest.raises(ValueError, match="model-00002"):
        fetch.plan_sizes(fetch.pick_files(index), [t for t in tree if "00002" not in t["path"]])


def test_dry_run_lists_and_downloads_nothing(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    rc, out = run(fake, tmp_path, "--dry-run")
    text = capsys.readouterr().out
    assert rc == 0
    assert fake.downloaded == []
    assert not out.exists()
    for n in fetch.pick_files(index):
        assert n in text
    assert "model-00004-of-00004" not in text
    assert "download" in text and "free" in text


def test_disk_check_stops_before_any_download(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    rc, out = run(fake, tmp_path, free_bytes=lambda p: 10)
    err = capsys.readouterr().err
    assert rc == 2
    assert fake.downloaded == []
    assert "not enough free disk space" in err
    assert not out.exists() or not any(out.iterdir())


def test_disk_check_counts_files_already_downloaded(remote, tmp_path):
    repo, _, index, tree = remote
    files = fetch.pick_files(index)
    plan = fetch.plan_sizes(files, tree)
    src = tmp_path / "work" / "src"
    src.mkdir(parents=True)
    for n in files:
        shutil.copy(repo / n, src / n)
    assert fetch.remaining_download(plan, str(src)) == 0
    (src / files[0]).write_bytes(b"short")
    assert fetch.remaining_download(plan, str(src)) == plan[0][1]


def test_full_run_converts_and_checks(remote, tmp_path, capsys):
    repo, shards, index, tree = remote
    fake = Fake(repo, index, tree)
    expected = learn(fake, tmp_path)
    capsys.readouterr()
    rc, out = run(fake, tmp_path, expected=expected)
    text = capsys.readouterr().out
    final = out / fetch.OUT_NAME
    assert rc == 0
    assert sorted(fake.downloaded) == fetch.pick_files(index)
    assert "model-00004-of-00004" not in fake.downloaded
    assert "CRUCIBLE_PLE_HOST=1" in text
    assert f"CRUCIBLE_PLE_BF16={final}" in text
    assert "--delete-source" in text
    assert (out / "src" / "model-00001-of-00003.safetensors").exists()
    data = np.fromfile(final, dtype=np.uint16, offset=final.stat().st_size - N * R * 320)
    assert np.array_equal(data.reshape(N * R, 160), shards.reshape(N * R, 160))


def test_wrong_size_is_refused(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    rc, out = run(Fake(repo, index, tree), tmp_path, expected=(1, "0" * 64))
    captured = capsys.readouterr()
    assert rc == 4
    assert "size" in captured.err
    assert "CRUCIBLE_PLE_BF16=" not in captured.out


def test_wrong_hash_is_refused(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    size, _ = learn(fake, tmp_path)
    capsys.readouterr()
    rc, out = run(fake, tmp_path, expected=(size, "0" * 64))
    captured = capsys.readouterr()
    assert rc == 4
    assert "sha256" in captured.err
    assert "CRUCIBLE_PLE_BF16=" not in captured.out


def test_delete_source_removes_only_the_downloads(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    expected = learn(fake, tmp_path)
    capsys.readouterr()
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "notes.txt").write_text("not ours")
    rc, out = run(fake, tmp_path, "--delete-source", expected=expected)
    text = capsys.readouterr().out
    assert rc == 0
    assert "freed" in text
    for n in fetch.pick_files(index):
        assert not (out / "src" / n).exists()
    assert (out / "src" / "notes.txt").read_text() == "not ours"
    assert (out / fetch.OUT_NAME).exists()


def test_second_run_with_delete_source_on_a_full_disk(remote, tmp_path, capsys):
    """The advice of the first run: run again with --delete-source. The disk is full by then."""
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    expected = learn(fake, tmp_path)
    rc, out = run(fake, tmp_path, expected=expected)
    first = capsys.readouterr().out
    assert rc == 0 and "--delete-source" in first
    fake.downloaded.clear()
    rc, out = run(fake, tmp_path, "--delete-source", expected=expected, free_bytes=lambda p: 0)
    text = capsys.readouterr().out
    assert rc == 0
    assert fake.downloaded == []
    assert not (out / "src").exists()
    assert (out / fetch.OUT_NAME).exists()
    assert "freed" in text
    assert f"CRUCIBLE_PLE_BF16={out / fetch.OUT_NAME}" in text


def test_second_run_without_delete_source_points_to_the_flag(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    expected = learn(fake, tmp_path)
    run(fake, tmp_path, expected=expected)
    capsys.readouterr()
    rc, out = run(fake, tmp_path, expected=expected)
    err = capsys.readouterr().err
    assert rc == 2
    assert "--delete-source" in err and "exists" in err
    assert (out / "src").exists()


def test_second_run_keeps_the_source_when_the_check_fails(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    size, _ = learn(fake, tmp_path)
    run(fake, tmp_path, expected=(size, "0" * 64))  # exit 4, files stay
    capsys.readouterr()
    rc, out = run(fake, tmp_path, "--delete-source", expected=(size, "0" * 64), free_bytes=lambda p: 0)
    err = capsys.readouterr().err
    assert rc == 4
    assert (out / "src").exists()
    assert fetch.OUT_NAME in err and fetch.OUT_NAME + ".sha256" in err


def test_download_error_exits_3_with_resume_hint(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)

    def boom(*a, **k):
        raise ConnectionError("network down")

    rc, out = run(fake, tmp_path, download=boom)
    err = capsys.readouterr().err
    assert rc == 3
    assert "download stopped" in err and "network down" in err and "Run the command again" in err


def test_sidecar_without_output_is_refused_before_any_work(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    work = tmp_path / "work"
    work.mkdir()
    (work / (fetch.OUT_NAME + ".sha256")).write_text("00  x\n")
    rc, _ = run(fake, tmp_path)
    err = capsys.readouterr().err
    assert rc == 2
    assert fake.downloaded == []
    assert fetch.OUT_NAME + ".sha256" in err and fetch.OUT_NAME in err


def test_output_without_sidecar_is_refused(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    work = tmp_path / "work"
    work.mkdir()
    (work / fetch.OUT_NAME).write_bytes(b"x")
    rc, _ = run(fake, tmp_path, "--delete-source")
    assert rc == 2
    assert fake.downloaded == []
    assert fetch.OUT_NAME + ".sha256" in capsys.readouterr().err


def test_disk_need_is_reduced_by_a_partial_output(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    files = fetch.pick_files(index)
    left = sum(s for _, s in fetch.plan_sizes(files, tree))
    work = tmp_path / "work"
    work.mkdir()
    (work / (fetch.OUT_NAME + ".part")).write_bytes(b"x" * 6000)
    # expected output 10,000 bytes, 6,000 already written: 4,000 + the downloads must fit.
    rc, _ = run(fake, tmp_path, "--dry-run", free_bytes=lambda p: left + 4000)
    assert rc == 0
    rc, _ = run(fake, tmp_path, "--dry-run", free_bytes=lambda p: left + 3999)
    assert rc == 2


@pytest.mark.parametrize("bad", ["../evil.safetensors", "/abs.safetensors", "sub/x.safetensors", "a..b"])
def test_pick_files_refuses_unsafe_names(bad):
    index = {"weight_map": {f"{PREFIX}.shard_0.weight": bad}}
    with pytest.raises(ValueError, match="unsafe"):
        fetch.pick_files(index)


def test_source_is_kept_when_the_check_fails(remote, tmp_path):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    rc, out = run(fake, tmp_path, "--delete-source", expected=(1, "0" * 64))
    assert rc == 4
    assert (out / "src").exists()


def test_existing_output_is_not_overwritten(remote, tmp_path, capsys):
    repo, _, index, tree = remote
    fake = Fake(repo, index, tree)
    out = tmp_path / "work"
    out.mkdir()
    (out / fetch.OUT_NAME).write_bytes(b"keep me")
    (out / (fetch.OUT_NAME + ".sha256")).write_text("00  x\n")
    rc, _ = run(fake, tmp_path)
    assert rc == 2
    assert (out / fetch.OUT_NAME).read_bytes() == b"keep me"
    assert fake.downloaded == []
    assert "exists" in capsys.readouterr().err


def test_low_memory_stops_with_a_message(remote, tmp_path, capsys, monkeypatch):
    import common
    repo, _, index, tree = remote
    monkeypatch.setattr(common, "MEM_START_KB", 10**12)
    fake = Fake(repo, index, tree)
    rc, _ = run(fake, tmp_path)
    assert rc == 2
    assert fake.downloaded == []
    assert "MemAvailable" in capsys.readouterr().err


def test_missing_huggingface_hub_gives_an_install_hint(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("huggingface_hub"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as e:
        fetch.hub_download("o/r", "rev", ["a"], "/nonexistent")
    assert e.value.code == 2
