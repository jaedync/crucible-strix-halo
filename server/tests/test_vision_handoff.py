"""Pixel files between the server and the engine: they exist while the engine works
on the request and are removed on every path after it."""
import base64
import io
import json
import os
import time

import pytest

Image = pytest.importorskip("PIL.Image")
from server.api import validate_request
from server.pixel_files import PixelStore
from server.vision import IMAGE_PAD, VisionSettings
from server.tests.test_backend_paths import collect, start_backend, wait_for  # noqa: F401 (fixture)


def png_url(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_request(count=2, max_tokens=3, last=7):
    parts = [dict(type="image_url", image_url=dict(url=png_url((index * 40, 9, 9)))) for index in range(count)]
    body = dict(messages=[dict(role="user", content=parts + [dict(type="text", text="x")])],
                max_tokens=max_tokens, spec=0)
    request = validate_request(body, "chat", "crucible", vision=VisionSettings())
    ids, offsets = [5], []
    for image in request.images:
        offsets.append(len(ids))
        ids += [IMAGE_PAD] * image.tokens
    return ids + [last], request, offsets


def start(start_backend, tmp_path, batch, options=None, **env):
    env = {"FAKE_PIXEL_LOG": str(tmp_path / "pixels.log"), "FAKE_CONTEXT": "4096",
           "FAKE_SLOT_CONTEXTS": "4096" if env.pop("one_slot", False) else "4096,4096", **env}
    backend = start_backend(batch, env=env, context=4096, **({"streams": len(env["FAKE_SLOT_CONTEXTS"].split(","))}
                                                            if batch else {}), **(options or {}))
    backend.pixels = PixelStore(root=tmp_path)
    return backend


def pixel_log(tmp_path):
    path = tmp_path / "pixels.log"
    return path.read_text().split() if path.exists() else []


def no_files(paths):
    return not any(os.path.exists(path) for path in paths)


@pytest.mark.parametrize("batch", [False, True])
def test_the_engine_reads_private_files_and_the_server_removes_them(start_backend, tmp_path, batch):
    backend = start(start_backend, tmp_path, batch)
    ids, request, offsets = image_request()
    job = backend.submit(ids, request, image_offsets=offsets)
    collect(job)
    images = job.payload["images"]
    paths = [image["path"] for image in images]
    assert all("rgb" not in image and os.path.isabs(image["path"]) for image in images)
    assert len(json.dumps(job.payload)) < 4096
    # The fake engine read each file while it held the request, and the pixels matched the digest.
    assert pixel_log(tmp_path) == ["ok", "ok"]
    wait_for(lambda: no_files(paths))
    directory = os.path.dirname(paths[0])
    backend.close()
    assert not os.path.exists(directory)


@pytest.mark.parametrize("batch", [False, True])
def test_a_cancelled_request_that_never_reached_the_engine_removes_its_files(start_backend, tmp_path, batch):
    backend = start(start_backend, tmp_path, batch, FAKE_DELAY=".01", one_slot=True)
    long_ids, long_request, long_offsets = image_request(count=1, max_tokens=60)
    first = backend.submit(long_ids, long_request, image_offsets=long_offsets)
    ids, request, offsets = image_request()
    waiting = backend.submit(ids, request, image_offsets=offsets)
    paths = [image["path"] for image in waiting.payload["images"]]
    assert not no_files(paths)
    waiting.cancelled.set()
    collect(first)
    wait_for(lambda: no_files(paths))
    assert pixel_log(tmp_path) == ["ok"]


def test_a_refused_submit_removes_its_files(start_backend, tmp_path):
    backend = start(start_backend, tmp_path, False)
    backend.fatal = "stopped for the test"
    ids, request, offsets = image_request()
    before = set(os.listdir(tmp_path))
    with pytest.raises(Exception):
        backend.submit(ids, request, image_offsets=offsets)
    directories = [name for name in os.listdir(tmp_path) if name.startswith("crucible-vision-")]
    assert all(os.listdir(tmp_path / name) == [] for name in directories)
    assert before <= set(os.listdir(tmp_path))
    backend.fatal = None


@pytest.mark.parametrize("batch", [False, True])
def test_a_full_dev_shm_sends_the_pixels_through_the_temp_folder(start_backend, tmp_path, monkeypatch, batch):
    from server.tests.test_pixel_files import FakeShm, files_under
    backend = start(start_backend, tmp_path, batch)
    shm, fallback = tmp_path / "shm", tmp_path / "tmp"
    shm.mkdir(), fallback.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fallback))
    backend.pixels = PixelStore(root=shm)
    FakeShm(monkeypatch, {shm: 4096})  # a /dev/shm with no room for one image
    ids, request, offsets = image_request()
    job = backend.submit(ids, request, image_offsets=offsets)
    collect(job)
    paths = [image["path"] for image in job.payload["images"]]
    assert all(path.startswith(str(fallback)) for path in paths)
    # The engine read the same bytes (their digests match), and the files leave after the request.
    assert pixel_log(tmp_path) == ["ok", "ok"]
    wait_for(lambda: no_files(paths))
    assert files_under(shm) == []
    backend.close()
    assert os.listdir(fallback) == []


def test_batch_encoder_slices_are_not_prompt_rows(start_backend, tmp_path):
    backend = start(start_backend, tmp_path, True)
    ids, request, offsets = image_request(count=2)
    started = time.monotonic()
    job = backend.submit(ids, request, image_offsets=offsets)
    collect(job)
    prefills = [entry for entry in backend.trace.since(0) if entry["op"] == "prefill"]
    # The fake engine reports three slices per image; those commands schedule no prompt rows.
    assert [entry["rows"] for entry in prefills] == [0] * 6 + [len(ids)]
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("argv, accepted", [
    (["--vision", "--max-body-mib", "1024"], True),
    (["--vision", "--max-body-mib", "1025"], False),
    (["--max-body-mib", "65"], False),
    (["--max-body-mib", "64"], True),
])
def test_image_sessions_may_raise_the_body_limit(monkeypatch, argv, accepted):
    """Many screenshots in one conversation need a body above 64 MiB; text servers keep the old range."""
    import server.__main__ as entry

    class Reached(BaseException):  # main() reports an Exception and returns 1
        pass

    def tokenizer(_path, **_options):
        raise Reached()
    monkeypatch.setattr(entry, "ModelTokenizer", tokenizer)
    monkeypatch.setattr("sys.argv", ["server", "/nonexistent"] + argv)
    with pytest.raises(Reached if accepted else SystemExit):
        entry.main()


def test_the_vision_body_default_holds_a_long_screenshot_session():
    from server.vision import VISION_BODY_MIB
    # 72 screenshots of 2560x1440 (3600 tokens each) fill a 262144-token context; at 2 MB of PNG each in base64
    # that is about 192 MB.
    assert VISION_BODY_MIB << 20 >= 72 * (2 << 20) * 4 // 3


@pytest.mark.parametrize("batch", [False, True])
def test_an_image_file_that_changes_after_the_open_fails_only_its_request(start_backend, tmp_path, batch):
    """Review F1: the engine checks an image file when it reads it. A changed file ends that request with an error;
    the worker, the peers and later image requests continue."""
    backend = start(start_backend, tmp_path, batch, FAKE_CHANGE_TOKEN="9", FAKE_DELAY=".002")
    peer_ids, peer_request, peer_offsets = image_request(count=1, max_tokens=40)
    peer = backend.submit(peer_ids, peer_request, image_offsets=peer_offsets)
    ids, request, offsets = image_request(count=2, last=9)
    bad = backend.submit(ids, request, image_offsets=offsets)
    paths = [image["path"] for image in bad.payload["images"]]
    collect(peer)  # the peer's own client reads its stream (a full queue would stall the serial worker)
    event = bad.events.get(timeout=30)
    while event["event"] == "token":
        event = bad.events.get(timeout=30)
    assert event["event"] == "error" and "/" not in event["error"]
    if batch:
        assert "image" in event["error"] and "Retry" in event["error"]
    assert not backend.fatal
    good_ids, good_request, good_offsets = image_request(count=1)
    collect(backend.submit(good_ids, good_request, image_offsets=good_offsets))
    wait_for(lambda: no_files(paths))
    assert "bad" in pixel_log(tmp_path)


def test_the_server_passes_its_context_to_the_image_settings(monkeypatch):
    import server.__main__ as entry
    seen = {}

    class Reached(BaseException):
        pass

    def tokenizer(_path, **_options):
        raise Reached()
    from server import vision
    original = vision.VisionSettings

    def capture(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(vision, "VisionSettings", capture)
    monkeypatch.setattr(entry, "ModelTokenizer", tokenizer)
    monkeypatch.setattr("sys.argv", ["server", "/nonexistent", "--vision", "--ctx", "40960"])
    with pytest.raises(Reached):
        entry.main()
    assert seen.get("max_image_tokens") == 40960


@pytest.mark.parametrize("mode", ["id", "generation", "part-before", "part-inside", "op"])
def test_a_request_error_event_outside_its_contract_stops_the_worker(start_backend, tmp_path, mode):
    """Review follow-up: the engine may end a request with its own error event only in a prefill command, for that
    request and its slot generation, before any decode pass of the command. Anything else is out of protocol, so the
    worker fails closed (backend.fatal) instead of releasing a slot that the engine may still use."""
    # Short chunks: interleave_min_rows=0 keeps their passes (the served default skips chunks below 300 rows).
    backend = start(start_backend, tmp_path, True, options=dict(prefill_interleave=8, interleave_min_rows=0),
                    FAKE_CHANGE_TOKEN="9", FAKE_IMAGE_ERROR_MODE=mode, FAKE_DELAY=".002")
    if mode == "op":
        # A lone speculative text request decodes with spec_step; that command carries the request error.
        job = backend.submit(*text_request(8, last=9, spec=7))
        wait_for(lambda: backend.fatal)
        assert job.events.get(timeout=3)["event"] in ("token", "error")
        return
    peer = backend.submit(*text_request(40, last=5))
    wait_for(lambda: any(entry["op"] in ("step", "verify_step") for entry in backend.trace.since(0)))
    ids, request, offsets = image_request(count=1, last=9)
    backend.submit(ids, request, image_offsets=offsets)
    wait_for(lambda: backend.fatal)
    while True:
        event = peer.events.get(timeout=3)
        if event["event"] != "token":
            break
    assert event["event"] == "error"


def text_request(max_tokens, last, spec=0):
    body = dict(prompt="x", max_tokens=max_tokens, spec=spec)
    return [1, 2, last], validate_request(body, "completion", "crucible")
