"""Release audit of image input (RA-2 and RA-3). Owner: the vision branch.

The tests skip when server/vision.py or Pillow is missing.
"""
import base64
import io
import json

import pytest

vision = pytest.importorskip("server.vision")
Image = pytest.importorskip("PIL.Image")
from server.api import ApiError, validate_request

# ServeLineBytes in engine/serve/serve_line.h: one limit for text and image lines.
ENGINE_LINE = 2 * 1024 * 1024


def png_url(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_an_accepted_image_request_fits_the_engine_line():
    """RA-2: twenty 2560x1440 images (72000 tokens) were a 281 MiB line. The line now carries file paths only.

    Before the image-file change the server refused six or more such images (HTTP 400), so an agent session with six screenshots
    in its history failed on every later turn. The request is accepted and its line is far below the engine limit.
    """
    parts = [dict(type="image_url", image_url=dict(url=png_url(2560, 1440)))] * 20
    body = dict(messages=[dict(role="user", content=parts + [dict(type="text", text="compare")])])
    request = validate_request(body, "chat", "crucible", vision=vision.VisionSettings())
    paths = ["/dev/shm/crucible-vision-4194304-abcdefgh/" + "0" * 32] * len(request.images)
    line = json.dumps(dict(images=vision.engine_images(request.images, [0] * len(request.images), paths)))
    assert len(line) < 16 * 1024 and len(line) * 256 // 20 < ENGINE_LINE // 4


def test_the_engine_line_limit_matches_the_engine_source():
    from pathlib import Path
    # the limit follows the context (serve_line.h), the same for text and image lines.
    root = Path(__file__).parents[2] / "engine/serve"
    assert "kServeVisionLine" not in (root / "serve.h").read_text() + (root / "serve_line.h").read_text()
    assert "inline constexpr size_t kServeLineFloorBytes = size_t(2) << 20;" in (root / "serve_line.h").read_text()


def test_the_decoder_never_starts_ghostscript(monkeypatch):
    """Pillow renders EPS with a Ghostscript subprocess. Only common raster formats may reach a decoder."""
    from PIL import EpsImagePlugin
    calls = []
    monkeypatch.setattr(EpsImagePlugin, "Ghostscript", lambda *args, **kwargs: calls.append(args) or None)
    eps = (b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 64 64\n"
           b"newpath 0 0 moveto 64 64 lineto stroke\nshowpage\n%%EOF\n")
    with pytest.raises(ApiError):
        vision.decode_bytes(eps, vision.VisionSettings())
    assert calls == []
