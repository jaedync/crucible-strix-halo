"""Image input: OpenAI image parts -> resized RGB, prompt placeholder expansion, and the engine payload.

Pixel limits and the resize rule follow Halogen 0.15.3 (tools/serve_api.py) and the transformers
qwen2_vl smart_resize, so image quality matches Halogen. The engine patchifies and normalizes.
"""
import base64
import binascii
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
from .api import ApiError

IMAGE_PAD = 248056
VIDEO_PAD = 248057
VISION_START = 248053
VISION_END = 248054
UNIT = 32  # patch 16 x spatial merge 2
MIN_PIXELS = 256 * 256
MAX_PIXELS = 2560 * 1440


@dataclass(frozen=True)
class VisionFamily:
    """The image placeholder of a model family: the template writes start, pad, end for each image part, and the
    server writes `tokens` pads in place of the one pad. row_end: each merged row has one more pad (the newline
    embedding of the tower). min_pixels and max_pixels: the resize range of the family's image processor."""
    start: int
    pad: int
    end: int
    min_pixels: int
    max_pixels: int
    row_end: bool = False


QWEN_VISION = VisionFamily(VISION_START, IMAGE_PAD, VISION_END, MIN_PIXELS, MAX_PIXELS)
# Hy4-flash (tencent/Hy4-flash config.json; transformers PR 50903 processing_hy_v4_flash.py _num_frame_tokens and
# image_processing_hy_v4_flash.py: patch 16, merge 2, 512 x 512 to 2048 x 2048 pixels, BICUBIC).
HY4_VISION = VisionFamily(120118, 120120, 120119, 512 * 512, 2048 * 2048, row_end=True)
HY4_TOWER_MISSING = "the engine does not run the Hy4-flash vision tower yet"
MAX_ASPECT = 200
# Decoding cost is bounded before the resize, as Halogen does.
DECODE_FACTOR = 4
# Per-request limits, checked before any pixel is decoded. Pixels reach the engine as private files, not in the
# JSON line (server/pixel_files.py), so the context window limits a long screenshot session. 256 is the engine's
# kMaxImages (engine/model/flashnext/multimodal.h).
MAX_IMAGES = 256
MAX_IMAGE_BYTES = 16 << 20
# The image tokens of one request must fit the server context (the engine's largest context by default). This also
# bounds the decoded pixels that a request holds: 3072 RGB bytes per token, about 315 MB at 102400 tokens.
MAX_IMAGE_TOKENS = 262144
# The default request body with --vision, and the flag's upper bound. A 262144-token context holds 72 screenshots
# of 2560x1440; each is often 1 to 2 MB of PNG, and base64 adds a third.
VISION_BODY_MIB = 256
VISION_BODY_MIB_MAX = 1024
# A checkpoint that stores the tower as plain HF tensors. The same text is in engine/model/flashnext/weight_validation.h.
TOWER_MISSING = (
    "this checkpoint stores the vision tower as plain BF16 tensors (for example model.visual.blocks.0.attn.qkv.weight), "
    "and the engine runs the EXL3 tower. Put the file vision_k6.safetensors in the model folder: "
    "hf download turboderp/Qwen3.8-Flash-Next-exl3 vision_k6.safetensors --revision 4.05bpw_h6_ng6 --local-dir <model folder>. "
    "The images are exact only when the weights of the HF tower are the stock weights; the engine does not compare them")
DECODE_ERROR = "The image could not be decoded."
ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")  # no EPS (Ghostscript) and no rare decoders


def tower_fault(checkpoint):
    """TOWER_MISSING when --vision cannot run on this folder, else None. Reads the index only; the engine checks the rest."""
    folder = Path(checkpoint)
    from .hy4_format import is_hy4
    if is_hy4(folder):
        return HY4_TOWER_MISSING
    if (folder / "vision_k6.safetensors").exists():
        return None
    try:
        names = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if "model.visual.blocks.0.attn.qkv.weight" in names and "model.visual.blocks.0.attn.q_proj.trellis" not in names:
        return TOWER_MISSING
    return None


@dataclass(frozen=True)
class VisionImage:
    height: int
    width: int
    rgb: bytes
    digest: str
    row_end: bool = False  # VisionFamily.row_end

    @property
    def tokens(self):
        return (self.height // UNIT) * (self.width // UNIT + self.row_end)


@dataclass(frozen=True)
class VisionSettings:
    max_pixels: int = MAX_PIXELS
    max_images: int = MAX_IMAGES
    max_image_bytes: int = MAX_IMAGE_BYTES
    max_image_tokens: int = MAX_IMAGE_TOKENS
    family: VisionFamily = QWEN_VISION


def smart_resize(height, width, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if min(height, width) <= 0:
        raise ApiError("The image has a zero dimension.", "messages")
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ApiError(f"The image aspect ratio must be below {MAX_ASPECT}:1.", "messages")
    h_bar = max(UNIT, round(height / UNIT) * UNIT)
    w_bar = max(UNIT, round(width / UNIT) * UNIT)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(UNIT, math.floor(height / beta / UNIT) * UNIT)
        w_bar = max(UNIT, math.floor(width / beta / UNIT) * UNIT)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / UNIT) * UNIT
        w_bar = math.ceil(width * beta / UNIT) * UNIT
    return h_bar, w_bar


def image_digest(height, width, rgb):
    """Identity of one resized image for the conversation cache: sha256 of its shape and pixels."""
    return hashlib.sha256(struct.pack("<II", height, width) + rgb).hexdigest()


def decode_url(url, settings, used_tokens=0):
    """A data: URL or bare base64 -> VisionImage. The server never fetches remote URLs."""
    if not isinstance(url, str) or not url:
        raise ApiError("image_url.url must be a nonempty string.", "messages")
    if url.startswith(("http://", "https://")):
        raise ApiError("This server does not fetch image URLs. Send the image as a data: URL "
                       "(data:image/png;base64,...).", "messages")
    payload = url
    if url.startswith("data:"):
        head, _, payload = url.partition(",")
        if ";base64" not in head:
            raise ApiError("Only base64 data: URLs are supported for images.", "messages")
    if len(payload) * 3 // 4 > settings.max_image_bytes:
        raise ApiError(f"The encoded image is larger than {settings.max_image_bytes} bytes.", "messages")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        raise ApiError("The image data is not valid base64.", "messages") from None
    if not raw:
        raise ApiError("The image data is empty.", "messages")
    return decode_bytes(raw, settings, used_tokens)


ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})
# Modes whose "transparency" entry convert("RGBA") reads correctly. Pillow cuts an I;16 tRNS key to 8 bits.
KEY_MODES = frozenset({"1", "L", "P", "RGB"})


def flatten_rgb(image, Image):
    """RGB of a loaded image. Transparent pixels go onto opaque white; convert("RGB") alone
    keeps their stored colour, often black. Images without transparency take the plain convert path.

    White matches Qwen's vision utility (qwen_vl_utils to_rgb) and transformers main. The pinned transformers
    v5.18.0 convert_to_rgb does a plain convert("RGB"), so no exact parity is claimed for transparent input.
    """
    keyed = image.mode in KEY_MODES and "transparency" in image.info
    if image.mode not in ALPHA_MODES and not keyed:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")


def decode_bytes(raw, settings, used_tokens=0):
    try:
        from PIL import Image
    except ImportError:
        raise ApiError("This server has no image decoder. Install Pillow to use --vision.", "messages") from None
    Image.MAX_IMAGE_PIXELS = settings.max_pixels * DECODE_FACTOR
    try:
        image = Image.open(io.BytesIO(raw), formats=ALLOWED_FORMATS)
        width0, height0 = image.size
        if width0 * height0 > settings.max_pixels * DECODE_FACTOR:
            raise ApiError(f"The image is {width0}x{height0}. The limit before resizing is "
                           f"{settings.max_pixels * DECODE_FACTOR} pixels.", "messages")
        family = settings.family
        height, width = smart_resize(height0, width0, min_pixels=family.min_pixels, max_pixels=settings.max_pixels)
        tokens = used_tokens + (height // UNIT) * (width // UNIT + family.row_end)
        if tokens > settings.max_image_tokens:
            # Clients such as Pi compact their history when the text says "exceeds the context window".
            raise ApiError(f"The images of this request need at least {tokens} tokens, which exceeds the context "
                           f"window of {settings.max_image_tokens} tokens. Send fewer or smaller images.", "messages")
        image.load()
        image = flatten_rgb(image, Image)
    except ApiError:
        raise
    except Image.DecompressionBombError:
        # Pillow refuses a header above twice its pixel limit before the size check above can run.
        raise ApiError(f"The image is too large. The limit before resizing is "
                       f"{settings.max_pixels * DECODE_FACTOR} pixels.", "messages") from None
    except Exception:
        # Pillow's text can hold object addresses; the client gets a fixed message.
        raise ApiError(DECODE_ERROR, "messages") from None
    if (width, height) != (width0, height0):
        image = image.resize((width, height), Image.BICUBIC)
    rgb = image.tobytes()
    if len(rgb) != height * width * 3:
        raise ApiError("The decoded image is not 3-channel RGB.", "messages")
    return VisionImage(height, width, rgb, image_digest(height, width, rgb), settings.family.row_end)


def image_url_of(item):
    """The URL of one image content part, or None for other parts."""
    kind = item.get("type")
    if kind == "image_url" or (kind is None and "image_url" in item):
        value = item.get("image_url")
        if isinstance(value, dict):
            if set(value) - {"url", "detail"}:
                raise ApiError("image_url supports url and detail only.", "messages")
            value = value.get("url")
        return value if isinstance(value, str) else ""
    if kind == "image" or (kind is None and "image" in item):
        value = item.get("image")
        return value if isinstance(value, str) else ""
    return None


def expand_pads(ids, images, family=QWEN_VISION):
    """One template placeholder per image becomes `tokens` placeholders (the IDs of `family`).

    The template writes <|vision_start|><|image_pad|><|vision_end|> for each image part. Only pads
    with that pair around them count. Any other pad ID is text the user or the model wrote; it stays
    one ordinary token. Returns (expanded IDs, first-token offset of each image).
    """
    if not images:
        return list(ids), []
    pads = [i for i in range(1, len(ids) - 1)
            if ids[i] == family.pad and ids[i - 1] == family.start and ids[i + 1] == family.end]
    if len(pads) != len(images):
        raise ApiError(f"The chat template wrote {len(pads)} image placeholders for {len(images)} images. "
                       "Do not type <|vision_start|><|image_pad|><|vision_end|> in a message with images.",
                       "messages")
    out, offsets, previous = [], [], 0
    for index, image in zip(pads, images):
        out.extend(ids[previous:index])
        offsets.append(len(out))
        out.extend([family.pad] * image.tokens)
        previous = index + 1
    out.extend(ids[previous:])
    return out, offsets


def engine_images(images, offsets, paths):
    """The engine protocol field: one object per image, in prompt order. The pixels are in the files at `paths`."""
    return [dict(offset=offset, height=image.height, width=image.width, digest=image.digest, path=path)
            for image, offset, path in zip(images, offsets, paths, strict=True)]
