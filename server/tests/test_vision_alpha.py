"""decode_bytes composites transparent images onto white, and keeps opaque images byte-identical."""
import io
import unittest
from unittest import mock
from server.vision import VisionSettings, decode_bytes, image_digest, smart_resize

try:
    from PIL import Image
except ImportError:  # Pillow is needed only for --vision.
    Image = None

SETTINGS = VisionSettings()
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def png(image, **params):
    buffer = io.BytesIO()
    image.save(buffer, "PNG", **params)
    return buffer.getvalue()


def pixels(decoded):
    rgb = decoded.rgb
    return [tuple(rgb[i:i + 3]) for i in range(0, len(rgb), 3)]


def pixel(decoded, x, y):
    i = (y * decoded.width + x) * 3
    return tuple(decoded.rgb[i:i + 3])


@unittest.skipIf(Image is None, "Pillow is not installed")
class TransparentImages(unittest.TestCase):
    def decode(self, image, **params):
        return decode_bytes(png(image, **params), SETTINGS)

    def test_rgba_transparent_background_black_shape(self):
        # The issue's reproduction: 256x128, transparent background, opaque black rectangle.
        image = Image.new("RGBA", (256, 128), (0, 0, 0, 0))
        image.paste((0, 0, 0, 255), (64, 32, 192, 96))
        decoded = self.decode(image)
        # smart_resize scales 256x128 up to 384x192; corners stay background, the centre stays shape.
        self.assertEqual((decoded.width, decoded.height), (384, 192))
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 383, 191), WHITE)
        self.assertEqual(pixel(decoded, 192, 96), BLACK)
        self.assertGreater(pixels(decoded).count(WHITE), 384 * 192 // 2)

    def test_rgba_no_resize_exact(self):
        image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        image.paste((0, 0, 0, 255), (64, 64, 192, 192))
        decoded = self.decode(image)
        self.assertEqual((decoded.width, decoded.height), (256, 256))
        self.assertEqual(set(pixels(decoded)), {WHITE, BLACK})
        self.assertEqual(pixels(decoded).count(BLACK), 128 * 128)

    def test_partial_alpha_over_white(self):
        # (c*a + 255*(255-a)) / 255 with a=128: 200 -> 227, 100 -> 177, 0 -> 127.
        image = Image.new("RGBA", (256, 256), (200, 100, 0, 128))
        decoded = self.decode(image)
        self.assertEqual(set(pixels(decoded)), {(227, 177, 127)})
        self.assertEqual(decoded.rgb, bytes([227, 177, 127]) * 256 * 256)

    def test_la_transparent_is_white(self):
        image = Image.new("LA", (256, 256), (0, 0))
        image.paste((0, 255), (64, 64, 192, 192))
        decoded = self.decode(image)
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 128, 128), BLACK)

    def test_pa_transparent_is_white(self):
        # Pillow cannot save PA as PNG; decode_bytes gets the image from Image.open, so patch it in.
        image = Image.new("PA", (256, 256), (0, 0))
        image.putpalette([0, 0, 0] * 256)
        image.paste((0, 255), (64, 64, 192, 192))
        self.assertEqual(image.mode, "PA")
        with mock.patch.object(Image, "open", return_value=image):
            decoded = decode_bytes(b"not used", SETTINGS)
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 128, 128), BLACK)

    def test_palette_with_transparency_index(self):
        image = Image.new("P", (256, 256), 0)
        image.putpalette([0, 0, 0, 0, 0, 0] + [0, 0, 0] * 254)
        image.paste(1, (64, 64, 192, 192))
        decoded = self.decode(image, transparency=0)
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 128, 128), BLACK)

    def test_rgb_trns_colour_key(self):
        image = Image.new("RGB", (256, 256), BLACK)
        image.paste((10, 10, 10), (64, 64, 192, 192))
        decoded = self.decode(image, transparency=BLACK)
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 128, 128), (10, 10, 10))

    def test_l_trns_colour_key(self):
        image = Image.new("L", (256, 256), 0)
        image.paste(10, (64, 64, 192, 192))
        decoded = self.decode(image, transparency=0)
        self.assertEqual(pixel(decoded, 0, 0), WHITE)
        self.assertEqual(pixel(decoded, 128, 128), (10, 10, 10))

    def test_transparent_resize_has_no_dark_fringe(self):
        # Composite happens before the resize: a large transparent image stays white after downscaling.
        image = Image.new("RGBA", (2048, 2048), (0, 0, 0, 0))
        decoded = self.decode(image)
        self.assertLess(decoded.width, 2048)
        self.assertEqual(set(pixels(decoded)), {WHITE})


    def test_i16_unused_key_keeps_opaque_black(self):
        # Pillow cuts a 16-bit tRNS key to 8 bits in convert("RGBA"): key 256 would erase every 0 pixel.
        decoded = self.decode(Image.new("I;16", (256, 256), 0), transparency=256)
        self.assertEqual(set(pixels(decoded)), {BLACK})

    def test_i16_key_keeps_opaque_zero_pixel(self):
        image = Image.new("I;16", (256, 256), 256)
        image.putpixel((128, 128), 0)
        decoded = self.decode(image, transparency=256)
        self.assertEqual(pixel(decoded, 128, 128), BLACK)


@unittest.skipIf(Image is None, "Pillow is not installed")
class OpaqueImagesUnchanged(unittest.TestCase):
    """The digest feeds the vision cache keys: opaque images must give the bytes of plain convert("RGB")."""

    def check(self, image):
        raw = png(image)
        # The trunk 849c8c45 pipeline: load, convert("RGB"), then a bicubic resize to the smart_resize size.
        reference = Image.open(io.BytesIO(raw))
        reference.load()
        reference = reference.convert("RGB")
        height, width = smart_resize(reference.height, reference.width, max_pixels=SETTINGS.max_pixels)
        if (width, height) != reference.size:
            reference = reference.resize((width, height), Image.BICUBIC)
        expected = reference.tobytes()
        decoded = decode_bytes(raw, SETTINGS)
        self.assertEqual((decoded.width, decoded.height), (width, height))
        self.assertEqual(decoded.rgb, expected)
        self.assertEqual(decoded.digest, image_digest(height, width, expected))

    def gradient(self, mode, size=(320, 224)):
        # 320x224 needs no resize; 96x64 is scaled up, so both resize paths are covered.
        width, height = size
        image = Image.new("RGB", size)
        image.putdata([(x % 256, (y * 3) % 256, (x + y) % 256) for y in range(height) for x in range(width)])
        return image if mode == "RGB" else image.convert(mode)

    def test_rgb(self):
        self.check(self.gradient("RGB"))
        self.check(self.gradient("RGB", (96, 64)))

    def test_l(self):
        self.check(self.gradient("L"))

    def test_p_without_transparency(self):
        image = self.gradient("RGB").quantize(64)
        self.assertNotIn("transparency", image.info)
        self.check(image)

    def test_rgba_fully_opaque(self):
        for size in ((320, 224), (96, 64)):
            image = self.gradient("RGB", size)
            image.putalpha(255)
            self.assertEqual(image.mode, "RGBA")
            self.check(image)


if __name__ == "__main__":
    unittest.main()
