#!/usr/bin/env python3

import binascii
import contextlib
import io
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import frame_report


def chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def png(width: int, height: int, color: int, channels: int, pixels: bytes) -> bytes:
    rows = b"".join(
        b"\0" + pixels[y * width * channels : (y + 1) * width * channels]
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return (
        frame_report.PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class FrameReportTests(unittest.TestCase):
    def test_decodes_rgb_and_ignores_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb_path = Path(directory) / "rgb.png"
            rgba_path = Path(directory) / "rgba.png"
            rgb_path.write_bytes(png(2, 1, 2, 3, bytes([1, 2, 3, 4, 5, 6])))
            rgba_path.write_bytes(png(1, 1, 6, 4, bytes([7, 8, 9, 0])))
            rgb = frame_report.decode_png(rgb_path)
            rgba = frame_report.decode_png(rgba_path)

        self.assertEqual((rgb.width, rgb.height, rgb.rgb), (2, 1, bytes([1, 2, 3, 4, 5, 6])))
        self.assertEqual(rgba.rgb, bytes([7, 8, 9]))

    def test_png_filters_reconstruct_rows(self):
        # First row uses Sub: [10, 10, 10] reconstructs [10, 20, 30].
        # Second uses Up: [1, 2, 3] reconstructs [11, 22, 33].
        raw = bytes([1, 10, 10, 10, 2, 1, 2, 3])
        self.assertEqual(
            frame_report.unfilter(raw, width=3, height=2, channels=1),
            bytes([10, 20, 30, 11, 22, 33]),
        )

    def test_measure_classifies_black_and_visible_images(self):
        black = frame_report.Image(2, 2, bytes([0, 0, 0] * 4))
        visible = frame_report.Image(2, 2, bytes([255, 255, 255] * 4))
        self.assertEqual(frame_report.measure(black, 8, 24).classification, "black")
        measured = frame_report.measure(visible, 8, 24)
        self.assertEqual(measured.classification, "visible")
        self.assertEqual(measured.non_black_ratio, 1.0)
        self.assertEqual(measured.content_ratio, 1.0)

    def test_crop_selects_the_requested_pixels(self):
        image = frame_report.Image(
            2,
            2,
            bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        )
        cropped = frame_report.crop_image(image, (1, 0, 1, 2))
        self.assertEqual((cropped.width, cropped.height), (1, 2))
        self.assertEqual(cropped.rgb, bytes([4, 5, 6, 10, 11, 12]))

    def test_difference_detects_static_and_changed_pixels(self):
        left = frame_report.Image(2, 1, bytes([0, 0, 0, 10, 10, 10]))
        same = frame_report.Image(2, 1, left.rgb)
        right = frame_report.Image(2, 1, bytes([20, 0, 0, 10, 10, 10]))
        self.assertEqual(frame_report.difference(left, same, 8).changed_ratio, 0.0)
        self.assertEqual(frame_report.difference(left, right, 8).changed_ratio, 0.5)

    def test_main_reports_a_capture_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frame-1.png").write_bytes(png(1, 1, 2, 3, bytes([0, 0, 0])))
            (folder / "frame-2.png").write_bytes(png(1, 1, 2, 3, bytes([0, 0, 0])))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = frame_report.main([str(folder)])

        self.assertEqual(result, 0)
        self.assertIn("2 black", output.getvalue())
        self.assertIn("captures are static", output.getvalue())


if __name__ == "__main__":
    unittest.main()
