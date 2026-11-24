#!/usr/bin/env python3
"""Measure black, near-black and static VIctory frame captures.

Capture frames with a diagnostics build:

  $env:VICTORY_CAPTURE = "F:\\VIctory-logs\\frames"
  $env:VICTORY_CAPTURE_FRAMES = "64,128,256,512"
  .\\target\\diag\\victory.exe run <title.exe> --execute

Analyse them without copying images into the repository:

  python tools/frame_report.py F:\\VIctory-logs\\frames

The decoder supports non-interlaced 8-bit grayscale, RGB, grayscale-alpha
and RGBA PNGs. VIctory writes 8-bit RGB captures.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FRAME_NUMBER = re.compile(r"frame-(\d+)")
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


@dataclass(frozen=True)
class Image:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class Metrics:
    pixels: int
    mean_luma: float
    p50_luma: int
    p95_luma: int
    max_channel: int
    non_black_ratio: float
    dark_ratio: float
    luma_stddev: float
    content_ratio: float
    classification: str


@dataclass(frozen=True)
class Difference:
    changed_ratio: float
    mean_channel_delta: float
    max_channel_delta: int


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    dl = abs(estimate - left)
    da = abs(estimate - above)
    dul = abs(estimate - upper_left)
    if dl <= da and dl <= dul:
        return left
    if da <= dul:
        return above
    return upper_left


def unfilter(raw: bytes, width: int, height: int, channels: int) -> bytes:
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"inflated image is {len(raw)} bytes, expected {expected}")
    output = bytearray(height * stride)
    source = 0
    for y in range(height):
        kind = raw[source]
        source += 1
        row = y * stride
        for x in range(stride):
            value = raw[source]
            source += 1
            left = output[row + x - channels] if x >= channels else 0
            above = output[row - stride + x] if y > 0 else 0
            upper_left = output[row - stride + x - channels] if y > 0 and x >= channels else 0
            if kind == 0:
                decoded = value
            elif kind == 1:
                decoded = value + left
            elif kind == 2:
                decoded = value + above
            elif kind == 3:
                decoded = value + ((left + above) // 2)
            elif kind == 4:
                decoded = value + paeth(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {kind}")
            output[row + x] = decoded & 0xFF
    return bytes(output)


def decode_png(path: Path) -> Image:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    at = len(PNG_SIGNATURE)
    width = height = channels = 0
    compressed = bytearray()
    saw_end = False
    while at + 12 <= len(data):
        length = struct.unpack_from(">I", data, at)[0]
        at += 4
        kind = data[at : at + 4]
        at += 4
        end = at + length
        if end + 4 > len(data):
            raise ValueError("truncated PNG chunk")
        payload = data[at:end]
        at = end + 4  # CRC is validated by the capture writer; skip it here.
        if kind == b"IHDR":
            if length != 13:
                raise ValueError("invalid IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if width == 0 or height == 0:
                raise ValueError("empty PNG")
            if depth != 8 or color not in CHANNELS:
                raise ValueError(f"unsupported PNG depth/color {depth}/{color}")
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("compressed, filtered or interlaced PNG mode is unsupported")
            channels = CHANNELS[color]
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
    if not saw_end or not width or not compressed:
        raise ValueError("PNG is missing IHDR, IDAT or IEND")
    pixels = unfilter(zlib.decompress(compressed), width, height, channels)
    rgb = bytearray(width * height * 3)
    for index in range(width * height):
        source = index * channels
        target = index * 3
        if channels in (1, 2):
            value = pixels[source]
            rgb[target : target + 3] = bytes((value, value, value))
        else:
            rgb[target : target + 3] = pixels[source : source + 3]
    return Image(width, height, bytes(rgb))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * min(1.0, max(0.0, fraction)))]


def crop_image(image: Image, box: tuple[int, int, int, int]) -> Image:
    x, y, width, height = box
    if x < 0 or y < 0 or width < 1 or height < 1:
        raise ValueError("crop coordinates must be non-negative and its size positive")
    if x + width > image.width or y + height > image.height:
        raise ValueError(
            f"crop {x},{y},{width},{height} exceeds {image.width}x{image.height} image"
        )
    rgb = bytearray(width * height * 3)
    for row in range(height):
        source = 3 * ((y + row) * image.width + x)
        target = 3 * row * width
        rgb[target : target + 3 * width] = image.rgb[source : source + 3 * width]
    return Image(width, height, bytes(rgb))


def measure(image: Image, black: int, dark: int) -> Metrics:
    pixels = image.width * image.height
    lumas: list[int] = []
    non_black = dark_pixels = 0
    max_channel = 0
    min_x, min_y = image.width, image.height
    max_x = max_y = -1
    for index in range(pixels):
        r, g, b = image.rgb[3 * index : 3 * index + 3]
        level = max(r, g, b)
        max_channel = max(max_channel, level)
        # Integer Rec. 709 luma, rounded to 8 bits.
        luma = (54 * r + 183 * g + 19 * b + 128) >> 8
        lumas.append(luma)
        if level >= black:
            non_black += 1
            x, y = index % image.width, index // image.width
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        if level < dark:
            dark_pixels += 1
    mean = sum(lumas) / pixels
    variance = sum((value - mean) ** 2 for value in lumas) / pixels
    non_black_ratio = non_black / pixels
    dark_ratio = dark_pixels / pixels
    if max_x < 0:
        content_ratio = 0.0
    else:
        content_ratio = ((max_x - min_x + 1) * (max_y - min_y + 1)) / pixels
    if non_black_ratio <= 0.001:
        classification = "black"
    elif non_black_ratio <= 0.01 or mean < 3.0:
        classification = "near-black"
    elif dark_ratio >= 0.98:
        classification = "very-dark"
    else:
        classification = "visible"
    return Metrics(
        pixels=pixels,
        mean_luma=mean,
        p50_luma=percentile(lumas, 0.50),
        p95_luma=percentile(lumas, 0.95),
        max_channel=max_channel,
        non_black_ratio=non_black_ratio,
        dark_ratio=dark_ratio,
        luma_stddev=math.sqrt(variance),
        content_ratio=content_ratio,
        classification=classification,
    )


def difference(left: Image, right: Image, threshold: int) -> Difference | None:
    if (left.width, left.height) != (right.width, right.height):
        return None
    pixels = left.width * left.height
    changed = total = peak = 0
    for index in range(pixels):
        offset = 3 * index
        delta = max(
            abs(left.rgb[offset + channel] - right.rgb[offset + channel])
            for channel in range(3)
        )
        changed += delta >= threshold
        total += sum(
            abs(left.rgb[offset + channel] - right.rgb[offset + channel])
            for channel in range(3)
        )
        peak = max(peak, delta)
    return Difference(changed / pixels, total / (pixels * 3), peak)


def image_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            paths.update(item.glob("*.png"))
        elif item.suffix.lower() == ".png":
            paths.add(item)
    return sorted(paths, key=sort_key)


def sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_NUMBER.search(path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name.lower())


def print_report(
    paths: list[Path],
    black: int,
    dark: int,
    delta: int,
    crop: tuple[int, int, int, int] | None = None,
) -> int:
    decoded: list[tuple[Path, Image, Metrics]] = []
    failures = 0
    print("frame                               size       state       mean  p50  p95  max  nonblack  dark     content")
    for path in paths:
        try:
            image = decode_png(path)
            if crop is not None:
                image = crop_image(image, crop)
            metrics = measure(image, black, dark)
        except (OSError, ValueError, zlib.error) as error:
            print(f"{path}: {error}", file=sys.stderr)
            failures += 1
            continue
        decoded.append((path, image, metrics))
        print(
            f"{path.name[:35]:35} {image.width:4}x{image.height:<4}"
            f" {metrics.classification:11} {metrics.mean_luma:6.1f}"
            f" {metrics.p50_luma:4d} {metrics.p95_luma:4d} {metrics.max_channel:4d}"
            f" {100 * metrics.non_black_ratio:8.3f}%"
            f" {100 * metrics.dark_ratio:7.2f}%"
            f" {100 * metrics.content_ratio:8.2f}%"
        )
    if not decoded:
        return 1

    states: dict[str, int] = {}
    for _, _, metrics in decoded:
        states[metrics.classification] = states.get(metrics.classification, 0) + 1
    print("\nsummary: " + ", ".join(f"{count} {state}" for state, count in sorted(states.items())))

    pairs = []
    print("\nframe-to-frame changes")
    for (left_path, left, _), (right_path, right, _) in zip(decoded, decoded[1:]):
        diff = difference(left, right, delta)
        if diff is None:
            print(f"  {left_path.name} -> {right_path.name}: size changed")
            continue
        pairs.append(diff)
        state = "static" if diff.changed_ratio <= 0.001 else "changing"
        print(
            f"  {left_path.name} -> {right_path.name}: {state},"
            f" {100 * diff.changed_ratio:.3f}% pixels changed,"
            f" mean delta {diff.mean_channel_delta:.2f}, max {diff.max_channel_delta}"
        )
    if pairs and all(pair.changed_ratio <= 0.001 for pair in pairs):
        print("\nverdict: captures are static (<=0.1% changed pixels per comparable pair)")
    elif all(metrics.classification in ("black", "near-black") for _, _, metrics in decoded):
        print("\nverdict: every decoded capture is black or near-black")
    else:
        print("\nverdict: visible or changing frame content was detected")
    return 2 if failures else 0


def crop_argument(text: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height") from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    if values[0] < 0 or values[1] < 0 or values[2] < 1 or values[3] < 1:
        raise argparse.ArgumentTypeError("crop origin must be non-negative and size positive")
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("inputs", type=Path, nargs="+", help="PNG files or capture directories")
    result.add_argument("--black", type=int, default=8, help="non-black channel threshold (default: 8)")
    result.add_argument("--dark", type=int, default=24, help="dark channel threshold (default: 24)")
    result.add_argument("--delta", type=int, default=8, help="changed-pixel channel delta (default: 8)")
    result.add_argument(
        "--crop",
        type=crop_argument,
        help="analyse only x,y,width,height (useful for screenshots with window borders)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("black", "dark", "delta"):
        value = getattr(args, name)
        if not 0 <= value <= 255:
            parser().error(f"--{name} must be between 0 and 255")
    paths = image_paths(args.inputs)
    if not paths:
        print("frame_report: no PNG files found", file=sys.stderr)
        return 1
    return print_report(paths, args.black, args.dark, args.delta, args.crop)


if __name__ == "__main__":
    raise SystemExit(main())
