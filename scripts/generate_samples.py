#!/usr/bin/env python3
"""
Generate 20 synthetic test images (224x224 RGB) for document authentication tests.

10 authentic: light background + colored rectangles simulating text/stamps.
10 forged: same structure + Gaussian noise patch or color-shifted region.

Saves to:
    data/samples/authentic/sample_00.png ... sample_09.png
    data/samples/forged/sample_00.png    ... sample_09.png

Uses numpy+PIL when available; falls back to stdlib-only (zlib/struct) PNG writer.

Usage:
    python scripts/generate_samples.py
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIDTH = 224
HEIGHT = 224
SEED = 42
N_PER_CLASS = 10
OUTPUT_DIR = Path("data/samples")


# ---------------------------------------------------------------------------
# Stdlib-only minimal PNG writer
# ---------------------------------------------------------------------------


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def _write_png_stdlib(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    """Write a 24-bit RGB PNG using only stdlib (zlib + struct)."""
    raw_rows = bytearray()
    for row in pixels:
        raw_rows.append(0)  # filter byte: None
        for r, g, b in row:
            raw_rows += bytes([r, g, b])

    compressed = zlib.compress(bytes(raw_rows), level=6)

    ihdr_data = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# Pixel-level drawing helpers (stdlib)
# ---------------------------------------------------------------------------


def _make_canvas(
    width: int, height: int, color: tuple[int, int, int]
) -> list[list[tuple[int, int, int]]]:
    return [[color] * width for _ in range(height)]


def _fill_rect(
    pixels: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            pixels[y][x] = color


def _add_noise_patch(
    pixels: list[list[tuple[int, int, int]]],
    rng: random.Random,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    intensity: int = 60,
) -> None:
    """Add random noise to a rectangular region."""
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            r, g, b = pixels[y][x]
            dr = rng.randint(-intensity, intensity)
            dg = rng.randint(-intensity, intensity)
            db = rng.randint(-intensity, intensity)
            pixels[y][x] = (
                max(0, min(255, r + dr)),
                max(0, min(255, g + dg)),
                max(0, min(255, b + db)),
            )


def _color_shift_patch(
    pixels: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    shift: tuple[int, int, int] = (80, -30, -30),
) -> None:
    """Shift colors in a rectangular region (simulates hue tampering)."""
    sr, sg, sb = shift
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            r, g, b = pixels[y][x]
            pixels[y][x] = (
                max(0, min(255, r + sr)),
                max(0, min(255, g + sg)),
                max(0, min(255, b + sb)),
            )


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------


def _make_authentic(rng: random.Random) -> list[list[tuple[int, int, int]]]:
    """Light document background with colored rectangles (text lines, stamps)."""
    bg_r = rng.randint(230, 255)
    bg_g = rng.randint(230, 255)
    bg_b = rng.randint(220, 255)
    pixels = _make_canvas(WIDTH, HEIGHT, (bg_r, bg_g, bg_b))

    # Document border
    _fill_rect(pixels, 10, 10, WIDTH - 10, HEIGHT - 10, (bg_r - 10, bg_g - 10, bg_b - 10))
    _fill_rect(pixels, 12, 12, WIDTH - 12, HEIGHT - 12, (bg_r, bg_g, bg_b))

    # Simulated text lines (thin dark rectangles)
    n_lines = rng.randint(5, 9)
    for i in range(n_lines):
        y = 30 + i * 18 + rng.randint(-2, 2)
        line_w = rng.randint(80, 180)
        x_start = rng.randint(15, 40)
        darkness = rng.randint(30, 90)
        _fill_rect(pixels, x_start, y, x_start + line_w, y + 4, (darkness, darkness, darkness + 10))

    # Simulated stamp (colored circle approximated as rectangle)
    stamp_x = rng.randint(120, 170)
    stamp_y = rng.randint(130, 170)
    stamp_r = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(100, 200))
    _fill_rect(pixels, stamp_x, stamp_y, stamp_x + 40, stamp_y + 40, stamp_r)
    # Inner lighter area
    _fill_rect(
        pixels,
        stamp_x + 5,
        stamp_y + 5,
        stamp_x + 35,
        stamp_y + 35,
        (min(255, stamp_r[0] + 60), min(255, stamp_r[1] + 60), min(255, stamp_r[2] + 40)),
    )

    return pixels


def _make_forged(rng: random.Random) -> list[list[tuple[int, int, int]]]:
    """Start from authentic layout, add visible perturbation."""
    pixels = _make_authentic(rng)

    perturbation = rng.choice(["noise", "color_shift"])
    # Place perturbation in a salient area
    px0 = rng.randint(40, 100)
    py0 = rng.randint(40, 100)
    px1 = px0 + rng.randint(40, 80)
    py1 = py0 + rng.randint(40, 80)

    if perturbation == "noise":
        _add_noise_patch(pixels, rng, px0, py0, px1, py1, intensity=70)
    else:
        shift = (
            rng.randint(60, 100),
            rng.randint(-60, -20),
            rng.randint(-60, -20),
        )
        _color_shift_patch(pixels, px0, py0, px1, py1, shift=shift)

    return pixels


# ---------------------------------------------------------------------------
# numpy + PIL path (preferred when available)
# ---------------------------------------------------------------------------


def _try_numpy_pil(output_dir: Path, rng_seed: int) -> bool:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False

    rng = np.random.default_rng(rng_seed)

    authentic_dir = output_dir / "authentic"
    forged_dir = output_dir / "forged"
    authentic_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for i in range(N_PER_CLASS):
        # Authentic
        bg = rng.integers(230, 256, size=3).tolist()
        img = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)
        # Border
        img[10 : HEIGHT - 10, 10 : WIDTH - 10] = [max(0, c - 10) for c in bg]
        img[12 : HEIGHT - 12, 12 : WIDTH - 12] = bg

        n_lines = int(rng.integers(5, 10))
        for j in range(n_lines):
            y = 30 + j * 18 + int(rng.integers(-2, 3))
            line_w = int(rng.integers(80, 181))
            x_start = int(rng.integers(15, 41))
            darkness = int(rng.integers(30, 91))
            y2 = min(HEIGHT, y + 4)
            x2 = min(WIDTH, x_start + line_w)
            img[y:y2, x_start:x2] = [darkness, darkness, darkness + 10]

        stamp_x = int(rng.integers(120, 171))
        stamp_y = int(rng.integers(130, 171))
        stamp_color = [
            int(rng.integers(0, 81)),
            int(rng.integers(0, 81)),
            int(rng.integers(100, 201)),
        ]
        img[stamp_y : stamp_y + 40, stamp_x : stamp_x + 40] = stamp_color
        img[stamp_y + 5 : stamp_y + 35, stamp_x + 5 : stamp_x + 35] = [
            min(255, stamp_color[0] + 60),
            min(255, stamp_color[1] + 60),
            min(255, stamp_color[2] + 40),
        ]

        Image.fromarray(img).save(authentic_dir / f"sample_{i:02d}.png")
        count += 1

    for i in range(N_PER_CLASS):
        # Forged: copy authentic logic with same seed offset, then perturb
        bg = rng.integers(230, 256, size=3).tolist()
        img = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)
        img[10 : HEIGHT - 10, 10 : WIDTH - 10] = [max(0, c - 10) for c in bg]
        img[12 : HEIGHT - 12, 12 : WIDTH - 12] = bg

        n_lines = int(rng.integers(5, 10))
        for j in range(n_lines):
            y = 30 + j * 18 + int(rng.integers(-2, 3))
            line_w = int(rng.integers(80, 181))
            x_start = int(rng.integers(15, 41))
            darkness = int(rng.integers(30, 91))
            y2 = min(HEIGHT, y + 4)
            x2 = min(WIDTH, x_start + line_w)
            img[y:y2, x_start:x2] = [darkness, darkness, darkness + 10]

        stamp_x = int(rng.integers(120, 171))
        stamp_y = int(rng.integers(130, 171))
        stamp_color = [
            int(rng.integers(0, 81)),
            int(rng.integers(0, 81)),
            int(rng.integers(100, 201)),
        ]
        img[stamp_y : stamp_y + 40, stamp_x : stamp_x + 40] = stamp_color
        img[stamp_y + 5 : stamp_y + 35, stamp_x + 5 : stamp_x + 35] = [
            min(255, stamp_color[0] + 60),
            min(255, stamp_color[1] + 60),
            min(255, stamp_color[2] + 40),
        ]

        # Perturbation
        px0 = int(rng.integers(40, 101))
        py0 = int(rng.integers(40, 101))
        px1 = min(WIDTH, px0 + int(rng.integers(40, 81)))
        py1 = min(HEIGHT, py0 + int(rng.integers(40, 81)))

        if i % 2 == 0:
            noise = rng.integers(-70, 71, size=(py1 - py0, px1 - px0, 3))
            patch = img[py0:py1, px0:px1].astype(np.int16) + noise
            img[py0:py1, px0:px1] = np.clip(patch, 0, 255).astype(np.uint8)
        else:
            shift = np.array([80, -30, -30], dtype=np.int16)
            patch = img[py0:py1, px0:px1].astype(np.int16) + shift
            img[py0:py1, px0:px1] = np.clip(patch, 0, 255).astype(np.uint8)

        Image.fromarray(img).save(forged_dir / f"sample_{i:02d}.png")
        count += 1

    print(f"Generated {count} samples in {output_dir}/")
    return True


# ---------------------------------------------------------------------------
# stdlib fallback path
# ---------------------------------------------------------------------------


def _generate_stdlib(output_dir: Path, seed: int) -> None:
    rng = random.Random(seed)

    authentic_dir = output_dir / "authentic"
    forged_dir = output_dir / "forged"
    authentic_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for i in range(N_PER_CLASS):
        pixels = _make_authentic(rng)
        _write_png_stdlib(authentic_dir / f"sample_{i:02d}.png", pixels)
        count += 1

    for i in range(N_PER_CLASS):
        pixels = _make_forged(rng)
        _write_png_stdlib(forged_dir / f"sample_{i:02d}.png", pixels)
        count += 1

    print(f"Generated {count} samples in {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = OUTPUT_DIR

    used_numpy = _try_numpy_pil(output_dir, SEED)
    if not used_numpy:
        _generate_stdlib(output_dir, SEED)


if __name__ == "__main__":
    main()
