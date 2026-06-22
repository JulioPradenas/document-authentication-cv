#!/usr/bin/env python3
"""Build a small balanced training set for the demo checkpoint (T1 path).

Generates synthetic "documents" with clearly separable forgeries so an
ImageNet-initialized EfficientNet-B0 can learn a real (non-random) decision
boundary in minutes on CPU/MPS. This is a toy task — visible perturbations,
not real document forensics — but it yields a genuine checkpoint that produces
verdicts ≠ 0.5 for the deployed dashboard.

Output layout (consumed by src.data.loader.create_dataloaders):
    data/train/authentic/auth_XXXX.png
    data/train/forged/forged_XXXX.png

Usage:
    uv run python scripts/build_training_data.py --n-per-class 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH = HEIGHT = 224


def _make_authentic(rng: np.random.Generator) -> np.ndarray:
    bg = rng.integers(228, 256, size=3).tolist()
    img = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)
    img[10 : HEIGHT - 10, 10 : WIDTH - 10] = [max(0, c - 12) for c in bg]
    img[12 : HEIGHT - 12, 12 : WIDTH - 12] = bg

    for j in range(int(rng.integers(5, 11))):
        y = 28 + j * 17 + int(rng.integers(-2, 3))
        x0 = int(rng.integers(15, 41))
        w = int(rng.integers(80, 181))
        d = int(rng.integers(25, 95))
        img[y : min(HEIGHT, y + 4), x0 : min(WIDTH, x0 + w)] = [d, d, min(255, d + 12)]

    sx, sy = int(rng.integers(118, 172)), int(rng.integers(125, 172))
    stamp = [int(rng.integers(0, 85)), int(rng.integers(0, 85)), int(rng.integers(100, 205))]
    img[sy : sy + 40, sx : sx + 40] = stamp
    img[sy + 5 : sy + 35, sx + 5 : sx + 35] = [min(255, c + 55) for c in stamp]
    return img


def _forge(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply one clearly visible perturbation to an authentic document."""
    out = img.copy()
    px0, py0 = int(rng.integers(35, 110)), int(rng.integers(35, 110))
    px1 = min(WIDTH, px0 + int(rng.integers(45, 90)))
    py1 = min(HEIGHT, py0 + int(rng.integers(45, 90)))
    kind = rng.integers(0, 3)

    if kind == 0:  # noise patch
        noise = rng.integers(-75, 76, size=(py1 - py0, px1 - px0, 3))
        patch = out[py0:py1, px0:px1].astype(np.int16) + noise
        out[py0:py1, px0:px1] = np.clip(patch, 0, 255).astype(np.uint8)
    elif kind == 1:  # color shift
        shift = np.array([85, -35, -35], dtype=np.int16)
        patch = out[py0:py1, px0:px1].astype(np.int16) + shift
        out[py0:py1, px0:px1] = np.clip(patch, 0, 255).astype(np.uint8)
    else:  # splice: paste a darker block (simulates pasted field)
        block = rng.integers(40, 120, size=3).tolist()
        out[py0:py1, px0:px1] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/train"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    auth_dir = args.out / "authentic"
    forged_dir = args.out / "forged"
    auth_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_per_class):
        Image.fromarray(_make_authentic(rng)).save(auth_dir / f"auth_{i:04d}.png")
    for i in range(args.n_per_class):
        base = _make_authentic(rng)
        Image.fromarray(_forge(base, rng)).save(forged_dir / f"forged_{i:04d}.png")

    print(f"Wrote {args.n_per_class} authentic + {args.n_per_class} forged to {args.out}/")


if __name__ == "__main__":
    main()
