#!/usr/bin/env python3
"""Generate synthetic forgeries from real authentic frames (T2-lite).

For every authentic frame in data/train/authentic/, applies one synthetic
forgery (cycling the 4 types and 3 intensities of SyntheticForgeryGenerator)
and writes it to data/train/forged/. The result is a balanced dataset whose
authentic class is real and whose forged class is synthetic.

Usage:
    uv run python scripts/build_t2_dataset.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from src.data.augmentation import ForgeryConfig, ForgeryType, Intensity, SyntheticForgeryGenerator

_INTENSITIES: list[Intensity] = ["mild", "medium", "strong"]


def build_forgeries(authentic_dir: Path, forged_dir: Path, seed: int = 42) -> int:
    """Write one synthetic forgery per authentic image. Returns the count."""
    if forged_dir.exists():
        shutil.rmtree(forged_dir)  # idempotent: avoid mixing in stale forgeries
    forged_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticForgeryGenerator(seed=seed)
    types = list(ForgeryType)

    paths = sorted(authentic_dir.glob("*.jpg")) + sorted(authentic_dir.glob("*.png"))
    count = 0
    for i, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        config = ForgeryConfig(
            forgery_type=types[i % len(types)],
            intensity=_INTENSITIES[i % len(_INTENSITIES)],
            seed=seed + i,
        )
        forged = generator.apply(rgb, config)
        cv2.imwrite(
            str(forged_dir / f"forged_{i:04d}.jpg"), cv2.cvtColor(forged, cv2.COLOR_RGB2BGR)
        )
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authentic", type=Path, default=Path("data/train/authentic"))
    ap.add_argument("--forged", type=Path, default=Path("data/train/forged"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = build_forgeries(args.authentic, args.forged, seed=args.seed)
    print(f"Wrote {n} synthetic forgeries to {args.forged}/")


if __name__ == "__main__":
    main()
