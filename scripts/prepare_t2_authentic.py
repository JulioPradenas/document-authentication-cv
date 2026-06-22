#!/usr/bin/env python3
"""Download ONE MIDV-500 document type and produce real authentic frames (T2-lite).

Disk-safe by design: reads `.tif` frames directly out of the downloaded zip
(never extracting the .mov video), downscales each to longest-side <= max_side,
writes them as JPGs to data/train/authentic/, then deletes the zip.

Usage:
    uv run python scripts/prepare_t2_authentic.py --type 01_alb_id --n 100
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".tif", ".tiff")


def downscale_rgb(img: np.ndarray, max_side: int = 384) -> np.ndarray:
    """Downscale an RGB image so its longest side is <= max_side (no upscaling)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def _zip_url(midv_type: str) -> str:
    from midv500.download_dataset import midv500_links

    for link in midv500_links:
        if link.rstrip("/").split("/")[-1] == f"{midv_type}.zip":
            return link
    available = ", ".join(sorted(link.split("/")[-1][:-4] for link in midv500_links))
    raise SystemExit(f"Unknown MIDV type {midv_type!r}. Available: {available}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="01_alb_id", help="MIDV-500 document type (zip basename)")
    ap.add_argument("--n", type=int, default=100, help="Number of frames to keep")
    ap.add_argument("--max-side", type=int, default=384, help="Longest side after downscale")
    ap.add_argument("--out", type=Path, default=Path("data/train/authentic"))
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    url = _zip_url(args.type)
    zip_path = Path("data") / f"{args.type}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded {zip_path.stat().st_size / 1e6:.0f} MB")

    saved = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m
                for m in zf.namelist()
                if "/images/" in m.lower() and m.lower().endswith(IMAGE_EXTS)
            ]
            members.sort()
            if not members:
                raise SystemExit("No image frames found under images/ in the zip.")
            step = max(1, len(members) // args.n)
            picked = members[::step][: args.n]
            for member in picked:
                raw = np.frombuffer(zf.read(member), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = downscale_rgb(rgb, args.max_side)
                cv2.imwrite(
                    str(out / f"auth_{saved:04d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                saved += 1
    finally:
        zip_path.unlink(missing_ok=True)  # reclaim disk no matter what

    print(f"Saved {saved} authentic frames to {out}/ ; removed {zip_path.name}")


if __name__ == "__main__":
    main()
