"""CI-safe tests for the T2-lite data pipeline (no network, no dataset)."""

from pathlib import Path

import cv2
import numpy as np

from scripts.build_t2_dataset import build_forgeries
from scripts.prepare_t2_authentic import downscale_rgb


def test_downscale_caps_longest_side():
    img = np.zeros((1000, 600, 3), dtype=np.uint8)
    out = downscale_rgb(img, max_side=384)
    assert max(out.shape[:2]) == 384
    assert out.shape[2] == 3


def test_downscale_is_noop_when_already_small():
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    out = downscale_rgb(img, max_side=384)
    assert out.shape == (200, 100, 3)


def test_build_forgeries_is_balanced_and_changes_pixels(tmp_path: Path):
    authentic = tmp_path / "authentic"
    authentic.mkdir()
    # A structured (non-uniform, saturated) image so the forgeries have material
    # to alter — on a flat gray image several forgery types are legitimately no-ops.
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    base[:, :112] = (200, 50, 50)
    base[:, 112:] = (50, 50, 200)
    base[80:140, 80:140] = (240, 220, 30)
    for i in range(3):
        cv2.imwrite(str(authentic / f"auth_{i:04d}.jpg"), base)

    forged = tmp_path / "forged"
    count = build_forgeries(authentic, forged, seed=1)

    assert count == 3
    assert len(list(forged.glob("*.jpg"))) == 3
    # At least one forgery must alter the flat authentic source. (Some types,
    # e.g. text_blur, are no-ops on a uniform image — that is expected.)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    differs = [
        not np.array_equal(cv2.imread(str(p)), base_bgr) for p in sorted(forged.glob("*.jpg"))
    ]
    assert any(differs)
