"""CI-safe tests for the T2-lite data pipeline (no network, no dataset)."""

import numpy as np

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
