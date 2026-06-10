"""Tests for dashboard/app.py helper functions and AppTest smoke tests."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("streamlit", reason="streamlit not installed")

# ---------------------------------------------------------------------------
# Helpers for creating fake data
# ---------------------------------------------------------------------------

def make_pil(size: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_fake_result(label: str = "forged", prob: float = 0.8) -> dict:
    return {
        "label": label,
        "probability": prob,
        "threshold": 0.5,
        "inference_ms": 12.3,
        "gradcam_b64": None,
        "most_activated_region": None,
    }


# ---------------------------------------------------------------------------
# 1. Helper function: pil_to_b64 / b64_to_pil
# ---------------------------------------------------------------------------

class TestImageHelpers:
    def test_pil_to_b64_is_valid_base64(self):
        from dashboard.app import pil_to_b64
        img = make_pil()
        b64 = pil_to_b64(img)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_b64_to_pil_roundtrip(self):
        from dashboard.app import b64_to_pil, pil_to_b64
        img = make_pil()
        b64 = pil_to_b64(img)
        restored = b64_to_pil(b64)
        assert restored.size == img.size

    def test_pil_to_b64_png_format(self):
        from dashboard.app import pil_to_b64
        img = make_pil()
        b64 = pil_to_b64(img, fmt="PNG")
        raw = base64.b64decode(b64)
        # PNG magic bytes
        assert raw[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# 2. _label_color helper
# ---------------------------------------------------------------------------

class TestLabelColor:
    def test_authentic_returns_green(self):
        from dashboard.app import _label_color
        assert _label_color("authentic") == "🟢"

    def test_forged_returns_red(self):
        from dashboard.app import _label_color
        assert _label_color("forged") == "🔴"


# ---------------------------------------------------------------------------
# 3. AppTest smoke tests — verify the app starts and renders key widgets
# ---------------------------------------------------------------------------

APP_PATH = str(Path(__file__).parent.parent / "dashboard" / "app.py")


@pytest.fixture()
def mock_predictor():
    pred = MagicMock()
    pred.model_info.return_value = {
        "architecture": "EfficientNet-B0",
        "total_params": 5288548,
        "trainable_params": 5288548,
        "checkpoint": "efficientnet_b0_best.pt",
        "device": "cpu",
        "input_size": [3, 224, 224],
        "classes": ["authentic", "forged"],
        "metadata": {},
    }
    pred.predict.return_value = make_fake_result()
    return pred


class TestAppSmoke:
    def test_app_runs_without_exception(self, mock_predictor):
        from streamlit.testing.v1 import AppTest
        with patch("dashboard.app.load_predictor", return_value=mock_predictor):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            assert not at.exception, f"App raised: {at.exception}"

    def test_sidebar_has_threshold_slider(self, mock_predictor):
        from streamlit.testing.v1 import AppTest
        with patch("dashboard.app.load_predictor", return_value=mock_predictor):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            slider_labels = [s.label for s in at.slider]
            assert any("threshold" in lbl.lower() for lbl in slider_labels)

    def test_three_tabs_present(self, mock_predictor):
        from streamlit.testing.v1 import AppTest
        with patch("dashboard.app.load_predictor", return_value=mock_predictor):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            # Tabs show as headers or tab elements — check headers
            headers = [h.value for h in at.header]
            expected = {"Upload a document image", "Demo — synthetic samples",
                        "Session statistics"}
            assert expected <= set(headers), f"Missing tabs, got: {headers}"

    def test_no_history_shows_info(self, mock_predictor):
        from streamlit.testing.v1 import AppTest
        with patch("dashboard.app.load_predictor", return_value=mock_predictor):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            info_texts = [i.value for i in at.info]
            assert any("no documents" in t.lower() for t in info_texts)

    def test_sidebar_shows_model_loaded_when_checkpoint_present(self, mock_predictor):
        """Sidebar success block is rendered when predictor is not None."""
        from streamlit.testing.v1 import AppTest
        with patch("dashboard.app.load_predictor", return_value=mock_predictor):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            assert not at.exception
            success_texts = [s.value for s in at.success]
            assert any("model loaded" in t.lower() for t in success_texts)
