"""Tests for src/reporting/pdf_report.py and the /report API endpoint."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("reportlab", reason="reportlab not installed")

from src.reporting.pdf_report import PDFReportGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_b64_image(size: int = 64, fmt: str = "PNG") -> str:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def make_result(
    label: str = "forged",
    prob: float = 0.87,
    with_gradcam: bool = False,
) -> dict:
    return {
        "label": label,
        "probability": prob,
        "threshold": 0.5,
        "inference_ms": 14.2,
        "gradcam_b64": make_b64_image() if with_gradcam else None,
        "most_activated_region": (
            {"cx": 112.0, "cy": 96.0, "mean_activation": 0.73} if with_gradcam else None
        ),
    }


MODEL_INFO = {
    "architecture": "EfficientNet-B0",
    "total_params": 5288548,
    "trainable_params": 5288548,
    "checkpoint": "efficientnet_b0_best.pt",
    "device": "cpu",
    "input_size": [3, 224, 224],
    "classes": ["authentic", "forged"],
    "metadata": {},
}


# ---------------------------------------------------------------------------
# 1. PDFReportGenerator
# ---------------------------------------------------------------------------


class TestPDFReportGenerator:
    def test_returns_valid_pdf_bytes(self):
        pdf = PDFReportGenerator().generate(make_result(), make_b64_image())
        assert pdf[:5] == b"%PDF-"

    def test_pdf_not_empty(self):
        pdf = PDFReportGenerator().generate(make_result(), make_b64_image())
        assert len(pdf) > 1000

    def test_authentic_label(self):
        pdf = PDFReportGenerator().generate(
            make_result(label="authentic", prob=0.12), make_b64_image()
        )
        assert pdf[:5] == b"%PDF-"

    def test_with_gradcam_overlay(self):
        pdf = PDFReportGenerator().generate(make_result(with_gradcam=True), make_b64_image())
        assert pdf[:5] == b"%PDF-"

    def test_gradcam_increases_size(self):
        gen = PDFReportGenerator()
        image_b64 = make_b64_image()
        without = gen.generate(make_result(with_gradcam=False), image_b64)
        with_cam = gen.generate(make_result(with_gradcam=True), image_b64)
        assert len(with_cam) > len(without)

    def test_with_model_info_and_filename(self):
        pdf = PDFReportGenerator().generate(
            make_result(),
            make_b64_image(),
            model_info=MODEL_INFO,
            filename="cedula_scan.jpg",
        )
        assert pdf[:5] == b"%PDF-"

    def test_jpeg_input_image(self):
        pdf = PDFReportGenerator().generate(make_result(), make_b64_image(fmt="JPEG"))
        assert pdf[:5] == b"%PDF-"

    def test_non_square_image_preserves_validity(self):
        arr = np.random.randint(0, 256, (40, 200, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        pdf = PDFReportGenerator().generate(make_result(), b64)
        assert pdf[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# 2. /report endpoint
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    from api.main import app

    pred = MagicMock()
    pred.predict.return_value = make_result()
    pred.model_info.return_value = MODEL_INFO
    with TestClient(app) as c:
        with patch("api.main._predictor", pred):
            yield c, pred


class TestReportEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.post("/report", json={"image_b64": make_b64_image()})
        assert resp.status_code == 200

    def test_content_type_pdf(self, client):
        c, _ = client
        resp = c.post("/report", json={"image_b64": make_b64_image()})
        assert resp.headers["content-type"] == "application/pdf"

    def test_body_is_pdf(self, client):
        c, _ = client
        resp = c.post("/report", json={"image_b64": make_b64_image()})
        assert resp.content[:5] == b"%PDF-"

    def test_content_disposition_attachment(self, client):
        c, _ = client
        resp = c.post("/report", json={"image_b64": make_b64_image()})
        assert "attachment" in resp.headers["content-disposition"]

    def test_predictor_receives_request_params(self, client):
        c, pred = client
        c.post(
            "/report",
            json={
                "image_b64": make_b64_image(),
                "threshold": 0.7,
                "return_gradcam": True,
                "gradcam_method": "eigencam",
            },
        )
        kwargs = pred.predict.call_args.kwargs
        assert kwargs["threshold"] == 0.7
        assert kwargs["return_gradcam"] is True
        assert kwargs["gradcam_method"] == "eigencam"

    def test_503_when_no_model(self):
        from api.main import app

        with TestClient(app) as c:
            with patch("api.main._predictor", None):
                resp = c.post("/report", json={"image_b64": make_b64_image()})
                assert resp.status_code == 503
