"""Tests for api/schemas.py, api/predictor.py, and api/main.py."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("torch", reason="torch not installed")

from fastapi.testclient import TestClient

from api.schemas import AuthenticateRequest, BatchAuthenticateRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_b64_image(size: int = 64) -> str:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_mock_predictor(prob: float = 0.8) -> MagicMock:
    pred = MagicMock()
    pred.predict.return_value = {
        "label": "forged" if prob >= 0.5 else "authentic",
        "probability": prob,
        "threshold": 0.5,
        "inference_ms": 12.3,
        "gradcam_b64": None,
        "most_activated_region": None,
    }
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
    return pred


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from api.main import app

    mock_pred = make_mock_predictor()
    # TestClient first (triggers lifespan), then patch _predictor with mock
    with TestClient(app) as c:
        with patch("api.main._predictor", mock_pred):
            yield c, mock_pred


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_valid_request(self):
        req = AuthenticateRequest(image_b64=make_b64_image())
        assert req.threshold == 0.5
        assert req.return_gradcam is False

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            AuthenticateRequest(image_b64="not-base64!!!")

    def test_data_uri_stripped(self):
        raw_b64 = make_b64_image()
        req = AuthenticateRequest(image_b64=f"data:image/jpeg;base64,{raw_b64}")
        assert not req.image_b64.startswith("data:")

    def test_invalid_gradcam_method_raises(self):
        with pytest.raises(Exception):
            AuthenticateRequest(image_b64=make_b64_image(), gradcam_method="invalid")

    def test_threshold_bounds(self):
        with pytest.raises(Exception):
            AuthenticateRequest(image_b64=make_b64_image(), threshold=1.5)

    def test_batch_min_length(self):
        with pytest.raises(Exception):
            BatchAuthenticateRequest(images=[])

    def test_batch_max_length(self):
        items = [AuthenticateRequest(image_b64=make_b64_image()) for _ in range(33)]
        with pytest.raises(Exception):
            BatchAuthenticateRequest(images=items)


# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_status_ok(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. GET /model/info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/model/info")
        assert resp.status_code == 200

    def test_architecture_field(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert data["architecture"] == "EfficientNet-B0"

    def test_classes_field(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert data["classes"] == ["authentic", "forged"]

    def test_input_size_field(self, client):
        c, _ = client
        data = c.get("/model/info").json()
        assert data["input_size"] == [3, 224, 224]


# ---------------------------------------------------------------------------
# 4. POST /authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.post("/authenticate", json={"image_b64": make_b64_image()})
        assert resp.status_code == 200

    def test_response_keys(self, client):
        c, _ = client
        data = c.post("/authenticate", json={"image_b64": make_b64_image()}).json()
        assert {"label", "probability", "threshold", "inference_ms"} <= data.keys()

    def test_label_forged_when_high_prob(self, client):
        c, _ = client
        data = c.post("/authenticate", json={"image_b64": make_b64_image()}).json()
        assert data["label"] == "forged"

    def test_probability_range(self, client):
        c, _ = client
        data = c.post("/authenticate", json={"image_b64": make_b64_image()}).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_gradcam_none_by_default(self, client):
        c, _ = client
        data = c.post("/authenticate", json={"image_b64": make_b64_image()}).json()
        assert data["gradcam_b64"] is None

    def test_custom_threshold_passed(self, client):
        c, mock_pred = client
        c.post("/authenticate", json={"image_b64": make_b64_image(), "threshold": 0.3})
        call_kwargs = mock_pred.predict.call_args.kwargs
        assert call_kwargs["threshold"] == 0.3

    def test_invalid_image_returns_422(self, client):
        c, mock_pred = client
        mock_pred.predict.side_effect = ValueError("decode error")
        resp = c.post("/authenticate", json={"image_b64": make_b64_image()})
        assert resp.status_code == 422

    def test_gradcam_requested_forwarded(self, client):
        c, mock_pred = client
        mock_pred.predict.return_value = {
            "label": "forged",
            "probability": 0.8,
            "threshold": 0.5,
            "inference_ms": 10.0,
            "gradcam_b64": "abc123",
            "most_activated_region": {
                "x0": 0,
                "y0": 0,
                "x1": 10,
                "y1": 10,
                "cx": 5,
                "cy": 5,
                "mean_activation": 0.7,
            },
        }
        data = c.post(
            "/authenticate",
            json={"image_b64": make_b64_image(), "return_gradcam": True},
        ).json()
        assert data["gradcam_b64"] == "abc123"


# ---------------------------------------------------------------------------
# 5. POST /authenticate/batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_returns_200(self, client):
        c, _ = client
        payload = {"images": [{"image_b64": make_b64_image()} for _ in range(3)]}
        resp = c.post("/authenticate/batch", json=payload)
        assert resp.status_code == 200

    def test_results_length(self, client):
        c, _ = client
        n = 4
        payload = {"images": [{"image_b64": make_b64_image()} for _ in range(n)]}
        data = c.post("/authenticate/batch", json=payload).json()
        assert len(data["results"]) == n

    def test_total_ms_present(self, client):
        c, _ = client
        payload = {"images": [{"image_b64": make_b64_image()}]}
        data = c.post("/authenticate/batch", json=payload).json()
        assert "total_ms" in data
        assert data["total_ms"] >= 0

    def test_batch_error_returns_422(self, client):
        c, mock_pred = client
        mock_pred.predict.side_effect = RuntimeError("crash")
        payload = {"images": [{"image_b64": make_b64_image()}]}
        resp = c.post("/authenticate/batch", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. Model not loaded
# ---------------------------------------------------------------------------


def test_health_when_no_model():
    from api.main import app

    with TestClient(app) as c:
        with patch("api.main._predictor", None):
            resp = c.get("/health")
            assert resp.json()["model_loaded"] is False


def test_authenticate_when_no_model():
    from api.main import app

    with TestClient(app) as c:
        with patch("api.main._predictor", None):
            resp = c.post("/authenticate", json={"image_b64": make_b64_image()})
            assert resp.status_code == 503
