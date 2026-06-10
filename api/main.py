"""Document Authentication API.

POST /authenticate        — classify a single document image
POST /authenticate/batch  — classify up to 32 images in one call
POST /report              — classify and return a PDF report
GET  /health              — liveness + model readiness check
GET  /model/info          — architecture, param counts, checkpoint metadata
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response

from api.predictor import DocumentPredictor
from api.schemas import (
    AuthenticateRequest,
    AuthenticateResponse,
    BatchAuthenticateRequest,
    BatchAuthenticateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.reporting.pdf_report import PDFReportGenerator

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------

CHECKPOINT = Path(os.getenv("MODEL_CHECKPOINT", "models/saved/efficientnet_b0_best.pt"))
DEVICE = os.getenv("MODEL_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

_predictor: DocumentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    if CHECKPOINT.exists():
        _predictor = DocumentPredictor(checkpoint=CHECKPOINT, device=DEVICE)
    else:
        _predictor = None
    yield
    _predictor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Authentication API",
    description="EfficientNet-B0 + Grad-CAM document forgery detection.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_predictor() -> DocumentPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model checkpoint not found: {CHECKPOINT}",
        )
    return _predictor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        device=DEVICE,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    predictor = _get_predictor()
    info = predictor.model_info()
    return ModelInfoResponse(**info)


@app.post(
    "/authenticate",
    response_model=AuthenticateResponse,
    status_code=status.HTTP_200_OK,
    tags=["authentication"],
)
def authenticate(req: AuthenticateRequest) -> AuthenticateResponse:
    predictor = _get_predictor()
    try:
        result = predictor.predict(
            image_b64=req.image_b64,
            threshold=req.threshold,
            return_gradcam=req.return_gradcam,
            gradcam_method=req.gradcam_method,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {exc}",
        ) from exc

    return AuthenticateResponse(**result)


@app.post(
    "/authenticate/batch",
    response_model=BatchAuthenticateResponse,
    status_code=status.HTTP_200_OK,
    tags=["authentication"],
)
def authenticate_batch(req: BatchAuthenticateRequest) -> BatchAuthenticateResponse:
    predictor = _get_predictor()
    t0 = time.perf_counter()
    results = []
    for item in req.images:
        try:
            r = predictor.predict(
                image_b64=item.image_b64,
                threshold=item.threshold,
                return_gradcam=item.return_gradcam,
                gradcam_method=item.gradcam_method,
            )
            results.append(AuthenticateResponse(**r))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Inference failed on item {len(results)}: {exc}",
            ) from exc

    return BatchAuthenticateResponse(
        results=results,
        total_ms=round((time.perf_counter() - t0) * 1e3, 2),
    )


@app.post("/report", status_code=status.HTTP_200_OK, tags=["authentication"])
def report(req: AuthenticateRequest) -> Response:
    """Classify the image and return a one-page PDF report."""
    predictor = _get_predictor()
    try:
        result = predictor.predict(
            image_b64=req.image_b64,
            threshold=req.threshold,
            return_gradcam=req.return_gradcam,
            gradcam_method=req.gradcam_method,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {exc}",
        ) from exc

    pdf_bytes = PDFReportGenerator().generate(
        result=result,
        image_b64=req.image_b64,
        model_info=predictor.model_info(),
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=informe_autenticacion.pdf"},
    )
