"""Pydantic schemas for the Document Authentication API."""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AuthenticateRequest(BaseModel):
    image_b64: str = Field(
        ...,
        description="Base64-encoded image (JPEG/PNG). No data-URI prefix required.",
    )
    return_gradcam: bool = Field(
        default=False,
        description="Include Grad-CAM heatmap (base64 PNG) in the response.",
    )
    gradcam_method: str = Field(
        default="gradcam++",
        description="CAM method: 'gradcam', 'gradcam++', 'eigencam', or 'ensemble'.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for 'forged' label.",
    )

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        if v.startswith("data:"):
            v = v.split(",", 1)[-1]
        try:
            base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("image_b64 is not valid base64") from exc
        return v

    @field_validator("gradcam_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"gradcam", "gradcam++", "eigencam", "ensemble"}
        if v not in allowed:
            raise ValueError(f"gradcam_method must be one of {sorted(allowed)}")
        return v


class AuthenticateResponse(BaseModel):
    label: str = Field(..., description="'authentic' or 'forged'")
    probability: float = Field(..., description="P(forged) ∈ [0, 1]")
    threshold: float
    gradcam_b64: str | None = Field(
        default=None,
        description="Base64 PNG of the Grad-CAM overlay (only when return_gradcam=True).",
    )
    most_activated_region: dict[str, Any] | None = Field(
        default=None,
        description="Bounding box of the highest-activation region.",
    )
    inference_ms: float = Field(..., description="Inference latency in milliseconds.")


class BatchAuthenticateRequest(BaseModel):
    images: list[AuthenticateRequest] = Field(
        ..., min_length=1, max_length=32, description="Up to 32 images per call."
    )


class BatchAuthenticateResponse(BaseModel):
    results: list[AuthenticateResponse]
    total_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    architecture: str
    total_params: int
    trainable_params: int
    checkpoint: str
    device: str
    input_size: list[int]
    classes: list[str]
    metadata: dict[str, Any]
