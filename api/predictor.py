"""DocumentPredictor: model loading + preprocessing + inference + optional Grad-CAM."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from PIL import Image

from src.explainability.gradcam import GradCAMExplainer
from src.explainability.visualizer import most_activated_region, overlay_heatmap
from src.models.classifier import DocumentClassifier
from src.preprocessing.pipeline import DocumentPreprocessor, PreprocessorConfig


class DocumentPredictor:
    """Wraps DocumentClassifier with preprocessing and optional Grad-CAM.

    Args:
        checkpoint: Path to the saved .pt checkpoint.
        device: 'cpu', 'cuda', or 'mps'.
        preprocessor_cfg: Optional PreprocessorConfig override.
    """

    def __init__(
        self,
        checkpoint: Path | str,
        device: str = "cpu",
        preprocessor_cfg: PreprocessorConfig | None = None,
    ) -> None:
        self.device = device
        self.checkpoint = Path(checkpoint)
        self.model = DocumentClassifier.load(self.checkpoint, device=device)
        self.model.eval()
        self.preprocessor = DocumentPreprocessor(preprocessor_cfg or PreprocessorConfig())
        self._explainer: GradCAMExplainer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_b64: str,
        threshold: float = 0.5,
        return_gradcam: bool = False,
        gradcam_method: str = "gradcam++",
    ) -> dict[str, Any]:
        """Run full inference pipeline on a base64-encoded image.

        Returns:
            dict with keys: label, probability, threshold, inference_ms,
            and optionally gradcam_b64, most_activated_region.
        """
        image_rgb = self._decode_image(image_b64)

        t0 = time.perf_counter()
        tensor = self._preprocess(image_rgb)
        prob = float(self.model.predict_proba(tensor.to(self.device)).item())
        inference_ms = (time.perf_counter() - t0) * 1e3

        result: dict[str, Any] = {
            "label": "forged" if prob >= threshold else "authentic",
            "probability": round(prob, 6),
            "threshold": threshold,
            "inference_ms": round(inference_ms, 2),
            "gradcam_b64": None,
            "most_activated_region": None,
        }

        if return_gradcam:
            explainer = self._get_explainer()
            _Method = Literal["gradcam", "gradcam++", "eigencam"]
            if gradcam_method == "ensemble":
                cam, _ = explainer.explain_ensemble(tensor)
            else:
                cam, _ = explainer.explain(tensor, method=cast(_Method, gradcam_method))
            overlay = overlay_heatmap(image_rgb, cam)
            result["gradcam_b64"] = self._encode_image(overlay)
            result["most_activated_region"] = most_activated_region(cam)

        return result

    def model_info(self) -> dict[str, Any]:
        metadata = {}
        ckpt_data = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt_data, dict):
            metadata = ckpt_data.get("metadata", {})

        return {
            "architecture": "EfficientNet-B0",
            "total_params": self.model.total_params(),
            "trainable_params": self.model.trainable_params(),
            "checkpoint": self.checkpoint.name,
            "device": self.device,
            "input_size": [3, 224, 224],
            "classes": ["authentic", "forged"],
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_image(self, image_b64: str) -> np.ndarray:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[-1]
        raw = base64.b64decode(image_b64)
        pil = Image.open(BytesIO(raw)).convert("RGB")
        return np.array(pil, dtype=np.uint8)

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        processed = self.preprocessor.process(image_rgb)
        if processed.dtype != np.float32:
            processed = processed.astype(np.float32) / 255.0
        tensor = torch.from_numpy(processed.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def _encode_image(self, image_rgb: np.ndarray) -> str:
        pil = Image.fromarray(image_rgb)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _get_explainer(self) -> GradCAMExplainer:
        if self._explainer is None:
            self._explainer = GradCAMExplainer(self.model)
        return self._explainer
