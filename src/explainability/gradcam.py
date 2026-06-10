"""Grad-CAM explainability for DocumentClassifier.

GradCAMExplainer wraps pytorch-grad-cam and supports three methods:
  - GradCAM     : classic gradient-weighted class activation mapping
  - GradCAM++   : better localization for small/multiple objects (e.g. stamps)
  - EigenCAM    : no target labels needed, fast for batch inference

All methods target the last convolutional block of EfficientNet-B0:
  model.backbone.features[-1]

Usage:
    explainer = GradCAMExplainer(model)
    cam = explainer.explain(image_tensor)       # (H, W) float32 in [0, 1]
    ensemble = explainer.explain_ensemble(img)  # average of all 3 methods
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.models.classifier import DocumentClassifier

Method = Literal["gradcam", "gradcam++", "eigencam"]

_CAM_CLASSES = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
}


class GradCAMExplainer:
    """Generates Grad-CAM heatmaps for DocumentClassifier predictions.

    Args:
        model: Loaded DocumentClassifier in eval mode.
    """

    def __init__(self, model: DocumentClassifier) -> None:
        self.model = model
        self.model.eval()
        # Target: last block of EfficientNet-B0 feature extractor
        self.target_layers = [model.backbone.features[-1]]

    def explain(
        self,
        image_tensor: torch.Tensor,
        method: Method = "gradcam++",
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """Compute CAM heatmap for a single image.

        Args:
            image_tensor: (1, 3, H, W) or (3, H, W) float32 tensor (ImageNet-normalized).
            method: CAM variant to use.
            threshold: Decision boundary for forgery classification.

        Returns:
            (cam, prob): cam is (H, W) float32 in [0, 1]; prob is forgery probability.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            prob = float(self.model(image_tensor).item())

        # BinaryClassifierOutputTarget forces gradients toward the forgery class
        targets = [BinaryClassifierOutputTarget(1)]

        cam_class = _CAM_CLASSES[method]
        with cam_class(model=self.model.backbone, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

        return grayscale_cam[0].astype(np.float32), prob

    def explain_ensemble(
        self,
        image_tensor: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """Average heatmap across all three CAM methods for a more stable result.

        Returns:
            (ensemble_cam, prob): same format as explain().
        """
        cams = []
        prob: float = 0.0
        for method in ("gradcam", "gradcam++", "eigencam"):
            cam, p = self.explain(image_tensor, method=method, threshold=threshold)
            cams.append(cam)
            prob = p

        ensemble = np.mean(cams, axis=0).astype(np.float32)
        return ensemble, prob

    def explain_batch(
        self,
        image_tensors: torch.Tensor,
        method: Method = "gradcam++",
    ) -> list[tuple[np.ndarray, float]]:
        """Explain a batch of images. Returns list of (cam, prob) tuples."""
        return [
            self.explain(image_tensors[i], method=method) for i in range(image_tensors.shape[0])
        ]
