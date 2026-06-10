"""Image quality assessment for input gating.

Before running inference, low-quality captures (blurry, over/under-exposed,
too small) should be rejected so the model is not asked to classify images
outside its competence — a common source of silent false positives.

ImageQualityAssessor computes four no-reference metrics and produces an
overall pass/fail verdict with per-check reasons:

  - sharpness  : variance of the Laplacian (focus measure)
  - exposure   : mean luminance + clipping fraction (over/under-exposure)
  - resolution : min(height, width) in pixels
  - contrast   : standard deviation of luminance

Thresholds are tuned for document photos and configurable via QualityConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class QualityConfig:
    min_sharpness: float = 100.0  # Laplacian variance; <100 ≈ visibly blurry
    min_resolution: int = 224  # shortest side in pixels (model input size)
    min_brightness: float = 40.0  # mean luminance (0–255)
    max_brightness: float = 220.0  # mean luminance (0–255)
    max_clipping: float = 0.35  # max fraction of pixels at 0 or 255
    min_contrast: float = 20.0  # std of luminance


@dataclass
class QualityReport:
    passed: bool
    sharpness: float
    brightness: float
    contrast: float
    clipping: float
    resolution: tuple[int, int]  # (height, width)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "sharpness": round(self.sharpness, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "clipping": round(self.clipping, 4),
            "resolution": list(self.resolution),
            "reasons": self.reasons,
        }


class ImageQualityAssessor:
    """Computes no-reference quality metrics and a pass/fail gate.

    Args:
        config: Threshold configuration. Defaults tuned for document photos.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def assess(self, image: np.ndarray) -> QualityReport:
        """Evaluate a single RGB image.

        Args:
            image: HxWx3 uint8 RGB array.

        Returns:
            QualityReport with metrics and a pass/fail verdict.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        sharpness = self._sharpness(gray)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        clipping = self._clipping(gray)

        cfg = self.config
        reasons: list[str] = []

        if sharpness < cfg.min_sharpness:
            reasons.append(f"imagen borrosa (nitidez {sharpness:.0f} < {cfg.min_sharpness:.0f})")
        if min(h, w) < cfg.min_resolution:
            reasons.append(f"resolución insuficiente ({min(h, w)}px < {cfg.min_resolution}px)")
        if brightness < cfg.min_brightness:
            reasons.append(f"imagen demasiado oscura (brillo {brightness:.0f})")
        elif brightness > cfg.max_brightness:
            reasons.append(f"imagen sobreexpuesta (brillo {brightness:.0f})")
        if clipping > cfg.max_clipping:
            reasons.append(f"recorte de tonos excesivo ({clipping:.0%})")
        if contrast < cfg.min_contrast:
            reasons.append(f"contraste insuficiente (std {contrast:.0f})")

        return QualityReport(
            passed=len(reasons) == 0,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            clipping=clipping,
            resolution=(h, w),
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        """Variance of the Laplacian — higher means sharper focus."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _clipping(gray: np.ndarray) -> float:
        """Fraction of pixels at the extremes (0 or 255)."""
        total = gray.size
        clipped = int(np.count_nonzero(gray <= 2) + np.count_nonzero(gray >= 253))
        return clipped / total
