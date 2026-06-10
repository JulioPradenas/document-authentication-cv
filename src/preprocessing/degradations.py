"""Controlled image degradations for robustness testing.

These transforms simulate real-world capture problems so we can measure how
much the model's accuracy degrades under each. Unlike augmentation.py (which
synthesizes *forgeries*), these preserve the authentic/forged label and only
reduce image *quality* — they answer "does the model still work on a bad photo?".

Each degradation takes a uint8 RGB image and a severity in [0, 1] and returns
a degraded uint8 RGB image of the same shape.
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np


class DegradationType(StrEnum):
    GAUSSIAN_BLUR = "gaussian_blur"
    JPEG_COMPRESSION = "jpeg_compression"
    GAUSSIAN_NOISE = "gaussian_noise"
    DOWNSCALE = "downscale"
    BRIGHTNESS = "brightness"


def gaussian_blur(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Out-of-focus / motion blur. severity scales the kernel size (3–21 px)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    k = int(3 + severity * 18)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)


def jpeg_compression(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Lossy compression artifacts. severity maps to JPEG quality 95→5."""
    severity = float(np.clip(severity, 0.0, 1.0))
    quality = int(95 - severity * 90)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    decoded: np.ndarray = cv2.imdecode(enc, cv2.IMREAD_COLOR)  # type: ignore[assignment]
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def gaussian_noise(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Sensor noise. severity scales the noise std (0–50)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    std = severity * 50.0
    noise = np.random.normal(0, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def downscale(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Low-resolution capture: downscale then upscale back to original size.

    severity scales the downscale factor (1.0× → 0.1×).
    """
    severity = float(np.clip(severity, 0.0, 1.0))
    h, w = image.shape[:2]
    factor = 1.0 - severity * 0.9
    small_w = max(1, int(w * factor))
    small_h = max(1, int(h * factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness(image: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """Under-exposure. severity scales darkening factor (1.0× → 0.2×)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    factor = 1.0 - severity * 0.8
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


_DEGRADATIONS = {
    DegradationType.GAUSSIAN_BLUR: gaussian_blur,
    DegradationType.JPEG_COMPRESSION: jpeg_compression,
    DegradationType.GAUSSIAN_NOISE: gaussian_noise,
    DegradationType.DOWNSCALE: downscale,
    DegradationType.BRIGHTNESS: brightness,
}


def apply_degradation(
    image: np.ndarray,
    degradation: DegradationType | str,
    severity: float = 0.5,
) -> np.ndarray:
    """Apply a named degradation at the given severity.

    Args:
        image: HxWx3 uint8 RGB array.
        degradation: DegradationType or its string value.
        severity: Strength in [0, 1].

    Returns:
        Degraded uint8 RGB array of the same shape.
    """
    key = DegradationType(degradation)
    return _DEGRADATIONS[key](image, severity)
