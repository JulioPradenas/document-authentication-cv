"""Heatmap visualization utilities for Grad-CAM output.

overlay_heatmap: superimposes a Grad-CAM heatmap over the original image.
"""

from __future__ import annotations

import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


def overlay_heatmap(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    use_rgb: bool = True,
) -> np.ndarray:
    """Superimpose a Grad-CAM heatmap over the original RGB image.

    Args:
        image_rgb: HxWx3 uint8 RGB array (original, not normalized).
        cam: HxW float32 in [0, 1] — Grad-CAM activation map.
        alpha: Blend factor for the heatmap overlay (0=no overlay, 1=heatmap only).
        colormap: OpenCV colormap for the heatmap (default: COLORMAP_JET).
        use_rgb: Return RGB (True) or BGR (False) output.

    Returns:
        HxWx3 uint8 array with heatmap superimposed.
    """
    # show_cam_on_image expects float32 image in [0, 1]
    img_float = image_rgb.astype(np.float32) / 255.0
    cam_resized = _resize_cam(cam, image_rgb.shape[:2])
    result = show_cam_on_image(img_float, cam_resized, use_rgb=use_rgb, colormap=colormap)
    return result.astype(np.uint8)


def cam_to_heatmap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a raw CAM (float32, [0,1]) to a coloured RGB heatmap image."""
    cam_uint8 = (cam * 255).clip(0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def most_activated_region(cam: np.ndarray, top_k_percent: float = 10.0) -> dict:
    """Return the bounding box and centroid of the most activated region.

    Args:
        cam: HxW float32 in [0, 1].
        top_k_percent: Percentage of highest-activation pixels to include.

    Returns:
        Dict with keys: x0, y0, x1, y1, cx, cy, mean_activation.
    """
    threshold = np.percentile(cam, 100 - top_k_percent)
    mask = (cam >= threshold).astype(np.uint8)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = cam.shape
        return {
            "x0": 0,
            "y0": 0,
            "x1": w,
            "y1": h,
            "cx": w // 2,
            "cy": h // 2,
            "mean_activation": float(cam.mean()),
        }

    return {
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()),
        "y1": int(ys.max()),
        "cx": int(xs.mean()),
        "cy": int(ys.mean()),
        "mean_activation": float(cam[mask == 1].mean()),
    }


def _resize_cam(cam: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize CAM to match the original image spatial dimensions."""
    h, w = target_hw
    if cam.shape == (h, w):
        return cam
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
