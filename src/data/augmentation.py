"""Synthetic forgery generator for document authentication training data.

Generates forged document images by applying 4 types of perturbations using
OpenCV and NumPy. No Albumentations dependency — that's used in the trainer.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import cv2
import numpy as np


class ForgeryType(StrEnum):
    TEXT_BLUR = "text_blur"
    COLOR_SHIFT = "color_shift"
    SPLICING = "splicing"
    HOLOGRAM_NOISE = "hologram_noise"


Intensity = Literal["mild", "medium", "strong"]

_BLUR_KSIZE: dict[Intensity, int] = {"mild": 5, "medium": 11, "strong": 21}
_HUE_SHIFT: dict[Intensity, int] = {"mild": 15, "medium": 30, "strong": 60}
_BLEND_ALPHA: dict[Intensity, float] = {"mild": 0.5, "medium": 0.75, "strong": 1.0}
_NOISE_SIGMA: dict[Intensity, int] = {"mild": 15, "medium": 30, "strong": 50}


@dataclass
class ForgeryConfig:
    forgery_type: ForgeryType
    intensity: Intensity = "medium"
    seed: int = 42


class SyntheticForgeryGenerator:
    """Generates synthetic document forgeries for training data augmentation."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(
        self,
        image: np.ndarray,
        config: ForgeryConfig,
        reference_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply forgery perturbation. Returns modified image (copy, not in-place).

        Args:
            image: HxWxC uint8 RGB array.
            config: Specifies forgery type, intensity, and optional seed override.
            reference_image: Source patch for splicing. If None, self-splicing is used.

        Returns:
            Modified copy of the input image.
        """
        # Re-seed per-config so results are reproducible when config.seed is set
        self.rng = np.random.default_rng(config.seed)

        dispatch = {
            ForgeryType.TEXT_BLUR: lambda img: self.apply_text_blur(img, config.intensity),
            ForgeryType.COLOR_SHIFT: lambda img: self.apply_color_shift(img, config.intensity),
            ForgeryType.SPLICING: lambda img: self.apply_splicing(
                img, config.intensity, reference_image
            ),
            ForgeryType.HOLOGRAM_NOISE: lambda img: self.apply_hologram_noise(
                img, config.intensity
            ),
        }
        return dispatch[config.forgery_type](image)

    # ------------------------------------------------------------------
    # Individual perturbation methods
    # ------------------------------------------------------------------

    def apply_text_blur(self, image: np.ndarray, intensity: Intensity) -> np.ndarray:
        """Localized Gaussian blur on a random rectangular region (upper 60% of image).

        Simulates altered text in a document field.
        """
        out = image.copy()
        h, w = out.shape[:2]

        region_w = int(self.rng.uniform(0.20, 0.40) * w)
        region_h = int(self.rng.uniform(0.05, 0.15) * h)

        # Constrain placement so the region stays within image bounds
        max_x = max(0, w - region_w)
        max_y = max(0, int(0.60 * h) - region_h)
        x = int(self.rng.integers(0, max_x + 1))
        y = int(self.rng.integers(0, max_y + 1))

        ksize = _BLUR_KSIZE[intensity]
        roi = out[y : y + region_h, x : x + region_w]
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        out[y : y + region_h, x : x + region_w] = blurred
        return out

    def apply_color_shift(self, image: np.ndarray, intensity: Intensity) -> np.ndarray:
        """Hue shift in a circular region (simulates tampered stamp/seal).

        Works in HSV space; shifts the H channel within a circular mask.
        """
        out = image.copy()
        h, w = out.shape[:2]

        min_dim = min(h, w)
        radius = int(self.rng.uniform(0.08, 0.15) * min_dim)

        cx = int(self.rng.integers(radius, w - radius + 1))
        cy = int(self.rng.integers(radius, h - radius + 1))

        # Build circular mask
        ys, xs = np.ogrid[:h, :w]
        mask = ((xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2).astype(np.uint8)

        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int32)

        hue_shift = _HUE_SHIFT[intensity]
        hsv[:, :, 0] = np.where(mask, (hsv[:, :, 0] + hue_shift) % 180, hsv[:, :, 0])

        hsv = hsv.astype(np.uint8)
        rgb_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        out[mask == 1] = rgb_shifted[mask == 1]
        return out

    def apply_splicing(
        self,
        image: np.ndarray,
        intensity: Intensity,
        reference: np.ndarray | None,
    ) -> np.ndarray:
        """Copy-paste a rectangular patch from reference (or self) onto image.

        If reference is None or same size as image, falls back to self-splicing
        from a non-overlapping region.
        """
        out = image.copy()
        h, w = out.shape[:2]

        patch_h = int(self.rng.uniform(0.15, 0.25) * h)
        patch_w = int(self.rng.uniform(0.15, 0.25) * w)

        # Destination region
        dst_x = int(self.rng.integers(0, w - patch_w + 1))
        dst_y = int(self.rng.integers(0, h - patch_h + 1))

        src_img = reference if reference is not None else image
        src_h, src_w = src_img.shape[:2]

        # Ensure source patch is within bounds
        src_x_max = max(0, src_w - patch_w)
        src_y_max = max(0, src_h - patch_h)

        if reference is None:
            # Self-splicing: pick a source region that doesn't overlap with destination
            # Try up to 20 candidate positions, fall back to any non-coincident region
            best_src_x, best_src_y = 0, 0
            for _ in range(20):
                cand_x = int(self.rng.integers(0, src_x_max + 1))
                cand_y = int(self.rng.integers(0, src_y_max + 1))
                # Accept if not the same top-left corner
                if cand_x != dst_x or cand_y != dst_y:
                    best_src_x, best_src_y = cand_x, cand_y
                    break
            src_x, src_y = best_src_x, best_src_y
        else:
            src_x = int(self.rng.integers(0, src_x_max + 1))
            src_y = int(self.rng.integers(0, src_y_max + 1))

        patch = src_img[src_y : src_y + patch_h, src_x : src_x + patch_w]

        # Resize patch to exact dimensions if source image was smaller
        if patch.shape[:2] != (patch_h, patch_w):
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)

        alpha = _BLEND_ALPHA[intensity]
        dst_region = out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w].astype(np.float32)
        blended = (alpha * patch.astype(np.float32) + (1 - alpha) * dst_region).astype(np.uint8)
        out[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = blended
        return out

    def apply_hologram_noise(self, image: np.ndarray, intensity: Intensity) -> np.ndarray:
        """Gaussian noise localized in a circular region (simulates tampered hologram)."""
        out = image.copy()
        h, w = out.shape[:2]

        min_dim = min(h, w)
        radius = int(self.rng.uniform(0.05, 0.12) * min_dim)

        cx = int(self.rng.integers(radius, w - radius + 1))
        cy = int(self.rng.integers(radius, h - radius + 1))

        ys, xs = np.ogrid[:h, :w]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2

        sigma = _NOISE_SIGMA[intensity]
        noise = self.rng.normal(0, sigma, (h, w, 3))

        noisy = out.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        out[mask] = noisy[mask]
        return out

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        images: list[np.ndarray],
        n_per_image: int = 1,
        intensities: list[Intensity] | None = None,
    ) -> list[tuple[np.ndarray, ForgeryConfig]]:
        """Generate n_per_image forgeries per input image, cycling through all 4 types.

        Args:
            images: List of HxWxC uint8 RGB arrays.
            n_per_image: Number of forged versions to produce per image.
            intensities: Optional list of intensities to cycle through. Defaults to
                         ["mild", "medium", "strong"] cycling.

        Returns:
            List of (forged_image, config) tuples, length == len(images) * n_per_image.
        """
        if intensities is None:
            intensities = ["mild", "medium", "strong"]

        all_types = list(ForgeryType)
        results: list[tuple[np.ndarray, ForgeryConfig]] = []

        for img_idx, img in enumerate(images):
            for j in range(n_per_image):
                forgery_type = all_types[(img_idx * n_per_image + j) % len(all_types)]
                intensity: Intensity = intensities[(img_idx * n_per_image + j) % len(intensities)]
                # Unique but deterministic seed per (image, perturbation) pair
                seed = self.seed + img_idx * 1000 + j
                config = ForgeryConfig(
                    forgery_type=forgery_type,
                    intensity=intensity,
                    seed=seed,
                )
                forged = self.apply(img, config)
                results.append((forged, config))

        return results
