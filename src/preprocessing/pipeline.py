"""OpenCV preprocessing pipeline for document images.

Pipeline steps (each individually togglable):
  1. Perspective correction  — homography if document corners are detectable
  2. Denoising               — cv2.fastNlMeansDenoisingColored
  3. CLAHE                   — contrast normalization in LAB L-channel
  4. Resize                  — bicubic to target_size (default 224×224)
  5. Normalize               — ImageNet mean/std → float32 tensor-ready array

Output: HxWx3 float32 in [0,1] after normalize, or uint8 before normalize.
"""

from dataclasses import dataclass

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessorConfig:
    target_size: tuple[int, int] = (224, 224)  # (width, height)
    perspective: bool = True
    denoise: bool = True
    clahe: bool = True
    resize: bool = True
    normalize: bool = True
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    # Denoising parameters (fastNlMeansDenoisingColored)
    denoise_h: int = 10
    denoise_template_size: int = 7
    denoise_search_size: int = 21


class DocumentPreprocessor:
    """Applies a configurable OpenCV preprocessing pipeline to document images.

    All methods accept and return HxWx3 uint8 RGB arrays, except `normalize`
    which returns HxWx3 float32.
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self.config = config or PreprocessorConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run the full pipeline on a single image.

        Args:
            image: HxWx3 uint8 RGB array.

        Returns:
            Processed array. float32 HxWx3 if normalize=True, else uint8 HxWx3.
        """
        img = image.copy()

        if self.config.perspective:
            img = self.correct_perspective(img)
        if self.config.denoise:
            img = self.denoise(img)
        if self.config.clahe:
            img = self.apply_clahe(img)
        if self.config.resize:
            img = self.resize(img)
        if self.config.normalize:
            return self.normalize(img)
        return img

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Attempt to detect document borders and apply perspective correction.

        Strategy: Canny + contour detection to find the largest quadrilateral.
        Falls back to returning the original image if no reliable quad is found.
        The minimum area threshold (20% of total image area) avoids correcting
        on noise or partial detections that would distort small documents.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Mild blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Sort by area descending, keep top 5 candidates
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        quad = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > 0.20 * h * w:
                    quad = approx
                    break

        if quad is None:
            return image

        # Order corners: top-left, top-right, bottom-right, bottom-left
        pts = quad.reshape(4, 2).astype(np.float32)
        pts_ordered = _order_points(pts)

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_ordered, dst)
        warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return warped

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Reduce sensor/compression noise while preserving edges.

        Uses fastNlMeansDenoisingColored which works in LAB space and is
        well-suited for photographic document images.
        """
        cfg = self.config
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=cfg.denoise_h,
            hColor=cfg.denoise_h,
            templateWindowSize=cfg.denoise_template_size,
            searchWindowSize=cfg.denoise_search_size,
        )
        return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Contrast-Limited Adaptive Histogram Equalization on the L channel.

        Works in LAB color space so hue/saturation are not affected — only
        local luminance contrast is enhanced. Critical for documents captured
        under uneven lighting (flash, shadows, office fluorescent).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target_size using bicubic interpolation."""
        w, h = self.config.target_size
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize to ImageNet mean/std. Input: uint8, Output: float32 HxWx3."""
        img = image.astype(np.float32) / 255.0
        return (img - _IMAGENET_MEAN) / _IMAGENET_STD

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def denormalize(image: np.ndarray) -> np.ndarray:
        """Reverse ImageNet normalization to uint8 for visualization."""
        img = image * _IMAGENET_STD + _IMAGENET_MEAN
        return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect
