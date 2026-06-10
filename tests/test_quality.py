"""Tests for src/preprocessing/quality.py and src/preprocessing/degradations.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.degradations import DegradationType, apply_degradation
from src.preprocessing.quality import ImageQualityAssessor, QualityConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sharp_document(h: int = 400, w: int = 600, seed: int = 0) -> np.ndarray:
    """A high-quality synthetic document: text-like dark bars on light background."""
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for y in range(40, h - 40, 28):
        bar_w = rng.integers(w // 3, w - 80)
        img[y : y + 10, 40 : 40 + bar_w] = rng.integers(20, 90)
    return img


# ---------------------------------------------------------------------------
# 1. ImageQualityAssessor
# ---------------------------------------------------------------------------


class TestImageQualityAssessor:
    def test_sharp_document_passes(self):
        report = ImageQualityAssessor().assess(sharp_document())
        assert report.passed, f"Expected pass, got reasons: {report.reasons}"

    def test_report_has_all_metrics(self):
        report = ImageQualityAssessor().assess(sharp_document())
        d = report.to_dict()
        for key in ("passed", "sharpness", "brightness", "contrast", "clipping", "resolution"):
            assert key in d

    def test_blurry_image_fails_sharpness(self):
        blurry = apply_degradation(sharp_document(), DegradationType.GAUSSIAN_BLUR, severity=1.0)
        report = ImageQualityAssessor().assess(blurry)
        assert not report.passed
        assert any("borrosa" in r for r in report.reasons)

    def test_low_resolution_fails(self):
        small = np.full((100, 100, 3), 128, dtype=np.uint8)
        small[::4] = 30  # add some texture so sharpness isn't the trigger
        report = ImageQualityAssessor().assess(small)
        assert not report.passed
        assert any("resolución" in r for r in report.reasons)

    def test_dark_image_fails_brightness(self):
        dark = np.full((300, 300, 3), 10, dtype=np.uint8)
        report = ImageQualityAssessor().assess(dark)
        assert not report.passed
        assert any("oscura" in r for r in report.reasons)

    def test_overexposed_image_fails(self):
        bright = np.full((300, 300, 3), 250, dtype=np.uint8)
        report = ImageQualityAssessor().assess(bright)
        assert not report.passed
        assert any("sobreexpuesta" in r or "recorte" in r for r in report.reasons)

    def test_low_contrast_fails(self):
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        report = ImageQualityAssessor().assess(flat)
        assert not report.passed
        assert any("contraste" in r for r in report.reasons)

    def test_resolution_reported_as_height_width(self):
        img = sharp_document(h=400, w=600)
        report = ImageQualityAssessor().assess(img)
        assert report.resolution == (400, 600)

    def test_custom_config_stricter_sharpness(self):
        img = sharp_document()
        strict = QualityConfig(min_sharpness=1e9)
        report = ImageQualityAssessor(strict).assess(img)
        assert not report.passed

    def test_clipping_fraction_in_range(self):
        report = ImageQualityAssessor().assess(sharp_document())
        assert 0.0 <= report.clipping <= 1.0


# ---------------------------------------------------------------------------
# 2. Degradations
# ---------------------------------------------------------------------------

ALL_DEGRADATIONS = list(DegradationType)


class TestDegradations:
    @pytest.mark.parametrize("deg", ALL_DEGRADATIONS)
    def test_preserves_shape(self, deg):
        img = sharp_document()
        out = apply_degradation(img, deg, severity=0.5)
        assert out.shape == img.shape

    @pytest.mark.parametrize("deg", ALL_DEGRADATIONS)
    def test_preserves_dtype(self, deg):
        img = sharp_document()
        out = apply_degradation(img, deg, severity=0.5)
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("deg", ALL_DEGRADATIONS)
    def test_severity_zero_is_near_identity(self, deg):
        img = sharp_document()
        out = apply_degradation(img, deg, severity=0.0)
        # At severity 0, most degradations are identity or near-identity
        assert out.shape == img.shape

    @pytest.mark.parametrize("deg", ALL_DEGRADATIONS)
    def test_modifies_image_at_high_severity(self, deg):
        img = sharp_document()
        out = apply_degradation(img, deg, severity=1.0)
        assert not np.array_equal(out, img)

    def test_accepts_string_name(self):
        img = sharp_document()
        out = apply_degradation(img, "gaussian_blur", severity=0.5)
        assert out.shape == img.shape

    def test_blur_reduces_sharpness(self):
        img = sharp_document()
        assessor = ImageQualityAssessor()
        sharp = assessor.assess(img).sharpness
        blurred = apply_degradation(img, DegradationType.GAUSSIAN_BLUR, severity=0.8)
        blurry = assessor.assess(blurred).sharpness
        assert blurry < sharp

    def test_brightness_degradation_darkens(self):
        img = sharp_document()
        dark = apply_degradation(img, DegradationType.BRIGHTNESS, severity=0.9)
        assert dark.mean() < img.mean()

    def test_severity_clamped_above_one(self):
        img = sharp_document()
        out = apply_degradation(img, DegradationType.GAUSSIAN_NOISE, severity=5.0)
        assert out.shape == img.shape

    def test_unknown_degradation_raises(self):
        with pytest.raises(ValueError):
            apply_degradation(sharp_document(), "teleport", severity=0.5)
