"""Tests for src/preprocessing/pipeline.py.

All images are synthetic (numpy arrays). No dataset files required.
Run: pytest tests/test_preprocessing.py -v
"""

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")

from src.preprocessing.pipeline import (
    DocumentPreprocessor,
    PreprocessorConfig,
    _order_points,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_image(h: int = 224, w: int = 224, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 240, (h, w, 3), dtype=np.uint8)


def make_document_image(h: int = 300, w: int = 400) -> np.ndarray:
    """Synthetic document: light background + dark border rectangle."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    # Border that looks like a document edge — helps perspective detection
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (30, 30, 30), 3)
    return img


@pytest.fixture
def default_preprocessor() -> DocumentPreprocessor:
    return DocumentPreprocessor()


@pytest.fixture
def square_img() -> np.ndarray:
    return make_image(224, 224)


@pytest.fixture
def large_img() -> np.ndarray:
    return make_image(480, 640, seed=7)


# ---------------------------------------------------------------------------
# 1. Output shape and dtype
# ---------------------------------------------------------------------------

def test_process_output_shape_normalized(default_preprocessor, large_img):
    result = default_preprocessor.process(large_img)
    assert result.shape == (224, 224, 3)


def test_process_output_dtype_normalized(default_preprocessor, large_img):
    result = default_preprocessor.process(large_img)
    assert result.dtype == np.float32


def test_process_no_normalize_returns_uint8(large_img):
    cfg = PreprocessorConfig(normalize=False)
    proc = DocumentPreprocessor(cfg)
    result = proc.process(large_img)
    assert result.dtype == np.uint8
    assert result.shape == (224, 224, 3)


def test_process_already_target_size(default_preprocessor, square_img):
    result = default_preprocessor.process(square_img)
    assert result.shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# 2. Individual steps
# ---------------------------------------------------------------------------

class TestResize:
    def test_resize_changes_dimensions(self):
        proc = DocumentPreprocessor(PreprocessorConfig(target_size=(128, 128)))
        img = make_image(300, 400)
        result = proc.resize(img)
        assert result.shape == (128, 128, 3)

    def test_resize_noop_when_already_target(self):
        proc = DocumentPreprocessor(PreprocessorConfig(target_size=(224, 224)))
        img = make_image(224, 224)
        result = proc.resize(img)
        assert result.shape == (224, 224, 3)

    def test_resize_custom_target(self):
        proc = DocumentPreprocessor(PreprocessorConfig(target_size=(64, 64)))
        img = make_image(100, 200)
        result = proc.resize(img)
        assert result.shape == (64, 64, 3)


class TestNormalize:
    def test_normalize_dtype(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.normalize(img)
        assert result.dtype == np.float32

    def test_normalize_range(self):
        # ImageNet-normalized values can go below 0 and above 1
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.normalize(img)
        # Range should be within roughly [-3, 3] for ImageNet stats
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_normalize_denormalize_roundtrip(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        normalized = proc.normalize(img)
        recovered = DocumentPreprocessor.denormalize(normalized)
        assert recovered.dtype == np.uint8
        assert recovered.shape == img.shape
        # Max error should be small (rounding only)
        assert np.max(np.abs(img.astype(np.int16) - recovered.astype(np.int16))) <= 2

    def test_all_zero_image_normalizes_to_imagenet_mean(self):
        proc = DocumentPreprocessor()
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = proc.normalize(img)
        # 0/255 = 0.0 → (0.0 - mean) / std = -mean/std
        expected = -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-5)


class TestCLAHE:
    def test_clahe_preserves_shape(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.apply_clahe(img)
        assert result.shape == img.shape

    def test_clahe_preserves_dtype(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.apply_clahe(img)
        assert result.dtype == np.uint8

    def test_clahe_changes_image(self):
        # Use a low-contrast image — CLAHE should change it
        proc = DocumentPreprocessor(PreprocessorConfig(clahe_clip_limit=4.0))
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        # Add a small gradient so it's not completely flat
        img[:, :, 0] += np.arange(224, dtype=np.uint8)[:, np.newaxis] // 4
        result = proc.apply_clahe(img)
        assert not np.array_equal(result, img)


class TestDenoise:
    def test_denoise_preserves_shape(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.denoise(img)
        assert result.shape == img.shape

    def test_denoise_preserves_dtype(self):
        proc = DocumentPreprocessor()
        img = make_image(224, 224)
        result = proc.denoise(img)
        assert result.dtype == np.uint8

    def test_denoise_reduces_noise(self):
        rng = np.random.default_rng(42)
        base = np.full((64, 64, 3), 150, dtype=np.uint8)
        noise = rng.integers(-40, 41, (64, 64, 3), dtype=np.int16)
        noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        proc = DocumentPreprocessor()
        denoised = proc.denoise(noisy)
        # Denoised should be closer to the base than the noisy input
        err_noisy = np.mean(np.abs(noisy.astype(np.int16) - base.astype(np.int16)))
        err_denoised = np.mean(np.abs(denoised.astype(np.int16) - base.astype(np.int16)))
        assert err_denoised < err_noisy


class TestPerspective:
    def test_perspective_preserves_shape(self):
        proc = DocumentPreprocessor()
        img = make_document_image()
        result = proc.correct_perspective(img)
        assert result.shape == img.shape

    def test_perspective_preserves_dtype(self):
        proc = DocumentPreprocessor()
        img = make_document_image()
        result = proc.correct_perspective(img)
        assert result.dtype == np.uint8

    def test_perspective_fallback_on_random_image(self):
        # Random noise has no detectable quad — should return input unchanged
        proc = DocumentPreprocessor()
        img = make_image(224, 224, seed=99)
        original = img.copy()
        result = proc.correct_perspective(img)
        assert result.shape == original.shape


# ---------------------------------------------------------------------------
# 3. Pipeline configuration flags
# ---------------------------------------------------------------------------

class TestPipelineFlags:
    def test_all_steps_disabled_except_normalize(self):
        cfg = PreprocessorConfig(
            perspective=False,
            denoise=False,
            clahe=False,
            resize=False,
            normalize=True,
        )
        proc = DocumentPreprocessor(cfg)
        img = make_image(224, 224)
        result = proc.process(img)
        assert result.dtype == np.float32
        assert result.shape == (224, 224, 3)

    def test_all_steps_disabled(self):
        cfg = PreprocessorConfig(
            perspective=False,
            denoise=False,
            clahe=False,
            resize=False,
            normalize=False,
        )
        proc = DocumentPreprocessor(cfg)
        img = make_image(300, 400)
        result = proc.process(img)
        assert result.dtype == np.uint8
        assert result.shape == (300, 400, 3)

    def test_only_resize_enabled(self):
        cfg = PreprocessorConfig(
            perspective=False,
            denoise=False,
            clahe=False,
            resize=True,
            normalize=False,
            target_size=(64, 64),
        )
        proc = DocumentPreprocessor(cfg)
        img = make_image(300, 400)
        result = proc.process(img)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. Input is not mutated
# ---------------------------------------------------------------------------

def test_process_does_not_mutate_input(default_preprocessor):
    img = make_image(300, 300)
    original = img.copy()
    default_preprocessor.process(img)
    assert np.array_equal(img, original)


# ---------------------------------------------------------------------------
# 5. Performance — under 100ms per image on CPU (generous margin for CI)
# ---------------------------------------------------------------------------

def test_processing_time_under_threshold():
    cfg = PreprocessorConfig(perspective=False)   # skip perspective for speed test
    proc = DocumentPreprocessor(cfg)
    img = make_image(480, 640)

    # Warm-up
    proc.process(img)

    N = 5
    start = time.perf_counter()
    for _ in range(N):
        proc.process(img)
    elapsed_ms = (time.perf_counter() - start) / N * 1000

    assert elapsed_ms < 500, (
        f"Pipeline too slow: {elapsed_ms:.1f}ms (target <100ms, CI limit 500ms)"
    )


# ---------------------------------------------------------------------------
# 6. Helper function
# ---------------------------------------------------------------------------

def test_order_points_top_left_has_smallest_sum():
    pts = np.array([[100, 100], [0, 100], [0, 0], [100, 0]], dtype=np.float32)
    ordered = _order_points(pts)
    assert tuple(ordered[0]) == (0.0, 0.0)   # top-left


def test_order_points_bottom_right_has_largest_sum():
    pts = np.array([[100, 100], [0, 100], [0, 0], [100, 0]], dtype=np.float32)
    ordered = _order_points(pts)
    assert tuple(ordered[2]) == (100.0, 100.0)   # bottom-right
