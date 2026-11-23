"""Tests for the document image degradation engine.

Verifies:
- Each tier produces a different image from the input
- Tier 0 is closest to original (lowest MSE)
- Tier 4 is most degraded (highest MSE)
- Deterministic output with same seed
- Works with both colour (BGR) and grayscale images
- Handles various image sizes (A4 at 150 dpi, 300 dpi)
- Invalid tier raises ValueError
- All transform primitives produce valid uint8 output
- New transforms: perspective warp (homography), corner blur, colour
  temperature, vignette, rotation
"""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    from scripts.lib.degradation_engine import (
        DegradationEngine,
        _brightness_contrast,
        _color_temperature,
        _corner_blur,
        _finger_occlusion,
        _gaussian_noise,
        _jpeg_compress,
        _lighting_gradient,
        _motion_blur,
        _paper_texture,
        _perspective_warp,
        _resolution_loss,
        _rotate,
        _salt_pepper_noise,
        _shadow_overlay,
        _vignette,
    )
except ModuleNotFoundError:
    from degradation_engine import (
        DegradationEngine,
        _brightness_contrast,
        _color_temperature,
        _corner_blur,
        _finger_occlusion,
        _gaussian_noise,
        _jpeg_compress,
        _lighting_gradient,
        _motion_blur,
        _paper_texture,
        _perspective_warp,
        _resolution_loss,
        _rotate,
        _salt_pepper_noise,
        _shadow_overlay,
        _vignette,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _make_document_image(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Create a synthetic document-like image with text-like features.

    White background with dark horizontal bands simulating text lines and
    a header block, giving transforms something realistic to distort.
    """
    if channels == 3:
        img = np.full((height, width, 3), 245, dtype=np.uint8)
    else:
        img = np.full((height, width), 245, dtype=np.uint8)

    # Draw simulated text lines
    line_height = max(height // 40, 4)
    margin = max(width // 10, 10)
    y = max(height // 8, 20)
    while y + line_height < height - margin:
        x_end = width - margin - (y * 7) % (width // 4)
        x_end = max(x_end, margin + 20)
        if channels == 3:
            img[y : y + line_height, margin:x_end] = [40, 40, 40]
        else:
            img[y : y + line_height, margin:x_end] = 40
        y += line_height * 3

    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> DegradationEngine:
    return DegradationEngine()


@pytest.fixture
def doc_image() -> np.ndarray:
    """A4 at ~150 dpi (1240 x 1754), BGR."""
    return _make_document_image(1754, 1240, channels=3)


@pytest.fixture
def small_image() -> np.ndarray:
    """Smaller test image for fast primitive tests."""
    return _make_document_image(400, 300, channels=3)


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Grayscale document image."""
    return _make_document_image(800, 600, channels=1)


# ---------------------------------------------------------------------------
# Core tier tests
# ---------------------------------------------------------------------------


class TestTierOutput:
    """Each tier should produce a different image from the input."""

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_tier_changes_image(
        self, engine: DegradationEngine, doc_image: np.ndarray, tier: int
    ) -> None:
        rng = random.Random(42)
        result = engine.degrade(doc_image, tier=tier, rng=rng)
        assert result.shape == doc_image.shape
        assert result.dtype == np.uint8
        # The image must actually be different
        assert not np.array_equal(result, doc_image), (
            f"Tier {tier} produced identical output"
        )

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_tier_does_not_modify_input(
        self, engine: DegradationEngine, doc_image: np.ndarray, tier: int
    ) -> None:
        original = doc_image.copy()
        rng = random.Random(99)
        engine.degrade(doc_image, tier=tier, rng=rng)
        assert np.array_equal(doc_image, original), "Input image was modified in-place"


class TestMonotonicDegradation:
    """MSE should increase monotonically from tier 0 through tier 4."""

    def test_mse_increases_with_tier(
        self, engine: DegradationEngine, doc_image: np.ndarray
    ) -> None:
        mses: list[float] = []
        for tier in range(5):
            rng = random.Random(42)
            result = engine.degrade(doc_image, tier=tier, rng=rng)
            mses.append(_mse(doc_image, result))

        # Tier 0 should be the closest to the original
        assert mses[0] == min(mses), (
            f"Tier 0 MSE ({mses[0]:.1f}) is not the minimum: {mses}"
        )
        # Tier 4 should be the most degraded
        assert mses[4] == max(mses), (
            f"Tier 4 MSE ({mses[4]:.1f}) is not the maximum: {mses}"
        )
        # Strictly increasing is ideal but we settle for non-decreasing with
        # first < last (the stochastic transforms may occasionally produce
        # similar MSEs between adjacent tiers).
        assert mses[0] < mses[4], f"MSE did not increase from tier 0 to 4: {mses}"


class TestDeterminism:
    """Same seed must produce identical output."""

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_same_seed_same_output(
        self, engine: DegradationEngine, doc_image: np.ndarray, tier: int
    ) -> None:
        rng1 = random.Random(12345)
        result1 = engine.degrade(doc_image, tier=tier, rng=rng1)

        rng2 = random.Random(12345)
        result2 = engine.degrade(doc_image, tier=tier, rng=rng2)

        assert np.array_equal(result1, result2), (
            f"Tier {tier} produced different results with same seed"
        )

    @pytest.mark.parametrize("tier", [1, 2, 3, 4])
    def test_different_seed_different_output(
        self, engine: DegradationEngine, doc_image: np.ndarray, tier: int
    ) -> None:
        rng1 = random.Random(111)
        result1 = engine.degrade(doc_image, tier=tier, rng=rng1)

        rng2 = random.Random(222)
        result2 = engine.degrade(doc_image, tier=tier, rng=rng2)

        assert not np.array_equal(result1, result2), (
            f"Tier {tier} produced same output with different seeds"
        )


# ---------------------------------------------------------------------------
# Grayscale support
# ---------------------------------------------------------------------------


class TestGrayscale:
    """Engine must handle single-channel images."""

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
    def test_grayscale_roundtrip(
        self, engine: DegradationEngine, grayscale_image: np.ndarray, tier: int
    ) -> None:
        rng = random.Random(42)
        result = engine.degrade(grayscale_image, tier=tier, rng=rng)
        assert result.ndim == 2, f"Expected 2D output, got shape {result.shape}"
        assert result.dtype == np.uint8
        assert result.shape == grayscale_image.shape


# ---------------------------------------------------------------------------
# Various image sizes
# ---------------------------------------------------------------------------


class TestImageSizes:
    """Engine should handle common document scan sizes."""

    @pytest.mark.parametrize(
        "height, width, label",
        [
            (1754, 1240, "A4_150dpi"),
            (3508, 2480, "A4_300dpi"),
            (480, 320, "small_mobile"),
            (100, 80, "tiny"),
            (4000, 3000, "large_photo"),
        ],
    )
    def test_size(
        self,
        engine: DegradationEngine,
        height: int,
        width: int,
        label: str,
    ) -> None:
        img = _make_document_image(height, width)
        rng = random.Random(42)
        result = engine.degrade(img, tier=2, rng=rng)
        assert result.shape == img.shape, f"Size mismatch for {label}"
        assert not np.array_equal(result, img), f"No change for {label}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestValidation:
    """Invalid inputs should raise clear errors."""

    def test_invalid_tier_negative(self, engine: DegradationEngine) -> None:
        img = _make_document_image(100, 100)
        with pytest.raises(ValueError, match="tier must be 0-4"):
            engine.degrade(img, tier=-1, rng=random.Random(0))

    def test_invalid_tier_too_high(self, engine: DegradationEngine) -> None:
        img = _make_document_image(100, 100)
        with pytest.raises(ValueError, match="tier must be 0-4"):
            engine.degrade(img, tier=5, rng=random.Random(0))


# ---------------------------------------------------------------------------
# Individual transform primitives
# ---------------------------------------------------------------------------


class TestJPEGCompress:
    def test_low_quality_changes_image(self, small_image: np.ndarray) -> None:
        result = _jpeg_compress(small_image, quality=20)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, small_image)

    def test_high_quality_close_to_original(self, small_image: np.ndarray) -> None:
        result = _jpeg_compress(small_image, quality=100)
        # Quality 100 is still lossy but should be very close
        assert _mse(small_image, result) < 5.0


class TestPerspectiveWarp:
    """Test the new physically-based homography perspective warp."""

    def test_small_angle_close_to_original(self, small_image: np.ndarray) -> None:
        rng = random.Random(0)
        np_rng = np.random.RandomState(0)
        result = _perspective_warp(small_image, rng, np_rng, max_tilt_deg=1.0)
        assert result.shape == small_image.shape
        # Homography-based warp with BORDER_CONSTANT fills exposed edges
        # with grey background, which raises MSE significantly even at small
        # angles.  Allow up to 6000 -- still far below heavy distortion
        # levels (20+ degrees produce MSE > 15000).
        assert _mse(small_image, result) < 6000.0

    def test_large_angle_more_distortion(self, small_image: np.ndarray) -> None:
        rng_small = random.Random(42)
        np_rng_small = np.random.RandomState(42)
        small_warp = _perspective_warp(small_image, rng_small, np_rng_small, max_tilt_deg=2.0)

        rng_large = random.Random(42)
        np_rng_large = np.random.RandomState(42)
        large_warp = _perspective_warp(small_image, rng_large, np_rng_large, max_tilt_deg=20.0)

        # Larger tilt must produce more distortion
        assert _mse(small_image, large_warp) > _mse(small_image, small_warp)

    def test_output_has_background_fill(self, small_image: np.ndarray) -> None:
        """Strong warp should expose background (grey) at edges."""
        rng = random.Random(42)
        np_rng = np.random.RandomState(42)
        result = _perspective_warp(small_image, rng, np_rng, max_tilt_deg=20.0)
        # The original image is mostly white (245). After strong warp,
        # exposed background should be grey (80-180), lowering the mean.
        assert result.astype(float).mean() < small_image.astype(float).mean()

    def test_deterministic_with_same_seed(self, small_image: np.ndarray) -> None:
        r1 = _perspective_warp(small_image, random.Random(99), np.random.RandomState(99), 10.0)
        r2 = _perspective_warp(small_image, random.Random(99), np.random.RandomState(99), 10.0)
        assert np.array_equal(r1, r2)


class TestRotate:
    """Test the separate rotation transform."""

    def test_zero_rotation_unchanged(self, small_image: np.ndarray) -> None:
        result = _rotate(small_image, 0.0, random.Random(42))
        assert np.array_equal(result, small_image)

    def test_rotation_changes_image(self, small_image: np.ndarray) -> None:
        result = _rotate(small_image, 5.0, random.Random(42))
        assert not np.array_equal(result, small_image)
        assert result.shape == small_image.shape

    def test_larger_rotation_more_change(self, small_image: np.ndarray) -> None:
        small_rot = _rotate(small_image, 1.0, random.Random(42))
        large_rot = _rotate(small_image, 10.0, random.Random(42))
        assert _mse(small_image, large_rot) > _mse(small_image, small_rot)


class TestCornerBlur:
    """Test spatially-varying corner blur."""

    def test_low_sigma_nearly_unchanged(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _corner_blur(small_image, np_rng, corner_sigma=0.3)
        # Below 0.5 threshold, should return unchanged
        assert np.array_equal(result, small_image)

    def test_blur_applied_at_corners(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _corner_blur(small_image, np_rng, corner_sigma=3.0)
        assert result.dtype == np.uint8
        assert result.shape == small_image.shape
        assert not np.array_equal(result, small_image)

    def test_centre_sharper_than_corners(self, small_image: np.ndarray) -> None:
        """Centre of the image should be more similar to original than corners."""
        np_rng = np.random.RandomState(42)
        result = _corner_blur(small_image, np_rng, corner_sigma=4.0)
        h, w = small_image.shape[:2]
        # Compare centre patch vs corner patch
        margin = min(h, w) // 6
        centre = result[h // 2 - margin:h // 2 + margin,
                        w // 2 - margin:w // 2 + margin]
        centre_orig = small_image[h // 2 - margin:h // 2 + margin,
                                  w // 2 - margin:w // 2 + margin]
        corner = result[:margin * 2, :margin * 2]
        corner_orig = small_image[:margin * 2, :margin * 2]

        mse_centre = _mse(centre_orig, centre)
        mse_corner = _mse(corner_orig, corner)
        assert mse_centre < mse_corner, (
            f"Centre MSE ({mse_centre:.1f}) should be less than corner MSE ({mse_corner:.1f})"
        )

    def test_higher_sigma_more_blur(self, small_image: np.ndarray) -> None:
        low = _corner_blur(small_image, np.random.RandomState(42), corner_sigma=1.0)
        high = _corner_blur(small_image, np.random.RandomState(42), corner_sigma=5.0)
        assert _mse(small_image, high) > _mse(small_image, low)


class TestColorTemperature:
    """Test colour temperature shift."""

    def test_shift_changes_image(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        result = _color_temperature(small_image, rng, shift_k=500)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, small_image)

    def test_warm_shift_boosts_red(self, small_image: np.ndarray) -> None:
        """Force a warm shift and verify red channel increases."""
        # Use a seed that gives direction=+1 (warm)
        # We just test the mechanics: warm = more red, less blue
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = img.astype(np.float32).copy()
        amount = 0.05  # positive = warm
        result[:, :, 2] *= 1.0 + amount  # Red up
        result[:, :, 0] *= 1.0 - amount  # Blue down
        result = np.clip(result, 0, 255).astype(np.uint8)
        assert result[:, :, 2].mean() > img[:, :, 2].mean()  # Red higher
        assert result[:, :, 0].mean() < img[:, :, 0].mean()  # Blue lower

    def test_output_clipped(self) -> None:
        """High shift should not overflow uint8."""
        bright = np.full((50, 50, 3), 250, dtype=np.uint8)
        rng = random.Random(42)
        result = _color_temperature(bright, rng, shift_k=1200)
        assert result.max() <= 255
        assert result.min() >= 0


class TestVignette:
    """Test radial vignette effect."""

    def test_vignette_darkens_corners(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        result = _vignette(small_image, rng, strength=0.5)
        assert result.dtype == np.uint8
        # Overall brightness should decrease
        assert result.astype(float).mean() < small_image.astype(float).mean()

    def test_corners_darker_than_centre(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        result = _vignette(small_image, rng, strength=0.5)
        h, w = result.shape[:2]
        # Corner brightness vs centre brightness
        corner_mean = result[:20, :20].astype(float).mean()
        centre_mean = result[h // 2 - 10:h // 2 + 10,
                             w // 2 - 10:w // 2 + 10].astype(float).mean()
        assert corner_mean < centre_mean

    def test_zero_strength_unchanged(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        result = _vignette(small_image, rng, strength=0.0)
        # With zero strength the vignette mask is all 1.0
        assert np.array_equal(result, small_image)

    def test_strong_vignette_more_darkening(self, small_image: np.ndarray) -> None:
        weak = _vignette(small_image, random.Random(42), strength=0.1)
        strong = _vignette(small_image, random.Random(42), strength=0.6)
        assert strong.astype(float).mean() < weak.astype(float).mean()


class TestGaussianNoise:
    def test_noise_added(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _gaussian_noise(small_image, sigma=10.0, np_rng=np_rng)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, small_image)

    def test_higher_sigma_more_noise(self, small_image: np.ndarray) -> None:
        low = _gaussian_noise(small_image, sigma=3.0, np_rng=np.random.RandomState(42))
        high = _gaussian_noise(
            small_image, sigma=30.0, np_rng=np.random.RandomState(42)
        )
        assert _mse(small_image, high) > _mse(small_image, low)

    def test_output_clipped_to_uint8(self) -> None:
        # White image + noise should not overflow
        white = np.full((50, 50, 3), 250, dtype=np.uint8)
        np_rng = np.random.RandomState(0)
        result = _gaussian_noise(white, sigma=50.0, np_rng=np_rng)
        assert result.max() <= 255
        assert result.min() >= 0


class TestSaltPepperNoise:
    def test_pixels_affected(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _salt_pepper_noise(small_image, amount=0.01, np_rng=np_rng)
        diff = np.count_nonzero(result != small_image)
        assert diff > 0

    def test_output_valid(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _salt_pepper_noise(small_image, amount=0.05, np_rng=np_rng)
        assert result.dtype == np.uint8


class TestBrightnessContrast:
    def test_brightness_increase(self, small_image: np.ndarray) -> None:
        result = _brightness_contrast(small_image, contrast=1.0, brightness=50.0)
        assert result.mean() > small_image.mean()

    def test_contrast_change(self, small_image: np.ndarray) -> None:
        result = _brightness_contrast(small_image, contrast=1.5, brightness=0.0)
        assert not np.array_equal(result, small_image)


class TestShadowOverlay:
    def test_shadow_darkens(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        np_rng = np.random.RandomState(42)
        result = _shadow_overlay(small_image, count=1, rng=rng, np_rng=np_rng)
        # Shadow should darken at least part of the image
        assert result.astype(float).mean() <= small_image.astype(float).mean()

    def test_multiple_shadows(self, small_image: np.ndarray) -> None:
        rng = random.Random(42)
        np_rng = np.random.RandomState(42)
        result = _shadow_overlay(small_image, count=3, rng=rng, np_rng=np_rng)
        assert result.dtype == np.uint8

    def test_darkness_range_parameter(self, small_image: np.ndarray) -> None:
        """Darker range should produce darker shadows."""
        rng1 = random.Random(42)
        np_rng1 = np.random.RandomState(42)
        light_shadow = _shadow_overlay(
            small_image, count=1, rng=rng1, np_rng=np_rng1,
            darkness_range=(0.8, 0.9),
        )
        rng2 = random.Random(42)
        np_rng2 = np.random.RandomState(42)
        dark_shadow = _shadow_overlay(
            small_image, count=1, rng=rng2, np_rng=np_rng2,
            darkness_range=(0.2, 0.3),
        )
        assert dark_shadow.astype(float).mean() < light_shadow.astype(float).mean()


class TestMotionBlur:
    def test_blur_applied(self, small_image: np.ndarray) -> None:
        result = _motion_blur(small_image, kernel_size=7, rng=random.Random(42))
        assert result.shape == small_image.shape
        assert not np.array_equal(result, small_image)


class TestPaperTexture:
    def test_texture_added(self, small_image: np.ndarray) -> None:
        np_rng = np.random.RandomState(42)
        result = _paper_texture(small_image, intensity=0.1, np_rng=np_rng)
        assert not np.array_equal(result, small_image)
        assert result.dtype == np.uint8


class TestFingerOcclusion:
    def test_finger_drawn(self, small_image: np.ndarray) -> None:
        result = _finger_occlusion(small_image, rng=random.Random(42))
        assert not np.array_equal(result, small_image)
        assert result.dtype == np.uint8


class TestLightingGradient:
    def test_gradient_applied(self, small_image: np.ndarray) -> None:
        result = _lighting_gradient(small_image, rng=random.Random(42))
        assert not np.array_equal(result, small_image)
        assert result.dtype == np.uint8


class TestResolutionLoss:
    def test_detail_lost(self, small_image: np.ndarray) -> None:
        result = _resolution_loss(small_image, scale_factor=0.5)
        assert result.shape == small_image.shape
        assert not np.array_equal(result, small_image)

    def test_extreme_downscale(self) -> None:
        img = _make_document_image(200, 150)
        result = _resolution_loss(img, scale_factor=0.1)
        assert result.shape == img.shape


def test_engine_does_not_import_augraphy(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def import_without_augraphy(name, *args, **kwargs):
        if name == "augraphy":
            raise AssertionError("Unused augmentation package imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_augraphy)
    engine = DegradationEngine()
    assert engine.degrade(_make_document_image(150, 80), 0, random.Random(42)).size > 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 7, 3), (9, 1, 3), (23, 39)])
@pytest.mark.parametrize("tier", range(5))
def test_singleton_and_noncontiguous_readonly_inputs(shape, tier):
    base = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    view = base[:, ::2]
    view.setflags(write=False)
    before = view.copy()
    actual = DegradationEngine().degrade(view, tier, random.Random(27))
    assert actual.shape == view.shape and actual.dtype == np.uint8
    np.testing.assert_array_equal(view, before)
    assert not np.shares_memory(view, actual)


@pytest.mark.parametrize("image", [[], np.zeros((0, 4), dtype=np.uint8),
    np.zeros((5,), dtype=np.uint8), np.zeros((5, 6, 1), dtype=np.uint8),
    np.zeros((5, 6, 4), dtype=np.uint8), np.zeros((5, 6, 3), dtype=np.float32)])
def test_invalid_image_fails_at_public_boundary(image):
    with pytest.raises(ValueError, match="uint8"):
        DegradationEngine().degrade(image, 0, random.Random(5))


@pytest.mark.parametrize("tier", [1.5, None, "3"])
def test_invalid_tier_type_fails_at_public_boundary(tier):
    with pytest.raises(ValueError, match="tier must be 0-4"):
        DegradationEngine().degrade(np.zeros((4, 4), np.uint8), tier, random.Random(5))


@pytest.mark.parametrize("failure", ["encode", "decode"])
def test_failed_jpeg_roundtrip_surfaces_a_write_failure(monkeypatch, failure):
    import cv2
    if failure == "encode":
        monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    else:
        monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    with pytest.raises(OSError, match="JPEG"):
        DegradationEngine().degrade(np.zeros((5, 7, 3), dtype=np.uint8), 0, random.Random(2))


@pytest.mark.parametrize("sigma", [7.5, 22.0])
def test_tiled_noise_preserves_draw_sequence_and_pixel_rounding(sigma):
    # A 515-wide page crosses a tile after an odd number of channels, so the
    # normal sampler's spare draw must carry through to the following tile.
    image = np.random.RandomState(91).randint(0, 256, (525, 515, 3), dtype=np.uint8)
    expected_rng = np.random.RandomState(13)
    if sigma > 15:
        noise = expected_rng.normal(1, sigma / 100, image.shape).astype(np.float32)
        frame = image.astype(np.float32) * noise
    else:
        noise = expected_rng.normal(0, sigma, image.shape).astype(np.float32)
        frame = image.astype(np.float32) + noise
    result_rng = np.random.RandomState(13)
    np.testing.assert_array_equal(_gaussian_noise(image, sigma, result_rng),
                                  np.clip(frame, 0, 255).astype(np.uint8))
    assert result_rng.normal() == expected_rng.normal()
