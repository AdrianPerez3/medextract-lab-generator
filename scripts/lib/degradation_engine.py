"""Document image degradation engine for synthetic lab report generation.

Applies realistic phone-camera degradation to clean rendered document images
across 5 quality tiers (0=clean scan through 4=bad photo).  Each tier composes
a deterministic sequence of OpenCV/numpy transforms that simulate real-world
capture artifacts.

The key design insight is that **perspective warp** is the single most
important transform for distinguishing photos from scans.  Every tier >= 1
applies a physically-based homography derived from camera intrinsics and
3D rotation angles, producing the characteristic keystone distortion seen in
phone photos of documents.

Additional transforms -- spatially-varying corner blur, colour temperature
shift, vignetting, realistic shadows, motion blur, noise, and JPEG
compression -- are layered on top in increasing severity to produce a
realistic distribution of phone-camera document images.

Tier overview
-------------
- **Tier 0** -- Clean scan (5%): JPEG quality 90-95 only.
- **Tier 1** -- Careful phone photo (20%): subtle perspective (3-6 deg),
  slight rotation, corner blur, mild brightness shift, light vignette,
  JPEG 80-90.
- **Tier 2** -- Casual phone photo (35%): moderate perspective (5-12 deg),
  rotation, corner blur, colour temperature shift, one shadow, noise,
  vignette, JPEG 65-80.
- **Tier 3** -- Quick snap (25%): strong perspective (8-18 deg), rotation,
  heavy corner blur, motion blur, colour temp shift, multiple shadows,
  noise, vignette, JPEG 55-70.
- **Tier 4** -- Bad photo (15%): extreme perspective (15-25 deg), rotation,
  very heavy corner blur, motion blur, heavy colour shift, heavy shadows
  + finger occlusion, noise, resolution loss, strong vignette, JPEG 40-60.

Usage
-----
    from scripts.lib.degradation_engine import DegradationEngine

    engine = DegradationEngine()
    rng = random.Random(42)
    degraded = engine.degrade(image, tier=2, rng=rng)

All transforms are deterministic given the same ``random.Random`` instance,
so output is reproducible across runs.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DegradationEngine:
    """Applies document-specific image degradation in 5 tiers.

    All methods are stateless -- the ``random.Random`` instance passed to
    :meth:`degrade` controls all stochastic decisions for reproducibility.
    """

    def degrade(
        self, image: np.ndarray, tier: int, rng: random.Random
    ) -> np.ndarray:
        """Apply degradation to a document image.

        Parameters
        ----------
        image:
            Numpy array ``(H, W, 3)`` uint8 BGR **or** ``(H, W)`` uint8
            grayscale.  The returned image always has the same number of
            channels as the input.
        tier:
            Integer 0-4 selecting the degradation severity.
        rng:
            ``random.Random`` instance used for all stochastic decisions.
            A deterministic numpy seed is derived from it so that numpy
            random calls are also reproducible.

        Returns
        -------
        np.ndarray
            Degraded image with the same dtype and channel count as
            *image*.
        """
        if not isinstance(tier, (int, np.integer)) or not 0 <= tier <= 4:
            raise ValueError(f"tier must be 0-4, got {tier}")

        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)
                or not image.size):
            raise ValueError("image must be a nonempty uint8 grayscale (H,W) or BGR (H,W,3) array")
        # The first transform creates a new image (JPEG decode or perspective warp).
        grayscale = image.ndim == 2
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if grayscale else image

        # Seed numpy from the stdlib rng for reproducibility.
        np_seed = rng.randint(0, 2**31 - 1)
        np_rng = np.random.RandomState(np_seed)

        dispatch = {
            0: self._degrade_tier0,
            1: self._degrade_tier1,
            2: self._degrade_tier2,
            3: self._degrade_tier3,
            4: self._degrade_tier4,
        }
        result = dispatch[tier](img, rng, np_rng)

        if grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return result

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _degrade_tier0(
        self,
        image: np.ndarray,
        rng: random.Random,
        np_rng: np.random.RandomState,
    ) -> np.ndarray:
        """Clean scan: slight JPEG compression only."""
        quality = rng.randint(90, 95)
        return _jpeg_compress(image, quality)

    def _degrade_tier1(
        self,
        image: np.ndarray,
        rng: random.Random,
        np_rng: np.random.RandomState,
    ) -> np.ndarray:
        """Careful phone photo -- subtle perspective, light effects."""
        img = image

        # Perspective warp: 3-6 degrees tilt (subtle keystone)
        max_tilt = rng.uniform(3.0, 6.0)
        img = _perspective_warp(img, rng, np_rng, max_tilt_deg=max_tilt)

        # Slight rotation: +/-2 degrees
        angle = rng.uniform(-2.0, 2.0)
        img = _rotate(img, angle, rng)

        # Corner blur: spatially varying, sigma 0-1.0 at corners
        corner_sigma = rng.uniform(0.0, 1.0)
        img = _corner_blur(img, np_rng, corner_sigma=corner_sigma)

        # Slight brightness variation +/-5%
        brightness_delta = rng.uniform(-12.75, 12.75)  # 5% of 255
        img = _brightness_contrast(img, contrast=1.0, brightness=brightness_delta)

        # Vignette: strength 0.1-0.2
        vignette_strength = rng.uniform(0.1, 0.2)
        img = _vignette(img, rng, strength=vignette_strength)

        # JPEG quality 80-90
        quality = rng.randint(80, 90)
        img = _jpeg_compress(img, quality)

        return img

    def _degrade_tier2(
        self,
        image: np.ndarray,
        rng: random.Random,
        np_rng: np.random.RandomState,
    ) -> np.ndarray:
        """Casual phone photo -- moderate perspective, shadows, noise."""
        img = image

        # Perspective warp: 5-12 degrees tilt
        max_tilt = rng.uniform(5.0, 12.0)
        img = _perspective_warp(img, rng, np_rng, max_tilt_deg=max_tilt)

        # Rotation: +/-4 degrees
        angle = rng.uniform(-4.0, 4.0)
        img = _rotate(img, angle, rng)

        # Corner blur: sigma 1.0-2.5 at corners
        corner_sigma = rng.uniform(1.0, 2.5)
        img = _corner_blur(img, np_rng, corner_sigma=corner_sigma)

        # Color temperature shift (+/-500K)
        img = _color_temperature(img, rng, shift_k=500)

        # One shadow (gradient from one side, intensity 0.2-0.4)
        img = _shadow_overlay(img, count=1, rng=rng, np_rng=np_rng,
                              darkness_range=(0.6, 0.8))

        # Gaussian noise sigma 5-12
        sigma = rng.uniform(5.0, 12.0)
        img = _gaussian_noise(img, sigma, np_rng)

        # Vignette: strength 0.2-0.3
        vignette_strength = rng.uniform(0.2, 0.3)
        img = _vignette(img, rng, strength=vignette_strength)

        # JPEG quality 65-80
        quality = rng.randint(65, 80)
        img = _jpeg_compress(img, quality)

        return img

    def _degrade_tier3(
        self,
        image: np.ndarray,
        rng: random.Random,
        np_rng: np.random.RandomState,
    ) -> np.ndarray:
        """Quick snap -- strong perspective, motion blur, multiple shadows."""
        img = image

        # Perspective warp: 8-18 degrees tilt
        max_tilt = rng.uniform(8.0, 18.0)
        img = _perspective_warp(img, rng, np_rng, max_tilt_deg=max_tilt)

        # Rotation: +/-6 degrees
        angle = rng.uniform(-6.0, 6.0)
        img = _rotate(img, angle, rng)

        # Corner blur: sigma 2.0-4.0 at corners
        corner_sigma = rng.uniform(2.0, 4.0)
        img = _corner_blur(img, np_rng, corner_sigma=corner_sigma)

        # Motion blur: kernel 3-5 pixels
        kernel_size = rng.choice([3, 5])
        img = _motion_blur(img, kernel_size, rng)

        # Color temperature shift +/-800K
        img = _color_temperature(img, rng, shift_k=800)

        # Multiple shadows (gradient + hand shadow)
        img = _shadow_overlay(img, count=rng.randint(2, 3), rng=rng,
                              np_rng=np_rng, darkness_range=(0.4, 0.7))

        # Noise sigma 10-20
        sigma = rng.uniform(10.0, 20.0)
        img = _gaussian_noise(img, sigma, np_rng)

        # Vignette: strength 0.3-0.5
        vignette_strength = rng.uniform(0.3, 0.5)
        img = _vignette(img, rng, strength=vignette_strength)

        # JPEG quality 55-70
        quality = rng.randint(55, 70)
        img = _jpeg_compress(img, quality)

        return img

    def _degrade_tier4(
        self,
        image: np.ndarray,
        rng: random.Random,
        np_rng: np.random.RandomState,
    ) -> np.ndarray:
        """Bad photo -- extreme perspective, heavy degradation."""
        img = image

        # Perspective warp: 15-25 degrees tilt
        max_tilt = rng.uniform(15.0, 25.0)
        img = _perspective_warp(img, rng, np_rng, max_tilt_deg=max_tilt)

        # Rotation: +/-10 degrees
        angle = rng.uniform(-10.0, 10.0)
        img = _rotate(img, angle, rng)

        # Strong corner blur: sigma 3.0-6.0
        corner_sigma = rng.uniform(3.0, 6.0)
        img = _corner_blur(img, np_rng, corner_sigma=corner_sigma)

        # Motion blur: kernel 5-9 pixels
        kernel_size = rng.choice([5, 7, 9])
        img = _motion_blur(img, kernel_size, rng)

        # Heavy color temperature shift
        img = _color_temperature(img, rng, shift_k=1200)

        # Heavy shadows + finger occlusion
        img = _shadow_overlay(img, count=rng.randint(2, 4), rng=rng,
                              np_rng=np_rng, darkness_range=(0.3, 0.6))
        img = _finger_occlusion(img, rng=rng)

        # Lighting gradient
        img = _lighting_gradient(img, rng=rng)

        # Noise sigma 15-30
        sigma = rng.uniform(15.0, 30.0)
        img = _gaussian_noise(img, sigma, np_rng)

        # Resolution loss: 50% downscale then upscale
        img = _resolution_loss(img, scale_factor=0.5)

        # Vignette: strength 0.4-0.6
        vignette_strength = rng.uniform(0.4, 0.6)
        img = _vignette(img, rng, strength=vignette_strength)

        # JPEG quality 40-60
        quality = rng.randint(40, 60)
        img = _jpeg_compress(img, quality)

        return img


# ---------------------------------------------------------------------------
# Transform primitives (module-private)
# ---------------------------------------------------------------------------


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG at *quality* then decode back."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        raise OSError("Could not encode JPEG degradation")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise OSError("Could not decode JPEG degradation")
    return decoded


def _perspective_warp(
    image: np.ndarray,
    rng: random.Random,
    np_rng: np.random.RandomState,
    max_tilt_deg: float = 10.0,
) -> np.ndarray:
    """Simulate camera viewing angle using a physically-based homography.

    Models a phone camera with realistic field-of-view (60-70 degrees)
    looking at a flat document from an angle.  The *max_tilt_deg* parameter
    controls maximum rotation around the X and Y axes, producing the
    keystone distortion characteristic of phone photos.

    The Y-axis tilt is biased towards forward tilt (phone tilted towards
    the photographer) which is the most common real-world scenario.
    """
    h, w = image.shape[:2]

    # Random tilt angles
    tilt_x = rng.uniform(-max_tilt_deg, max_tilt_deg)
    # Bias: phone tilts forward more than sideways
    tilt_y = rng.uniform(-max_tilt_deg * 0.8, max_tilt_deg * 0.3)
    # Small in-plane rotation component
    rotation = rng.uniform(-max_tilt_deg * 0.3, max_tilt_deg * 0.3)

    # Camera intrinsics: phone camera ~60-70 degree FOV
    fov = rng.uniform(60.0, 70.0)
    f = w / (2.0 * math.tan(math.radians(fov / 2.0)))
    cx, cy = w / 2.0, h / 2.0

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    # Build 3D rotation matrix from Euler angles
    rx = math.radians(tilt_x)
    ry = math.radians(tilt_y)
    rz = math.radians(rotation)

    Rx = np.array(
        [[1, 0, 0],
         [0, math.cos(rx), -math.sin(rx)],
         [0, math.sin(rx), math.cos(rx)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[math.cos(ry), 0, math.sin(ry)],
         [0, 1, 0],
         [-math.sin(ry), 0, math.cos(ry)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[math.cos(rz), -math.sin(rz), 0],
         [math.sin(rz), math.cos(rz), 0],
         [0, 0, 1]],
        dtype=np.float64,
    )

    R = Rz @ Ry @ Rx

    # Homography: H = K * R * K^-1
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv

    # Background colour simulating desk/surface
    bg_color = rng.randint(80, 180)

    warped = cv2.warpPerspective(
        image,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(bg_color, bg_color, bg_color),
    )
    return warped


def _rotate(
    image: np.ndarray, angle_deg: float, rng: random.Random
) -> np.ndarray:
    """Rotate image by *angle_deg* degrees around its centre.

    Uses a grey background consistent with a desk surface.
    """
    if abs(angle_deg) < 0.1:
        return image

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    bg_color = rng.randint(80, 180)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(bg_color, bg_color, bg_color),
    )


def _row_blocks(image: np.ndarray):
    """Bound float scratch space to about 256k pixels per tile, in scan order."""
    height, width = image.shape[:2]
    rows_per_block = max(1, 262_144 // width)
    for start in range(0, height, rows_per_block):
        yield slice(start, min(start + rows_per_block, height))


def _squared_radius(x_squared, rows, cy, max_r):
    """Squared normalized distance, preserving the simulator's float32 rounding."""
    y = np.arange(rows.start, rows.stop, dtype=np.float32)[:, None] - cy
    distance = np.sqrt(x_squared + y ** 2) / max_r
    return distance ** 2


def _corner_blur(
    image: np.ndarray,
    np_rng: np.random.RandomState,
    corner_sigma: float = 2.0,
) -> np.ndarray:
    """Apply spatially-varying blur: sharp centre, blurry corners.

    Simulates the depth-of-field falloff of phone cameras where the centre
    of the document is in focus but corners (further from the focal plane
    due to perspective) are progressively blurred.

    Uses vectorised alpha-blending between the sharp original and a
    Gaussian-blurred version, weighted by squared radial distance from the
    image centre.
    """
    if corner_sigma < 0.5:
        return image

    h, w = image.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx**2 + cy**2)

    x_squared = (np.arange(w, dtype=np.float32) - cx) ** 2
    k = max(3, int(corner_sigma * 6) | 1)
    blurred = cv2.GaussianBlur(image, (k, k), corner_sigma)
    for rows in _row_blocks(image):
        alpha = np.clip(_squared_radius(x_squared, rows, cy, max_r), 0, 1)
        sharp_weight = 1.0 - alpha
        # One channel at a time keeps the floating-point buffers tile-sized.
        for channel in range(3):
            plane = blurred[rows, :, channel]
            plane[:] = image[rows, :, channel] * sharp_weight + plane * alpha
    return blurred


def _color_temperature(
    image: np.ndarray, rng: random.Random, shift_k: int = 500
) -> np.ndarray:
    """Shift colour temperature to simulate warm/cool lighting.

    Positive shift = warmer (boost red, reduce blue).
    Negative shift = cooler (boost blue, reduce red).
    The *shift_k* parameter sets the maximum shift in pseudo-Kelvins,
    normalised to a channel scaling factor.
    """
    direction = rng.choice([-1, 1])
    amount = rng.uniform(0.01, shift_k / 10000.0) * direction

    # Only 256 input values exist per channel. Apply the three float32
    # multipliers to those levels once instead of allocating a float image.
    levels = np.arange(256, dtype=np.float32)[:, None]
    scales = np.array([1.0 - amount, 1.0 + amount * 0.1, 1.0 + amount], dtype=np.float32)
    table = np.clip(levels * scales, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table[:, None, :])


def _vignette(
    image: np.ndarray, rng: random.Random, strength: float = 0.3
) -> np.ndarray:
    """Apply radial vignetting (darkened corners).

    Simulates the natural light falloff of phone camera lenses.
    *strength* controls the maximum darkening at the corners (0 = none,
    1 = fully black corners).

    The vignette centre is slightly jittered to avoid perfectly symmetric
    results.
    """
    h, w = image.shape[:2]
    # Slight jitter on centre for realism
    cx = w / 2.0 + rng.uniform(-w * 0.02, w * 0.02)
    cy = h / 2.0 + rng.uniform(-h * 0.02, h * 0.02)
    max_r = math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)

    x_squared = (np.arange(w, dtype=np.float32) - cx) ** 2
    result = np.empty_like(image)
    for rows in _row_blocks(image):
        mask = np.clip(1.0 - strength * _squared_radius(x_squared, rows, cy, max_r), 0, 1)
        for channel in range(3):
            result[rows, :, channel] = image[rows, :, channel] * mask
    return result


def _gaussian_noise(
    image: np.ndarray, sigma: float, np_rng: np.random.RandomState
) -> np.ndarray:
    """Add Gaussian noise with standard deviation *sigma*.

    For high sigma (>15), uses multiplicative noise to preserve relative
    brightness and avoid blown-out highlights from additive clipping.
    """
    multiplicative = sigma > 15
    mean, sd = (1.0, sigma / 100.0) if multiplicative else (0.0, sigma)
    result = np.empty_like(image)
    for rows in _row_blocks(image):
        tile = image[rows]
        # RandomState keeps its cached normal across calls. Row order preserves
        # the full-frame noise sequence, including at odd tile boundaries.
        noise = np_rng.normal(mean, sd, tile.shape).astype(np.float32)
        if multiplicative:
            noise *= tile
        else:
            noise += tile
        result[rows] = np.clip(noise, 0, 255, out=noise)
    return result


def _brightness_contrast(
    image: np.ndarray, contrast: float, brightness: float
) -> np.ndarray:
    """Adjust brightness and contrast via ``alpha * pixel + beta``."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def _shadow_overlay(
    image: np.ndarray,
    count: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    darkness_range: tuple[float, float] = (0.4, 0.8),
) -> np.ndarray:
    """Overlay *count* gradient shadow bands on the image.

    *darkness_range* controls min/max brightness multiplier at the darkest
    point of each shadow (lower = darker).
    """
    h, w = image.shape[:2]
    masks = []
    for _ in range(count):
        darkness = rng.uniform(*darkness_range)
        horizontal = rng.random() < 0.5
        size = w if horizontal else h
        start, end = rng.randint(0, size // 2), rng.randint(size // 2, size)
        gradient = np.ones(size, dtype=np.float32)
        gradient[start:end] = np.linspace(1.0, darkness, end - start)
        masks.append((horizontal, gradient))

    result = np.empty_like(image)
    for rows in _row_blocks(image):
        tile = image[rows].astype(np.float32)
        for horizontal, gradient in masks:
            tile *= gradient[None, :, None] if horizontal else gradient[rows, None, None]
        result[rows] = np.clip(tile, 0, 255, out=tile)
    return result


def _motion_blur(
    image: np.ndarray, kernel_size: int, rng: random.Random
) -> np.ndarray:
    """Apply directional motion blur with a random angle."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    # Horizontal, vertical, or diagonal
    direction = rng.choice(["h", "v", "d"])
    if direction == "h":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "v":
        kernel[:, kernel_size // 2] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)

    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _finger_occlusion(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Draw a dark elliptical shape at a random edge to simulate a finger."""
    h, w = image.shape[:2]
    result = image.copy()

    # Choose an edge: 0=top, 1=right, 2=bottom, 3=left
    edge = rng.randint(0, 3)

    # Finger dimensions proportional to image size
    finger_w = rng.randint(int(w * 0.06), int(w * 0.12))
    finger_h = rng.randint(int(h * 0.08), int(h * 0.18))

    if edge == 0:  # top
        cx = rng.randint(int(w * 0.2), int(w * 0.8))
        cy = rng.randint(0, int(h * 0.05))
    elif edge == 1:  # right
        cx = rng.randint(int(w * 0.95), w)
        cy = rng.randint(int(h * 0.2), int(h * 0.8))
        finger_w, finger_h = finger_h, finger_w  # rotate orientation
    elif edge == 2:  # bottom
        cx = rng.randint(int(w * 0.2), int(w * 0.8))
        cy = rng.randint(int(h * 0.95), h)
    else:  # left
        cx = rng.randint(0, int(w * 0.05))
        cy = rng.randint(int(h * 0.2), int(h * 0.8))
        finger_w, finger_h = finger_h, finger_w

    # Skin-tone colour (BGR) with slight variation
    skin_b = rng.randint(100, 140)
    skin_g = rng.randint(120, 170)
    skin_r = rng.randint(150, 210)

    # Draw filled ellipse with slight rotation
    angle = rng.uniform(-15, 15)
    cv2.ellipse(
        result,
        (cx, cy),
        (finger_w, finger_h),
        angle,
        0,
        360,
        (skin_b, skin_g, skin_r),
        -1,
        cv2.LINE_AA,
    )

    return result


def _lighting_gradient(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a full-image lighting gradient (corner-to-corner)."""
    h, w = image.shape[:2]

    # Gradient direction
    direction = rng.choice(["tl_br", "tr_bl", "t_b", "l_r"])
    dark_level = rng.uniform(0.5, 0.75)

    if direction == "tr_bl":
        xs = np.linspace(dark_level, 1.0, w)
    elif direction == "t_b":
        xs = np.ones(1)
    else:
        xs = np.linspace(1.0, dark_level, w)
    ys = np.ones(h) if direction == "l_r" else np.linspace(1.0, dark_level, h)
    result = np.empty_like(image)
    for rows in _row_blocks(image):
        # Form the original float64 product before reducing precision.
        mask = (ys[rows, None] * xs).astype(np.float32)
        for channel in range(3):
            result[rows, :, channel] = image[rows, :, channel] * mask
    return result


def _resolution_loss(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Downscale by *scale_factor* then upscale back to original size."""
    h, w = image.shape[:2]
    small_h = max(int(h * scale_factor), 1)
    small_w = max(int(w * scale_factor), 1)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
