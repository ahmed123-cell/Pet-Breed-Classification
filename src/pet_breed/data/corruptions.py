"""
Corruption suite for simulating real-world pet photo domain shift.

Implements 5 corruption types at 3 severity levels:
1. gaussian_blur (out-of-focus phone camera)
2. brightness_shift (indoor low-light vs direct desert sun)
3. jpeg_compression (messaging app re-encoding artifacts)
4. downscale_upscale (low-resolution older handsets)
5. motion_blur (moving pet / camera shake)
"""

import io
from typing import Literal, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.signal import convolve2d

CorruptionType = Literal[
    "gaussian_blur",
    "brightness_shift",
    "jpeg_compression",
    "downscale_upscale",
    "motion_blur",
]

CORRUPTION_TYPES = [
    "gaussian_blur",
    "brightness_shift",
    "jpeg_compression",
    "downscale_upscale",
    "motion_blur",
]


def apply_gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian defocus blur with increasing kernel radius."""
    radii = {1: 1.5, 2: 3.0, 3: 5.5}
    radius = radii.get(severity, 2.0)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_shift(image: Image.Image, severity: int) -> Image.Image:
    """
    Apply brightness shift:
    Severity 1: Low-light evening indoor (0.6x)
    Severity 2: Harsh midday direct sun (1.7x)
    Severity 3: Extreme underexposure / night indoor (0.35x)
    """
    factors = {1: 0.6, 2: 1.7, 3: 0.35}
    factor = factors.get(severity, 1.0)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_jpeg_compression(image: Image.Image, severity: int) -> Image.Image:
    """Simulate WhatsApp/Telegram re-compression artifacts with low JPEG quality."""
    qualities = {1: 40, 2: 25, 3: 10}
    quality = qualities.get(severity, 30)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_downscale_upscale(image: Image.Image, severity: int) -> Image.Image:
    """Downscale to low resolution (e.g. 96x96) and upscale back to simulate old sensors."""
    orig_size = image.size
    scales = {1: 128, 2: 96, 3: 64}
    target_dim = scales.get(severity, 96)
    
    # Maintain aspect ratio for downscaling
    w, h = orig_size
    if w < h:
        new_w, new_h = target_dim, int(h * target_dim / w)
    else:
        new_w, new_h = int(w * target_dim / h), target_dim
    new_w, new_h = max(new_w, 16), max(new_h, 16)

    downscaled = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return downscaled.resize(orig_size, resample=Image.Resampling.NEAREST)


def _create_motion_blur_kernel(kernel_size: int, angle_deg: float = 45.0) -> np.ndarray:
    """Generate a directional motion blur kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = np.deg2rad(angle_deg)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * cos_val))
        y = int(round(center - i * sin_val))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    s = np.sum(kernel)
    if s > 0:
        kernel /= s
    else:
        kernel[center, center] = 1.0
    return kernel


def apply_motion_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply directional motion blur simulating moving pets."""
    sizes = {1: 7, 2: 13, 3: 21}
    k_size = sizes.get(severity, 11)
    kernel = _create_motion_blur_kernel(k_size, angle_deg=35.0)

    img_arr = np.array(image.convert("RGB"), dtype=np.float32)
    channels = []
    for c in range(3):
        convolved = convolve2d(img_arr[:, :, c], kernel, mode="same", boundary="symm")
        channels.append(convolved)
    blurred_arr = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(blurred_arr)


def apply_corruption(
    image: Image.Image, corruption_type: CorruptionType, severity: int
) -> Image.Image:
    """
    Apply specified corruption and severity level (1, 2, or 3) to a PIL Image.
    """
    if severity not in [1, 2, 3]:
        raise ValueError(f"Severity must be 1, 2, or 3. Got {severity}")

    image_rgb = image.convert("RGB")
    if corruption_type == "gaussian_blur":
        return apply_gaussian_blur(image_rgb, severity)
    elif corruption_type == "brightness_shift":
        return apply_brightness_shift(image_rgb, severity)
    elif corruption_type == "jpeg_compression":
        return apply_jpeg_compression(image_rgb, severity)
    elif corruption_type == "downscale_upscale":
        return apply_downscale_upscale(image_rgb, severity)
    elif corruption_type == "motion_blur":
        return apply_motion_blur(image_rgb, severity)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
