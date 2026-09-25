"""
Cheap statistical pixel drift detector.
Computes low-cost distributional shifts across image brightness, contrast, and Laplacian sharpness.
"""

from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.stats import ks_2samp, wasserstein_distance


def extract_pixel_features(image: Image.Image) -> Dict[str, float]:
    """
    Extract low-cost pixel-level statistics from an image:
    1. Brightness: mean grayscale intensity [0..255]
    2. Contrast: standard deviation of grayscale intensity
    3. Sharpness: variance of Laplacian convolution (measure of edge detail/blur)
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Laplacian kernel for sharpness estimation
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve2d(gray, laplacian_kernel, mode="valid")
    sharpness = float(np.var(lap))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def extract_batch_pixel_features(images: List[Image.Image]) -> Dict[str, np.ndarray]:
    """Extract pixel statistics vectors across a batch of images."""
    brightnesses = []
    contrasts = []
    sharpnesses = []

    for img in images:
        feats = extract_pixel_features(img)
        brightnesses.append(feats["brightness"])
        contrasts.append(feats["contrast"])
        sharpnesses.append(feats["sharpness"])

    return {
        "brightness": np.array(brightnesses),
        "contrast": np.array(contrasts),
        "sharpness": np.array(sharpnesses),
    }


class CheapPixelDriftDetector:
    """
    Statistical pixel drift detector using Wasserstein distance and KS 2-sample tests against a reference baseline.
    """

    def __init__(self, reference_images: List[Image.Image], threshold_pvalue: float = 0.01):
        self.reference_feats = extract_batch_pixel_features(reference_images)
        self.threshold_pvalue = threshold_pvalue

    def score_batch(self, current_images: List[Image.Image]) -> Dict[str, any]:
        """
        Evaluate current batch against reference baseline distributions.
        """
        curr_feats = extract_batch_pixel_features(current_images)
        metrics = {}
        drift_detected = False

        for feature_name in ["brightness", "contrast", "sharpness"]:
            ref_vec = self.reference_feats[feature_name]
            curr_vec = curr_feats[feature_name]

            w_dist = float(wasserstein_distance(ref_vec, curr_vec))
            ks_res = ks_2samp(ref_vec, curr_vec)
            p_val = float(ks_res.pvalue)

            feature_drift = p_val < self.threshold_pvalue
            if feature_drift:
                drift_detected = True

            metrics[feature_name] = {
                "wasserstein_distance": round(w_dist, 4),
                "ks_pvalue": round(p_val, 6),
                "drift_detected": bool(feature_drift),
            }

        return {
            "detector": "cheap_pixel_statistical",
            "drift_detected": bool(drift_detected),
            "features": metrics,
        }
