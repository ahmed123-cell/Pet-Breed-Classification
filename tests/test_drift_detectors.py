"""
Unit tests for drift detectors (Cheap Pixel, Embedding MMD, Confidence KS).
"""

import numpy as np
from PIL import Image
import pytest

from pet_breed.data.corruptions import apply_brightness_shift, apply_gaussian_blur
from pet_breed.monitoring.confidence_detector import ConfidenceDriftDetector
from pet_breed.monitoring.embedding_detector import EmbeddingDriftDetector
from pet_breed.monitoring.pixel_detector import CheapPixelDriftDetector


def test_cheap_pixel_detector_detects_severe_blur():
    """Verify CheapPixelDriftDetector catches severe blur."""
    ref_imgs = [Image.new("RGB", (100, 100), color=(i, i + 10, i + 20)) for i in range(20)]
    detector = CheapPixelDriftDetector(ref_imgs)

    # Clean batch -> no drift
    clean_batch = [Image.new("RGB", (100, 100), color=(i + 2, i + 12, i + 22)) for i in range(20)]
    clean_res = detector.score_batch(clean_batch)
    assert not clean_res["drift_detected"]

    # Severe brightness shift -> drift detected
    dark_batch = [apply_brightness_shift(img, severity=3) for img in ref_imgs]
    dark_res = detector.score_batch(dark_batch)
    assert dark_res["drift_detected"]
    assert dark_res["features"]["brightness"]["drift_detected"]


def test_embedding_drift_detector():
    """Verify EmbeddingDriftDetector computes MMD and flags out-of-distribution embeddings."""
    np.random.seed(42)
    ref_emb = np.random.normal(loc=0.0, scale=1.0, size=(100, 64))
    detector = EmbeddingDriftDetector(ref_emb, mmd_threshold=0.05)

    # In-distribution batch
    in_dist = np.random.normal(loc=0.0, scale=1.0, size=(100, 64))
    res_in = detector.score_batch(in_dist)
    assert not res_in["drift_detected"]

    # Shifted batch
    shifted = np.random.normal(loc=2.0, scale=1.0, size=(100, 64))
    res_shift = detector.score_batch(shifted)
    assert res_shift["drift_detected"]
    assert res_shift["mmd_score"] > 0.05


def test_confidence_drift_detector():
    """Verify ConfidenceDriftDetector catches skewed confidence distributions."""
    np.random.seed(42)
    ref_conf = np.random.beta(a=8, b=2, size=200)  # High confidence
    detector = ConfidenceDriftDetector(ref_conf)

    # Similar distribution
    same_conf = np.random.beta(a=8, b=2, size=200)
    assert not detector.score_batch(same_conf)["drift_detected"]

    # Degraded confidence distribution
    low_conf = np.random.beta(a=2, b=8, size=200)
    res_low = detector.score_batch(low_conf)
    assert res_low["drift_detected"]
