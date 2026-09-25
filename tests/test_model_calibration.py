"""
Unit tests for temperature scaling calibration and selective abstention evaluation.
"""

import numpy as np
import pytest
import torch

from pet_breed.calibration.abstention import evaluate_abstention, find_optimal_threshold
from pet_breed.calibration.temperature_scaling import TemperatureScaler, compute_ece


def test_temperature_scaler_optimization():
    """Verify TemperatureScaler converges to an optimal temperature and improves ECE."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Overconfident uncalibrated logits
    num_samples = 200
    num_classes = 37
    logits = torch.randn(num_samples, num_classes) * 5.0  # High scale causes overconfidence
    labels = torch.randint(0, num_classes, (num_samples,))

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels)
    assert temperature > 0.0, "Temperature must be strictly positive"

    # Compute uncalibrated vs calibrated ECE
    logits_np = logits.numpy()
    labels_np = labels.numpy()

    uncal_probs = np.exp(logits_np) / np.sum(np.exp(logits_np), axis=1, keepdims=True)
    cal_logits = logits_np / temperature
    cal_probs = np.exp(cal_logits) / np.sum(np.exp(cal_logits), axis=1, keepdims=True)

    ece_uncal, _, _, _ = compute_ece(uncal_probs, labels_np)
    ece_cal, _, _, _ = compute_ece(cal_probs, labels_np)

    print(f"Test ECE before: {ece_uncal:.4f}, after: {ece_cal:.4f}, Learned T: {temperature:.4f}")
    assert ece_cal <= ece_uncal + 1e-4, "Temperature scaling should improve or preserve calibration error"


def test_abstention_metrics():
    """Verify abstention policy correctly computes coverage and selective accuracy."""
    probs = np.array([
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
        [0.40, 0.35, 0.25],  # Low confidence
        [0.45, 0.30, 0.25],  # Low confidence
    ])
    labels = np.array([0, 0, 1, 0])

    # With threshold 0.70, only the first two are answered
    res = evaluate_abstention(probs, labels, threshold=0.70)
    assert res["coverage"] == 0.50, f"Expected 0.50 coverage, got {res['coverage']}"
    assert res["selective_accuracy"] == 1.0, f"Expected 1.0 selective accuracy, got {res['selective_accuracy']}"
    assert res["num_answered"] == 2
