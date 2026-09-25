"""
Abstention policy evaluator for selective classification.
Computes coverage, selective accuracy, and determines optimal confidence thresholds.
"""

from typing import Dict, List, Tuple
import numpy as np


def evaluate_abstention(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.65,
) -> Dict[str, float]:
    """
    Evaluate selective prediction metrics at a given confidence threshold.
    
    Returns:
        coverage: fraction of samples where max_prob >= threshold
        selective_accuracy: accuracy computed ONLY on the answered (covered) samples
        top1_accuracy: standard unconstrained accuracy on all samples
        abstention_rate: 1.0 - coverage
    """
    confidences = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)

    total_samples = len(labels)
    answered_mask = confidences >= threshold
    num_answered = np.sum(answered_mask)

    top1_accuracy = float(np.mean(predictions == labels))
    coverage = float(num_answered / total_samples) if total_samples > 0 else 0.0

    if num_answered > 0:
        selective_accuracy = float(
            np.mean(predictions[answered_mask] == labels[answered_mask])
        )
    else:
        selective_accuracy = 0.0

    return {
        "threshold": threshold,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "top1_accuracy": top1_accuracy,
        "abstention_rate": 1.0 - coverage,
        "num_answered": int(num_answered),
        "total_samples": int(total_samples),
    }


def find_optimal_threshold(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    target_selective_accuracy: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Scan thresholds in [0.30, 0.99] to find the minimum threshold achieving target selective accuracy.
    """
    thresholds = np.linspace(0.30, 0.99, 70)
    best_threshold = 0.65
    best_metrics = evaluate_abstention(calibrated_probs, labels, threshold=best_threshold)

    for th in thresholds:
        m = evaluate_abstention(calibrated_probs, labels, threshold=float(th))
        if m["selective_accuracy"] >= target_selective_accuracy and m["coverage"] >= 0.50:
            return float(th), m

    return best_threshold, best_metrics
