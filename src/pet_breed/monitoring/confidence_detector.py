"""
Output confidence distribution drift detector.
Monitors shifts in calibrated confidence distributions and abstention rates.
"""

from typing import Dict
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


class ConfidenceDriftDetector:
    """
    Monitors output confidence distribution shifts vs reference baseline.
    """

    def __init__(self, reference_confidences: np.ndarray, ks_pvalue_threshold: float = 0.01):
        self.reference_confs = reference_confidences
        self.ks_threshold = ks_pvalue_threshold

    def score_batch(self, current_confidences: np.ndarray) -> Dict[str, any]:
        """
        Evaluate current batch confidences against baseline.
        """
        w_dist = float(wasserstein_distance(self.reference_confs, current_confidences))
        ks_res = ks_2samp(self.reference_confs, current_confidences)
        p_val = float(ks_res.pvalue)

        mean_ref = float(np.mean(self.reference_confs))
        mean_curr = float(np.mean(current_confidences))

        drift_detected = p_val < self.ks_threshold

        return {
            "detector": "confidence_distribution",
            "wasserstein_distance": round(w_dist, 4),
            "ks_pvalue": round(p_val, 6),
            "mean_reference_confidence": round(mean_ref, 4),
            "mean_current_confidence": round(mean_curr, 4),
            "drift_detected": bool(drift_detected),
        }
