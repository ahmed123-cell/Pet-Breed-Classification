"""
Penultimate-layer embedding drift detector using Maximum Mean Discrepancy (MMD) and Domain Classifier.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def compute_rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """Compute RBF Gaussian kernel matrix between X and Y."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    dist_sq = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(Y ** 2, axis=1, keepdims=True).T
        - 2.0 * np.dot(X, Y.T)
    )
    return np.exp(-gamma * np.maximum(dist_sq, 0.0))


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD^2) between reference embeddings X and batch embeddings Y.
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 0.0

    K_XX = compute_rbf_kernel(X, X, gamma)
    K_YY = compute_rbf_kernel(Y, Y, gamma)
    K_XY = compute_rbf_kernel(X, Y, gamma)

    # Unbiased estimator of MMD^2
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd_sq = (
        np.sum(K_XX) / (n * (n - 1) if n > 1 else 1)
        + np.sum(K_YY) / (m * (m - 1) if m > 1 else 1)
        - 2.0 * np.sum(K_XY) / (n * m)
    )
    return float(max(0.0, mmd_sq))


class EmbeddingDriftDetector:
    """
    Feature embedding drift detector operating on penultimate backbone representations.
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        mmd_threshold: float = 0.015,
        domain_classifier_auc_threshold: float = 0.65,
    ):
        self.reference_embeddings = reference_embeddings
        self.mmd_threshold = mmd_threshold
        self.auc_threshold = domain_classifier_auc_threshold

    def score_batch(self, current_embeddings: np.ndarray) -> Dict[str, any]:
        """
        Evaluate batch embeddings against reference embeddings via MMD and Domain Classifier AUC.
        """
        mmd_score = compute_mmd(self.reference_embeddings, current_embeddings)

        # Domain Classifier (discriminate between Reference=0 and Current=1)
        n_ref = len(self.reference_embeddings)
        n_curr = len(current_embeddings)
        
        # Subsample to balance if sizes differ heavily
        min_n = min(n_ref, n_curr, 200)
        idx_ref = np.random.choice(n_ref, min_n, replace=False)
        idx_curr = np.random.choice(n_curr, min_n, replace=False)

        X = np.vstack([self.reference_embeddings[idx_ref], current_embeddings[idx_curr]])
        y = np.hstack([np.zeros(min_n), np.ones(min_n)])

        try:
            clf = LogisticRegression(max_iter=200, C=1.0)
            scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
            auc_score = float(np.mean(scores))
        except Exception:
            auc_score = 0.50

        drift_detected = (mmd_score >= self.mmd_threshold) or (auc_score >= self.auc_threshold)

        return {
            "detector": "embedding_penultimate_mmd",
            "mmd_score": round(mmd_score, 6),
            "mmd_threshold": self.mmd_threshold,
            "domain_classifier_auc": round(auc_score, 4),
            "auc_threshold": self.auc_threshold,
            "drift_detected": bool(drift_detected),
        }
