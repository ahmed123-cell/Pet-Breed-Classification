"""
Temperature scaling calibration and reliability diagram generation.
"""

from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pet_breed.config import REPORTS_DIR


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) and bin statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_counts.append(0)

    return (
        float(ece),
        np.array(bin_accs),
        np.array(bin_confs),
        np.array(bin_counts),
    )


class TemperatureScaler(nn.Module):
    """
    Learns a scalar temperature T on validation logits to minimize NLL.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature parameter using L-BFGS optimizer.
        """
        logits_tensor = val_logits.detach().clone()
        labels_tensor = val_labels.detach().clone()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_tensor), labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def plot_reliability_diagrams(
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    save_path: Path = REPORTS_DIR / "calibration.png",
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Generate and save side-by-side reliability diagrams before and after temperature scaling.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ece_uncal, accs_uncal, confs_uncal, _ = compute_ece(
        uncalibrated_probs, labels, n_bins=n_bins
    )
    ece_cal, accs_cal, confs_cal, _ = compute_ece(
        calibrated_probs, labels, n_bins=n_bins
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bin_centers = np.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)

    # Plot 1: Uncalibrated
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    axes[0].bar(
        bin_centers,
        accs_uncal,
        width=1.0 / n_bins,
        alpha=0.7,
        color="#e74c3c",
        edgecolor="black",
        label="Observed Accuracy",
    )
    axes[0].set_title(f"Before Calibration (Raw Softmax)\nECE = {ece_uncal:.4f}")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Calibrated
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    axes[1].bar(
        bin_centers,
        accs_cal,
        width=1.0 / n_bins,
        alpha=0.7,
        color="#2ecc71",
        edgecolor="black",
        label="Observed Accuracy",
    )
    axes[1].set_title(f"After Calibration (Temperature Scaled)\nECE = {ece_cal:.4f}")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Calibration reliability diagram saved to {save_path}")
    return {"ece_uncalibrated": ece_uncal, "ece_calibrated": ece_cal}
