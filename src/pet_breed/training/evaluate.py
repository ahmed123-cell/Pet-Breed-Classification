"""
Evaluation metrics computation: Top-1 accuracy, Macro F1, ECE, Coverage, and Selective Accuracy.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_breed.calibration.abstention import evaluate_abstention
from pet_breed.calibration.temperature_scaling import compute_ece


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pass all samples through model to collect full logits tensor and ground truth labels.
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels, _ in tqdm(dataloader, desc="Collecting logits", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.detach().cpu().clone())
        all_labels.append(labels.detach().cpu().clone())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return logits_cat, labels_cat


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    abstention_threshold: float = 0.65,
) -> Dict[str, float]:
    """
    Comprehensive evaluation returning:
    - top1: Top-1 Accuracy
    - f1_macro: Macro F1-Score
    - ece_uncalibrated: Raw ECE before temperature scaling
    - ece: Calibrated ECE
    - coverage: Answered fraction at threshold
    - selective_accuracy: Accuracy on answered fraction
    - temperature: Learned/applied temperature
    """
    logits, labels = collect_logits_and_labels(model, dataloader, device)
    logits_np = logits.numpy()
    labels_np = labels.numpy()

    # Raw softmax probabilities
    raw_exp = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    raw_probs = raw_exp / np.sum(raw_exp, axis=1, keepdims=True)

    # Temperature scaled probabilities
    scaled_logits_np = logits_np / max(temperature, 1e-6)
    scaled_exp = np.exp(scaled_logits_np - np.max(scaled_logits_np, axis=1, keepdims=True))
    calibrated_probs = scaled_exp / np.sum(scaled_exp, axis=1, keepdims=True)

    # Top-1 & Macro F1
    preds = np.argmax(calibrated_probs, axis=1)
    top1 = float(np.mean(preds == labels_np))
    f1_macro = float(f1_score(labels_np, preds, average="macro"))

    # ECE
    ece_uncal, _, _, _ = compute_ece(raw_probs, labels_np)
    ece_cal, _, _, _ = compute_ece(calibrated_probs, labels_np)

    # Selective Abstention
    abst_metrics = evaluate_abstention(
        calibrated_probs, labels_np, threshold=abstention_threshold
    )

    return {
        "top1": round(top1, 4),
        "f1_macro": round(f1_macro, 4),
        "ece_uncalibrated": round(ece_uncal, 4),
        "ece": round(ece_cal, 4),
        "coverage": round(abst_metrics["coverage"], 4),
        "selective_accuracy": round(abst_metrics["selective_accuracy"], 4),
        "abstention_rate": round(abst_metrics["abstention_rate"], 4),
        "temperature": round(float(temperature), 4),
        "threshold": round(float(abstention_threshold), 4),
    }
