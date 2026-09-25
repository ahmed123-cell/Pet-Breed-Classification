"""
Automated closed-loop retraining pipeline triggered by drift alerts.
Fine-tunes model on drifting data distribution and enforces production quality gate.
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import get_eval_transform, get_train_transform
from pet_breed.retraining.quality_gate import evaluate_quality_gate
from pet_breed.training.evaluate import evaluate_model
from pet_breed.training.mlflow_tracker import log_run_to_mlflow


def run_retraining_trigger(
    corruption_trigger: str = "gaussian_blur",
    severity: int = 2,
    fine_tune_epochs: int = 3,
    lr: float = 5e-5,
    production_top1: float = 0.9140,
) -> Dict[str, any]:
    """
    Execute closed-loop retraining triggered by drift alert.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" DRIFT ALERT TRIGGERED: Retraining Pipeline Initiated")
    print(f" Detected Drift Domain: {corruption_trigger} (Severity {severity})")
    print(f"=======================================================\n")

    # 1. Prepare Augmented Dataset including detected drift samples
    train_clean_ds = PetBreedDataset(split="train", transform=get_train_transform())
    train_corrupt_ds = PetBreedDataset(
        split="test_corrupted",
        corruption=corruption_trigger,
        severity=severity,
        transform=get_train_transform(),
    )
    combined_train_ds = (
        ConcatDataset([train_clean_ds, train_corrupt_ds])
        if len(train_corrupt_ds) > 0
        else train_clean_ds
    )
    train_loader = DataLoader(
        combined_train_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True
    )

    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    # 2. Load Current Production Baseline
    prod_checkpoint = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    model = get_model(backbone_name="resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=True)
    if prod_checkpoint.exists():
        state = torch.load(prod_checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=False)
    model.to(device)

    # 3. Fine-tune candidate model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, fine_tune_epochs + 1):
        total_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        print(f"Retrain Epoch {epoch}/{fine_tune_epochs} | Loss: {total_loss/len(combined_train_ds):.4f}")

    # 4. Evaluate Candidate on Holdout Validation
    candidate_metrics = evaluate_model(model, val_loader, device)
    candidate_top1 = candidate_metrics["top1"]
    print(f"Retrained Candidate Top-1: {candidate_top1:.4f} (Production Target: {production_top1:.4f})")

    # 5. Production Quality Gate Check
    passed, gate_result = evaluate_quality_gate(
        candidate_top1=candidate_top1,
        production_top1=production_top1,
        tolerance=0.01,
    )

    # 6. Log Retraining Experiment to MLflow
    retrain_export_path = EXPORTED_MODELS_DIR / "resnet50_retrained_candidate.pth"
    torch.save(
        {
            "backbone": "resnet50",
            "state_dict": model.state_dict(),
            "trigger_corruption": corruption_trigger,
            "severity": severity,
            "metrics": candidate_metrics,
            "quality_gate": gate_result,
        },
        retrain_export_path,
    )

    run_id = log_run_to_mlflow(
        run_name=f"retrain_{corruption_trigger}_s{severity}",
        params={
            "pipeline": "automated_drift_retraining",
            "trigger_corruption": corruption_trigger,
            "severity": severity,
            "epochs": fine_tune_epochs,
            "lr": lr,
            "quality_gate_passed": passed,
        },
        metrics=candidate_metrics,
        model=model if passed else None,
        artifacts={"candidate_checkpoint": retrain_export_path},
        register_model_name="PetBreedClassifier" if passed else None,
    )

    return {
        "run_id": run_id,
        "quality_gate_passed": passed,
        "gate_result": gate_result,
        "metrics": candidate_metrics,
    }


if __name__ == "__main__":
    run_retraining_trigger()
