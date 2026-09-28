"""
Training, validation, temperature calibration, and MLflow tracking pipeline.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_breed.calibration.abstention import find_optimal_threshold
from pet_breed.calibration.temperature_scaling import (
    TemperatureScaler,
    plot_reliability_diagrams,
)
from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
    EXPORTED_MODELS_DIR,
    REPORTS_DIR,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import get_eval_transform, get_train_transform
from pet_breed.training.evaluate import collect_logits_and_labels, evaluate_model
from pet_breed.training.mlflow_tracker import log_run_to_mlflow


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def run_training_experiment(
    backbone: str = "resnet50",
    lr: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = 5,
    seed: int = DEFAULT_SEED,
    freeze_backbone: bool = False,
    register_model_name: Optional[str] = None,
) -> Dict[str, any]:
    """
    Execute full training, calibration, and MLflow logging cycle for a backbone.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Experiment: {backbone} | LR: {lr} | Batch Size: {batch_size} | Device: {device} ---")

    # Datasets and Loaders
    train_ds = PetBreedDataset(split="train", transform=get_train_transform())
    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # Initialize model
    model = get_model(backbone_name=backbone, num_classes=DEFAULT_NUM_CLASSES, pretrained=True)
    
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    model.to(device)

    # Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train Loop
    best_val_acc = 0.0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINTS_DIR / f"{backbone}_best.pth"

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        # Validation accuracy check
        val_eval = evaluate_model(model, val_loader, device)
        val_acc = val_eval["top1"]
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Top-1: {val_acc:.4f} (F1: {val_eval['f1_macro']:.4f})")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": backbone,
                    "model_state_dict": model.state_dict(),
                    "val_top1": val_acc,
                },
                best_checkpoint_path,
            )

    # Load best checkpoint for temperature scaling calibration
    if best_checkpoint_path.exists():
        chk = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])

    print("\nFitting Temperature Scaling calibration on validation fold...")
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)
    print(f"Learned Optimal Temperature: {temperature:.4f}")

    # Compute uncalibrated and calibrated probabilities
    val_logits_np = val_logits.numpy()
    val_labels_np = val_labels.numpy()
    
    uncal_exp = np.exp(val_logits_np - np.max(val_logits_np, axis=1, keepdims=True))
    uncal_probs = uncal_exp / np.sum(uncal_exp, axis=1, keepdims=True)
    
    cal_logits_np = val_logits_np / max(temperature, 1e-6)
    cal_exp = np.exp(cal_logits_np - np.max(cal_logits_np, axis=1, keepdims=True))
    cal_probs = cal_exp / np.sum(cal_exp, axis=1, keepdims=True)

    # Save Reliability Diagram
    calib_plot_path = REPORTS_DIR / f"calibration_{backbone}.png"
    plot_reliability_diagrams(
        uncal_probs, cal_probs, val_labels_np, save_path=calib_plot_path
    )
    if backbone == "resnet50":
        # Save standard reference report
        plot_reliability_diagrams(
            uncal_probs, cal_probs, val_labels_np, save_path=REPORTS_DIR / "calibration.png"
        )

    # Determine abstention threshold
    optimal_th, abst_summary = find_optimal_threshold(cal_probs, val_labels_np, target_selective_accuracy=0.95)
    print(f"Abstention Analysis at Threshold {optimal_th:.2f} -> Coverage: {abst_summary['coverage']:.2%}, Selective Acc: {abst_summary['selective_accuracy']:.2%}")

    # Comprehensive evaluation
    final_metrics = evaluate_model(
        model, val_loader, device, temperature=temperature, abstention_threshold=optimal_th
    )

    # Export canonical weights
    export_path = EXPORTED_MODELS_DIR / f"{backbone}_calibrated.pth"
    torch.save(
        {
            "backbone": backbone,
            "num_classes": DEFAULT_NUM_CLASSES,
            "state_dict": model.state_dict(),
            "temperature": temperature,
            "abstention_threshold": optimal_th,
            "metrics": final_metrics,
        },
        export_path,
    )

    # Log to MLflow
    params = {
        "backbone": backbone,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "temperature": round(temperature, 4),
        "abstention_threshold": round(optimal_th, 4),
    }

    artifacts = {
        "checkpoint": best_checkpoint_path,
        "calibration_plot": calib_plot_path,
        "exported_model": export_path,
    }

    run_name = f"{backbone}_lr{lr}_bs{batch_size}"
    run_id = log_run_to_mlflow(
        run_name=run_name,
        params=params,
        metrics=final_metrics,
        model=model,
        artifacts=artifacts,
        register_model_name=register_model_name,
    )

    return {
        "run_id": run_id,
        "backbone": backbone,
        "metrics": final_metrics,
        "temperature": temperature,
        "threshold": optimal_th,
        "export_path": str(export_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pet Breed Classifier")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--register", type=str, default=None)
    args = parser.parse_args()

    run_training_experiment(
        backbone=args.backbone,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
        register_model_name=args.register,
    )
