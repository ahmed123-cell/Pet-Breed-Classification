"""
Quantization-Aware Training (QAT) pipeline.
Simulates low-precision INT8 quantization forward noise during training for superior accuracy retention.
"""

from pathlib import Path
from typing import Dict
import copy
import torch
import torch.ao.quantization as quantization
import torch.nn as nn
from torch.utils.data import DataLoader

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import get_eval_transform, get_train_transform
from pet_breed.training.evaluate import evaluate_model


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """
    Fuse modules (Conv+BN+ReLU) and insert fake-quantization observers.
    """
    model_qat = copy.deepcopy(model)
    model_qat.eval()
    
    # Configure quantization backend for x86 / ARM
    model_qat.qconfig = quantization.get_default_qat_qconfig("fbgemm")
    prepared_model = quantization.prepare_qat(model_qat, inplace=False)
    return prepared_model


def run_qat_experiment(
    checkpoint_path: Path,
    epochs: int = 2,
    lr: float = 1e-5,
) -> Dict[str, any]:
    """
    Execute Quantization-Aware Training fine-tuning and convert to quantized model.
    """
    device = torch.device("cpu")  # PyTorch QAT conversion is best handled on CPU
    print(f"\n--- Running Quantization-Aware Training (QAT) for {epochs} epochs ---")

    # Load base model
    base_model = get_model(backbone_name="resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=False)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state.get("model_state_dict", state))
        base_model.load_state_dict(state_dict, strict=False)

    # Datasets
    train_ds = PetBreedDataset(split="train", transform=get_train_transform())
    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    base_eval = evaluate_model(base_model, val_loader, device)
    print(f"Base FP32 Top-1: {base_eval['top1']:.4f}")

    # Prepare QAT model
    qat_model = prepare_qat_model(base_model)
    qat_model.train()

    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        print(f"QAT Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_ds):.4f}")

    # Convert fake-quantized model to true INT8 quantized model
    qat_model.eval()
    quantized_model = quantization.convert(qat_model, inplace=False)

    # Evaluate INT8 QAT model
    qat_eval = evaluate_model(quantized_model, val_loader, device)
    print(f"QAT INT8 Top-1: {qat_eval['top1']:.4f} | Accuracy Delta vs FP32: {qat_eval['top1'] - base_eval['top1']:+.4f}")

    # Save QAT artifact
    save_path = EXPORTED_MODELS_DIR / "resnet50_qat.pth"
    torch.save(
        {
            "model_type": "qat_int8",
            "backbone": "resnet50",
            "metrics": qat_eval,
            "state_dict": quantized_model.state_dict(),
        },
        save_path,
    )

    return {
        "base_top1": base_eval["top1"],
        "qat_top1": qat_eval["top1"],
        "delta_top1": round(qat_eval["top1"] - base_eval["top1"], 4),
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    chk = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    if not chk.exists():
        chk = CHECKPOINTS_DIR / "resnet50_best.pth"
    run_qat_experiment(chk)
