"""
Structured channel pruning for ResNet backbone.
Applies L1-norm structured channel pruning on Conv2d layers and measures accuracy/sparsity impact.
"""

from pathlib import Path
from typing import Dict, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from pet_breed.config import (
    CHECKPOINTS_DIR,
    EXPORTED_MODELS_DIR,
    DEFAULT_NUM_CLASSES,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.backbones import PetClassifierBackbone, get_model
from pet_breed.models.transforms import get_eval_transform, get_train_transform
from pet_breed.training.evaluate import evaluate_model


def apply_structured_pruning(
    model: PetClassifierBackbone,
    amount: float = 0.25,
) -> PetClassifierBackbone:
    """
    Apply structured L1-norm channel pruning across all Conv2d modules in the backbone.
    amount: fraction of channels to prune (e.g. 0.25 = 25% sparsity)
    """
    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune channels (dim 0 = output channels / filters)
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, "weight")  # Make pruning permanent

    return pruned_model


def calculate_model_sparsity(model: nn.Module) -> float:
    """Calculate actual fraction of zeroed weights across the network."""
    total_params = 0
    zero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        zero_params += torch.sum(p == 0).item()
    return float(zero_params / total_params) if total_params > 0 else 0.0


def run_pruning_experiment(
    checkpoint_path: Path,
    amount: float = 0.30,
    fine_tune_epochs: int = 1,
    lr: float = 1e-5,
) -> Dict[str, any]:
    """
    Load base model, apply structured pruning, optionally fine-tune, and evaluate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Structured Channel Pruning ({amount * 100:.0f}% sparsity) ---")

    # Load baseline model
    base_model = get_model(backbone_name="resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=False)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state.get("model_state_dict", state))
        base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device)

    # Eval dataset
    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    base_eval = evaluate_model(base_model, val_loader, device)
    print(f"Baseline Top-1: {base_eval['top1']:.4f}")

    # Prune
    pruned_model = apply_structured_pruning(base_model, amount=amount)
    pruned_model.to(device)
    actual_sparsity = calculate_model_sparsity(pruned_model)

    # Post-pruning eval
    pruned_eval_pre = evaluate_model(pruned_model, val_loader, device)
    print(f"Pruned (Raw) Top-1: {pruned_eval_pre['top1']:.4f} | Sparsity: {actual_sparsity:.2%}")

    # Optional brief fine-tuning to recover accuracy
    if fine_tune_epochs > 0:
        train_ds = PetBreedDataset(split="train", transform=get_train_transform())
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        pruned_model.train()
        for epoch in range(fine_tune_epochs):
            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = pruned_model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

    final_eval = evaluate_model(pruned_model, val_loader, device)
    print(f"Pruned (Recovered) Top-1: {final_eval['top1']:.4f} | Delta: {final_eval['top1'] - base_eval['top1']:+.4f}")

    # Save pruned model
    save_path = EXPORTED_MODELS_DIR / "resnet50_pruned.pth"
    torch.save(
        {
            "backbone": "resnet50",
            "sparsity": actual_sparsity,
            "target_amount": amount,
            "state_dict": pruned_model.state_dict(),
            "metrics": final_eval,
        },
        save_path,
    )

    return {
        "sparsity": actual_sparsity,
        "baseline_top1": base_eval["top1"],
        "pruned_top1": final_eval["top1"],
        "delta_top1": round(final_eval["top1"] - base_eval["top1"], 4),
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    chk = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    if not chk.exists():
        chk = CHECKPOINTS_DIR / "resnet50_best.pth"
    run_pruning_experiment(chk)
