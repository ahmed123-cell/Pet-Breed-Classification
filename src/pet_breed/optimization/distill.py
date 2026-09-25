"""
Knowledge Distillation: Transferring representations from ResNet-50 (Teacher, ~25M params)
to MobileNetV3-Small (Student, ~2.5M params).
"""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import get_eval_transform, get_train_transform
from pet_breed.training.evaluate import evaluate_model


class DistillationLoss(nn.Module):
    """
    Combined Hard Cross-Entropy Loss and Soft KL-Divergence Distillation Loss.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hard_loss = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        return self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss


def run_distillation(
    teacher_checkpoint: Path,
    epochs: int = 4,
    lr: float = 3e-4,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> Dict[str, any]:
    """
    Train MobileNetV3-Small student guided by ResNet-50 teacher logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Knowledge Distillation: ResNet-50 -> MobileNetV3-Small (T={temperature}, alpha={alpha}) ---")

    # Load Teacher (ResNet-50)
    teacher = get_model(backbone_name="resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=False)
    if teacher_checkpoint.exists():
        state = torch.load(teacher_checkpoint, map_location=device)
        state_dict = state.get("state_dict", state.get("model_state_dict", state))
        teacher.load_state_dict(state_dict, strict=False)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Initialize Student (MobileNetV3-Small)
    student = get_model(backbone_name="mobilenet_v3_small", num_classes=DEFAULT_NUM_CLASSES, pretrained=True)
    student.to(device)

    # Datasets
    train_ds = PetBreedDataset(split="train", transform=get_train_transform())
    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    teacher_eval = evaluate_model(teacher, val_loader, device)
    print(f"Teacher (ResNet-50) Top-1: {teacher_eval['top1']:.4f}")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    distill_criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    best_student_acc = 0.0
    best_student_state = None

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0

        for images, labels, _ in tqdm(train_loader, desc=f"Distill Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distill_criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()
        val_eval = evaluate_model(student, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | Distill Loss: {total_loss/len(train_ds):.4f} | Student Top-1: {val_eval['top1']:.4f}")

        if val_eval["top1"] >= best_student_acc:
            best_student_acc = val_eval["top1"]
            best_student_state = student.state_dict().copy()

    if best_student_state is not None:
        student.load_state_dict(best_student_state)

    final_student_eval = evaluate_model(student, val_loader, device)
    print(f"Final Distilled Student Top-1: {final_student_eval['top1']:.4f} (Teacher: {teacher_eval['top1']:.4f})")

    save_path = EXPORTED_MODELS_DIR / "mobilenetv3_small_distilled.pth"
    torch.save(
        {
            "backbone": "mobilenet_v3_small",
            "teacher_backbone": "resnet50",
            "temperature": temperature,
            "alpha": alpha,
            "metrics": final_student_eval,
            "state_dict": student.state_dict(),
        },
        save_path,
    )

    return {
        "teacher_top1": teacher_eval["top1"],
        "student_top1": final_student_eval["top1"],
        "delta_top1": round(final_student_eval["top1"] - teacher_eval["top1"], 4),
        "save_path": str(save_path),
    }


if __name__ == "__main__":
    chk = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    if not chk.exists():
        chk = CHECKPOINTS_DIR / "resnet50_best.pth"
    run_distillation(chk)
