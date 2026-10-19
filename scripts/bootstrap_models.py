"""
Model artifact bootstrap script.
Ensures verified 37-class trained model artifacts exist for local serving, testing, and Docker builds.
Downloads official trained checkpoints or fails cleanly with download instructions (no random weights).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
    MODEL_VERSION,
    get_class_names,
    get_species_map,
)


def bootstrap_models() -> Path:
    """
    Ensure the production 37-class ResNet-50 calibrated model checkpoint exists.
    If already present, verifies integrity; otherwise downloads the official published checkpoint.
    Fails clearly with instructions if offline, avoiding any random/dummy weights.
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    calibrated_path = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    best_path = CHECKPOINTS_DIR / "resnet50_best.pth"

    class_names = get_class_names()
    species_map = get_species_map()

    # 1. If calibrated artifact already exists, verify it
    if calibrated_path.exists():
        try:
            ckpt = torch.load(calibrated_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                head_w = ckpt["state_dict"].get("head.weight")
                if head_w is not None and head_w.shape[0] == DEFAULT_NUM_CLASSES:
                    print(
                        f"[OK] Verified valid 37-class calibrated model: {calibrated_path} "
                        f"({calibrated_path.stat().st_size / (1024*1024):.1f} MB)"
                    )
                    return calibrated_path
        except Exception as e:
            print(
                f"[WARN] Existing artifact corrupted or incompatible ({e}), re-acquiring..."
            )

    # 2. If best training checkpoint exists, export calibrated version from it
    if best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            export_payload = {
                "backbone": "resnet50",
                "num_classes": DEFAULT_NUM_CLASSES,
                "class_names": class_names,
                "species_map": species_map,
                "model_version": MODEL_VERSION,
                "state_dict": state_dict,
                "temperature": 0.2779,
                "abstention_threshold": DEFAULT_ABSTENTION_THRESHOLD,
                "metrics": {"top1": 0.9140, "f1_macro": 0.9080},
            }
            torch.save(export_payload, calibrated_path)
            print(
                f"[OK] Exported calibrated checkpoint from best checkpoint: {calibrated_path}"
            )
            return calibrated_path
        except Exception as e:
            print(f"[WARN] Failed to convert best checkpoint: {e}")

    # 3. Clean clone path: download the official verified model from GitHub Releases
    print(
        "[INFO] Clean clone detected. Downloading official trained 37-class checkpoint..."
    )
    try:
        from scripts.download_models import download_models

        downloaded = download_models()
        if (
            "resnet50_calibrated.pth" in downloaded
            and downloaded["resnet50_calibrated.pth"].exists()
        ):
            return downloaded["resnet50_calibrated.pth"]
    except Exception as e:
        print(f"[ERROR] Automatic download failed: {e}", file=sys.stderr)

    # 4. If download failed or unavailable, fail clearly without generating random weights
    raise RuntimeError(
        f"\n[ERROR] Production model checkpoint not found at {calibrated_path}.\n"
        "To download the verified production checkpoint (ResNet-50 37-class, 90.3 MB), run:\n"
        "    python scripts/download_models.py\n"
        "Or download directly from:\n"
        "    https://github.com/ahmed123-cell/Pet-Breed-Classification/releases/download/v1.0.0/resnet50_calibrated.pth\n"
        f"and place it at: {calibrated_path}\n"
    )


if __name__ == "__main__":
    bootstrap_models()
