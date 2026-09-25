"""
PyTorch Dataset implementation consuming the JSON manifest.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from pet_breed.config import MANIFEST_PATH, PROJECT_ROOT


class PetBreedDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset loaded from manifest.json.
    Ensures safe RGB conversion for Grayscale, CMYK, and 4-channel RGBA images.
    """

    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        split: str = "train",
        corruption: Optional[str] = None,
        severity: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        # Filter by split and optional corruption/severity
        self.records: List[Dict] = []
        for r in all_records:
            if split == "all" or r["split"] == split:
                if corruption is not None and r.get("corruption") != corruption:
                    continue
                if severity is not None and r.get("severity") != severity:
                    continue
                self.records.append(r)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self.records[idx]
        img_full_path = PROJECT_ROOT / record["path"]

        # Ensure robust loading across RGB, CMYK, Grayscale, RGBA
        with Image.open(img_full_path) as raw_img:
            img = raw_img.convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            from torchvision.transforms import functional as TF
            img_tensor = TF.to_tensor(img)

        target = record["class_index"]
        image_id = record["image_id"]

        return img_tensor, target, image_id
