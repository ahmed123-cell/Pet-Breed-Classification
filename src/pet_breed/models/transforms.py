"""
Canonical preprocessing pipelines for training, evaluation, and production serving.
Ensures zero train/serve preprocessing skew.
"""

from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as T

from pet_breed.config import (
    CROP_PCT,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transform(
    image_size: int = IMAGE_SIZE,
    crop_pct: float = CROP_PCT,
) -> T.Compose:
    """
    Training transform: includes data augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter).
    Only used during training.
    """
    return T.Compose([
        T.RandomResizedCrop(
            image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(
    image_size: int = IMAGE_SIZE,
    crop_pct: float = CROP_PCT,
) -> T.Compose:
    """
    Canonical evaluation & serving transform:
    Deterministic Resize -> CenterCrop -> ToTensor -> Normalize.
    Used identically in offline evaluation, batch scoring, and production API serving.
    """
    resize_dim = int(image_size / crop_pct)  # 224 / 0.875 = 256
    return T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CanonicalPreprocessor:
    """
    Callable preprocessor wrapping the canonical eval transform for production deployment.
    Accepts PIL Images (RGB, RGBA, CMYK, Grayscale) or file buffers.
    """

    def __init__(self, image_size: int = IMAGE_SIZE, crop_pct: float = CROP_PCT):
        self.transform = get_eval_transform(image_size=image_size, crop_pct=crop_pct)

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Convert any PIL image mode to RGB and apply standard eval transform."""
        rgb_image = image.convert("RGB")
        return self.transform(rgb_image)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess a list of PIL Images into a batched Tensor (B, C, H, W)."""
        tensors = [self.preprocess_pil(img) for img in images]
        return torch.stack(tensors, dim=0)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_pil(image)
