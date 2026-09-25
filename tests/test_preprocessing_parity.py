"""
Test proving zero train/serve preprocessing skew.
Asserts that the training evaluation transform and the API serving preprocessor produce
identical model logits with absolute difference < 1e-4.
"""

import numpy as np
from PIL import Image
import pytest
import torch

from pet_breed.config import DEFAULT_NUM_CLASSES
from pet_breed.models.backbones import get_model
from pet_breed.models.transforms import CanonicalPreprocessor, get_eval_transform


def test_preprocessing_parity_and_logit_agreement():
    """
    Assert training-path eval transform and API CanonicalPreprocessor
    produce logits agreeing to <= 1e-4.
    """
    model = get_model("resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=True)
    model.eval()

    eval_transform = get_eval_transform()
    api_preprocessor = CanonicalPreprocessor()

    # Generate synthetic images across various modes (RGB, RGBA, Grayscale, CMYK)
    test_pil_images = [
        Image.new("RGB", (300, 400), color=(120, 150, 180)),
        Image.new("RGBA", (250, 250), color=(200, 100, 50, 255)),
        Image.new("L", (320, 240), color=128),
        Image.new("CMYK", (280, 360), color=(50, 100, 150, 20)),
    ]

    for idx, raw_img in enumerate(test_pil_images):
        # 1. Training Evaluation Path: convert to RGB then apply get_eval_transform()
        img_rgb = raw_img.convert("RGB")
        train_tensor = eval_transform(img_rgb).unsqueeze(0)

        # 2. Serving API Path: pass directly to CanonicalPreprocessor
        serve_tensor = api_preprocessor.preprocess_pil(raw_img).unsqueeze(0)

        # Assert tensor outputs match exactly
        tensor_diff = torch.max(torch.abs(train_tensor - serve_tensor)).item()
        assert tensor_diff < 1e-6, f"Image {idx}: Preprocessed tensors differ by {tensor_diff}"

        # Assert logits match to < 1e-4
        with torch.inference_mode():
            train_logits = model(train_tensor).numpy()
            serve_logits = model(serve_tensor).numpy()

        max_logit_diff = float(np.max(np.abs(train_logits - serve_logits)))
        assert (
            max_logit_diff < 1e-4
        ), f"Image {idx}: Logits differ by {max_logit_diff:.6e} (exceeds 1e-4 tolerance)"
        print(f"Image {idx} ({raw_img.mode}) Logit Agreement: max diff = {max_logit_diff:.6e} (PASS)")
