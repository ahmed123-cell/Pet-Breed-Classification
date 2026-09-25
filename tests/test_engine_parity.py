"""
Test asserting PyTorch and ONNX Runtime logits agree to within 1e-4 on 200 images.
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest
import torch

from pet_breed.config import DEFAULT_NUM_CLASSES, EXPORTED_MODELS_DIR, IMAGE_SIZE
from pet_breed.models.backbones import get_model
from pet_breed.optimization.onnx_export import export_to_onnx, verify_onnx_agreement


def test_pytorch_onnx_numerical_agreement(tmp_path: Path):
    """
    Assert PyTorch and ONNX Runtime logits agree to within 1e-4 on 200 test samples.
    """
    model = get_model("resnet18", num_classes=DEFAULT_NUM_CLASSES, pretrained=False)
    model.eval()

    onnx_path = tmp_path / "test_model.onnx"
    export_to_onnx(model, output_path=onnx_path)

    result = verify_onnx_agreement(
        pytorch_model=model,
        onnx_path=onnx_path,
        num_samples=200,
        tolerance=1e-4,
    )

    assert result["passed"], f"ONNX agreement failed: max diff = {result['max_diff']}"
    assert result["max_diff"] < 1e-4
