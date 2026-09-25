import sys
import os
from pathlib import Path
from typing import Dict, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from pet_breed.config import (
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
    IMAGE_SIZE,
)
from pet_breed.models.backbones import get_model


def export_to_onnx(
    model: nn.Module,
    output_path: Path = EXPORTED_MODELS_DIR / "model_fp32.onnx",
    input_shape: Tuple[int, int, int, int] = (1, 3, IMAGE_SIZE, IMAGE_SIZE),
    dynamic_axes: bool = True,
) -> Path:
    """
    Export PyTorch model to standard ONNX FP32 format with dynamic batching.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape, requires_grad=False)
    
    dyn_axes_config = (
        {"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
        if dynamic_axes
        else None
    )

    print(f"Exporting model to ONNX at {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dyn_axes_config,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model successfully exported and checked: {output_path}")
    return output_path


def verify_onnx_agreement(
    pytorch_model: nn.Module,
    onnx_path: Path,
    num_samples: int = 200,
    tolerance: float = 1e-4,
) -> Dict[str, any]:
    """
    Verify that PyTorch model and ONNX Runtime produce logits agreeing to within tolerance (1e-4).
    """
    pytorch_model.eval()
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    passed = True

    for _ in range(num_samples):
        # Generate random input representing normalized image
        test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        
        # PyTorch logits
        with torch.inference_mode():
            torch_logits = pytorch_model(test_input).cpu().numpy()

        # ONNX Runtime logits
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_logits = ort_session.run(None, ort_inputs)[0]

        diff = np.max(np.abs(torch_logits - ort_logits))
        if diff > max_diff:
            max_diff = float(diff)

        if diff > tolerance:
            passed = False

    print(f"Logit Agreement over {num_samples} samples: Max Diff = {max_diff:.6e} (Tolerance: {tolerance}) -> Passed: {passed}")
    return {
        "num_samples": num_samples,
        "max_diff": max_diff,
        "tolerance": tolerance,
        "passed": passed,
    }


if __name__ == "__main__":
    from pet_breed.models.backbones import get_model
    model = get_model(backbone_name="resnet50", num_classes=DEFAULT_NUM_CLASSES, pretrained=True)
    out_path = export_to_onnx(model)
    verify_onnx_agreement(model, out_path, num_samples=50)
