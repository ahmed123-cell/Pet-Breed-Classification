"""
OpenVINO IR model conversion and optimized CPU inference runner.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import openvino as ov
import torch
import torch.nn as nn

from pet_breed.config import EXPORTED_MODELS_DIR, IMAGE_SIZE


def convert_onnx_to_openvino(
    onnx_path: Path = EXPORTED_MODELS_DIR / "model_fp32.onnx",
    output_dir: Path = EXPORTED_MODELS_DIR / "openvino",
) -> Path:
    """
    Convert ONNX model to OpenVINO Intermediate Representation (IR: .xml and .bin).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "model.xml"

    print(f"Converting {onnx_path} to OpenVINO IR at {xml_path}...")
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))

    print(f"OpenVINO IR model exported successfully to {xml_path}")
    return xml_path


class OpenVINOInferenceRunner:
    """
    High-performance CPU inference runner using Intel OpenVINO runtime.
    """

    def __init__(self, xml_path: Path = EXPORTED_MODELS_DIR / "openvino" / "model.xml"):
        self.xml_path = Path(xml_path)
        self.core = ov.Core()
        self.model = self.core.read_model(str(self.xml_path))
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = self.compiled_model.input(0).get_any_name()
        self.output_tensor_name = self.compiled_model.output(0).get_any_name()

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed batch/single tensor (B, C, H, W).
        Returns raw output logits.
        """
        if isinstance(input_array, torch.Tensor):
            input_array = input_array.cpu().numpy()
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)

        results = self.infer_request.infer({self.input_tensor_name: input_array})
        return list(results.values())[0]


if __name__ == "__main__":
    convert_onnx_to_openvino()
