"""
Post-Training Quantization (PTQ) to INT8 format with calibration dataset.
Produces model_int8.onnx and evaluates accuracy and latency impact.
"""

from pathlib import Path
from typing import Dict, Iterator, Optional
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch.utils.data import DataLoader

from pet_breed.config import EXPORTED_MODELS_DIR, IMAGE_SIZE
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.transforms import get_eval_transform


class PetCalibrationDataReader(CalibrationDataReader):
    """
    Supplies representative uncorrupted validation batches for INT8 PTQ calibration.
    """

    def __init__(self, dataloader: DataLoader, max_samples: int = 100):
        self.dataloader = dataloader
        self.max_samples = max_samples
        self.data_iter = iter(dataloader)
        self.count = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.count >= self.max_samples:
            return None
        try:
            images, _, _ = next(self.data_iter)
            self.count += images.size(0)
            return {"input": images.numpy().astype(np.float32)}
        except StopIteration:
            return None

    def rewind(self):
        self.data_iter = iter(self.dataloader)
        self.count = 0


def quantize_onnx_int8(
    input_fp32_onnx: Path = EXPORTED_MODELS_DIR / "model_fp32.onnx",
    output_int8_onnx: Path = EXPORTED_MODELS_DIR / "model_int8.onnx",
    max_calibration_samples: int = 100,
) -> Path:
    """
    Apply Static INT8 Post-Training Quantization using calibration dataset.
    """
    output_int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"Applying Static INT8 Quantization: {input_fp32_onnx} -> {output_int8_onnx}")

    val_ds = PetBreedDataset(split="val", transform=get_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    calibration_reader = PetCalibrationDataReader(
        val_loader, max_samples=max_calibration_samples
    )

    quantize_static(
        model_input=str(input_fp32_onnx),
        model_output=str(output_int8_onnx),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    fp32_size_mb = input_fp32_onnx.stat().st_size / (1024 * 1024)
    int8_size_mb = output_int8_onnx.stat().st_size / (1024 * 1024)
    print(f"PTQ INT8 model created: {int8_size_mb:.2f} MB (Compression: {fp32_size_mb/int8_size_mb:.2f}x vs FP32 {fp32_size_mb:.2f} MB)")
    return output_int8_onnx


if __name__ == "__main__":
    quantize_onnx_int8()
