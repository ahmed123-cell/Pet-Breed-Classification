"""
TensorRT FP16 conversion script and Triton Model Repository configuration with dynamic batching.
"""

import os
from pathlib import Path
from typing import Dict, Optional

from pet_breed.config import EXPORTED_MODELS_DIR, MODELS_DIR


def generate_triton_config(
    model_repo_dir: Path = MODELS_DIR / "triton_repository",
    model_name: str = "pet_breed_tensorrt",
    max_batch_size: int = 64,
) -> Path:
    """
    Generate Triton Inference Server model repository directory structure and config.pbtxt.
    """
    model_dir = model_repo_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    config_content = f"""name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

dynamic_batching {{
  max_queue_delay_microseconds: 5000
}}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 37 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
    config_file = model_dir / "config.pbtxt"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"Generated Triton configuration at {config_file}")
    return config_file


def build_tensorrt_engine_command(
    onnx_path: Path = EXPORTED_MODELS_DIR / "model_fp32.onnx",
    output_engine_path: Path = EXPORTED_MODELS_DIR / "model_fp16.engine",
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 64,
) -> str:
    """
    Generate the trtexec command for building an optimized TensorRT FP16 engine with dynamic shapes.
    """
    output_engine_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f"trtexec --onnx={onnx_path} "
        f"--saveEngine={output_engine_path} "
        f"{'--fp16 ' if fp16 else ''}"
        f"--minShapes=input:{min_batch}x3x224x224 "
        f"--optShapes=input:{opt_batch}x3x224x224 "
        f"--maxShapes=input:{max_batch}x3x224x224"
    )
    return cmd


if __name__ == "__main__":
    generate_triton_config()
    print("TensorRT Build Command:")
    print(build_tensorrt_engine_command())
