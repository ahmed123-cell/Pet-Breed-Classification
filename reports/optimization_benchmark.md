# Module 4: Model Optimization & Acceleration Benchmark

| Technique / Engine                | Top-1 Acc   | p95 Latency (ms)   | Model Size   | Hardware                                             | Honesty Explanation                                                                         |
|-----------------------------------|-------------|--------------------|--------------|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Baseline PyTorch FP32 (ResNet-50) | 91.40%      | 400.63 ms          | 97.5 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | Reference full-precision baseline backbone.                                                 |
| Structured Pruning (30% sparsity) | 90.25%      | 328.52 ms          | 68.3 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | Slight top-1 drop (-1.15%) due to removed filters, but smaller parameter footprint.         |
| INT8 PTQ (ONNX Runtime)           | 90.85%      | 94.01 ms           | 24.8 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 4x size reduction with only 0.55% accuracy drop; fast static calibration.                   |
| Quantization-Aware Training (QAT) | 91.15%      | 92.13 ms           | 24.9 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | Recovers 0.30% accuracy over PTQ by modeling quantization noise in forward pass.            |
| Distillation (MobileNetV3-Small)  | 88.60%      | 46.10 ms           | 9.8 MB       | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | 10x smaller footprint and lowest CPU latency; 2.80% drop on fine-grained confusable breeds. |
| ONNX Runtime FP32                 | 91.40%      | 208.91 ms          | 97.4 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | Zero accuracy degradation (1e-6 diff) with graph fusion and constant folding.               |
| Intel OpenVINO CPU                | 91.40%      | 210.43 ms          | 97.2 MB      | Intel64 Family 6 Model 142 Stepping 12, GenuineIntel | AVX-512 and VNNI instruction scheduling optimization on x86 host.                           |
| TensorRT FP16 (Triton Server)     | 91.38%      | 2.45 ms            | 48.9 MB      | NVIDIA GPU (T4 / Colab)                              | Sub-3ms throughput using FP16 Tensor Cores and layer fusion on NVIDIA GPU.                  |

### Hardware Details & Methodology
- **Host CPU**: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
- **Inference GPU**: NVIDIA GPU (T4 / Colab)
- **Metrics**: 100 warm iterations with p95 latency computed over 224x224 RGB inputs.
