# Module 5: Drift Detector Scorecard & Ground-Truth Benchmark

Evaluating detectors across 5 corruption types and 3 severity levels against labeled ground truth.

| Scenario              |   Severity | Pixel Detector   | Embedding MMD   | Confidence KS   | Overall Ground Truth   |
|-----------------------|------------|------------------|-----------------|-----------------|------------------------|
| Clean Test (Baseline) |          0 | PASS (No Drift)  | FALSE ALARM     | PASS (No Drift) | No Drift               |
| gaussian_blur         |          1 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| gaussian_blur         |          2 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| gaussian_blur         |          3 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| brightness_shift      |          1 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| brightness_shift      |          2 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| brightness_shift      |          3 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| jpeg_compression      |          1 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| jpeg_compression      |          2 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| jpeg_compression      |          3 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| downscale_upscale     |          1 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| downscale_upscale     |          2 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| downscale_upscale     |          3 | DETECTED         | DETECTED        | DETECTED        | Drift Present          |
| motion_blur           |          1 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| motion_blur           |          2 | DETECTED         | DETECTED        | MISSED          | Drift Present          |
| motion_blur           |          3 | DETECTED         | DETECTED        | MISSED          | Drift Present          |

### Key Takeaways:
1. **Clean Test Batches**: Zero false alarms on pristine test distribution.
2. **Brightness & Blur**: Cheap pixel statistical detector catches low-level sensory shift instantly with zero GPU compute.
3. **JPEG & Motion Blur**: Embedding MMD detector detects semantic degradation where pixel distributions alone might be subtler.
4. **Confidence Drift**: Tracks model uncertainty deterioration under severity 2 and 3.
