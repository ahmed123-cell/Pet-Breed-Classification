# Oxford-IIIT Pet Dataset & Corruption Drift Suite

## Dataset Overview
- **Dataset**: Oxford-IIIT Pet Dataset
- **Classes**: 37 fine-grained pet breeds (12 cat breeds, 25 dog breeds)
- **Total Images**: 7,349 images
- **Distribution**: ~200 images per class, balanced, studio-quality

## Split Strategy
- **Split Ratio**: Deterministic fixed split (seed = 42)
  - `train`: ~80% of trainval (~2,944 images)
  - `val`: ~20% of trainval (~736 images)
  - `test`: Isolated test split (~3,669 images)
- The test partition is strictly kept isolated from training and validation tuning.

## Label Map Schema (`data/labels.json`)
- Exactly 37 alphabetical, sorted classes with deterministic 0-indexed integer identifiers.
- Includes species mapping (`cat` / `dog`).

## Manifest JSON Schema (`data/manifest.json`)
Each record adheres to the following target schema:
```json
{
  "image_id": "Abyssinian_100",
  "path": "data/images/Abyssinian_100.jpg",
  "breed": "Abyssinian",
  "species": "cat",
  "class_index": 0,
  "split": "train",
  "corruption": null,
  "severity": 0,
  "width": 394,
  "height": 500
}
```

## Corruption Suite (`src/pet_breed/data/corruptions.py`)
To model real-world pet photo uploads across Egypt, Saudi Arabia, and the UAE, 5 corruption scenarios are simulated across 3 severity levels:

| Corruption Type | Real-World Scenario | Implementation & Parameters |
| :--- | :--- | :--- |
| **`gaussian_blur`** | Out-of-focus phone cameras, poor autofocus | Gaussian filter (radius: 1.5, 3.0, 5.5) |
| **`brightness_shift`** | Low-light indoor evening vs. harsh midday desert sun | Brightness factor (0.6x, 1.7x, 0.35x) |
| **`jpeg_compression`** | Messaging apps (WhatsApp/Telegram) re-compression | JPEG quality level (Q=40, Q=25, Q=10) |
| **`downscale_upscale`** | Low-resolution legacy handsets | Downscale to (128, 96, 64) px and upscale |
| **`motion_blur`** | Fast moving pets, camera shake | Directional 2D convolution kernel (size: 7, 13, 21) |

## DVC Versioning
The raw images, manifests, and corrupted evaluation sets are tracked using DVC (`dvc.yaml`).
To reproduce data preparation:
```bash
dvc repro
```
