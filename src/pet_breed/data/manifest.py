"""
Manifest generator creating validated JSON records for train/val/test and corrupted splits.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from tqdm import tqdm

from pet_breed.config import (
    ANNOTATIONS_DIR,
    CORRUPTED_DIR,
    DEFAULT_SEED,
    IMAGES_DIR,
    LABELS_PATH,
    MANIFEST_PATH,
    get_class_names,
    get_species_map,
)
from pet_breed.data.corruptions import (
    CORRUPTION_TYPES,
    apply_corruption,
)


def _get_breed_from_filename(filename: str) -> str:
    """
    Extract standardized breed name from Oxford-IIIT Pet image filename.
    Filenames are formatted as: 'Abyssinian_100.jpg' or 'american_bulldog_101.jpg'
    """
    stem = Path(filename).stem
    # Split from right on last underscore followed by numbers
    parts = stem.rsplit("_", 1)
    raw_breed = parts[0]
    
    # Map raw_breed string to canonical class name from labels.json
    classes = get_class_names()
    # Normalize comparison by removing underscores and lowercase
    norm_map = {c.lower().replace(" ", "_"): c for c in classes}
    
    if raw_breed in norm_map:
        return norm_map[raw_breed]
    if raw_breed.lower() in norm_map:
        return norm_map[raw_breed.lower()]
    
    # Direct match fallback
    for c in classes:
        if c.lower().replace(" ", "") == raw_breed.lower().replace("_", ""):
            return c
            
    raise ValueError(f"Could not map filename stem '{stem}' (raw breed: '{raw_breed}') to a known class.")


def build_manifest(
    val_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
    generate_corrupted_test: bool = True,
) -> List[Dict]:
    """
    Build complete dataset manifest with train, val, test, and corrupted drift sets.
    """
    random.seed(seed)
    class_names = get_class_names()
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    species_map = get_species_map()

    # Read original trainval.txt and test.txt if annotations exist, or partition cleanly
    trainval_list_file = ANNOTATIONS_DIR / "trainval.txt"
    test_list_file = ANNOTATIONS_DIR / "test.txt"

    trainval_stems = set()
    test_stems = set()

    if trainval_list_file.exists() and test_list_file.exists():
        with open(trainval_list_file, "r") as f:
            for line in f:
                if line.strip():
                    trainval_stems.add(line.strip().split()[0])
        with open(test_list_file, "r") as f:
            for line in f:
                if line.strip():
                    test_stems.add(line.strip().split()[0])
    else:
        # Fallback if text files aren't present: deterministic hash partition
        all_imgs = sorted(list(IMAGES_DIR.glob("*.jpg")))
        for img in all_imgs:
            stem = img.stem
            if random.random() < 0.5:
                trainval_stems.add(stem)
            else:
                test_stems.add(stem)

    # Carve validation fold out of trainval with fixed seed
    sorted_trainval = sorted(list(trainval_stems))
    random.shuffle(sorted_trainval)
    val_count = int(len(sorted_trainval) * val_ratio)
    val_stems = set(sorted_trainval[:val_count])
    train_stems = set(sorted_trainval[val_count:])

    records = []

    # Process clean images
    all_images = sorted(list(IMAGES_DIR.glob("*.jpg")))
    print(f"Generating manifest records for {len(all_images)} clean images...")

    for img_path in tqdm(all_images, desc="Clean manifest"):
        stem = img_path.stem
        try:
            breed = _get_breed_from_filename(stem)
        except ValueError:
            continue

        if stem in train_stems:
            split = "train"
        elif stem in val_stems:
            split = "val"
        elif stem in test_stems:
            split = "test"
        else:
            split = "train"

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        record = {
            "image_id": stem,
            "path": str(img_path.relative_to(MANIFEST_PATH.parent.parent)).replace("\\", "/"),
            "breed": breed,
            "species": species_map.get(breed, "unknown"),
            "class_index": class_to_idx[breed],
            "split": split,
            "corruption": None,
            "severity": 0,
            "width": w,
            "height": h,
        }
        records.append(record)

    # Generate corrupted test sets
    if generate_corrupted_test:
        print("Generating corrupted test sets for drift evaluation...")
        CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
        test_records = [r for r in records if r["split"] == "test"]

        for corr_type in CORRUPTION_TYPES:
            for severity in [1, 2, 3]:
                corr_subfolder = CORRUPTED_DIR / f"{corr_type}_s{severity}"
                corr_subfolder.mkdir(parents=True, exist_ok=True)

                for r in tqdm(
                    test_records, desc=f"Corrupting {corr_type} (severity {severity})"
                ):
                    orig_img_path = MANIFEST_PATH.parent.parent / r["path"]
                    dest_path = corr_subfolder / f"{r['image_id']}.jpg"

                    if not dest_path.exists():
                        try:
                            with Image.open(orig_img_path) as img:
                                corrupted_img = apply_corruption(
                                    img, corr_type, severity
                                )
                                corrupted_img.save(dest_path, "JPEG", quality=90)
                        except Exception as e:
                            print(f"Failed to corrupt {orig_img_path}: {e}")
                            continue

                    corr_record = {
                        "image_id": f"{r['image_id']}_{corr_type}_s{severity}",
                        "path": str(
                            dest_path.relative_to(MANIFEST_PATH.parent.parent)
                        ).replace("\\", "/"),
                        "breed": r["breed"],
                        "species": r["species"],
                        "class_index": r["class_index"],
                        "split": "test_corrupted",
                        "corruption": corr_type,
                        "severity": severity,
                        "width": r["width"],
                        "height": r["height"],
                    }
                    records.append(corr_record)

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Manifest saved to {MANIFEST_PATH} with {len(records)} total records.")
    return records


if __name__ == "__main__":
    build_manifest()
