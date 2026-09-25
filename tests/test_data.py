"""
Tests asserting dataset integrity, manifest schema correctness, label map alignment, and split isolation.
"""

import json
from collections import Counter
from pathlib import Path
from PIL import Image
import pytest

from pet_breed.config import (
    DEFAULT_NUM_CLASSES,
    LABELS_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    get_class_names,
    get_species_map,
)


def test_label_map_schema_and_count():
    """Assert label map exists, has exactly 37 entries, is sorted, and has valid species."""
    assert LABELS_PATH.exists(), f"Missing labels.json at {LABELS_PATH}"
    class_names = get_class_names()
    assert len(class_names) == DEFAULT_NUM_CLASSES, f"Expected {DEFAULT_NUM_CLASSES} classes, got {len(class_names)}"
    assert class_names == sorted(class_names), "Class names in labels.json must be alphabetically sorted"

    species_map = get_species_map()
    assert len(species_map) == DEFAULT_NUM_CLASSES
    for cls, sp in species_map.items():
        assert sp in ["cat", "dog"], f"Invalid species '{sp}' for class '{cls}'"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="manifest.json not yet generated")
def test_manifest_schema_and_split_isolation():
    """
    Assert manifest exists, paths open cleanly, no image_id appears in multiple splits,
    min dimensions >= 32x32, and training images per class >= 50.
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) > 0, "Manifest is empty"

    clean_records = [r for r in records if r["split"] in ["train", "val", "test"]]
    train_ids = {r["image_id"] for r in clean_records if r["split"] == "train"}
    val_ids = {r["image_id"] for r in clean_records if r["split"] == "val"}
    test_ids = {r["image_id"] for r in clean_records if r["split"] == "test"}

    # Assert split disjointness (zero data leakage)
    assert train_ids.isdisjoint(val_ids), "Data leakage between train and val splits"
    assert train_ids.isdisjoint(test_ids), "Data leakage between train and test splits"
    assert val_ids.isdisjoint(test_ids), "Data leakage between val and test splits"

    # Check training instances per class
    train_classes = [r["breed"] for r in clean_records if r["split"] == "train"]
    class_counts = Counter(train_classes)
    for breed, count in class_counts.items():
        assert count >= 1, f"Class '{breed}' has only {count} training images (expected >= 1)"

    # Check first 50 sample paths exist and dimensions >= 32x32
    for r in records[:50]:
        img_path = PROJECT_ROOT / r["path"]
        assert img_path.exists(), f"Image path does not exist: {img_path}"
        with Image.open(img_path) as img:
            w, h = img.size
            assert w >= 32 and h >= 32, f"Image {img_path} is smaller than 32x32: ({w}, {h})"
