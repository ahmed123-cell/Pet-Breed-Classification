"""
Central configuration for Pet Breed Classification MLOps System.
"""

from pathlib import Path
from typing import List, Dict
import json

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CORRUPTED_DIR = DATA_DIR / "corrupted"
MANIFEST_PATH = DATA_DIR / "manifest.json"
LABELS_PATH = DATA_DIR / "labels.json"
SCORING_OUTPUT_DIR = DATA_DIR / "scoring" / "output"

# Model & Artifact Paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
CROP_PCT = 0.875  # Standard center crop proportion: 224 / 256 = 0.875

# Default training hyperparameters
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_CLASSES = 37
DEFAULT_LEARNING_RATE = 1e-4

# Default serving & calibration settings
DEFAULT_ABSTENTION_THRESHOLD = 0.65
MODEL_VERSION = "v1.0.0"


def load_label_map() -> Dict[str, any]:
    """Load sorted classes and species mapping from committed labels.json."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Label map not found at {LABELS_PATH}")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_names() -> List[str]:
    """Get the 37 sorted class names."""
    data = load_label_map()
    return data["classes"]


def get_species_map() -> Dict[str, str]:
    """Get mapping from class name to 'cat' or 'dog'."""
    data = load_label_map()
    return data["species_map"]
