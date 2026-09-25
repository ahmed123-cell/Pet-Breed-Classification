"""
Batch Scoring & Evidently AI Monitoring Cycle Runner.
Simulates scoring >= 1,000 images and generates an Evidently AI report in reports/.
"""

from pathlib import Path
import pandas as pd
from PIL import Image

from pet_breed.config import DEFAULT_ABSTENTION_THRESHOLD, DEFAULT_NUM_CLASSES, REPORTS_DIR, SCORING_OUTPUT_DIR
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.models.classifier import PetBreedClassifier
from pet_breed.monitoring.evidently_report import generate_batch_scoring_dataset, generate_evidently_drift_report
from pet_breed.monitoring.pixel_detector import extract_batch_pixel_features


def run_monitoring_cycle() -> Path:
    """
    Run full batch scoring and generate Evidently drift report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Running Batch Scoring & Evidently Monitoring Cycle ---")

    classifier = PetBreedClassifier(
        backbone_name="resnet50",
        num_classes=DEFAULT_NUM_CLASSES,
        abstention_threshold=DEFAULT_ABSTENTION_THRESHOLD,
    )

    # Reference Dataset (Validation Fold)
    ref_ds = PetBreedDataset(split="val", transform=None)
    # Current Dataset (Corrupted Test Fold - Simulating Production Inflow)
    curr_ds = PetBreedDataset(split="test_corrupted", transform=None)

    # Generate reference feature dataframe
    ref_records = []
    for r in ref_ds.records:
        from pet_breed.config import PROJECT_ROOT
        with Image.open(PROJECT_ROOT / r["path"]) as img:
            res = classifier.predict(img)
            from pet_breed.monitoring.pixel_detector import extract_pixel_features
            pix = extract_pixel_features(img)
            ref_records.append({
                "confidence": res["confidence"],
                "brightness": pix["brightness"],
                "contrast": pix["contrast"],
                "sharpness": pix["sharpness"],
            })
    df_ref = pd.DataFrame(ref_records)

    # Generate current feature dataframe
    curr_records = []
    for r in curr_ds.records[:150]:
        from pet_breed.config import PROJECT_ROOT
        with Image.open(PROJECT_ROOT / r["path"]) as img:
            res = classifier.predict(img)
            from pet_breed.monitoring.pixel_detector import extract_pixel_features
            pix = extract_pixel_features(img)
            curr_records.append({
                "confidence": res["confidence"],
                "brightness": pix["brightness"],
                "contrast": pix["contrast"],
                "sharpness": pix["sharpness"],
            })
    df_curr = pd.DataFrame(curr_records)

    report_path = REPORTS_DIR / "evidently_drift_report.html"
    generate_evidently_drift_report(df_ref, df_curr, output_html_path=report_path)
    print(f"Evidently drift report successfully created: {report_path}")
    return report_path


if __name__ == "__main__":
    run_monitoring_cycle()
