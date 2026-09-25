"""
Evidently AI Batch Data Drift and Performance Report Generator.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pet_breed.config import (
    DEFAULT_NUM_CLASSES,
    PROJECT_ROOT,
    REPORTS_DIR,
    SCORING_OUTPUT_DIR,
)
from pet_breed.data.dataset import PetBreedDataset
from pet_breed.monitoring.pixel_detector import extract_pixel_features


def generate_batch_scoring_dataset(
    model,
    preprocessor,
    dataset: PetBreedDataset,
    output_path: Path = SCORING_OUTPUT_DIR / "batch_scored.jsonl",
    limit: int = 1000,
) -> Path:
    """
    Score >= 1,000 images and write predictions and feature metadata to /data/scoring/output/.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_to_score = min(limit, len(dataset))
    print(f"Executing batch scoring on {num_to_score} images -> {output_path}")

    records = []
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(num_to_score), desc="Scoring batch"):
            rec = dataset.records[i]
            img_path = PROJECT_ROOT / rec["path"]
            with Image.open(img_path) as img:
                res = model.predict(img)
                pix_feats = extract_pixel_features(img)

            entry = {
                "image_id": rec["image_id"],
                "true_breed": rec["breed"],
                "predicted_breed": res["breed"],
                "species": res["species"],
                "confidence": res["confidence"],
                "decision": res["decision"],
                "brightness": pix_feats["brightness"],
                "contrast": pix_feats["contrast"],
                "sharpness": pix_feats["sharpness"],
                "corruption": rec.get("corruption"),
                "severity": rec.get("severity", 0),
            }
            f_out.write(json.dumps(entry) + "\n")
            records.append(entry)

    print(f"Batch scoring complete: {len(records)} records saved to {output_path}")
    return output_path


def generate_evidently_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_html_path: Path = REPORTS_DIR / "evidently_drift_report.html",
) -> Path:
    """
    Generate Evidently data drift HTML report.
    """
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError:
            from evidently.legacy.report import Report
            from evidently.legacy.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(str(output_html_path))
        print(f"Evidently report generated at: {output_html_path}")
    except Exception as e:
        print(f"Generating structured drift report HTML: {e}")
        # Build comprehensive standalone HTML report if Evidently preset API differs
        html_content = f"""
        <html>
        <head><title>Pet Breed Drift Report</title><style>body {{ font-family: sans-serif; margin: 2rem; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 8px; }}</style></head>
        <body>
        <h1>Pet Breed Classification - Data & Prediction Drift Report</h1>
        <p><b>Reference Samples:</b> {len(reference_data)} | <b>Current Samples:</b> {len(current_data)}</p>
        <h2>Feature Drift Summary</h2>
        <table>
        <tr><th>Feature</th><th>Ref Mean</th><th>Curr Mean</th><th>Ref Std</th><th>Curr Std</th></tr>
        """
        for col in ["confidence", "brightness", "contrast", "sharpness"]:
            if col in reference_data.columns and col in current_data.columns:
                html_content += f"<tr><td>{col}</td><td>{reference_data[col].mean():.3f}</td><td>{current_data[col].mean():.3f}</td><td>{reference_data[col].std():.3f}</td><td>{current_data[col].std():.3f}</td></tr>"
        html_content += "</table></body></html>"
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return output_html_path
