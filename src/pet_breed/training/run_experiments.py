"""
Multi-backbone MLflow Experiment Suite Runner.
Executes >= 5 runs across ResNet-50, ResNet-18, and MobileNetV3-Small,
logging hyperparameters, metrics, calibration plots, and registering PetBreedClassifier.
"""

from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from pet_breed.config import REPORTS_DIR
from pet_breed.training.train import run_training_experiment
from pet_breed.training.mlflow_tracker import promote_model_to_production


def run_all_experiments() -> List[Dict]:
    """
    Execute 5 MLflow training runs across 3 backbones with varied learning rates and architectures.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=======================================================")
    print(" Running Multi-Backbone MLflow Experimentation Suite")
    print("=======================================================\n")

    experiments = [
        {"backbone": "resnet50", "lr": 1e-4, "batch_size": 16, "epochs": 2, "register": "PetBreedClassifier"},
        {"backbone": "resnet50", "lr": 5e-5, "batch_size": 16, "epochs": 2, "register": None},
        {"backbone": "resnet18", "lr": 1e-4, "batch_size": 16, "epochs": 2, "register": None},
        {"backbone": "resnet18", "lr": 2e-4, "batch_size": 16, "epochs": 2, "register": None},
        {"backbone": "mobilenet_v3_small", "lr": 3e-4, "batch_size": 16, "epochs": 2, "register": None},
    ]

    results = []
    for exp in experiments:
        res = run_training_experiment(
            backbone=exp["backbone"],
            lr=exp["lr"],
            batch_size=exp["batch_size"],
            epochs=exp["epochs"],
            register_model_name=exp["register"],
        )
        results.append({
            "run_id": res["run_id"][:8],
            "backbone": exp["backbone"],
            "lr": exp["lr"],
            "batch_size": exp["batch_size"],
            "top1": res["metrics"]["top1"],
            "f1_macro": res["metrics"]["f1_macro"],
            "ece_calibrated": res["metrics"]["ece"],
            "temperature": res["metrics"]["temperature"],
            "coverage": res["metrics"]["coverage"],
            "selective_acc": res["metrics"]["selective_accuracy"],
        })

    # Promote best model in MLflow Registry
    try:
        promote_model_to_production(model_name="PetBreedClassifier", stage="Production")
    except Exception as e:
        print(f"Registry promotion note: {e}")

    # Generate MLflow Comparison Summary Image & Markdown
    df = pd.DataFrame(results)
    
    # Plot MLflow Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    run_labels = [f"{r['backbone']}\nlr={r['lr']}" for r in results]
    top1_vals = [r["top1"] for r in results]
    f1_vals = [r["f1_macro"] for r in results]
    
    x = range(len(results))
    width = 0.35
    ax.bar([i - width/2 for i in x], top1_vals, width=width, label="Top-1 Accuracy", color="#3498db")
    ax.bar([i + width/2 for i in x], f1_vals, width=width, label="Macro F1", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylim([0.0, 1.05])
    ax.set_title("MLflow Multi-Backbone Experiment Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    
    mlflow_comp_path = REPORTS_DIR / "mlflow_comparison.png"
    plt.tight_layout()
    fig.savefig(mlflow_comp_path, dpi=300)
    plt.close(fig)
    print(f"MLflow comparison plot saved to {mlflow_comp_path}")

    # Print markdown table
    headers = ["Run ID", "Backbone", "LR", "Batch Size", "Top-1", "Macro F1", "ECE (Calib)", "T", "Coverage", "Selective Acc"]
    table_rows = [
        [r["run_id"], r["backbone"], r["lr"], r["batch_size"], r["top1"], r["f1_macro"], r["ece_calibrated"], r["temperature"], r["coverage"], r["selective_acc"]]
        for r in results
    ]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github") + "\n")
    return results


if __name__ == "__main__":
    run_all_experiments()
