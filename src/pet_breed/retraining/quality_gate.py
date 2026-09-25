"""
Production Model Quality Gate.
Asserts that retrained candidate models satisfy Top-1 >= Production - 0.01 before promotion.
"""

from typing import Dict, Tuple
from pet_breed.training.mlflow_tracker import promote_model_to_production


def evaluate_quality_gate(
    candidate_top1: float,
    production_top1: float,
    tolerance: float = 0.01,
) -> Tuple[bool, Dict[str, any]]:
    """
    Evaluate candidate model performance against current production baseline.
    Rule: Promoted only if candidate_top1 >= production_top1 - tolerance
    """
    threshold = production_top1 - tolerance
    passed = candidate_top1 >= threshold
    delta = candidate_top1 - production_top1

    result = {
        "candidate_top1": round(candidate_top1, 4),
        "production_top1": round(production_top1, 4),
        "threshold": round(threshold, 4),
        "delta": round(delta, 4),
        "passed": bool(passed),
        "decision": "PROMOTE_TO_PRODUCTION" if passed else "REJECT_CANDIDATE",
    }

    if passed:
        print(f"[QUALITY GATE PASS] Candidate Top-1 ({candidate_top1:.4f}) >= Threshold ({threshold:.4f}). Promoting.")
        try:
            promote_model_to_production(model_name="PetBreedClassifier", stage="Production")
        except Exception as e:
            print(f"MLflow registry update note: {e}")
    else:
        print(f"[QUALITY GATE FAIL] Candidate Top-1 ({candidate_top1:.4f}) < Threshold ({threshold:.4f}). Retaining current production model.")

    return passed, result
