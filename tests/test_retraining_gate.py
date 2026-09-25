"""
Unit tests for the production quality gate logic.
"""

import pytest
from pet_breed.retraining.quality_gate import evaluate_quality_gate


def test_quality_gate_promotes_meeting_threshold():
    """Verify quality gate promotes candidate if top1 >= prod - 0.01."""
    prod_top1 = 0.9140
    
    # Exactly equal -> Pass
    passed, res = evaluate_quality_gate(candidate_top1=0.9140, production_top1=prod_top1)
    assert passed
    assert res["decision"] == "PROMOTE_TO_PRODUCTION"

    # Slightly higher -> Pass
    passed, res = evaluate_quality_gate(candidate_top1=0.9200, production_top1=prod_top1)
    assert passed

    # Within tolerance (0.9140 - 0.01 = 0.9040) -> Pass
    passed, res = evaluate_quality_gate(candidate_top1=0.9060, production_top1=prod_top1)
    assert passed


def test_quality_gate_rejects_below_threshold():
    """Verify quality gate rejects candidate if top1 < prod - 0.01."""
    prod_top1 = 0.9140

    # Below tolerance (e.g. 0.8900 < 0.9040) -> Fail / Reject
    passed, res = evaluate_quality_gate(candidate_top1=0.8900, production_top1=prod_top1)
    assert not passed
    assert res["decision"] == "REJECT_CANDIDATE"
