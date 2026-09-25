"""
Pydantic schemas for API requests, responses, and health status.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    breed: str = Field(..., description="Pet breed name")
    species: str = Field(..., description="'cat' or 'dog'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence probability")


class PredictionResponse(BaseModel):
    breed: str = Field(..., description="Top-1 predicted breed")
    species: str = Field(..., description="Species: 'cat' or 'dog'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated top-1 confidence score")
    top_3: List[TopPrediction] = Field(..., description="Top 3 ranked predictions with calibrated probabilities")
    decision: Literal["confident", "uncertain"] = Field(
        ..., description="'confident' if top-1 confidence >= abstention threshold, else 'uncertain'"
    )
    model_version: str = Field(..., description="Deployed model version identifier")


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_version: str
    num_classes: int
    abstention_threshold: float
