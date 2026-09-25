"""
Prometheus Metrics Exporter tracking request rates, latencies (p95), prediction counts,
confidence score distributions, and drift detection scores.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# Request & Latency metrics
REQUEST_COUNT = Counter(
    "pet_breed_requests_total",
    "Total HTTP inference requests received",
    ["endpoint", "http_status"],
)

INFERENCE_LATENCY = Histogram(
    "pet_breed_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Model Prediction metrics
PREDICTIONS_BY_BREED = Counter(
    "pet_breed_predictions_total",
    "Total count of predictions by breed",
    ["breed", "species", "decision"],
)

CONFIDENCE_DISTRIBUTION = Histogram(
    "pet_breed_confidence_distribution",
    "Calibrated confidence score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

ABSTENTION_DECISIONS = Counter(
    "pet_breed_abstention_total",
    "Abstention outcomes (confident vs uncertain)",
    ["decision"],
)

# Drift Monitoring Gauges
EMBEDDING_MMD_DRIFT_SCORE = Gauge(
    "pet_breed_embedding_drift_mmd",
    "Maximum Mean Discrepancy (MMD) feature embedding drift score on latest batch",
)

PIXEL_DRIFT_SCORE = Gauge(
    "pet_breed_pixel_drift_score",
    "Cheap statistical pixel distribution drift score (Wasserstein distance)",
    ["metric"],
)

CONFIDENCE_DRIFT_SCORE = Gauge(
    "pet_breed_confidence_drift_ks",
    "Kolmogorov-Smirnov distance on prediction confidence distribution",
)


def record_prediction_metrics(
    breed: str,
    species: str,
    confidence: float,
    decision: str,
    latency_seconds: float,
) -> None:
    """Record metrics for an individual prediction request."""
    PREDICTIONS_BY_BREED.labels(breed=breed, species=species, decision=decision).inc()
    CONFIDENCE_DISTRIBUTION.observe(confidence)
    ABSTENTION_DECISIONS.labels(decision=decision).inc()
    INFERENCE_LATENCY.observe(latency_seconds)


def get_prometheus_metrics() -> bytes:
    """Export Prometheus format metrics byte string."""
    return generate_latest()
