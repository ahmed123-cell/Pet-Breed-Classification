"""
Production FastAPI Application for Pet Breed Classification.
"""

import io
import time
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST

from pet_breed.config import (
    CHECKPOINTS_DIR,
    DEFAULT_ABSTENTION_THRESHOLD,
    DEFAULT_NUM_CLASSES,
    EXPORTED_MODELS_DIR,
    MODEL_VERSION,
)
from pet_breed.models.classifier import PetBreedClassifier
from pet_breed.monitoring.metrics_exporter import (
    REQUEST_COUNT,
    get_prometheus_metrics,
    record_prediction_metrics,
)
from pet_breed.serving.schemas import HealthResponse, PredictionResponse

# Global classifier instance loaded ONCE at startup
classifier: Optional[PetBreedClassifier] = None

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights, preprocessor, and calibration parameters once at application startup."""
    global classifier
    print("Initializing PetBreedClassifier...")
    weights_path = EXPORTED_MODELS_DIR / "resnet50_calibrated.pth"
    if not weights_path.exists():
        weights_path = CHECKPOINTS_DIR / "resnet50_best.pth"

    classifier = PetBreedClassifier(
        backbone_name="resnet50",
        num_classes=DEFAULT_NUM_CLASSES,
        weights_path=weights_path if weights_path.exists() else None,
        temperature=1.0,
        abstention_threshold=DEFAULT_ABSTENTION_THRESHOLD,
        model_version=MODEL_VERSION,
    )
    print(f"PetBreedClassifier initialized with model version: {MODEL_VERSION}")
    yield
    print("Shutting down PetBreedClassifier service.")


app = FastAPI(
    title="Pet Breed Classification API",
    description="Fine-grained Pet Breed Classification Service with Calibrated Abstention",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health():
    """Health check endpoint used by Docker and load balancers."""
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized",
        )
    return HealthResponse(
        status="healthy",
        model_version=classifier.model_version,
        num_classes=classifier.num_classes,
        abstention_threshold=classifier.abstention_threshold,
    )


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_prometheus_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Validation error: invalid image payload or oversized file"},
        200: {"description": "Successful breed classification"},
    },
)
async def predict(file: UploadFile = File(...)):
    """
    Predict pet breed from uploaded multipart image file.
    Robustly handles RGB, RGBA (4-channel), CMYK, and Grayscale formats.
    Returns calibrated confidence and abstention decision ('confident' or 'uncertain').
    """
    t0 = time.perf_counter()

    # 1. Read file contents & validate size
    try:
        contents = await file.read()
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", http_status="422").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if len(contents) == 0:
        REQUEST_COUNT.labels(endpoint="/predict", http_status="422").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty file uploaded.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        REQUEST_COUNT.labels(endpoint="/predict", http_status="422").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
        )

    # 2. Decode and validate image
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify image integrity
        # Re-open after verify() since verify() closes the file descriptor
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, ValueError, OSError) as e:
        REQUEST_COUNT.labels(endpoint="/predict", http_status="422").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file is not a valid image: {str(e)}",
        )

    # 3. Model Inference
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready.",
        )

    try:
        prediction_result = classifier.predict(image)
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", http_status="422").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image through inference pipeline: {str(e)}",
        )

    latency = time.perf_counter() - t0
    REQUEST_COUNT.labels(endpoint="/predict", http_status="200").inc()
    record_prediction_metrics(
        breed=prediction_result["breed"],
        species=prediction_result["species"],
        confidence=prediction_result["confidence"],
        decision=prediction_result["decision"],
        latency_seconds=latency,
    )

    return PredictionResponse(**prediction_result)
