"""
API integration tests for FastAPI endpoints: /predict, /health, /metrics.
Verifies robust handling of 4-channel RGBA PNG, grayscale, CMYK, invalid files (422), and zero 500 errors.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from pet_breed.serving.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health returns 200 with model version and class count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert data["num_classes"] == 37


def test_predict_standard_rgb(client: TestClient):
    """Verify /predict returns valid schema on standard RGB image."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict", files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "breed" in data
    assert data["species"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_3"]) == 3
    assert data["decision"] in ["confident", "uncertain"]
    assert "model_version" in data


def test_predict_4channel_rgba_png(client: TestClient):
    """Verify 4-channel PNG with alpha channel returns 200 (never 500)."""
    img_rgba = Image.new("RGBA", (250, 250), color=(128, 64, 32, 200))
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict", files={"file": ("test_alpha.png", buf, "image/png")}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "breed" in data


def test_predict_grayscale_and_cmyk(client: TestClient):
    """Verify Grayscale and CMYK images return 200."""
    # Grayscale
    img_l = Image.new("L", (180, 180), color=100)
    buf_l = io.BytesIO()
    img_l.save(buf_l, format="JPEG")
    buf_l.seek(0)
    res_l = client.post("/predict", files={"file": ("test_gray.jpg", buf_l, "image/jpeg")})
    assert res_l.status_code == 200

    # CMYK
    img_cmyk = Image.new("CMYK", (200, 200), color=(10, 20, 30, 40))
    buf_cmyk = io.BytesIO()
    img_cmyk.save(buf_cmyk, format="JPEG")
    buf_cmyk.seek(0)
    res_cmyk = client.post("/predict", files={"file": ("test_cmyk.jpg", buf_cmyk, "image/jpeg")})
    assert res_cmyk.status_code == 200


def test_predict_invalid_files_returns_422(client: TestClient):
    """Verify non-image text file and corrupted image return 422 (never 500)."""
    # Text file masquerading as image
    fake_buf = io.BytesIO(b"This is not a pet picture, just plain text")
    res_fake = client.post(
        "/predict", files={"file": ("corrupt.jpg", fake_buf, "image/jpeg")}
    )
    assert res_fake.status_code == 422

    # Empty file
    empty_buf = io.BytesIO(b"")
    res_empty = client.post(
        "/predict", files={"file": ("empty.jpg", empty_buf, "image/jpeg")}
    )
    assert res_empty.status_code == 422


def test_metrics_endpoint(client: TestClient):
    """Verify /metrics exposes Prometheus metrics."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "pet_breed_requests_total" in res.text


def test_ui_and_samples_endpoints(client: TestClient):
    """Verify / and /ui serve the HTML web interface and /samples serves test images."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "Pet Breed Classification AI" in res_root.text

    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Pet Breed Classification AI" in res_ui.text

    # Verify sample image route
    res_sample = client.get("/samples/Bengal_2.jpg")
    assert res_sample.status_code == 200
    assert "image/jpeg" in res_sample.headers.get("content-type", "")

