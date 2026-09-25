FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install project
COPY pyproject.toml .
COPY src ./src
COPY data/labels.json ./data/labels.json

RUN pip install --upgrade pip && \
    pip install .

# Copy model artifacts if available
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "pet_breed.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
