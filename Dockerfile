# ===========================================================================
# MedRax Dockerfile
# ===========================================================================
# Multi-stage build for the MedRax medical image analysis platform.
#
# Build:   docker build -t medrax .
# Run:     docker run -p 7860:7860 --gpus all --env-file .env medrax
#
# For CPU-only (slow):
#          docker run -p 7860:7860 --env-file .env medrax
# ===========================================================================

# ── Stage 1: Base with Python and system dependencies ─────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by Pillow, pydicom, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Install Python dependencies ──────────────────────────────────
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Application ─────────────────────────────────────────────────
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY config.py app.py ./
COPY models/ ./models/
COPY services/ ./services/
COPY ui/ ./ui/
COPY utils/ ./utils/
COPY assets/ ./assets/

# Create data directories
RUN mkdir -p data/uploads data/reports data/logs

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Run the application
# Use 0.0.0.0 to accept connections from outside the container
CMD ["python", "app.py", "--host", "0.0.0.0"]
