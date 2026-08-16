"""
MedRax Configuration Module
============================
Centralized configuration management for the MedRax medical image analysis platform.

Loads settings from environment variables (via .env file) with sensible defaults,
validates critical values, and exposes a frozen Settings singleton used across
all modules.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file — must happen before any os.getenv() calls below
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# Project root — everything is relative to the directory containing this file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables.

    Attributes are grouped by domain:
      - Hugging Face authentication
      - Model configuration
      - Inference parameters
      - Server / UI settings
      - Path configuration
      - Google Cloud Platform (optional)
    """

    # ── Hugging Face ──────────────────────────────────────────────────────
    hf_token: str = ""
    model_id: str = "google/medgemma-4b-it"

    # ── Device & Precision ────────────────────────────────────────────────
    device: str = "cpu"
    torch_dtype: str = "float32"
    use_quantization: bool = False  # Enable 4-bit quantization (bitsandbytes)

    # ── Generation Parameters ─────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False

    # ── Gradio Server ─────────────────────────────────────────────────────
    server_name: str = "127.0.0.1"
    port: int = 7860

    # ── Paths ─────────────────────────────────────────────────────────────
    upload_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    report_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "reports")
    assets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "assets")

    # ── Google Cloud Platform (optional — used in Phase 7) ────────────────
    gcp_project_id: Optional[str] = None
    gcs_bucket_name: Optional[str] = None
    gcp_region: str = "us-central1"
    enable_cloud_logging: bool = False
    enable_gcs_storage: bool = False

    # ── Supported Image Formats ───────────────────────────────────────────
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".dcm", ".dicom")
    max_image_size_mb: float = 50.0
    max_image_dimension: int = 4096

    def __post_init__(self) -> None:
        """Validate settings and ensure required directories exist."""
        # Create data directories if they don't exist
        object.__setattr__(self, "upload_dir", Path(self.upload_dir))
        object.__setattr__(self, "report_dir", Path(self.report_dir))
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Warn about missing HF token
        if not self.hf_token:
            print(
                "[WARNING] HF_TOKEN is not set. MedGemma is a gated model and "
                "requires a Hugging Face token with accepted license terms.",
                file=sys.stderr,
            )

    @property
    def max_image_size_bytes(self) -> int:
        """Maximum image file size in bytes."""
        return int(self.max_image_size_mb * 1024 * 1024)

    @property
    def gcs_enabled(self) -> bool:
        """Whether Google Cloud Storage integration is active."""
        return self.enable_gcs_storage and bool(self.gcs_bucket_name)


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.strip().lower() in ("true", "1", "yes")


def _load_settings() -> Settings:
    """Build a Settings instance from environment variables with defaults."""
    return Settings(
        # Hugging Face
        hf_token=os.getenv("HF_TOKEN", ""),
        model_id=os.getenv("MODEL_ID", "google/medgemma-4b-it"),
        # Device & Precision
        device=os.getenv("DEVICE", "cpu"),
        torch_dtype=os.getenv("TORCH_DTYPE", "float32"),
        use_quantization=_parse_bool(os.getenv("USE_QUANTIZATION", "false")),
        # Generation
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        do_sample=_parse_bool(os.getenv("DO_SAMPLE", "false")),
        # Server
        server_name=os.getenv("SERVER_NAME", "127.0.0.1"),
        port=int(os.getenv("PORT", "7860")),
        # GCP
        gcp_project_id=os.getenv("GCP_PROJECT_ID"),
        gcs_bucket_name=os.getenv("GCS_BUCKET_NAME"),
        gcp_region=os.getenv("GCP_REGION", "us-central1"),
        enable_cloud_logging=_parse_bool(os.getenv("ENABLE_CLOUD_LOGGING", "false")),
        enable_gcs_storage=_parse_bool(os.getenv("ENABLE_GCS_STORAGE", "false")),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import as:  from config import settings
# ---------------------------------------------------------------------------
settings = _load_settings()
