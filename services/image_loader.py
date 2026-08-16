"""
MedRax Image Loader Service
=============================
Loads medical images from various sources and formats into PIL Image objects.

Supported formats:
  - Standard images: JPEG, PNG
  - Medical imaging: DICOM (.dcm, .dicom)

Responsibilities:
  - File type detection and validation
  - File size enforcement
  - DICOM pixel data extraction and windowing
  - Saving uploaded files to the configured upload directory
  - Optional loading from Google Cloud Storage (Phase 7)
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or is invalid."""


class ImageLoader:
    """Service for loading and validating medical images.

    This class handles the first step of the pipeline: getting raw files
    from the user into a consistent ``PIL.Image.Image`` format that
    downstream modules can work with.
    """

    def __init__(self) -> None:
        self._supported_extensions = settings.supported_extensions
        self._max_size_bytes = settings.max_image_size_bytes
        self._upload_dir = settings.upload_dir

    # ── Public API ────────────────────────────────────────────────────────

    def load(self, file_path: str | Path) -> Image.Image:
        """Load an image from a local file path.

        Args:
            file_path: Path to the image file (JPEG, PNG, or DICOM).

        Returns:
            A PIL Image in its original mode (RGB conversion is handled
            by ``utils.image.prepare_image_for_model``).

        Raises:
            ImageLoadError: If the file doesn't exist, is too large,
                has an unsupported format, or is corrupt.
        """
        path = Path(file_path)
        self._validate_file(path)

        extension = path.suffix.lower()
        if extension in (".dcm", ".dicom"):
            image = self._load_dicom(path)
        else:
            image = self._load_standard(path)

        logger.info(
            "Loaded image: %s (%dx%d, mode=%s)",
            path.name,
            image.size[0],
            image.size[1],
            image.mode,
        )
        return image

    def save_upload(self, file_path: str | Path) -> Path:
        """Copy an uploaded file to the upload directory with a unique name.

        The file is renamed using a timestamp and hash to avoid collisions
        and preserve the original extension.

        Args:
            file_path: Path to the uploaded file.

        Returns:
            The new path within the upload directory.
        """
        source = Path(file_path)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_hash = self._short_hash(source)
        new_name = f"{timestamp}_{file_hash}{source.suffix.lower()}"
        destination = self._upload_dir / new_name

        shutil.copy2(source, destination)
        logger.info("Saved upload: %s → %s", source.name, destination.name)
        return destination

    def load_from_gcs(self, gcs_uri: str) -> Image.Image:
        """Load an image from Google Cloud Storage.

        Args:
            gcs_uri: A ``gs://bucket/path/to/image`` URI.

        Returns:
            A PIL Image.

        Raises:
            ImageLoadError: If GCS is not configured or the download fails.
        """
        if not settings.gcs_enabled:
            raise ImageLoadError(
                "Google Cloud Storage is not enabled. "
                "Set ENABLE_GCS_STORAGE=true and GCS_BUCKET_NAME in .env."
            )

        try:
            from google.cloud import storage as gcs

            # Parse gs://bucket/path
            if not gcs_uri.startswith("gs://"):
                raise ImageLoadError(f"Invalid GCS URI: {gcs_uri}")

            parts = gcs_uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""

            client = gcs.Client(project=settings.gcp_project_id)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download to upload directory
            extension = Path(blob_name).suffix or ".png"
            local_path = self._upload_dir / f"gcs_{self._timestamp()}{extension}"
            blob.download_to_filename(str(local_path))

            logger.info("Downloaded from GCS: %s → %s", gcs_uri, local_path.name)
            return self.load(local_path)

        except ImportError:
            raise ImageLoadError(
                "google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            )
        except Exception as exc:
            raise ImageLoadError(f"Failed to load from GCS: {exc}") from exc

    # ── Private Helpers ───────────────────────────────────────────────────

    def _validate_file(self, path: Path) -> None:
        """Validate that a file exists, is supported, and is within size limits."""
        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")

        if not path.is_file():
            raise ImageLoadError(f"Not a file: {path}")

        extension = path.suffix.lower()
        if extension not in self._supported_extensions:
            raise ImageLoadError(
                f"Unsupported format: '{extension}'. "
                f"Supported: {', '.join(self._supported_extensions)}"
            )

        file_size = path.stat().st_size
        if file_size > self._max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            raise ImageLoadError(
                f"File too large: {size_mb:.1f} MB "
                f"(max: {settings.max_image_size_mb:.0f} MB)"
            )

        if file_size == 0:
            raise ImageLoadError(f"File is empty: {path}")

    def _load_standard(self, path: Path) -> Image.Image:
        """Load a JPEG or PNG image."""
        try:
            image = Image.open(path)
            image.load()  # Force-read pixel data into memory
            return image
        except Exception as exc:
            raise ImageLoadError(
                f"Failed to load image '{path.name}': {exc}"
            ) from exc

    def _load_dicom(self, path: Path) -> Image.Image:
        """Load a DICOM file and convert it to a PIL Image.

        Handles windowing (Window Center / Window Width) to produce
        a visually meaningful grayscale image from the raw pixel data.
        """
        try:
            import pydicom
        except ImportError:
            raise ImageLoadError(
                "pydicom is not installed. Run: pip install pydicom"
            )

        try:
            ds = pydicom.dcmread(str(path))
        except Exception as exc:
            raise ImageLoadError(
                f"Failed to read DICOM file '{path.name}': {exc}"
            ) from exc

        if not hasattr(ds, "pixel_array"):
            raise ImageLoadError(
                f"DICOM file '{path.name}' does not contain pixel data."
            )

        pixel_array = ds.pixel_array.astype(np.float64)

        # Apply DICOM windowing for proper display
        pixel_array = self._apply_dicom_windowing(ds, pixel_array)

        # Normalize to 0–255 uint8
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)

        pixel_array = pixel_array.astype(np.uint8)

        # Handle multi-frame DICOM (take first frame)
        if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
            logger.info("Multi-frame DICOM detected, using first frame.")
            pixel_array = pixel_array[0]

        image = Image.fromarray(pixel_array)
        logger.info(
            "DICOM loaded: %s (Modality=%s, size=%dx%d)",
            path.name,
            getattr(ds, "Modality", "Unknown"),
            image.size[0],
            image.size[1],
        )
        return image

    @staticmethod
    def _apply_dicom_windowing(
        ds: object,
        pixel_array: np.ndarray,
    ) -> np.ndarray:
        """Apply DICOM Window Center/Width to pixel data.

        This maps the raw Hounsfield unit (CT) or intensity values to
        a display range, which is critical for producing clinically
        meaningful images.
        """
        window_center = getattr(ds, "WindowCenter", None)
        window_width = getattr(ds, "WindowWidth", None)

        if window_center is None or window_width is None:
            return pixel_array

        # Handle multi-valued Window Center/Width (take first value)
        if hasattr(window_center, "__iter__"):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)

        if hasattr(window_width, "__iter__"):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)

        # Apply rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        # Window clipping
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, lower, upper)

        return pixel_array

    @staticmethod
    def _short_hash(path: Path, length: int = 8) -> str:
        """Generate a short hash of a file's contents for unique naming."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:length]

    @staticmethod
    def _timestamp() -> str:
        """Generate a UTC timestamp string for file naming."""
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
