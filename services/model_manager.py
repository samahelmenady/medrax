"""
MedRax Model Manager
=====================
Manages the lifecycle of the MedGemma model: loading, caching, and resource cleanup.

Architecture:
  - Singleton pattern — the model is loaded once and reused across requests.
  - Supports multiple precision modes: float32, bfloat16, float16, and 4-bit quantization.
  - Handles Hugging Face authentication for the gated MedGemma model.
  - Thread-safe lazy initialization with proper error reporting.

Usage:
    from services.model_manager import ModelManager

    manager = ModelManager()
    manager.load_model()
    model = manager.get_model()
    processor = manager.get_processor()
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when the model or processor fails to load."""


class ModelManager:
    """Singleton manager for the MedGemma model and processor.

    The model is loaded lazily on the first call to ``load_model()`` and
    cached for subsequent use.  This avoids loading the ~8 GB model
    at import time and allows the application to start quickly.
    """

    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelManager:
        """Ensure only one ModelManager instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model = None
        self._processor = None
        self._model_id: str = settings.model_id
        self._device: str = settings.device
        self._is_loaded: bool = False
        self._initialized = True

    # ── Public API ────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the MedGemma model and processor into memory.

        This method is idempotent — calling it multiple times has no effect
        if the model is already loaded.

        Raises:
            ModelLoadError: If loading fails due to authentication,
                missing libraries, or insufficient resources.
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping.")
            return

        with self._lock:
            if self._is_loaded:
                return

            logger.info("Loading model: %s", self._model_id)
            logger.info("Device: %s | Dtype: %s | Quantization: %s",
                        self._device, settings.torch_dtype, settings.use_quantization)

            try:
                self._load_processor()
                self._load_model_weights()
                self._is_loaded = True
                logger.info("✓ Model loaded successfully.")
                self._log_memory_usage()
            except Exception as exc:
                self._model = None
                self._processor = None
                self._is_loaded = False
                logger.error("✗ Failed to load model: %s", exc)
                raise ModelLoadError(f"Failed to load model: {exc}") from exc

    def get_model(self):
        """Get the loaded model instance.

        Returns:
            The loaded ``AutoModelForImageTextToText`` instance.

        Raises:
            ModelLoadError: If the model has not been loaded yet.
        """
        if not self._is_loaded or self._model is None:
            raise ModelLoadError(
                "Model is not loaded. Call load_model() first."
            )
        return self._model

    def get_processor(self):
        """Get the loaded processor instance.

        Returns:
            The loaded ``AutoProcessor`` instance.

        Raises:
            ModelLoadError: If the processor has not been loaded yet.
        """
        if not self._is_loaded or self._processor is None:
            raise ModelLoadError(
                "Processor is not loaded. Call load_model() first."
            )
        return self._processor

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded and ready for inference."""
        return self._is_loaded

    @property
    def model_id(self) -> str:
        """The Hugging Face model identifier."""
        return self._model_id

    @property
    def device_info(self) -> str:
        """Human-readable string describing the current compute device."""
        if self._model is not None:
            try:
                device = next(self._model.parameters()).device
                return str(device)
            except StopIteration:
                pass
        return self._device

    def unload_model(self) -> None:
        """Release the model and processor from memory.

        Useful for freeing GPU VRAM when the model is no longer needed.
        """
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            self._is_loaded = False

            # Force garbage collection and CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded and memory released.")

    # ── Private Helpers ───────────────────────────────────────────────────

    def _get_torch_dtype(self) -> torch.dtype:
        """Map the string dtype from config to a torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype_str = settings.torch_dtype.lower()
        if dtype_str not in dtype_map:
            logger.warning(
                "Unknown torch_dtype '%s', falling back to float32.", dtype_str
            )
            return torch.float32
        return dtype_map[dtype_str]

    def _get_auth_token(self) -> Optional[str]:
        """Get the Hugging Face authentication token."""
        token = settings.hf_token
        if not token:
            logger.warning(
                "No HF_TOKEN provided. MedGemma is a gated model — "
                "authentication will likely fail."
            )
            return None
        return token

    def _load_processor(self) -> None:
        """Load the AutoProcessor for MedGemma."""
        from transformers import AutoProcessor

        logger.info("Loading processor...")
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            token=self._get_auth_token(),
        )
        logger.info("✓ Processor loaded.")

    def _load_model_weights(self) -> None:
        """Load the model weights with the configured precision and device."""
        from transformers import AutoModelForImageTextToText

        load_kwargs = {
            "pretrained_model_name_or_path": self._model_id,
            "token": self._get_auth_token(),
        }

        # Configure quantization (4-bit via bitsandbytes)
        if settings.use_quantization:
            load_kwargs.update(self._get_quantization_config())
        else:
            load_kwargs["torch_dtype"] = self._get_torch_dtype()

        # Configure device mapping
        if self._device == "auto" or (
            self._device != "cpu" and torch.cuda.is_available()
        ):
            load_kwargs["device_map"] = "auto"
            logger.info("Using automatic device mapping (GPU).")
        elif self._device == "cpu":
            load_kwargs["device_map"] = "cpu"
            logger.info("Using CPU — inference will be slow.")
        else:
            load_kwargs["device_map"] = self._device

        logger.info("Loading model weights (this may take several minutes)...")
        self._model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
        self._model.eval()  # Set to evaluation mode

    def _get_quantization_config(self) -> dict:
        """Build quantization configuration for 4-bit loading."""
        try:
            from transformers import BitsAndBytesConfig

            logger.info("Enabling 4-bit quantization (bitsandbytes).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            return {"quantization_config": bnb_config, "device_map": "auto"}
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — falling back to full precision. "
                "Install with: pip install bitsandbytes"
            )
            return {"torch_dtype": self._get_torch_dtype()}

    def _log_memory_usage(self) -> None:
        """Log GPU memory usage after model loading."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(
                "GPU memory — Allocated: %.2f GB | Reserved: %.2f GB",
                allocated,
                reserved,
            )
        else:
            logger.info("Running on CPU — GPU memory stats not available.")
