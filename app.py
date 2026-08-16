"""
MedRax — AI-Powered Medical Image Analysis
=============================================
Application entry point.

This script:
  1. Loads configuration from ``.env``
  2. Initializes the structured logging system
  3. Optionally pre-loads the MedGemma model for fast first-request response
  4. Creates the Gradio web interface
  5. Launches the server on the configured host/port
  6. Handles graceful shutdown (SIGINT / SIGTERM)

Usage:
    python app.py                  # Start the server
    python app.py --preload        # Pre-load model before serving
    python app.py --share          # Create a public Gradio link
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Ensure the project root is on sys.path so all packages resolve correctly
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MedRax — AI-Powered Medical Image Analysis",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=False,
        help="Pre-load the MedGemma model at startup (slower start, faster first request).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Server host (default: {settings.server_name}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Server port (default: {settings.port}).",
    )
    return parser.parse_args()


def _preload_model() -> None:
    """Pre-load the MedGemma model so the first request is fast."""
    from services.model_manager import ModelManager

    logger.info("Pre-loading MedGemma model...")
    try:
        manager = ModelManager()
        manager.load_model()
        logger.info("✓ Model pre-loaded and ready for inference.")
    except Exception as exc:
        logger.error(
            "✗ Model pre-loading failed: %s. "
            "The model will be loaded on the first request instead.",
            exc,
        )


def _setup_signal_handlers() -> None:
    """Register handlers for graceful shutdown."""

    def _handle_signal(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully...", sig_name)

        # Attempt to unload the model and free GPU memory
        try:
            from services.model_manager import ModelManager

            manager = ModelManager()
            if manager.is_loaded:
                manager.unload_model()
                logger.info("Model unloaded.")
        except Exception:
            pass

        logger.info("MedRax shut down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    """Application entry point."""
    args = _parse_args()

    # ── Banner ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  MedRax — AI-Powered Medical Image Analysis")
    logger.info("=" * 60)
    logger.info("  Model:    %s", settings.model_id)
    logger.info("  Device:   %s", settings.device)
    logger.info("  Dtype:    %s", settings.torch_dtype)
    logger.info("  Quant:    %s", "4-bit" if settings.use_quantization else "None")
    logger.info("  GCS:      %s", "Enabled" if settings.gcs_enabled else "Disabled")
    logger.info("=" * 60)

    # ── Signal handlers ───────────────────────────────────────────────────
    _setup_signal_handlers()

    # ── Optional model pre-load ───────────────────────────────────────────
    if args.preload:
        _preload_model()

    # ── Create the Gradio interface ───────────────────────────────────────
    from ui.interface import create_interface

    interface = create_interface()

    # ── Resolve host/port ─────────────────────────────────────────────────
    host = args.host or settings.server_name
    port = args.port or settings.port

    # ── Launch ────────────────────────────────────────────────────────────
    logger.info("Starting Gradio server at http://%s:%d", host, port)

    interface.launch(
        server_name=host,
        server_port=port,
        share=args.share,
        show_error=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
