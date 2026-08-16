"""
MedRax Image Utilities
=======================
Low-level image preprocessing and validation functions.

These utilities sit between the image loader (which reads files from disk/DICOM)
and the inference service (which feeds images to MedGemma).  MedGemma's
``AutoProcessor`` handles its own internal normalization, so the preprocessing
here focuses on:
  - Ensuring images are in RGB mode (required by MedGemma's SigLIP encoder)
  - Validating dimensions are within safe limits
  - Optional resizing for consistency and memory efficiency
"""

from __future__ import annotations

from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def validate_image_dimensions(
    image: Image.Image,
    max_dimension: int = 4096,
) -> bool:
    """Check whether an image's dimensions are within acceptable limits.

    Args:
        image: A PIL Image to validate.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        ``True`` if both width and height are ≤ ``max_dimension``.
    """
    width, height = image.size
    if width > max_dimension or height > max_dimension:
        logger.warning(
            "Image dimensions %dx%d exceed maximum %d. "
            "Consider resizing before inference.",
            width,
            height,
            max_dimension,
        )
        return False
    if width < 10 or height < 10:
        logger.warning(
            "Image dimensions %dx%d are suspiciously small.", width, height
        )
        return False
    return True


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to RGB mode if it isn't already.

    MedGemma's SigLIP vision encoder requires 3-channel RGB input.
    DICOM images often arrive as grayscale ('L') or 16-bit ('I;16'),
    and some PNGs may include an alpha channel ('RGBA').

    Args:
        image: Any PIL Image.

    Returns:
        The image converted to RGB mode.
    """
    if image.mode == "RGB":
        return image

    original_mode = image.mode
    logger.info("Converting image from %s to RGB.", original_mode)

    # Handle palette-based images
    if image.mode == "P":
        image = image.convert("RGBA")

    # Handle alpha channel
    if image.mode == "RGBA":
        # Composite onto a white background to remove transparency
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background

    # Handle grayscale, 16-bit, and other modes
    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    max_dimension: int = 2048,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize an image so its longest side is at most ``max_dimension`` pixels.

    Aspect ratio is preserved.  Images already within limits are returned
    unchanged.

    Args:
        image: A PIL Image.
        max_dimension: Maximum allowed size for the longest side.
        resample: Resampling filter (default: LANCZOS for high quality).

    Returns:
        The (possibly resized) image.
    """
    width, height = image.size
    longest_side = max(width, height)

    if longest_side <= max_dimension:
        return image

    scale = max_dimension / longest_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(
        "Resizing image from %dx%d to %dx%d (scale=%.2f).",
        width,
        height,
        new_width,
        new_height,
        scale,
    )
    return image.resize((new_width, new_height), resample=resample)


def prepare_image_for_model(
    image: Image.Image,
    max_dimension: int = 2048,
) -> Image.Image:
    """Full preprocessing pipeline: validate → convert to RGB → resize.

    This is the single entry point that other modules should call
    before passing an image to the inference service.

    Args:
        image: A raw PIL Image from the image loader.
        max_dimension: Maximum dimension for resizing.

    Returns:
        A preprocessed PIL Image ready for MedGemma.

    Raises:
        ValueError: If the image is invalid or corrupt.
    """
    if image is None:
        raise ValueError("Received None instead of a PIL Image.")

    # Validate basic integrity
    try:
        image.verify()
        # verify() can invalidate the image object, so re-open from data
        image = image.copy()
    except Exception:
        # If verify fails, the image data may still be usable
        # (verify is strict about metadata), so we continue cautiously
        pass

    # Reload if verify consumed the pixel data
    try:
        image.load()
    except Exception as exc:
        raise ValueError(f"Image data is corrupt or unreadable: {exc}") from exc

    # Convert to RGB
    image = ensure_rgb(image)

    # Resize if necessary
    image = resize_image(image, max_dimension=max_dimension)

    logger.debug(
        "Image prepared: size=%dx%d, mode=%s.",
        image.size[0],
        image.size[1],
        image.mode,
    )
    return image
