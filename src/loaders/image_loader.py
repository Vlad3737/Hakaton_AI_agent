"""Image loader and processor."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator


class ImageLoader:
    """Image loader and processor."""

    def __init__(self, image_dir: Path | str):
        """Initialize loader.

        Args:
            image_dir: Path to image directory
        """
        self.image_dir = Path(image_dir)
        self.images: dict[str, dict] = {}

    def load_image(self, image_path: Path | str) -> dict:
        """Load single image.

        Args:
            image_path: Path to image

        Returns:
            Dict with image and metadata
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.debug(f"Loading image: {path}")

        # Open image
        with Image.open(path) as img:
            # Convert to RGB if needed (for PNG with transparency)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Get base64 for multimodal processing
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Additional metadata
            width, height = img.size

        self.images[path.name] = {
            "path": path,
            "name": path.name,
            "format": path.suffix.lower()[1:],
            "width": width,
            "height": height,
            "base64": img_base64,
            "image": img,
        }

        return self.images[path.name]

    def load_all_images(self, extensions: list[str] | None = None) -> dict[str, dict]:
        """Load all images from directory.

        Args:
            extensions: List of extensions (default: pdf, png, jpg, jpeg, gif, bmp)

        Returns:
            Dict {name: image data}
        """
        if extensions is None:
            extensions = [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp"]

        for ext in extensions:
            for image_path in self.image_dir.glob(f"**/*{ext}"):
                if image_path.name not in self.images:
                    try:
                        self.load_image(image_path)
                    except Exception as e:
                        logger.warning(f"Error loading {image_path}: {e}")

        logger.info(f"Loaded {len(self.images)} images")
        return self.images

    def get_image(self, name: str) -> dict | None:
        """Get image by name."""
        if not self.images:
            self.load_all_images()

        # Normalize name
        name_lower = name.lower()
        for key, img_data in self.images.items():
            if key.lower() == name_lower:
                return img_data
        return None

    def get_image_by_filename(self, filename: str) -> dict | None:
        """Get image by filename (for LaTeX \includegraphics).

        Args:
            filename: Filename from \includegraphics{filename}

        Returns:
            Image data or None
        """
        if not self.images:
            self.load_all_images()

        # Remove extension if present
        name = Path(filename).stem

        for key, img_data in self.images.items():
            # Compare without extension
            if Path(key).stem.lower() == name.lower():
                return img_data

            # Direct comparison with extension
            if key.lower() == name.lower():
                return img_data

        return None

    def get_base64_by_filename(self, filename: str) -> str | None:
        """Get base64 image by filename."""
        img = self.get_image_by_filename(filename)
        return img["base64"] if img else None

    def __enter__(self) -> ImageLoader:
        """Context manager entry."""
        self.load_all_images()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.images.clear()


def load_image_base64(image_path: Path) -> str:
    """Load single image and return base64.

    Args:
        image_path: Path to image

    Returns:
        Base64-encoded string
    """
    loader = ImageLoader(image_path.parent)
    img_data = loader.load_image(image_path)
    return img_data["base64"]


def encode_image_to_base64(image_path: Path | str) -> str:
    """Encode image to base64.

    Args:
        image_path: Path to image

    Returns:
        Base64-encoded string
    """
    path = Path(image_path)

    with Image.open(path) as img:
        # Convert if needed
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
