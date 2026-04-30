"""PDF file loader and parser."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # pymupdf
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


class PDFLoader:
    """PDF document loader with text and image extraction."""

    def __init__(self, pdf_path: Path | str):
        """Initialize loader.

        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.doc = None
        self.text_chunks: list[str] = []
        self.images: list[dict] = []

    def load(self) -> None:
        """Load PDF document."""
        logger.info(f"Loading PDF: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        logger.info(f"Loaded {len(self.doc)} pages")

    def extract_text(self) -> str:
        """Extract all text from PDF."""
        if self.doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")

        full_text = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            full_text.append(page.get_text())
        return "\n\n".join(full_text)

    def extract_text_with_pages(self) -> list[dict]:
        """Extract text with page numbers.

        Returns:
            List of dicts with text and page number
        """
        if self.doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")

        result = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            result.append({
                "page": page_num + 1,
                "text": text,
            })
        return result

    def extract_images(self) -> list[dict]:
        """Extract all images from PDF.

        Returns:
            List of images with metadata
        """
        if self.doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")

        images = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img_ext = base_image["ext"]

                    # Encode to base64 for multimodal processing
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                    images.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "bytes": img_bytes,
                        "base64": img_base64,
                        "format": img_ext,
                    })
                except Exception as e:
                    logger.warning(f"Error extracting image {img_idx} on page {page_num}: {e}")

        self.images = images
        logger.info(f"Extracted {len(images)} images")
        return images

    def extract_figure_captions(self) -> list[dict]:
        """Extract figure captions (simple heuristic).

        Returns:
            List of figures with captions
        """
        if self.doc is None:
            raise RuntimeError("PDF not loaded. Call load() first.")

        figures = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text")

            # Simple heuristic: search for "Figure", "Fig.", "Fig"
            lines = text.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if any(keyword in line.lower() for keyword in ["figure", "fig.", "fig", "figure."]):
                    # Next lines are caption
                    caption_lines = [line]
                    j = i + 1
                    while j < len(lines) and len(caption_lines) < 10:
                        next_line = lines[j].strip()
                        if not next_line or next_line.lower().startswith(("fig", "figure")):
                            break
                        caption_lines.append(next_line)
                        j += 1
                    figures.append({
                        "page": page_num + 1,
                        "caption": " ".join(caption_lines),
                        "position": i,
                    })
                i += 1

        return figures

    def extract_image_by_filename(self, filename: str) -> str | None:
        """Search and extract image by filename.

        Args:
            filename: Filename (e.g., "Braiding.pdf")

        Returns:
            Base64-encoded image string or None
        """
        if not self.images:
            self.extract_images()

        # Search for image by name
        for img in self.images:
            # Images in PDF are stored with these names
            if filename.lower() in str(img.get("stream", "")).lower():
                return img["base64"]
        return None

    def __enter__(self) -> PDFLoader:
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.doc:
            self.doc.close()


def load_pdf_with_images(pdf_path: Path) -> tuple[str, list[dict]]:
    """Load PDF with text and images.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple (text, list of images)
    """
    loader = PDFLoader(pdf_path)
    loader.load()
    text = loader.extract_text()
    images = loader.extract_images()
    return text, images
