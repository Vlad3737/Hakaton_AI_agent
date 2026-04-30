"""LaTeX file loader and parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


class TeXLoader:
    """LaTeX document loader and parser."""

    def __init__(self, tex_dir: Path | str):
        """Initialize loader.

        Args:
            tex_dir: Path to LaTeX source directory
        """
        self.tex_dir = Path(tex_dir)
        self.main_tex: Path | None = None
        self.all_tex_files: list[Path] = []
        self.image_files: list[Path] = []
        self.preamble = ""
        self.body = ""
        self.tables: list[dict] = []
        self.equations: list[dict] = []
        self.figure_refs: dict[str, str] = {}

    def discover_files(self) -> None:
        """Discover all files in directory."""
        logger.info(f"Discovering files in: {self.tex_dir}")

        # Find all .tex files
        self.all_tex_files = sorted(self.tex_dir.glob("*.tex"))

        # Determine main file (by \documentclass presence or by name)
        for tex_file in self.all_tex_files:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            if "\\documentclass" in content:
                self.main_tex = tex_file
                break

        # If not found, take first alphabetically
        if not self.main_tex and self.all_tex_files:
            self.main_tex = self.all_tex_files[0]

        logger.info(f"Main file: {self.main_tex}")

        # Find all images
        image_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]
        for ext in image_extensions:
            self.image_files.extend(self.tex_dir.glob(f"**/*{ext}"))
        self.image_files = list(set(self.image_files))  # Remove duplicates

        logger.info(f"Found {len(self.image_files)} images")

    def read_main_tex(self) -> str:
        """Read main .tex file."""
        if not self.main_tex:
            raise RuntimeError("Main .tex file not found. Call discover_files() first.")

        content = self.main_tex.read_text(encoding="utf-8", errors="ignore")
        self.body = content
        return content

    def extract_preamble(self, content: str | None = None) -> str:
        """Extract preamble (before \\begin{document})."""
        if content is None:
            content = self.body

        match = re.search(r"\\begin\{document\}", content)
        if match:
            self.preamble = content[: match.start()]
        else:
            self.preamble = content
        return self.preamble

    def extract_body(self, content: str | None = None) -> str:
        """Extract body (after \\begin{document})."""
        if content is None:
            content = self.body

        match = re.search(r"\\begin\{document\}", content)
        if match:
            end_match = re.search(r"\\end\{document\}", content[match.start():])
            if end_match:
                self.body = content[match.start() + match.end(): match.start() + end_match.start()]
            else:
                self.body = content[match.start() + match.end():]
        else:
            self.body = ""
        return self.body

    def extract_figures(self, content: str | None = None) -> list[dict]:
        """Extract figures from document.

        Args:
            content: LaTeX content (optional)

        Returns:
            List of figures with parameters
        """
        if content is None:
            content = self.body

        figures = []

        # Pattern for \includegraphics
        include_pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"

        # Pattern for figure environment
        figure_pattern = r"\\begin\{figure(?:\*?)\}(.*?)\\end\{figure(?:\*?)\}"

        for match in re.finditer(figure_pattern, content, re.DOTALL):
            figure_env = match.group(1)

            # Extract image(s)
            img_matches = re.findall(include_pattern, figure_env)

            # Extract caption
            caption_match = re.search(r"\\caption\{([^}]*)\}", figure_env)
            caption = caption_match.group(1) if caption_match else ""

            # Extract label
            label_match = re.search(r"\\label\{([^}]*)\}", figure_env)
            label = label_match.group(1) if label_match else ""

            figures.append({
                "images": img_matches,
                "caption": caption,
                "label": label,
                "full_env": figure_env,
            })

            for img_file in img_matches:
                self.figure_refs[img_file] = label or caption[:50]

        logger.info(f"Extracted {len(figures)} figures")
        return figures

    def extract_equations(self, content: str | None = None) -> list[dict]:
        """Extract equations from document.

        Args:
            content: LaTeX content (optional)

        Returns:
            List of equations
        """
        if content is None:
            content = self.body

        equations = []

        # Equations in $...$
        inline_pattern = r"\$([^$]+)\$"
        for match in re.finditer(inline_pattern, content):
            equations.append({
                "type": "inline",
                "content": match.group(1),
                "position": match.start(),
            })

        # Equations in \[...\]
        display_pattern = r"\\\[(.*?)\\\]"
        for match in re.finditer(display_pattern, content, re.DOTALL):
            equations.append({
                "type": "display",
                "content": match.group(1).strip(),
                "position": match.start(),
            })

        # Equations in \begin{equation}...\end{equation}
        equation_env_pattern = r"\\begin\{equation\}(.*?)\\end\{equation\}"
        for match in re.finditer(equation_env_pattern, content, re.DOTALL):
            equations.append({
                "type": "equation",
                "content": match.group(1).strip(),
                "position": match.start(),
            })

        logger.info(f"Extracted {len(equations)} equations")
        return equations

    def extract_tables(self, content: str | None = None) -> list[dict]:
        """Extract tables from document.

        Args:
            content: LaTeX content (optional)

        Returns:
            List of tables
        """
        if content is None:
            content = self.body

        tables = []

        table_pattern = r"\\begin\{table(?:\*?)\}(.*?)\\end\{table(?:\*?)\}"
        for match in re.finditer(table_pattern, content, re.DOTALL):
            table_env = match.group(1)

            caption_match = re.search(r"\\caption\{([^}]*)\}", table_env)
            caption = caption_match.group(1) if caption_match else ""

            tables.append({
                "content": table_env,
                "caption": caption,
            })

        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def read_all_tex(self) -> str:
        """Read all .tex files and combine."""
        all_content = []

        for tex_file in self.all_tex_files:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            all_content.append(f"\n\n%% File: {tex_file.name}\n\n{content}")

        return "\n".join(all_content)

    def get_images(self) -> list[Path]:
        """Get list of image files."""
        if not self.image_files:
            self.discover_files()
        return self.image_files

    def get_image_by_name(self, name: str) -> Path | None:
        """Find image by name."""
        for img in self.image_files:
            if img.name.lower() == name.lower():
                return img
        return None

    def __enter__(self) -> TeXLoader:
        """Context manager entry."""
        self.discover_files()
        self.read_main_tex()
        self.extract_preamble()
        self.extract_body()
        self.extract_figures()
        self.extract_equations()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass


def load_tex_with_images(tex_dir: Path) -> tuple[str, list[Path], dict[str, str]]:
    """Load LaTeX with text, images, and references.

    Args:
        tex_dir: Path to LaTeX source directory

    Returns:
        Tuple (text, list of images, figure references)
    """
    loader = TeXLoader(tex_dir)
    loader.discover_files()
    loader.read_main_tex()
    loader.extract_figures()

    images = loader.get_images()
    refs = loader.figure_refs

    return loader.body, images, refs
