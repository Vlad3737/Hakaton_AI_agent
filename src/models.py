"""Общие структуры данных проекта.

Контракт между всеми разработчиками. Менять только по согласованию команды.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Figure:
    """Описывает один рисунок в статье."""

    file_path: str
    caption: str
    figure_number: Optional[str] = None
    base64: Optional[str] = None


@dataclass
class Question:
    number: int
    text: str


@dataclass
class RetrievalContext:
    text_chunks: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
