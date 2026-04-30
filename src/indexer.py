"""STUB Dev1. Заменить на реальную реализацию ArticleIndex + create_article_index."""

from __future__ import annotations

from typing import Optional

from src.models import Figure


class ArticleIndex:
    def __init__(self) -> None:
        self._figure_map: dict[str, Figure] = {}

    def search_text(self, query: str, k: int = 4) -> list[str]:
        return [f"[stub chunk for query: {query[:60]}]"] * k

    def search_figures(self, query: str, k: int = 2) -> list[Figure]:
        return []

    def get_figure_by_number(self, number: str) -> Optional[Figure]:
        return self._figure_map.get(number)


def create_article_index(data_dir: str, persist_dir: str) -> Optional[ArticleIndex]:
    return ArticleIndex()
