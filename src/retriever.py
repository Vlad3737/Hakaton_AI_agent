"""STUB Dev2. Заменить на реальный retrieve_context."""

from __future__ import annotations

from src.indexer import ArticleIndex
from src.models import RetrievalContext


def retrieve_context(
    question: str,
    index: ArticleIndex,
    top_k_text: int = 4,
    top_k_fig: int = 2,
) -> RetrievalContext:
    chunks = index.search_text(question, k=top_k_text)
    figures = index.search_figures(question, k=top_k_fig)
    return RetrievalContext(text_chunks=chunks, figures=figures)
