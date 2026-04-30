"""STUB Dev2. Заменить на реальный answer_question_sync с GigaChat-2-Max."""

from __future__ import annotations

import time

from src.models import RetrievalContext


def answer_question_sync(question: str, context: RetrievalContext) -> str:
    time.sleep(0.2)
    return f"[stub answer for: {question[:80]}] (chunks={len(context.text_chunks)}, figures={len(context.figures)})"
