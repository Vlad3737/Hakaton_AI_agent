"""Парсер файла с вопросами."""

from __future__ import annotations

import re
from pathlib import Path

from src.models import Question

_QUESTION_HEAD = re.compile(r"^\s*(\d+)[\.\)]\s*(.*)$")


def parse_questions(filepath: str | Path) -> list[Question]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"questions file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"questions file is empty: {path}")

    questions: list[Question] = []
    current_num: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_num is not None:
            text = " ".join(s.strip() for s in current_lines).strip()
            if text:
                questions.append(Question(number=current_num, text=text))

    for line in raw.splitlines():
        m = _QUESTION_HEAD.match(line)
        if m:
            flush()
            current_num = int(m.group(1))
            current_lines = [m.group(2)]
        else:
            if current_num is not None:
                current_lines.append(line)
    flush()

    if not questions:
        raise ValueError(f"no questions parsed from {path}")

    return questions
