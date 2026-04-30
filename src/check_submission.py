"""Локальная самопроверка: .env, output/answers.txt, число ответов = число вопросов."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
QUESTIONS_FILE = ROOT / "data" / "questions.txt"
ANSWERS_FILE = ROOT / "output" / "answers.txt"


class SubmissionCheckError(RuntimeError):
    pass


def check_submission(
    env_file: Path = ENV_FILE,
    questions_file: Path = QUESTIONS_FILE,
    answers_file: Path = ANSWERS_FILE,
) -> int:
    """Возвращает число ответов. Бросает SubmissionCheckError при провале."""
    if not env_file.exists():
        raise SubmissionCheckError(f"missing {env_file}")
    if not questions_file.exists():
        raise SubmissionCheckError(f"missing {questions_file}")
    if not answers_file.exists():
        raise SubmissionCheckError(f"missing {answers_file}")

    questions = [
        ln for ln in questions_file.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    answers_text = answers_file.read_text(encoding="utf-8")
    headers = re.findall(r"^##\s*Answer\s+(\d+)\s*$", answers_text, flags=re.MULTILINE)

    # Считаем число вопросов так же, как question_parser: по головам "N." / "N)".
    num_questions = sum(1 for ln in questions if re.match(r"^\s*\d+[\.\)]", ln))
    if num_questions == 0:
        num_questions = len(questions)

    if len(headers) != num_questions:
        raise SubmissionCheckError(
            f"answers={len(headers)} != questions={num_questions}"
        )
    expected = [str(i) for i in range(1, num_questions + 1)]
    if headers != expected:
        raise SubmissionCheckError(f"answer indices not sequential: {headers}")

    return len(headers)


def main() -> None:
    try:
        n = check_submission()
    except SubmissionCheckError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    print(f"[OK] {n} answers")


if __name__ == "__main__":
    main()
