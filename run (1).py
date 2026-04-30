"""Точка входа: `python run.py` — генерация `output/answers.txt`.

Оркестрирует пайплайн:
    1) load_config -> .env, логгер, дедлайны;
    2) parse_questions(data/questions.txt);
    3) create_article_index(data_dir, persist_dir) — синхронно, до event loop;
    4) async-обработка вопросов с семафором и общим дедлайном;
    5) write_answers(output/answers.txt);
    6) check_submission.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from loguru import logger

from src.answerer import answer_question_sync
from src.config import Config, load_config
from src.indexer import ArticleIndex, create_article_index
from src.models import Question
from src.question_parser import parse_questions
from src.retriever import retrieve_context
from src.utils.check_submission import SubmissionCheckError, check_submission


def write_answers(answers: list[str], questions: list[Question], out_dir: str) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / "answers.txt"
    with out.open("w", encoding="utf-8") as fh:
        for q, ans in zip(questions, answers):
            text = (ans or "").strip() or "no answer"
            fh.write(f"## Answer {q.number}\n{text}\n\n")
    return out


async def _process_one(
    q: Question,
    index: ArticleIndex,
    semaphore: asyncio.Semaphore,
    deadline: float,
    per_q_budget: float,
) -> tuple[int, str]:
    now = time.monotonic()
    timeout = min(per_q_budget, deadline - now)
    if timeout <= 0:
        logger.warning(f"Q{q.number}: deadline exceeded before start")
        return q.number, "no answer"

    try:
        async with semaphore:
            now = time.monotonic()
            timeout = min(per_q_budget, deadline - now)
            if timeout <= 0:
                return q.number, "no answer"
            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            ctx = await loop.run_in_executor(
                None, retrieve_context, q.text, index, 4, 2
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return q.number, "no answer"
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, answer_question_sync, q.text, ctx),
                timeout=min(timeout, remaining),
            )
            logger.info(f"Q{q.number} done in {time.monotonic() - t0:.1f}s")
            return q.number, answer
    except asyncio.TimeoutError:
        logger.warning(f"Q{q.number}: timeout")
        return q.number, "no answer"
    except Exception as e:
        logger.exception(f"Q{q.number}: failed: {e}")
        return q.number, "no answer"


async def _process_all(
    questions: list[Question], index: ArticleIndex, cfg: Config, deadline: float
) -> list[str]:
    semaphore = asyncio.Semaphore(cfg.max_concurrent_requests)
    remaining = max(deadline - time.monotonic(), 1.0)
    per_q_budget = (remaining / max(len(questions), 1)) * 0.9 * cfg.max_concurrent_requests

    tasks = [
        _process_one(q, index, semaphore, deadline, per_q_budget) for q in questions
    ]
    results = await asyncio.gather(*tasks)
    answers_map = {num: ans for num, ans in results}
    return [answers_map.get(q.number, "no answer") for q in questions]


def main() -> None:
    cfg = load_config()
    overall_start = time.monotonic()
    deadline = overall_start + cfg.max_total_time_seconds

    questions = parse_questions(os.path.join(cfg.data_dir, "questions.txt"))
    logger.info(f"loaded {len(questions)} questions")

    with tempfile.TemporaryDirectory(prefix="chroma_") as persist_dir:
        logger.info("indexing article...")
        index = create_article_index(cfg.data_dir, persist_dir)

        if index is None:
            logger.error("indexing failed; writing 'no answer' for all questions")
            answers: list[str] = ["no answer"] * len(questions)
        else:
            logger.info(
                f"indexing done in {time.monotonic() - overall_start:.1f}s; "
                f"deadline in {deadline - time.monotonic():.0f}s"
            )
            answers = asyncio.run(_process_all(questions, index, cfg, deadline))

    out_path = write_answers(answers, questions, cfg.output_dir)
    logger.info(f"answers written: {out_path}")

    try:
        n = check_submission(
            env_file=Path(".env"),
            questions_file=Path(cfg.data_dir) / "questions.txt",
            answers_file=out_path,
        )
        logger.info(f"submission check OK: {n} answers")
    except SubmissionCheckError as e:
        logger.error(f"submission check FAILED: {e}")


if __name__ == "__main__":
    main()
