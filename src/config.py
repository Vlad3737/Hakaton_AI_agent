"""Конфигурация и инициализация логгера."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger


@dataclass
class Config:
    gigachat_credentials: str
    gigachat_scope: str
    data_dir: str = "data"
    output_dir: str = "output"
    max_total_time_seconds: int = 900
    max_concurrent_requests: int = 2
    model_name: str = "GigaChat-2-Max"


def _setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_config() -> Config:
    load_dotenv()
    _setup_logger()

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")
    if not credentials:
        raise RuntimeError("GIGACHAT_CREDENTIALS is not set in environment / .env")
    if not scope:
        raise RuntimeError("GIGACHAT_SCOPE is not set in environment / .env")

    return Config(
        gigachat_credentials=credentials,
        gigachat_scope=scope,
        data_dir=os.getenv("DATA_DIR", "data"),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        max_total_time_seconds=int(os.getenv("MAX_TOTAL_TIME_SECONDS", "900")),
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "2")),
        model_name=os.getenv("MODEL_NAME", "GigaChat-2-Max"),
    )
