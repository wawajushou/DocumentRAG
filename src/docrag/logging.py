"""日志初始化。控制台输出 WARNING 以上，完整日志写文件，ERROR 单独一份。"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(log_dir: Path | str = "./logs", level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Windows 控制台默认 GBK，先尝试切 UTF-8；失败就退回纯文本输出
    with contextlib.suppress(AttributeError, OSError):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        colorize=False,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
    )
    logger.add(
        log_path / "docrag_{time:YYYY-MM-DD}.log",
        level=level,
        encoding="utf-8",
        rotation="1 day",
        retention="14 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        encoding="utf-8",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {name}:{function}:{line} | {message}\n{exception}",
    )
    _CONFIGURED = True


def get_logger():
    return logger
