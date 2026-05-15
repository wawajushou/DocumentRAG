"""
utils/logger.py

日志配置模块 - 使用 loguru 实现结构化日志

特点：
1. 控制台输出：保留 Rich Panel 用于用户展示，日志文件用于调试
2. 日志轮转：按天轮转，保留 7 天
3. 分级存储：ERROR 级别单独存储
4. 结构化格式：包含时间、模块、函数、行号
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: str = "./logs",
    log_level: str = "DEBUG",
    rotation: str = "1 day",
    retention: str = "7 days",
) -> None:
    """
    配置日志系统

    Args:
        log_dir: 日志文件目录
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        rotation: 日志轮转策略
        retention: 日志保留策略
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 移除默认的控制台处理器（避免与 Rich 输出冲突）
    logger.remove()

    # 添加控制台处理器（仅输出 WARNING 及以上级别）
    logger.add(
        sys.stderr,
        level="WARNING",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # 添加文件处理器（输出所有级别）
    logger.add(
        log_path / "rag_{time:YYYY-MM-DD}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    # 添加 ERROR 级别单独存储
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )


def get_logger():
    """
    获取 logger 实例

    Returns:
        logger: loguru logger 实例
    """
    return logger
