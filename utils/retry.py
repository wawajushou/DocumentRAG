"""
utils/retry.py

重试装饰器模块 - 使用 tenacity 实现智能重试

特点：
1. 指数退避：等待时间指数增长，避免频繁重试
2. 条件重试：仅对可重试的异常进行重试
3. 日志记录：记录重试次数和原因
4. 可配置：支持自定义重试次数、等待时间等
"""

from typing import Type, Tuple, Callable, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
from loguru import logger


# 可重试的异常类型（网络错误、限流等）
RETRIABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
)


def create_api_retry_decorator(
    max_attempts: int = 3,
    min_wait: int = 2,
    max_wait: int = 10,
    retriable_exceptions: Tuple[Type[Exception], ...] = RETRIABLE_EXCEPTIONS,
) -> Callable:
    """
    创建 API 调用重试装饰器

    Args:
        max_attempts: 最大重试次数
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        retriable_exceptions: 可重试的异常类型

    Returns:
        装饰器函数
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retriable_exceptions),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


# 预定义的重试装饰器
api_retry = create_api_retry_decorator(max_attempts=3, min_wait=2, max_wait=10)
