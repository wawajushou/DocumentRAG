"""LLM 客户端接口。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class LLMClient(Protocol):
    """只暴露 RAG 需要的两个能力。"""

    @property
    def name(self) -> str:
        """用于日志和界面展示，例如 'deepseek/deepseek-flash'。"""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> str:
        """一次性返回完整回答。thinking=None 跟随配置，False 强制关闭。"""

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> Iterator[str]:
        """逐块返回回答文本。"""
