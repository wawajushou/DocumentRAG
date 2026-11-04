"""检索器接口。"""

from __future__ import annotations

from typing import Protocol

from docrag.types import Hit


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, k: int) -> list[Hit]:
        """返回按相关度降序排列的结果。"""
