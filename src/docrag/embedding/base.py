"""Embedding 接口。

刻意区分 embed_documents / embed_query：
Qwen3-Embedding 是 instruction-aware 的，查询侧要加指令前缀，
文档侧不能加。旧代码用 HuggingFaceEmbeddings 一把梭，天然拿不到这个收益。
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    @property
    def dim(self) -> int:
        """向量维度。"""

    @property
    def name(self) -> str:
        """模型标识，用于写进向量库元数据，检测到变化时提示重建。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档块。"""

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询。"""
