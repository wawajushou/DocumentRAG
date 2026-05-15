"""
retrievers/base.py

检索器抽象基类 - 定义统一的检索接口

设计模式：策略模式 (Strategy Pattern)
- 所有检索器都实现相同的接口
- HybridRetriever 组合多个检索器
- 新增检索方式只需实现此基类
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from langchain_core.documents import Document


class RetrievalResult:
    """
    检索结果封装

    包含文档和对应的相似度分数，用于后续的融合和过滤。

    Attributes:
        document: 检索到的文档
        score: 相似度分数（0-1，越高越相似）
    """

    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

    def __repr__(self):
        return f"RetrievalResult(score={self.score:.4f}, content={self.document.page_content[:50]}...)"


class BaseRetriever(ABC):
    """
    检索器抽象基类

    所有检索器（稠密向量、BM25、混合等）都必须实现此接口。

    使用方法：
        class MyCustomRetriever(BaseRetriever):
            @property
            def retriever_name(self) -> str:
                return "MyCustom"

            def retrieve(self, query, k=5):
                # 实现检索逻辑
                ...
    """

    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """返回检索器名称，用于日志"""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            RetrievalResult 列表，按分数降序排列
        """
        ...
