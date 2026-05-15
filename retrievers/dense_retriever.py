"""
retrievers/dense_retriever.py

稠密向量检索器 - 基于 Embedding 模型的语义检索

使用 ChromaDB 进行向量相似度搜索。
"""

from typing import List, Optional
from langchain_core.documents import Document

from retrievers.base import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """
    稠密向量检索器

    使用 Embedding 模型将查询和文档转换为向量，
    然后通过向量相似度（余弦相似度）进行检索。

    优点：
    - 能够理解语义相似性（如"汽车"和"车辆"）
    - 对同义词、近义词有较好的召回效果

    缺点：
    - 对精确匹配的专有名词（如错误码、芯片型号）效果有限
    - 依赖 Embedding 模型的质量

    使用方法：
        retriever = DenseRetriever(vectorstore)
        results = retriever.retrieve("仪表盘通信协议", k=5)
    """

    def __init__(self, vectorstore):
        """
        初始化稠密向量检索器

        Args:
            vectorstore: ChromaDB 向量存储实例
        """
        self._vectorstore = vectorstore

    @property
    def retriever_name(self) -> str:
        """返回检索器名称"""
        return "Dense (向量检索)"

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        执行稠密向量检索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            RetrievalResult 列表，按相似度分数降序排列
        """
        if self._vectorstore is None:
            return []

        # 使用 ChromaDB 的 similarity_search_with_relevance_scores
        # 返回 (document, score) 元组，score 是余弦相似度（0-1，越高越相似）
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        retrieval_results = []
        for doc, score in results:
            retrieval_results.append(RetrievalResult(
                document=doc,
                score=score,
            ))

        return retrieval_results
