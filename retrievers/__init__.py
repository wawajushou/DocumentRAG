"""
retrievers/__init__.py

检索器模块 - 提供多种检索策略

包含：
- BaseRetriever: 检索器抽象基类
- RetrievalResult: 检索结果封装
- DenseRetriever: 稠密向量检索器
- BM25Retriever: BM25 关键词检索器
- HybridRetriever: 混合检索器（RRF 融合）
"""

from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
]
