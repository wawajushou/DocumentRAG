"""
retrievers/bm25_retriever.py

BM25 关键词检索器 - 基于词频的经典检索算法

BM25 (Best Matching 25) 是一种基于词频的检索算法，
特别适合精确匹配专有名词、错误码、芯片型号等。

算法原理：
1. 对文档进行分词
2. 统计查询词在每个文档中的词频（TF）
3. 计算逆文档频率（IDF）
4. 综合 TF 和 IDF 计算相关性分数
"""

import hashlib
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from retrievers.base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """
    BM25 关键词检索器

    使用 BM25 算法进行基于关键词的检索。

    优点：
    - 对精确匹配的专有名词效果极佳
    - 不需要 GPU，计算速度快
    - 无须训练，开箱即用

    缺点：
    - 无法理解语义相似性（如"汽车"和"车辆"会被视为不同词）
    - 对分词质量敏感

    使用方法：
        retriever = BM25Retriever()
        retriever.build_index(documents)
        results = retriever.retrieve("CAN总线", k=5)
    """

    def __init__(self):
        """初始化 BM25 检索器"""
        self._documents: List[Document] = []      # 存储所有文档
        self._bm25: Optional[BM25Okapi] = None    # BM25 索引
        self._doc_hashes: set = set()              # 用于去重

    @property
    def retriever_name(self) -> str:
        """返回检索器名称"""
        return "BM25 (关键词检索)"

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        使用 jieba 精确模式进行中文分词，
        比字符级分词更准确，能识别专业术语。

        Args:
            text: 原始文本

        Returns:
            分词结果列表
        """
        import jieba

        # jieba 精确模式分词
        tokens = jieba.lcut(text)

        # 过滤空白和单字符（保留有意义的单字词如 "车"、"灯"）
        result = []
        for token in tokens:
            token = token.strip().lower()
            if token and len(token) >= 1:
                result.append(token)

        return result

    def build_index(self, documents: List[Document]) -> None:
        """
        构建 BM25 索引

        Args:
            documents: 文档列表
        """
        # 去重
        new_docs = []
        for doc in documents:
            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc_hash not in self._doc_hashes:
                self._doc_hashes.add(doc_hash)
                new_docs.append(doc)

        if not new_docs:
            return

        self._documents.extend(new_docs)

        # 对所有文档进行分词
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self._documents]

        # 构建 BM25 索引
        self._bm25 = BM25Okapi(tokenized_docs)

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        执行 BM25 检索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            RetrievalResult 列表，按 BM25 分数降序排列
        """
        if self._bm25 is None or len(self._documents) == 0:
            return []

        # 对查询进行分词
        query_tokens = self._tokenize(query)

        # 计算 BM25 分数
        scores = self._bm25.get_scores(query_tokens)

        # 获取 Top-K 结果
        top_k_indices = scores.argsort()[-k:][::-1]

        retrieval_results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # 只返回分数大于 0 的结果
                retrieval_results.append(RetrievalResult(
                    document=self._documents[idx],
                    score=float(scores[idx]),
                ))

        return retrieval_results
