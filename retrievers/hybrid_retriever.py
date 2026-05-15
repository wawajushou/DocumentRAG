"""
retrievers/hybrid_retriever.py

混合检索器 - 融合稠密向量检索和 BM25 关键词检索

使用 RRF (Reciprocal Rank Fusion) 算法合并两路检索结果。

RRF 算法原理：
对于每个文档 d，计算其在所有检索器中的 RRF 分数：
    RRF_score(d) = Σ 1/(k + rank_i(d))

其中：
- k 是平滑常数（通常取 60，来自原始论文）
- rank_i(d) 是文档 d 在第 i 个检索器中的排名（从 1 开始）

最终按 RRF 分数降序排列，取 Top-K 结果。

优点：
- 不需要对不同检索器的分数进行归一化
- 对排名靠前的文档给予更高权重
- 简单有效，被广泛使用（如 Elasticsearch、RAG-Fusion）
"""

from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document

from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    融合稠密向量检索和 BM25 关键词检索的优势：
    - 稠密向量：理解语义相似性
    - BM25：精确匹配专有名词

    使用 RRF 算法合并两路结果。

    使用方法：
        hybrid = HybridRetriever(dense_retriever, bm25_retriever)
        results = hybrid.retrieve("仪表盘通信协议", k=5)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """
        初始化混合检索器

        Args:
            dense_retriever: 稠密向量检索器
            bm25_retriever: BM25 检索器
            rrf_k: RRF 平滑常数（默认 60，来自原始论文）
            dense_weight: 稠密检索权重（用于调整两路的重要性）
            bm25_weight: BM25 检索权重
        """
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    @property
    def retriever_name(self) -> str:
        """返回检索器名称"""
        return "Hybrid (混合检索 + RRF)"

    def _rrf_fusion(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
    ) -> List[Tuple[Document, float]]:
        """
        RRF 融合算法

        将两路检索结果的排名转换为 RRF 分数，然后合并。

        Args:
            dense_results: 稠密检索结果
            bm25_results: BM25 检索结果

        Returns:
            (文档, RRF分数) 元组列表，按分数降序排列
        """
        # 文档 ID -> (文档对象, RRF 分数)
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        # 处理稠密检索结果
        for rank, result in enumerate(dense_results):
            doc_id = self._get_doc_id(result.document)
            rrf_score = self._dense_weight / (self._rrf_k + rank + 1)

            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0], doc_scores[doc_id][1] + rrf_score)
            else:
                doc_scores[doc_id] = (result.document, rrf_score)

        # 处理 BM25 检索结果
        for rank, result in enumerate(bm25_results):
            doc_id = self._get_doc_id(result.document)
            rrf_score = self._bm25_weight / (self._rrf_k + rank + 1)

            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0], doc_scores[doc_id][1] + rrf_score)
            else:
                doc_scores[doc_id] = (result.document, rrf_score)

        # 按 RRF 分数降序排列
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results

    def _get_doc_id(self, doc: Document) -> str:
        """
        生成文档的唯一标识

        使用内容哈希作为文档 ID，用于 RRF 融合时的去重。
        """
        import hashlib
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        执行混合检索

        流程：
        1. 分别执行稠密检索和 BM25 检索
        2. 使用 RRF 算法合并结果
        3. 返回 Top-K 结果

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            RetrievalResult 列表，按 RRF 分数降序排列
        """
        # 1. 分别执行两路检索（每路取更多结果，以便 RRF 融合后有足够候选）
        dense_results = self._dense_retriever.retrieve(query, k=k*2)
        bm25_results = self._bm25_retriever.retrieve(query, k=k*2)

        # 2. RRF 融合
        fused_results = self._rrf_fusion(dense_results, bm25_results)

        # 3. 取 Top-K
        retrieval_results = []
        for doc, score in fused_results[:k]:
            retrieval_results.append(RetrievalResult(
                document=doc,
                score=score,
            ))

        return retrieval_results
