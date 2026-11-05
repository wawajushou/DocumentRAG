"""Dense + BM25 混合检索，RRF 融合。

RRF 只看排名，不需要把两种分数归一化到同一量纲。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docrag.types import Hit

if TYPE_CHECKING:  # 仅类型检查时导入，避免 fuse() 的单元测试被 jieba/chromadb 拖累
    from docrag.retrieve.bm25 import BM25Index
    from docrag.store.chroma import ChromaStore


class HybridRetriever:
    def __init__(
        self,
        store: ChromaStore,
        bm25: BM25Index,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self._store = store
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    @property
    def name(self) -> str:
        return "hybrid"

    def dense(self, query: str, k: int) -> list[Hit]:
        vector = self._store.embedder.embed_query(query)
        return [
            Hit(chunk=chunk, score=score, channels=("dense",))
            for chunk, score in self._store.search(vector, k)
        ]

    def retrieve(self, query: str, k: int, candidate_k: int = 30) -> list[Hit]:
        dense_hits = self.dense(query, candidate_k)
        bm25_hits = self._bm25.retrieve(query, candidate_k)
        return fuse(
            dense_hits,
            bm25_hits,
            k=k,
            rrf_k=self._rrf_k,
            dense_weight=self._dense_weight,
            bm25_weight=self._bm25_weight,
        )


def fuse(
    dense_hits: list[Hit],
    bm25_hits: list[Hit],
    k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[Hit]:
    """RRF 融合。

    score(d) = Σ weight_i / (rrf_k + rank_i(d))
    """
    merged: dict[str, Hit] = {}

    channels = (("dense", dense_weight, dense_hits), ("bm25", bm25_weight, bm25_hits))
    for channel, weight, hits in channels:
        for rank, hit in enumerate(hits, start=1):
            gain = weight / (rrf_k + rank)
            tags = tuple(dict.fromkeys((*hit.channels, channel)))
            existing = merged.get(hit.chunk.id)
            if existing is None:
                merged[hit.chunk.id] = Hit(chunk=hit.chunk, score=gain, channels=tags)
            else:
                existing.score += gain
                existing.channels = tuple(dict.fromkeys(existing.channels + tags))

    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ordered[:k]
