"""可选的重排层。

默认关闭：bge-reranker-v2-m3 在纯 CPU 上对 30 个候选打分约 1-3 秒，
对交互式问答偏慢。文档量大或召回噪声明显时再打开。
"""

from __future__ import annotations

from loguru import logger

from docrag.config import Settings
from docrag.embedding.local import resolve_model_path
from docrag.types import Hit


class Reranker:
    def __init__(self, settings: Settings):
        from sentence_transformers import CrossEncoder

        path = resolve_model_path(settings.rerank_model, settings.model_dir, settings.model_source)
        self._model = CrossEncoder(str(path), device=settings.embedding_device)
        self._batch = settings.embedding_batch_size
        self._name = settings.rerank_model
        logger.info(f"Reranker ready: {self._name}")

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict(
            [(query, hit.chunk.text) for hit in hits],
            batch_size=self._batch,
            show_progress_bar=False,
        )
        rescored = [
            Hit(chunk=hit.chunk, score=float(score), channels=hit.channels + ("rerank",))
            for hit, score in zip(hits, scores, strict=False)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:k]
