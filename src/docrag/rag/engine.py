"""检索增强生成引擎。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from loguru import logger

from docrag.config import Settings
from docrag.embedding.base import Embedder
from docrag.llm.base import LLMClient
from docrag.rag.prompts import ANSWER_PROMPT, REWRITE_PROMPT, build_context
from docrag.retrieve.bm25 import BM25Index
from docrag.retrieve.hybrid import HybridRetriever
from docrag.retrieve.rerank import Reranker
from docrag.store.chroma import ChromaStore
from docrag.types import Answer, Hit

_CITE_RE = re.compile(r"\[(\d+)\]")
_NUMBERING_RE = re.compile(r"^\s*(?:\d+[.)、]|[-*•])\s*")


@dataclass(slots=True)
class Prepared:
    question: str
    hits: list[Hit] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)


class RAGEngine:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: Embedder,
        store: ChromaStore,
        bm25: BM25Index,
        reranker: Reranker | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.retriever = HybridRetriever(
            store=store,
            bm25=bm25,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )

    # ---------- 检索 ----------

    def prepare(self, question: str) -> Prepared:
        sub_queries = self._rewrite(question) if self.settings.rewrite_enabled else [question]
        hits = self._retrieve(sub_queries)
        logger.info(f"prepare: {len(sub_queries)} queries -> {len(hits)} hits")
        return Prepared(question=question, hits=hits, sub_queries=sub_queries)

    def _rewrite(self, question: str) -> list[str]:
        try:
            raw = self.llm.complete(
                [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                temperature=0.2,
                thinking=False,  # 查询重写不需要思考模式，省一次长等待
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"查询重写失败，退回原问题: {exc}")
            return [question]

        candidates = []
        for line in raw.splitlines():
            line = _NUMBERING_RE.sub("", line.strip()).strip("\"'“”‘’ ")
            if not 3 <= len(line) <= 60 or line == question:
                continue
            if any(_jaccard(line, existing) > 0.7 for existing in candidates):
                continue
            candidates.append(line)

        return [question, *candidates[:3]]

    def _retrieve(self, queries: list[str]) -> list[Hit]:
        if self.store.count() == 0:
            return []

        merged: dict[str, Hit] = {}
        for query in queries:
            for hit in self.retriever.retrieve(query, k=self.settings.candidate_k):
                existing = merged.get(hit.chunk.id)
                if existing is None or hit.score > existing.score:
                    merged[hit.chunk.id] = Hit(
                        chunk=hit.chunk,
                        score=hit.score,
                        channels=tuple(dict.fromkeys(
                            (existing.channels if existing else ()) + hit.channels
                        )),
                    )
                else:
                    existing.channels = tuple(
                        dict.fromkeys(existing.channels + hit.channels)
                    )

        hits = sorted(merged.values(), key=lambda item: item.score, reverse=True)

        if self.settings.rerank_enabled and self.reranker is not None:
            hits = self.reranker.rerank(queries[0], hits, self.settings.retrieve_top_k)
        else:
            # 全局截断，避免多路召回把上下文撑爆
            hits = hits[: self.settings.retrieve_top_k]

        if self.settings.expand_neighbors:
            hits = self._expand_neighbors(hits)

        return hits

    def _expand_neighbors(self, hits: list[Hit]) -> list[Hit]:
        """长表格分片时，把命中块的相邻分片一起带上，保证表格内容完整。"""
        expanded: list[Hit] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk.id not in seen:
                expanded.append(hit)
                seen.add(hit.chunk.id)
            if hit.chunk.content_type != "table":
                continue
            for sibling in self.store.table_siblings(hit.chunk):
                if sibling.id in seen:
                    continue
                expanded.append(
                    Hit(chunk=sibling, score=hit.score * 0.98, channels=("sibling",))
                )
                seen.add(sibling.id)
        return expanded[: self.settings.retrieve_top_k * 3]

    # ---------- 生成 ----------

    def stream(self, prepared: Prepared) -> Iterator[str]:
        if not prepared.hits:
            yield "当前知识库里没有检索到相关内容，请先把文档放进 data 目录并执行同步。"
            return

        prompt = ANSWER_PROMPT.format(
            context=build_context(prepared.hits),
            question=prepared.question,
        )
        yield from self.llm.stream([{"role": "user", "content": prompt}])

    def ask(self, question: str) -> Answer:
        prepared = self.prepare(question)
        if not prepared.hits:
            return Answer(
                text="当前知识库里没有检索到相关内容，请先把文档放进 data 目录并执行同步。",
                sub_queries=prepared.sub_queries,
            )

        prompt = ANSWER_PROMPT.format(
            context=build_context(prepared.hits),
            question=question,
        )
        text = self.llm.complete([{"role": "user", "content": prompt}])
        return Answer(
            text=text,
            hits=prepared.hits,
            sub_queries=prepared.sub_queries,
            cited=_extract_citations(text, len(prepared.hits)),
        )


def _extract_citations(text: str, total: int) -> list[int]:
    numbers = {int(match) for match in _CITE_RE.findall(text)}
    return sorted(number for number in numbers if 1 <= number <= total)


def _jaccard(a: str, b: str) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
