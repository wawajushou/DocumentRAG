"""Chroma 向量库封装。

直接使用 chromadb，不经过 langchain：
- 显式指定 cosine 空间（旧代码默认 L2，和归一化向量不匹配）
- metadata 只存 Chroma 支持的基础类型（page 用 -1 表示未知）
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from loguru import logger

from docrag.embedding.base import Embedder
from docrag.types import Chunk

COLLECTION = "docrag"


class ChromaStore:
    def __init__(self, storage_dir: Path, embedder: Embedder):
        storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(storage_dir / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def count(self) -> int:
        return self._collection.count()

    def wipe(self) -> None:
        logger.warning("清空向量库（embedding 模型已变更）")
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[self._metadata(c) for c in chunks],
        )

    def delete_file(self, file: str) -> None:
        self._collection.delete(where={"file": file})

    def iter_chunks(self, batch: int = 500):
        """按批遍历全部块，供 BM25 建索引使用。"""
        total = self.count()
        offset = 0
        while offset < total:
            data = self._collection.get(
                limit=batch,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = data.get("ids") or []
            if not ids:
                break
            for chunk_id, text, meta in zip(
                ids, data["documents"] or [], data["metadatas"] or [], strict=False
            ):
                yield self._to_chunk(chunk_id, text, meta)
            offset += len(ids)

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        hits = []
        for chunk_id, text, meta, distance in zip(ids, docs, metas, dists, strict=False):
            # cosine distance -> similarity
            hits.append((self._to_chunk(chunk_id, text, meta), 1.0 - float(distance)))
        return hits

    def table_siblings(self, chunk: Chunk, max_parts: int = 8) -> list[Chunk]:
        """取同一张表被切分后的连续分片。

        长表格在切分时变成多个 chunk_index 连续的块，只召回其中一片会导致
        模型得出"资料里没有完整清单"的错误结论。
        """
        window = 12
        data = self._collection.get(
            where={
                "$and": [
                    {"file": {"$eq": chunk.file}},
                    {"chunk_index": {"$gte": chunk.chunk_index - window}},
                    {"chunk_index": {"$lte": chunk.chunk_index + window}},
                ]
            },
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        siblings = [
            self._to_chunk(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                ids, data["documents"] or [], data["metadatas"] or [], strict=False
            )
            if meta.get("content_type") == "table"
        ]
        by_index = {item.chunk_index: item for item in siblings}

        run = [chunk]
        index = chunk.chunk_index - 1
        while index in by_index:
            run.insert(0, by_index[index])
            index -= 1
        index = chunk.chunk_index + 1
        while index in by_index:
            run.append(by_index[index])
            index += 1
        return run[:max_parts]

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        return {
            "file": chunk.file,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "header": chunk.header,
            "content_type": chunk.content_type,
            "page": chunk.page if chunk.page is not None else -1,
        }

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
        raw_page = meta.get("page", -1)
        page = int(raw_page) if raw_page is not None else -1
        return Chunk(
            id=chunk_id,
            text=text,
            doc_id=meta.get("doc_id", ""),
            file=meta.get("file", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            header=meta.get("header", ""),
            content_type=meta.get("content_type", "text"),
            page=page if page > 0 else None,
        )
