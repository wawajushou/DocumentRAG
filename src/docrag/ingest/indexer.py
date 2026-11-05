"""增量索引。

修改文件时先删除旧块再写入；删除或重命名的文件会同步清理。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

from docrag.config import Settings
from docrag.embedding.base import Embedder
from docrag.ingest.manifest import Manifest
from docrag.ingest.readers import read_document, scan_documents
from docrag.ingest.splitter import SPLITTER_VERSION, split_pages
from docrag.store.chroma import ChromaStore
from docrag.types import Chunk

EMBED_BATCH = 64


@dataclass(slots=True)
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: int = 0
    chunks: int = 0
    seconds: float = 0.0

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.removed)


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaStore,
        manifest: Manifest,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.manifest = manifest

    def sync(self, progress: Callable[[str], None] | None = None) -> SyncReport:
        started = time.perf_counter()
        report = SyncReport()
        notify = progress or logger.info

        if self.manifest.needs_rebuild:
            notify("embedding 或切分参数已变更，重建整个向量库")
            self.store.wipe()
            self.manifest.reset()

        files = scan_documents(self.settings.data_dir)
        present = {
            path.relative_to(self.settings.data_dir).as_posix() for path in files
        }

        # 1. 清理已删除的文件
        for name in list(self.manifest.files):
            if name not in present:
                self.store.delete_file(name)
                self.manifest.remove(name)
                report.removed.append(name)
                notify(f"移除: {name}")

        # 2. 处理新增 / 修改
        for path in files:
            rel = path.relative_to(self.settings.data_dir).as_posix()
            try:
                pages = read_document(path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"读取失败 {rel}: {exc}")
                continue

            text = "\n".join(page.text for page in pages)
            if not text.strip():
                logger.warning(f"跳过空文档: {rel}")
                continue

            doc_id = hashlib.sha256(f"{rel}\0{text}".encode()).hexdigest()[:16]
            entry = self.manifest.get(rel)
            if entry and entry.doc_id == doc_id:
                report.skipped += 1
                continue

            is_update = entry is not None
            notify(f"{'更新' if is_update else '新增'}: {rel}")

            drafts = split_pages(
                pages,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
                max_chunk_chars=self.settings.max_chunk_chars,
            )
            chunks = [
                Chunk(
                    id=f"{doc_id}:{index}",
                    text=draft.text,
                    doc_id=doc_id,
                    file=rel,
                    chunk_index=index,
                    header=draft.header,
                    content_type=draft.content_type,
                    page=draft.page,
                )
                for index, draft in enumerate(drafts)
            ]

            # 先删旧块再写新块：单进程顺序执行，不会出现新旧版本共存
            self.store.delete_file(rel)
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                vectors = self.embedder.embed_documents(
                    [self._embed_text(chunk) for chunk in batch]
                )
                self.store.upsert(batch, vectors)

            self.manifest.put(rel, doc_id, len(chunks))
            report.chunks += len(chunks)
            (report.updated if is_update else report.added).append(rel)

        self.manifest.save()
        report.seconds = time.perf_counter() - started
        logger.info(
            f"索引完成: 新增 {len(report.added)} / 更新 {len(report.updated)} / "
            f"删除 {len(report.removed)} / 跳过 {report.skipped} / "
            f"{report.chunks} 块 / {report.seconds:.1f}s"
        )
        return report

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        """把章节标题拼进 embedding 文本，提升上下文召回。"""
        return f"{chunk.header}\n{chunk.text}" if chunk.header else chunk.text


def build_manifest(settings: Settings, embedder: Embedder) -> Manifest:
    strategy = (
        f"cs={settings.chunk_size};ov={settings.chunk_overlap};"
        f"mc={settings.max_chunk_chars};v={SPLITTER_VERSION}"
    )
    return Manifest(settings.storage_dir / "manifest.json", embedder.name, strategy)
