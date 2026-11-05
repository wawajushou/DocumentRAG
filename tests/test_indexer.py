"""增量索引测试：改文档不产生重复块、删文件会清理。"""

from __future__ import annotations

from pathlib import Path

from docrag.config import Settings
from docrag.ingest.indexer import Indexer
from docrag.ingest.manifest import Manifest
from docrag.types import Chunk


class FakeEmbedder:
    name = "fake@4"
    dim = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def delete_file(self, file: str) -> None:
        for key in [k for k, v in self.chunks.items() if v.file == file]:
            del self.chunks[key]

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def count(self) -> int:
        return len(self.chunks)

    def wipe(self) -> None:
        self.chunks.clear()


def _build(tmp_path: Path, embedder_name: str = "fake@4"):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        storage_dir=tmp_path / "storage",
        chunk_size=200,
        chunk_overlap=20,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    embedder = FakeEmbedder()
    embedder.name = embedder_name
    store = FakeStore()
    manifest = Manifest(settings.storage_dir / "manifest.json", embedder.name)
    return settings, embedder, store, manifest


def test_initial_sync_adds_chunks(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    (settings.data_dir / "a.txt").write_text("第一段内容。\n\n第二段内容。", encoding="utf-8")

    indexer = Indexer(settings, embedder, store, manifest)
    report = indexer.sync()

    assert report.added == ["a.txt"]
    assert store.count() > 0


def test_unchanged_file_is_skipped(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    (settings.data_dir / "a.txt").write_text("内容不变。", encoding="utf-8")

    Indexer(settings, embedder, store, manifest).sync()
    before = store.count()

    reloaded = Manifest(settings.storage_dir / "manifest.json", embedder.name)
    report = Indexer(settings, embedder, store, reloaded).sync()

    assert report.skipped == 1
    assert store.count() == before


def test_modified_file_replaces_old_chunks(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    path = settings.data_dir / "a.txt"
    path.write_text("旧版本内容。", encoding="utf-8")
    Indexer(settings, embedder, store, manifest).sync()
    before = store.count()

    path.write_text("完全不同的新版本内容，长度也不一样。" * 20, encoding="utf-8")
    reloaded = Manifest(settings.storage_dir / "manifest.json", embedder.name)
    report = Indexer(settings, embedder, store, reloaded).sync()

    assert report.updated == ["a.txt"]
    # 关键：不应该出现旧块和新块共存
    assert all("旧版本" not in chunk.text for chunk in store.chunks.values())
    assert store.count() != before


def test_removed_file_cleans_chunks(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    path = settings.data_dir / "a.txt"
    path.write_text("待删除文档。", encoding="utf-8")
    Indexer(settings, embedder, store, manifest).sync()
    assert store.count() > 0

    path.unlink()
    reloaded = Manifest(settings.storage_dir / "manifest.json", embedder.name)
    report = Indexer(settings, embedder, store, reloaded).sync()

    assert report.removed == ["a.txt"]
    assert store.count() == 0


def test_embedding_change_wipes_store(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    (settings.data_dir / "a.txt").write_text("一些内容。", encoding="utf-8")
    Indexer(settings, embedder, store, manifest).sync()
    assert store.count() > 0

    # 换模型：manifest 记录的指纹和当前 embedder 不一致，应清库重建
    new_embedder = FakeEmbedder()
    new_embedder.name = "another@8"
    reloaded = Manifest(settings.storage_dir / "manifest.json", new_embedder.name)
    report = Indexer(settings, new_embedder, store, reloaded).sync()

    assert report.added == ["a.txt"]  # 清库后当作新增处理
    assert store.count() > 0


def test_same_text_in_two_files_not_merged(tmp_path):
    settings, embedder, store, manifest = _build(tmp_path)
    same = "两个文档里完全相同的通用段落。"
    (settings.data_dir / "a.txt").write_text(same, encoding="utf-8")
    (settings.data_dir / "b.txt").write_text(same, encoding="utf-8")

    Indexer(settings, embedder, store, manifest).sync()

    files = {chunk.file for chunk in store.chunks.values()}
    assert files == {"a.txt", "b.txt"}
    assert len({chunk.id for chunk in store.chunks.values()}) == store.count()
