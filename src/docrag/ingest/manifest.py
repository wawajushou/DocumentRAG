"""文件清单：记录每个文档的内容哈希和块数量。

作用：
1. 增量更新只处理真正改过的文件（内容哈希，不是 mtime）
2. 文档删除 / 改名时能清理对应的旧向量
3. 记录 embedding 指纹，换模型时强制全量重建
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class FileEntry:
    doc_id: str
    chunks: int
    updated_at: str


class Manifest:
    VERSION = 1

    def __init__(self, path: Path, embedding: str):
        self.path = path
        self.embedding = embedding
        self.files: dict[str, FileEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, item in raw.get("files", {}).items():
            self.files[name] = FileEntry(
                doc_id=item["doc_id"],
                chunks=item.get("chunks", 0),
                updated_at=item.get("updated_at", ""),
            )

    @property
    def embedding_changed(self) -> bool:
        if not self.path.exists():
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return raw.get("embedding") != self.embedding

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "embedding": self.embedding,
            "files": {name: asdict(entry) for name, entry in sorted(self.files.items())},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, name: str) -> FileEntry | None:
        return self.files.get(name)

    def put(self, name: str, doc_id: str, chunks: int) -> None:
        self.files[name] = FileEntry(
            doc_id=doc_id,
            chunks=chunks,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def remove(self, name: str) -> None:
        self.files.pop(name, None)

    def reset(self) -> None:
        self.files.clear()
