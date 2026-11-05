"""核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class Chunk:
    """可检索的文本块。id = doc_id + 块序号，doc_id 由文件名与内容哈希生成。"""

    id: str
    text: str
    doc_id: str
    file: str
    chunk_index: int
    header: str = ""
    content_type: str = "text"  # text | table | code | list
    page: int | None = None


@dataclass(slots=True)
class Hit:
    """一条检索结果。"""

    chunk: Chunk
    score: float
    channels: tuple[str, ...] = ()  # 命中来源：dense / bm25 / rerank


@dataclass(slots=True)
class Answer:
    """一次问答的完整结果。"""

    text: str
    hits: list[Hit] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)  # 模型实际引用的来源序号（1-based）
