"""结构感知切分。

- 保留表格与代码块的完整结构
- 依据 docx 标题样式维护章节栈，块上带 "章 > 节 > 小节" 的 header
- 超长表格按行切分并重复表头
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from docrag.ingest.readers import Page

Kind = Literal["text", "table", "code", "list"]

# 正则兜底：Markdown、1.2.3、一、、第一章
_HEADER_PATTERNS = (
    re.compile(r"^#{1,6}\s+\S.{0,58}$"),
    re.compile(r"^\d+(?:\.\d+)*[\.\s]\s*[^\d\s].{0,30}$"),
    re.compile(r"^[一二三四五六七八九十]+[、.]\s*(?![一二三四五六七八九十百千\d])\S.{0,30}$"),
    re.compile(r"^第[一二三四五六七八九十百\d]+[章节篇]\s*\S.{0,40}$"),
)

_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.、)])\s+\S")
_MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# 切分策略版本：改了切分逻辑就 +1，让 manifest 自动触发重建
SPLITTER_VERSION = 2


def _is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    # 以句读符号结尾的通常是正文，不是章节标题
    if stripped.endswith(("。", "；", "，", "：", "？", "！", ":", ";", ",", ".")):
        return False
    return any(pattern.match(stripped) for pattern in _HEADER_PATTERNS)


def _guess_level(line: str) -> int:
    """纯文本标题没有样式信息，按编号点数推断层级。"""
    match = re.match(r"^(\d+(?:\.\d+)*)", line.strip())
    if match:
        return min(match.group(1).count(".") + 1, 6)
    return 1


def header_level(line: str) -> int | None:
    """返回标题级别；不是标题返回 None。"""
    stripped = line.strip()
    markdown = _MARKDOWN_HEADER_RE.match(stripped)
    if markdown:
        return len(markdown.group(1))
    if _is_header(stripped):
        return _guess_level(stripped)
    return None


@dataclass(slots=True, frozen=True)
class Draft:
    text: str
    header: str = ""
    content_type: Kind = "text"
    page: int | None = None


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # 只去行尾空白
    return text


def _blocks(text: str) -> list[tuple[str, Kind]]:
    """把一页文本拆成结构块：代码块、表格、普通段落。"""
    lines = _clean(text).split("\n")
    result: list[tuple[str, Kind]] = []
    buffer: list[str] = []
    kind: Kind = "text"
    in_code = False

    def flush() -> None:
        nonlocal buffer, kind
        body = "\n".join(buffer).strip("\n")
        if body.strip():
            result.append((body, kind))
        buffer = []
        kind = "text"

    for line in lines:
        stripped = line.strip()

        # 代码块整体保留
        if stripped.startswith("```"):
            if in_code:
                buffer.append(line)
                flush()
                in_code = False
            else:
                flush()
                kind = "code"
                buffer.append(line)
                in_code = True
            continue
        if in_code:
            buffer.append(line)
            continue

        # 连续含 | 的行视为表格
        is_table_line = "|" in line and line.count("|") >= 2
        if is_table_line:
            if kind != "table":
                flush()
                kind = "table"
            buffer.append(line)
            continue
        if kind == "table":
            flush()

        if not stripped:
            flush()
            continue

        buffer.append(line)
        if _LIST_RE.match(line):
            kind = "list"

    flush()
    return result


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """超长文本的兜底切分：优先在句末断，其次硬切。"""
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            cut = max(window.rfind(ch) for ch in "。！？；\n") if window else -1
            if cut > chunk_size // 2:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _split_table(text: str, max_chars: int) -> list[str]:
    """超长表格按行切分，每片都重复表头。

    直接硬切会把表头和数据行切断，检索时看不懂列的语义。
    """
    if len(text) <= max_chars:
        return [text]

    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 3:
        return _hard_split(text, max_chars, 0)

    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    buffer: list[str] = []
    for row in rows:
        candidate = len("\n".join(header + buffer + [row]))
        if buffer and candidate > max_chars:
            pieces.append("\n".join(header + buffer))
            buffer = [row]
        else:
            buffer.append(row)
    if buffer:
        pieces.append("\n".join(header + buffer))
    return pieces


def _pack(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把短段落聚合成接近 chunk_size 的块，避免碎成一句一块。"""
    packed: list[str] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > chunk_size:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_hard_split(para, chunk_size, overlap))
            continue
        if not buffer:
            buffer = para
        elif len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}"
        else:
            packed.append(buffer)
            tail = buffer[-overlap:] if overlap else ""
            merged = f"{tail}\n{para}".strip() if tail else para
            if len(merged) > chunk_size:
                packed.extend(_hard_split(merged, chunk_size, overlap))
                buffer = ""
            else:
                buffer = merged
    if buffer:
        packed.append(buffer)
    return packed


def split_pages(
    pages: list[Page],
    chunk_size: int = 400,
    overlap: int = 60,
    max_chunk_chars: int = 1200,
) -> list[Draft]:
    drafts: list[Draft] = []
    stack: dict[int, str] = {}

    def current_header() -> str:
        if not stack:
            return ""
        levels = sorted(stack)[-3:]  # 最多保留 3 级，避免 header 过长
        return " > ".join(stack[level] for level in levels)

    def apply_header(line: str, level: int) -> None:
        stripped = line.strip()
        markdown = _MARKDOWN_HEADER_RE.match(stripped)
        title = markdown.group(2).strip() if markdown else stripped
        stack[level] = title
        for deeper in [key for key in stack if key > level]:
            del stack[deeper]

    def flush(paragraphs: list[tuple[str, Kind]], page_number: int | None) -> None:
        texts = [text for text, kind in paragraphs if kind in ("text", "list")]
        # 只有标题、没有正文的块直接丢弃，避免空壳块占用检索名额
        if not any(not text.strip().startswith("#") for text in texts):
            return
        for chunk in _pack(texts, chunk_size, overlap):
            drafts.append(
                Draft(
                    text=chunk,
                    header=current_header(),
                    content_type=(
                        "list" if _LIST_RE.match(chunk.split("\n")[0]) else "text"
                    ),
                    page=page_number,
                )
            )

    for page in pages:
        paragraphs: list[tuple[str, Kind]] = []

        for block_text, kind in _blocks(page.text):
            if kind in ("table", "code"):
                flush(paragraphs, page.page)
                paragraphs = []
                if kind == "table":
                    pieces = _split_table(block_text, max_chunk_chars)
                elif len(block_text) <= max_chunk_chars:
                    pieces = [block_text]
                else:
                    pieces = _hard_split(block_text, max_chunk_chars, 0)

                for piece in pieces:
                    drafts.append(
                        Draft(
                            text=piece,
                            header=current_header(),
                            content_type=kind,
                            page=page.page,
                        )
                    )
                continue

            for line in block_text.split("\n"):
                level = header_level(line)
                if level is not None:
                    if level <= 2:
                        # 一级/二级标题作为强边界：先收尾，标题只进 header 元数据
                        flush(paragraphs, page.page)
                        paragraphs = []
                        apply_header(line, level)
                    else:
                        # 三级以下不断开，标题保留在正文里作为小节标记
                        apply_header(line, level)
                        paragraphs.append((line, kind))
                    continue
                paragraphs.append((line, kind))

        flush(paragraphs, page.page)

    return [draft for draft in drafts if draft.text.strip()]
