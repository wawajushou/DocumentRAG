"""文档读取：docx / pdf / txt / md。

docx 直接解析 document.xml：表格转 Markdown，heading 样式转 Markdown 标题，
toc 样式丢弃。PDF 页码从 1 开始。
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt", ".md"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class UnsupportedFormat(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class Page:
    text: str
    page: int | None = None


def read_document(path: Path) -> list[Page]:
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return [Page(text=_read_text(path))]

    if suffix == ".docx":
        return [Page(text=_read_docx(path))]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Page(text=page.extract_text() or "", page=index + 1)
            for index, page in enumerate(reader.pages)
        ]

    raise UnsupportedFormat(f"不支持的格式: {suffix}")


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


# ----------------------------- docx 解析 -----------------------------

def _load_style_levels(archive: zipfile.ZipFile) -> dict[str, int]:
    """styleId -> 标题级别。

    >0 表示标题级别，0 表示目录，-1 表示普通段落。
    Word 中文版的 styleId 常常是 "2"/"3" 这样的数字，
    必须通过 styles.xml 里的 name 才能知道它其实是 heading 1/2。
    """
    try:
        root = ElementTree.fromstring(archive.read("word/styles.xml"))
    except KeyError:
        return {}

    levels: dict[str, int] = {}
    for style in root.findall(f"{_W}style"):
        style_id = style.get(f"{_W}styleId") or ""
        name_node = style.find(f"{_W}name")
        name = (name_node.get(f"{_W}val") or "").strip().lower() if name_node is not None else ""

        heading = re.match(r"heading\s*(\d+)", name)
        if heading:
            levels[style_id] = int(heading.group(1))
        elif name in ("title", "se-title"):
            levels[style_id] = 1
        elif name.startswith("toc"):
            levels[style_id] = 0
        else:
            levels[style_id] = -1
    return levels


def _style_level(node: ElementTree.Element, levels: dict[str, int]) -> int:
    properties = node.find(f"{_W}pPr")
    if properties is None:
        return -1
    style = properties.find(f"{_W}pStyle")
    if style is None:
        return -1
    return levels.get(style.get(f"{_W}val") or "", -1)


def _is_toc_line(text: str) -> bool:
    """兜底：没有 toc 样式、但形如 "2\\t目录\\t2" 的目录条目。"""
    if "\t" not in text:
        return False
    if not re.match(r"^\s*\d+(?:\.\d+)*\t", text):
        return False
    tail = text.rsplit("\t", 1)[-1].strip()
    return tail.isdigit() or "未定义书签" in tail


def _paragraph_text(node: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == f"{_W}t":
            parts.append(child.text or "")
        elif child.tag == f"{_W}br":
            parts.append("\n")
        elif child.tag == f"{_W}tab":
            parts.append("\t")
    return "".join(parts)


def _table_text(node: ElementTree.Element) -> str:
    rows: list[list[str]] = []
    for row in node.findall(f"{_W}tr"):
        cells = []
        for cell in row.findall(f"{_W}tc"):
            text = _paragraph_text(cell).replace("\n", " ").strip()
            cells.append(text.replace("|", "/"))  # 单元格内的竖线会破坏表格
        rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    lines: list[str] = []
    for index, row in enumerate(rows):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if index == 0:
            lines.append("|" + "|".join([" --- "] * width) + "|")
    return "\n".join(lines)


def _walk(node: ElementTree.Element, out: list[str], levels: dict[str, int]) -> None:
    for child in node:
        if child.tag == f"{_W}p":
            text = _paragraph_text(child).strip()
            if not text:
                continue

            level = _style_level(child, levels)
            if level == 0:          # 目录
                continue
            if level > 0:           # 标题
                out.append("#" * min(level, 6) + " " + text)
                continue
            if _is_toc_line(text):  # 目录兜底
                continue
            out.append(text)

        elif child.tag == f"{_W}tbl":
            table = _table_text(child)
            if table:
                out.append(table)

        elif child.tag == f"{_W}sdt":
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                _walk(content, out, levels)


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
        levels = _load_style_levels(archive)

    root = ElementTree.fromstring(xml)
    body = root.find(f"{_W}body")
    if body is None:
        return ""

    out: list[str] = []
    _walk(body, out, levels)
    return "\n\n".join(out)


def scan_documents(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
