"""文档读取与切分。

这里刻意不导入 indexer：indexer 依赖 loguru/chromadb 等重模块，
让"只想用切分"的调用方（测试、脚本）保持轻量。
"""

from docrag.ingest.manifest import Manifest
from docrag.ingest.readers import Page, read_document, scan_documents
from docrag.ingest.splitter import Draft, split_pages

__all__ = ["Manifest", "Page", "read_document", "scan_documents", "Draft", "split_pages"]
