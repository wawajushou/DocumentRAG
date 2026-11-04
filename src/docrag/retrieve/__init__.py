"""检索层（延迟导入，便于单独测试 RRF 融合）。"""

from docrag.retrieve.base import Retriever

__all__ = ["Retriever", "BM25Index", "HybridRetriever", "fuse"]


def __getattr__(name: str):
    if name == "BM25Index":
        from docrag.retrieve.bm25 import BM25Index

        return BM25Index
    if name == "HybridRetriever":
        from docrag.retrieve.hybrid import HybridRetriever

        return HybridRetriever
    if name == "fuse":
        from docrag.retrieve.hybrid import fuse

        return fuse
    raise AttributeError(name)
