"""Embedding 层（延迟导入，避免 status 之类的轻量命令也加载模型）。"""

from docrag.embedding.base import Embedder

__all__ = ["Embedder", "LocalEmbedder", "create_embedder"]


def create_embedder(settings):
    from docrag.embedding.local import LocalEmbedder

    return LocalEmbedder(settings)


def __getattr__(name: str):
    if name == "LocalEmbedder":
        from docrag.embedding.local import LocalEmbedder

        return LocalEmbedder
    raise AttributeError(name)
