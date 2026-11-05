"""本地 SentenceTransformer 实现。

处理查询侧 instruction、向量归一化和 MRL 降维。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from docrag.config import Settings

# bge 系列的查询前缀；文档侧不加
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _query_instruction(model_id: str) -> str:
    return BGE_QUERY_INSTRUCTION if "bge" in model_id.lower() else ""


def resolve_model_path(model_id: str, model_dir: Path, source: str) -> Path:
    """优先复用已下载的模型，避免每次启动都联网检查。"""
    slug = model_id.replace("/", "__").replace(":", "_")
    marker = model_dir / f"{slug}.path"
    if marker.exists():
        cached = Path(marker.read_text(encoding="utf-8").strip())
        if (cached / "config.json").exists():
            return cached

    model_dir.mkdir(parents=True, exist_ok=True)
    path: Path | None = None

    if source == "modelscope":
        try:
            from modelscope import snapshot_download

            path = Path(snapshot_download(model_id, cache_dir=str(model_dir)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ModelScope 下载失败（{exc}），回退 HuggingFace")

    if path is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, cache_dir=str(model_dir)))

    marker.write_text(str(path), encoding="utf-8")
    return path


class LocalEmbedder:
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self._settings = settings
        self._name = settings.embedding_model
        path = resolve_model_path(settings.embedding_model, settings.model_dir, settings.model_source)

        started = time.perf_counter()
        model_kwargs = {"dtype": torch.float32} if settings.embedding_device == "cpu" else {}
        try:
            self._model = SentenceTransformer(
                str(path),
                device=settings.embedding_device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )
        except TypeError:
            # 新版 sentence-transformers 移除了 trust_remote_code 参数
            self._model = SentenceTransformer(
                str(path),
                device=settings.embedding_device,
                model_kwargs=model_kwargs,
            )

        # sentence-transformers 6.x 把方法名从 get_sentence_embedding_dimension
        # 改成了 get_embedding_dimension，两个方法名都兼容
        dim_reader = getattr(self._model, "get_embedding_dimension", None)
        if dim_reader is None:
            dim_reader = self._model.get_sentence_embedding_dimension
        native_dim = int(dim_reader())
        requested = settings.embedding_dimensions
        self._dim = requested if requested and requested < native_dim else native_dim
        self._instruction = _query_instruction(settings.embedding_model)
        self._prompt_name_supported = True

        logger.info(
            f"Embedding ready: {self._name} dim={self._dim} "
            f"device={settings.embedding_device} load={time.perf_counter() - started:.1f}s"
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"{self._name}@{self._dim}"

    def _postprocess(self, vectors: np.ndarray) -> list[list[float]]:
        if self._dim < vectors.shape[1]:
            vectors = vectors[:, : self._dim]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._postprocess(np.asarray(vectors))

    def embed_query(self, text: str) -> list[float]:
        kwargs = dict(
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if self._instruction and self._prompt_name_supported:
            try:
                # 使用模型自带的 prompt 模板，此时传原始 query
                vectors = self._model.encode([text], prompt_name="query", **kwargs)
                return self._postprocess(np.asarray(vectors))[0]
            except (KeyError, ValueError):
                # 模型没有内置 prompt 配置时退回手写前缀
                self._prompt_name_supported = False

        payload = f"{self._instruction}{text}" if self._instruction else text
        vectors = self._model.encode([payload], **kwargs)
        return self._postprocess(np.asarray(vectors))[0]
