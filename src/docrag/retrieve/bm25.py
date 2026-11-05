"""BM25 关键词检索。

支持加载领域词典（data/dict.txt），改善专有名词的分词效果。
"""

from __future__ import annotations

import logging
import string
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from docrag.types import Chunk, Hit

jieba.setLogLevel(logging.WARNING)

_PUNCT = set(string.punctuation) | set("，。！？；：、“”‘’（）《》【】—…·")
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "为", "对", "从", "到",
    "把", "被", "让", "使", "可", "以", "有", "无", "不", "中", "上", "下",
    "个", "等", "这", "那", "其", "之", "而", "并", "就", "都", "也", "还",
}

_DICT_LOADED: set[str] = set()


def _load_userdict(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    key = str(path.resolve())
    if key in _DICT_LOADED:
        return
    jieba.load_userdict(str(path))
    _DICT_LOADED.add(key)


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.lcut(text):
        token = token.strip().lower()
        if not token or token in _STOPWORDS:
            continue
        if all(ch in _PUNCT for ch in token):
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], userdict: Path | None = None):
        _load_userdict(userdict)
        self._chunks = chunks
        # 章节标题里往往包含关键术语，一起参与 BM25 匹配
        self._tokens = [
            tokenize(f"{chunk.header}\n{chunk.text}" if chunk.header else chunk.text)
            for chunk in chunks
        ]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    @property
    def name(self) -> str:
        return "bm25"

    def __len__(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(chunk=self._chunks[i], score=float(scores[i]), channels=("bm25",))
            for i in ranked
            if scores[i] > 0
        ]
