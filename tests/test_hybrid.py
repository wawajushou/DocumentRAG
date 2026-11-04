"""RRF 融合测试。"""

from docrag.retrieve.hybrid import fuse
from docrag.types import Chunk, Hit


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, text=f"内容{chunk_id}", doc_id="d", file="a.docx", chunk_index=0)


def test_rrf_prefers_document_hit_by_both_channels():
    dense = [Hit(chunk=_chunk("a"), score=0.9)]
    bm25 = [Hit(chunk=_chunk("b"), score=12.0), Hit(chunk=_chunk("a"), score=5.0)]

    result = fuse(dense, bm25, k=2, rrf_k=60)

    assert result[0].chunk.id == "a"
    assert set(result[0].channels) == {"dense", "bm25"}


def test_rrf_ignores_raw_score_scale():
    # BM25 分数量级远大于余弦相似度，RRF 只看排名，第一名应并列
    dense = [Hit(chunk=_chunk("x"), score=0.99)]
    bm25 = [Hit(chunk=_chunk("x"), score=9999.0)]
    result = fuse(dense, bm25, k=1, rrf_k=60)
    expected = 1 / 61 + 1 / 61
    assert abs(result[0].score - expected) < 1e-9


def test_same_content_different_docs_not_merged():
    # 旧实现用内容 md5 当 id，会把不同文档里的相同段落合并，这里按 id 区分
    first = Chunk(id="d1:0", text="通用免责声明", doc_id="d1", file="a.docx", chunk_index=0)
    second = Chunk(id="d2:0", text="通用免责声明", doc_id="d2", file="b.docx", chunk_index=0)
    result = fuse(
        [Hit(chunk=first, score=0.5), Hit(chunk=second, score=0.4)],
        [],
        k=5,
    )
    assert {hit.chunk.id for hit in result} == {"d1:0", "d2:0"}


def test_weight_zero_disables_channel():
    dense = [Hit(chunk=_chunk("a"), score=0.9)]
    bm25 = [Hit(chunk=_chunk("b"), score=9.0)]
    result = fuse(dense, bm25, k=1, dense_weight=1.0, bm25_weight=0.0)
    assert result[0].chunk.id == "a"
