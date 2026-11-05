"""切分逻辑测试：结构保留、标题识别、超长兜底。"""

from docrag.ingest.readers import Page
from docrag.ingest.splitter import split_pages


def test_table_kept_as_one_chunk():
    text = "说明如下：\n\n| 信号 | 周期 | 说明 |\n|---|---|---|\n| 车速 | 20ms | 轮速传感器 |\n| 转速 | 10ms | 曲轴信号 |"
    drafts = split_pages([Page(text=text)], chunk_size=400)
    table = [d for d in drafts if d.content_type == "table"]
    assert len(table) == 1
    assert "车速" in table[0].text and "转速" in table[0].text


def test_code_block_indent_preserved():
    text = "示例：\n\n```python\ndef calc(x):\n    if x > 0:\n        return x * 2\n```"
    drafts = split_pages([Page(text=text)], chunk_size=400)
    code = [d for d in drafts if d.content_type == "code"]
    assert len(code) == 1
    assert "    if x > 0:" in code[0].text


def test_header_attached_to_following_chunks():
    text = "1.1 车速信号\n车速信号由轮速传感器提供。\n\n1.2 转速信号\n转速信号由曲轴位置传感器提供。"
    drafts = split_pages([Page(text=text)], chunk_size=400)
    headers = {d.header for d in drafts}
    assert "1.1 车速信号" in headers
    assert "1.2 转速信号" in headers


def test_chinese_number_body_not_treated_as_header():
    text = "一、概述\n本章说明系统结构。\n\n十、二十个通道同时上报时，需要做优先级仲裁。"
    drafts = split_pages([Page(text=text)], chunk_size=400)
    headers = {d.header for d in drafts}
    assert "一、概述" in headers
    assert all("二十个通道" not in h for h in headers)


def test_long_text_is_split_and_overlapped():
    text = "。".join(["这是一个用于测试的超长句子"] * 100) + "。"
    drafts = split_pages([Page(text=text)], chunk_size=200, overlap=40)
    assert len(drafts) > 3
    assert all(len(d.text) <= 220 for d in drafts)


def test_pdf_page_number_kept():
    drafts = split_pages([Page(text="第一页内容", page=1)])
    assert drafts[0].page == 1
