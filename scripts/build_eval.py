"""生成评测集：按一级章节分批让 LLM 出题。

产物为 data/eval.jsonl，每行包含 question、answer、section、type。
问题要求不能直接复述章节标题，否则评测集区分度不足。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docrag.config import get_settings  # noqa: E402
from docrag.ingest.readers import read_document, scan_documents  # noqa: E402
from docrag.llm import create_llm  # noqa: E402

SKIP_SECTIONS = {"版本记录", "目录", "项目介绍"}

PROMPT = """你是技术文档的测试工程师。请根据下面的规范内容，出 {count} 道用于评测 RAG 问答系统的问题。

要求：
1. 问题要像真实工程师会问的，**不要直接复述章节标题**。反例："工作电压范围是多少"（照抄标题）。正例："电瓶电压掉到 7V 时仪表还能正常显示吗"。
2. 覆盖不同题型：数值/阈值、操作流程、故障处理、状态切换、配置对比。
3. 答案必须能从下面的内容中直接找到依据，不能需要文档外的知识。
4. 问题之间不要重复，不要都集中在同一小节。
5. 每题输出一个 JSON 对象，字段：
   - question：问题文本
   - answer：标准答案，简洁准确，保留关键数值和步骤
   - section：答案所在章节，必须从下面的【章节列表】里原样选一个
   - type：数值 / 流程 / 故障 / 对比 / 其他
6. 只输出 JSON 数组，不要 Markdown 代码块，不要任何解释。

【章节列表】
{sections}

【规范内容】
{content}
"""


def split_by_h1(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title = "文档开头"
    buffer: list[str] = []
    for line in text.split("\n"):
        if line.startswith("# "):
            if buffer:
                sections.append((title, "\n".join(buffer)))
            title = line[2:].strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections.append((title, "\n".join(buffer)))
    return sections


def collect_headings(text: str) -> list[str]:
    headings = []
    for line in text.split("\n"):
        if re.match(r"^#{2,3}\s+\S", line):
            headings.append(line.strip().lstrip("#").strip())
    return headings


def parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./data/eval.jsonl")
    parser.add_argument("--per-section", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=20000)
    args = parser.parse_args()

    settings = get_settings()
    llm = create_llm(settings)

    files = scan_documents(Path(args.data))
    if not files:
        raise SystemExit("data 目录下没有文档")

    text = "\n".join(page.text for page in read_document(files[0]))
    headings = collect_headings(text)
    heading_block = "\n".join(f"- {h}" for h in headings[:120])

    sections = [
        (title, body)
        for title, body in split_by_h1(text)
        if title not in SKIP_SECTIONS and len(body) > 500
    ]
    print(f"文档 {len(text)} 字，一级章节 {len(sections)} 个，候选标题 {len(headings)} 个")

    all_items: list[dict] = []
    for index, (title, body) in enumerate(sections, start=1):
        content = body[: args.max_chars]
        prompt = PROMPT.format(
            count=args.per_section,
            sections=heading_block,
            content=content,
        )
        try:
            raw = llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index}/{len(sections)}] {title}: 失败 {exc}")
            continue

        items = parse_json_array(raw)
        valid = [item for item in items if item.get("question") and item.get("answer")]
        for item in valid:
            item["source_section"] = title
        all_items.extend(valid)
        print(f"  [{index}/{len(sections)}] {title}: +{len(valid)}")

    # 去重（问题文本完全相同的）
    seen = set()
    unique = []
    for item in all_items:
        key = item["question"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in unique:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n共生成 {len(unique)} 题 -> {output}")


if __name__ == "__main__":
    main()
