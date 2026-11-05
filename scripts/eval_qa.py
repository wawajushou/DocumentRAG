"""RAG 端到端评测。

对 data/eval.jsonl 里的每题跑完整流程（检索 + 生成），再用 LLM 对比标准答案
判断正确性，同时统计检索命中率和引用情况。

输出 reports/*.json，用于不同版本之间横向对比。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docrag.config import get_settings  # noqa: E402
from docrag.embedding import create_embedder  # noqa: E402
from docrag.llm import create_llm  # noqa: E402
from docrag.rag import RAGEngine  # noqa: E402
from docrag.retrieve.bm25 import BM25Index  # noqa: E402
from docrag.store import ChromaStore  # noqa: E402

JUDGE_PROMPT = """你是严格的评测员。请对比【标准答案】和【系统答案】，判断系统答案是否正确。

判定规则：
- 正确：核心结论一致，关键数值/步骤正确，允许表述不同、允许补充正确信息
- 部分正确：结论方向对，但缺少关键数值/步骤，或有次要错误
- 错误：结论错误、关键数值错误、答非所问，或标准答案有而系统答"无法确定"

只输出一行 JSON：{{"verdict": "正确|部分正确|错误", "reason": "20字以内理由"}}

【标准答案】
{reference}

【系统答案】
{answer}
"""


def parse_verdict(raw: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("verdict", "未知"), data.get("reason", "")
        except json.JSONDecodeError:
            pass
    return "未知", raw.strip()[:40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="./data/eval.jsonl")
    parser.add_argument("--out", default="./data/reports/eval.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    if args.no_rewrite:
        settings = settings.model_copy(update={"rewrite_enabled": False})
    if args.top_k:
        settings = settings.model_copy(update={"retrieve_top_k": args.top_k})

    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = ChromaStore(settings.storage_dir, embedder)
    chunks = list(store.iter_chunks())
    bm25 = BM25Index(chunks, userdict=settings.data_dir / "dict.txt")
    engine = RAGEngine(settings, llm, embedder, store, bm25, reranker=None)

    items = [
        json.loads(line)
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        items = items[: args.limit]

    print(f"评测 {len(items)} 题，top_k={settings.retrieve_top_k}，rewrite={settings.rewrite_enabled}\n")

    results = []
    counts = {"正确": 0, "部分正确": 0, "错误": 0, "未知": 0}
    retrieval_hits = 0
    started = time.perf_counter()

    for index, item in enumerate(items, start=1):
        question = item["question"]
        answer = engine.ask(question)

        expect = (item.get("section") or "").strip()
        hit = bool(expect) and any(expect in h.chunk.header for h in answer.hits)
        retrieval_hits += int(hit)

        try:
            raw = llm.complete(
                [
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            reference=item.get("answer", ""),
                            answer=answer.text,
                        ),
                    }
                ],
                temperature=0.0,
                thinking=False,
            )
            verdict, reason = parse_verdict(raw)
        except Exception as exc:  # noqa: BLE001
            verdict, reason = "未知", f"judge failed: {exc}"

        counts[verdict] = counts.get(verdict, 0) + 1

        results.append(
            {
                "question": question,
                "reference": item.get("answer", ""),
                "answer": answer.text,
                "verdict": verdict,
                "reason": reason,
                "retrieval_hit": hit,
                "expect_section": expect,
                "sources": [
                    {"header": h.chunk.header, "file": h.chunk.file}
                    for h in answer.hits[:5]
                ],
                "cited": answer.cited,
            }
        )
        mark = {"正确": "+", "部分正确": "~", "错误": "-"}.get(verdict, "?")
        print(f"[{index:2}/{len(items)}] {mark} {question[:44]}")
        if verdict != "正确":
            print(f"         起因: {reason}")

    elapsed = time.perf_counter() - started
    total = len(items)
    summary = {
        "total": total,
        "retrieval_hit_rate": round(retrieval_hits / total, 3) if total else 0,
        "correct": counts["正确"],
        "partial": counts["部分正确"],
        "wrong": counts["错误"],
        "accuracy": round(counts["正确"] / total, 3) if total else 0,
        "soft_accuracy": round((counts["正确"] + counts["部分正确"] * 0.5) / total, 3) if total else 0,
        "seconds": round(elapsed, 1),
        "config": {
            "top_k": settings.retrieve_top_k,
            "rewrite": settings.rewrite_enabled,
            "expand_neighbors": settings.expand_neighbors,
            "rerank": settings.rerank_enabled,
            "chunk_size": settings.chunk_size,
        },
        "results": results,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 汇总 ================")
    print(f"检索命中率 : {summary['retrieval_hit_rate']:.1%}")
    print(f"完全正确   : {counts['正确']}/{total}")
    print(f"部分正确   : {counts['部分正确']}/{total}")
    print(f"错误       : {counts['错误']}/{total}")
    print(f"严格准确率 : {summary['accuracy']:.1%}")
    print(f"宽松准确率 : {summary['soft_accuracy']:.1%}")
    print(f"耗时       : {elapsed:.1f}s")
    print(f"报告       : {output}")


if __name__ == "__main__":
    main()
