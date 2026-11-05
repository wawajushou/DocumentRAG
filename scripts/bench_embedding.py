"""Embedding 选型基准测试。

直接复用项目自己的 readers / splitter，保证和实际索引的切分完全一致。

用法：
  python scripts/bench_embedding.py --data ./data --models bge-small,bge-base
  python scripts/bench_embedding.py --data ./data --queries data/queries.jsonl

queries.jsonl 每行：
  {"query": "仪表的工作电压范围是多少？", "expect_header": "工作电压范围"}
  expect_header 是期望命中的章节关键词，用于粗略计算 recall@k。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docrag.ingest.readers import read_document, scan_documents  # noqa: E402
from docrag.ingest.splitter import split_pages  # noqa: E402

MODELS: dict[str, dict] = {
    "bge-small": {
        "ms": "AI-ModelScope/bge-small-zh-v1.5",
        "hf": "BAAI/bge-small-zh-v1.5",
        "instruction": "为这个句子生成表示以用于检索相关文章：",
    },
    "bge-base": {
        "ms": "AI-ModelScope/bge-base-zh-v1.5",
        "hf": "BAAI/bge-base-zh-v1.5",
        "instruction": "为这个句子生成表示以用于检索相关文章：",
    },
    "bge-m3": {
        "ms": "BAAI/bge-m3",
        "hf": "BAAI/bge-m3",
        "instruction": "",
    },
}

DEFAULT_QUERIES = [
    "系统的输入电压范围是多少？",
    "主要功能模块有哪些？",
    "日志保留多少天？",
    "配置项如何修改？",
    "异常情况下如何处理？",
]


def resolve_model(key: str, cache_dir: Path) -> str:
    spec = MODELS[key]
    slug = spec["hf"].replace("/", "__")
    marker = cache_dir / f"{slug}.path"
    if marker.exists():
        cached = Path(marker.read_text(encoding="utf-8").strip())
        if (cached / "config.json").exists():
            return str(cached)

    try:
        from modelscope import snapshot_download

        path = Path(snapshot_download(spec["ms"], cache_dir=str(cache_dir)))
        print(f"  [modelscope] {spec['ms']} -> {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [modelscope failed] {exc}; 尝试 HuggingFace")
        from huggingface_hub import snapshot_download as hf_download

        path = Path(hf_download(spec["hf"], cache_dir=str(cache_dir)))
        print(f"  [huggingface] {spec['hf']} -> {path}")

    marker.write_text(str(path), encoding="utf-8")
    return str(path)


def load_corpus(data_dir: Path, chunk_size: int, overlap: int, max_chars: int):
    files = scan_documents(data_dir)
    if not files:
        raise SystemExit(f"目录里没有可读文档：{data_dir.resolve()}")

    records = []
    started = time.perf_counter()
    for path in files:
        pages = read_document(path)
        drafts = split_pages(pages, chunk_size=chunk_size, overlap=overlap, max_chunk_chars=max_chars)
        for index, draft in enumerate(drafts):
            records.append(
                {
                    "text": draft.text,
                    "embed_text": f"{draft.header}\n{draft.text}" if draft.header else draft.text,
                    "file": path.name,
                    "index": index,
                    "header": draft.header,
                    "kind": draft.content_type,
                }
            )
    return records, time.perf_counter() - started


def encode(model, texts, batch_size, is_query=False, instruction="", prompt_style=False):
    kwargs = dict(
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if is_query and prompt_style:
        try:
            return model.encode(texts, prompt_name="query", **kwargs)
        except Exception:  # noqa: BLE001
            pass
    payload = [f"{instruction}{t}" for t in texts] if is_query and instruction else texts
    return model.encode(payload, **kwargs)


def run_model(key, records, queries, cache_dir, batch_size, top_k, use_prompt):
    spec = MODELS[key]
    print(f"\n=== {key} ({spec['hf']}) ===")

    started = time.perf_counter()
    path = resolve_model(key, cache_dir)
    import torch
    from sentence_transformers import SentenceTransformer

    # CPU 上强制 fp32，bfloat16 权重在缺少 AVX512-BF16 的机器上会明显变慢
    model = SentenceTransformer(
        path, device="cpu", model_kwargs={"dtype": torch.float32}
    )
    load_sec = time.perf_counter() - started
    print(f"  模型加载: {load_sec:.1f}s")

    texts = [record["embed_text"] for record in records]
    started = time.perf_counter()
    chunk_emb = encode(model, texts, batch_size)
    index_sec = time.perf_counter() - started
    per_chunk = index_sec / max(len(texts), 1) * 1000
    print(f"  索引: {len(texts)} 块 / {index_sec:.1f}s ({per_chunk:.0f} ms/块)")

    q_texts = [q["query"] for q in queries]
    started = time.perf_counter()
    q_emb = encode(
        model, q_texts, batch_size, is_query=True,
        instruction=spec["instruction"], prompt_style=use_prompt,
    )
    query_ms = (time.perf_counter() - started) / max(len(q_texts), 1) * 1000
    print(f"  查询编码: {query_ms:.0f} ms/条")

    import numpy as np

    scores = np.asarray(chunk_emb) @ np.asarray(q_emb).T
    results = []
    header_hits = 0
    labelled = 0
    for qi, query in enumerate(queries):
        order = np.argsort(-scores[:, qi])[:top_k]
        rows = [
            {
                "score": round(float(scores[i, qi]), 4),
                "file": records[i]["file"],
                "index": records[i]["index"],
                "header": records[i]["header"],
                "kind": records[i]["kind"],
                "preview": records[i]["text"][:100].replace("\n", " "),
            }
            for i in order
        ]
        expect = query.get("expect_header")
        got = bool(expect) and any(expect in row["header"] for row in rows)
        if expect:
            labelled += 1
            header_hits += int(got)
        results.append({"query": query["query"], "hit": got, "topk": rows})

    return {
        "model": key,
        "hf_id": spec["hf"],
        "dim": int(np.asarray(chunk_emb).shape[1]),
        "chunks": len(texts),
        "load_sec": round(load_sec, 1),
        "index_sec": round(index_sec, 1),
        "per_chunk_ms": round(per_chunk, 1),
        "query_ms": round(query_ms, 1),
        "recall": header_hits / labelled if labelled else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--models", default="bge-small,bge-base")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cache-dir", default="./models")
    parser.add_argument("--out", default=None)
    parser.add_argument("--prompt-name", action="store_true", help="优先使用模型自带的 query prompt")
    args = parser.parse_args()

    data_dir = Path(args.data)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取文档: {data_dir.resolve()}")
    records, read_sec = load_corpus(data_dir, args.chunk_size, args.overlap, args.max_chars)
    kinds = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    print(f"  {len(records)} 个文本块，切分耗时 {read_sec:.1f}s，分布 {kinds}")
    if not records:
        raise SystemExit("没有切出任何文本块")

    if args.queries:
        queries = [
            json.loads(line)
            for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        queries = [{"query": q, "expect_header": ""} for q in DEFAULT_QUERIES]

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"未知模型: {unknown}，可选: {list(MODELS)}")

    reports = []
    for key in keys:
        try:
            reports.append(
                run_model(
                    key, records, queries, cache_dir,
                    args.batch_size, args.top_k, args.prompt_name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {key}: {type(exc).__name__}: {exc}")

    print("\n\n================ 汇总 ================")
    header = f"{'模型':<14}{'维度':>6}{'块数':>8}{'索引(s)':>10}{'ms/块':>8}{'查询(ms)':>10}{'recall@k':>10}"
    print(header)
    print("-" * 76)
    for report in reports:
        recall = "-" if report["recall"] is None else f"{report['recall']:.2f}"
        print(
            f"{report['model']:<14}{report['dim']:>6}{report['chunks']:>8}"
            f"{report['index_sec']:>10.1f}{report['per_chunk_ms']:>8.0f}"
            f"{report['query_ms']:>10.0f}{recall:>10}"
        )

    for report in reports:
        print(f"\n--- {report['model']} 召回对比 ---")
        for item in report["results"]:
            flag = "命中" if item["hit"] else "未中"
            print(f"[{flag}] {item['query']}")
            for row in item["topk"][:3]:
                print(
                    f"      {row['score']:.3f}  [{row['kind']}] "
                    f"{row['header'][:60]} | {row['preview'][:70]}"
                )

    if args.out:
        Path(args.out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
