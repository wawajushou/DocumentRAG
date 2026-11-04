# docrag

本地技术文档 RAG 问答系统。面向汽车仪表盘规范、操作手册、硬件协议这类**术语密集、表格多**的中文文档，纯 CPU 可跑。

## 这一版改了什么

v1 的代码能跑起来，但有一批"看起来做了、实际没生效"的问题。v2 是重写，主要变化：

| 方面 | v1 | v2 |
|---|---|---|
| LLM | `deepseek-chat` / `mimo-v2-pro`（**均已下线**） | `deepseek-flash` / `mimo-v2.6-pro` |
| 重试 | 捕获内置 `ConnectionError`，实际永不触发 | 捕获 openai 真实异常；流式只在建连阶段重试 |
| Embedding | `bge-small` + 800 字块（超 512 token 被静默截断） | `bge-small` + 400 字块 + query instruction + 归一化；长文档可选 Qwen3 |
| 向量库 | 默认 L2 + 未归一化 | cosine + 归一化 |
| 增量更新 | 只追加，改文档产生重复块 | 按文件先删后写，删除/改名同步清理 |
| 文档 ID | 内容 md5（跨文档同段落被合并） | `文件名 + 内容哈希`，稳定且唯一 |
| 切分 | 预处理把缩进/对齐压平 | 只去行尾空白，表格/代码整体保留 |
| 章节标题 | `position=0`，实际永远失效 | 逐行识别，随块写入元数据 |
| 多路召回 | 最多 20 块全进 prompt | 全局 top-k 截断 |
| 噪声阈值 | `min_score=0.01` 对 RRF 无效 | 移除假阈值 |
| 来源展示 | 列出全部召回结果 | 只列模型实际引用的 `[n]` |
| 依赖 | 25 个（含 langchain 全家桶） | 13 个，去掉 langchain |
| 启动交互 | `pick` 超时线程抢占 stdin，非 TTY 卡 10 秒 | 命令行参数，不再交互阻塞 |

## 安装

需要 Python 3.11+（推荐 3.13）。

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .
```

CPU 版 torch 体积较大，首次安装会比较久。国内建议先配置镜像：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 配置

```bash
copy .env.example .env
```

最少填一个 API Key：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
```

Key 只从环境变量 / `.env` 读取，代码里没有任何硬编码，`.env` 已在 `.gitignore` 中。

## 使用

把文档放进 `data/`（支持 `.docx` / `.pdf` / `.txt` / `.md`），然后：

```bash
docrag sync              # 建索引
docrag                   # 交互问答
docrag ask "背光亮度怎么调"   # 单次提问
docrag status            # 查看状态
docrag --provider mimo   # 临时切换到 MiMo
```

交互模式下可用 `/update` 重新同步、`/status` 查看状态、`/quit` 退出。

### 领域词典（可选）

BM25 对专有名词敏感。把术语表放到 `data/dict.txt`（每行一个词），可以显著改善 `CAN总线`、`IPK`、`DTC` 这类词的切分：

```bash
copy examples\dict.txt data\dict.txt
```

## 模型现状（2026-11）

| 提供商 | 旧模型（已失效） | 当前模型 | 备注 |
|---|---|---|---|
| DeepSeek | `deepseek-chat`（2026-07-24 弃用） | `deepseek-flash` | 100 万上下文，支持 `thinking` |
| 小米 MiMo | `mimo-v2-pro`（2026-06-30 下线）<br>`mimo-v2.5-pro`（2026-10-21 下线） | `mimo-v2.6-pro` | 参数用 `max_completion_tokens` |

默认关闭思考模式（`LLM_THINKING=off`），RAG 问答响应更快。复杂推导可以临时开：

```env
LLM_THINKING=on
LLM_REASONING_EFFORT=high
```

## 换 Embedding 模型

改 `EMBEDDING_MODEL` 后直接 `docrag sync`，程序会检测到指纹变化并自动重建向量库，不需要手动删目录。

在本机（i5-12450H / 16GB / 无独显）用这份文档（2245 块）实测：

| 模型 | 维度 | 索引耗时 | 查询延迟 | top-1 命中 | 结论 |
|---|---|---|---|---|---|
| `BAAI/bge-small-zh-v1.5` | 512 | 133 s | 17 ms | 14/14 | **默认，推荐** |
| `BAAI/bge-base-zh-v1.5` | 768 | 669 s | 105 ms | 13/14 | 备选，效果没有更好 |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | ≈20 min | 880 ms | 14/14 | 有 GPU 再考虑 |

结论有点反直觉：对结构清晰、术语明确的技术文档，**最小的 bge-small 就够了**，因为 BM25 负责专有名词、章节标题负责上下文，稠密向量只需要补语义那一部分。

**Qwen3 在 CPU 上有一个大坑**：模型权重是 bfloat16，而大多数消费级 CPU 没有 AVX512-BF16，会退化成模拟运算，慢 4-5 倍（实测 2.8 秒/块 vs fp32 的 0.58 秒/块）。本项目在 CPU 上强制用 float32 加载，但如果要用 Qwen3，仍然建议上 GPU。

选型可以用 `scripts/bench_embedding.py` 在自己的文档上实测：

```bash
python scripts/bench_embedding.py --data ./data --queries data/queries.jsonl --models bge-small,bge-base
```

## 目录结构

```
src/docrag/
├── config.py            配置（pydantic-settings）
├── types.py             Chunk / Hit / Answer
├── cli.py               命令行入口
├── llm/                 OpenAI 兼容客户端（DeepSeek / MiMo 共用）
├── embedding/           SentenceTransformer 封装（query instruction / 归一化）
├── ingest/              读取、切分、清单、增量索引
├── retrieve/            BM25、RRF 混合检索、可选重排
├── rag/                 Prompt 与生成引擎
└── store/               Chroma 封装（cosine）
tests/                   单元测试
scripts/bench_embedding.py   Embedding 选型基准
```

## 已知取舍

- **重排默认关闭**：`bge-reranker-v2-m3` 在纯 CPU 上对 30 个候选打分约 1-3 秒，交互体验下降。文档量大时可以开。
- **PDF 页码**：来自 pypdf，已转成 1-based。扫描件（无文本层）需要先 OCR。
- **docx 图片不索引**：只提取文字和表格。图里的信息需要单独整理成文字。
- **超长表格**：超过 `MAX_CHUNK_CHARS` 的表格按行切片，每片重复表头，结构不丢。少数单元格本身超长的表格仍会被截断。
- **索引是单机的**：Chroma 本地持久化，多进程同时写会锁文件。同一时间只跑一个 `docrag` 实例。
