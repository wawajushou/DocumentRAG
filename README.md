# docrag

本地技术文档 RAG 问答系统。面向技术规范、操作手册这类表格多、术语密的中文文档，纯 CPU 可运行。

## 特性

- 混合检索：BGE 向量 + BM25 + RRF 融合
- 结构感知切分：保留 Word 表格与章节层级，表格超长时按行切分并重复表头
- 增量索引：按内容哈希判断变更，修改文档先删旧块，删除文件自动清理
- 支持 DeepSeek 与小米 MiMo（均为 OpenAI 兼容协议）
- Embedding 本地运行，无需 GPU

## 安装

要求 Python 3.11+。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

首次安装会下载 CPU 版 torch，约 1.4 GB。国内建议配置镜像：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## 配置

```bash
copy .env.example .env
```

填入至少一个 API Key：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
```

## 使用

将文档放入 `data/`（支持 docx / pdf / txt / md）：

```bash
docrag sync              # 建索引
docrag                   # 交互问答
docrag ask "问题"         # 单次提问
docrag status            # 索引状态
docrag --provider mimo   # 切换提供商
```

交互模式支持 `/update`、`/status`、`/quit`。

Windows 用户可直接双击 `start.bat`，非命令行用户见 [USAGE.md](USAGE.md)。

### 领域词典

在 `data/dict.txt` 中维护术语表（每行一个词），可改善 BM25 对专有名词的切分。该文件不进入仓库。

## 目录结构

```
src/docrag/
├── config.py       配置
├── cli.py          命令行入口
├── llm/            LLM 客户端
├── embedding/      向量模型封装
├── ingest/         文档读取、切分、增量索引
├── retrieve/       BM25、RRF 混合检索、重排
├── rag/            Prompt 与生成引擎
└── store/          Chroma 封装
scripts/            评测与模型选型
tests/              单元测试
```

## 模型

| 用途 | 默认 |
|---|---|
| LLM | `deepseek-flash`，可切换 `mimo-v2.6-pro` |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Rerank | 默认关闭 |

旧的 `deepseek-chat` 与 `mimo-v2-pro` 已下线。

本机（i5-12450H，无独显）实测：

| 模型 | 索引 1099 块 | 查询 | 结论 |
|---|---|---|---|
| bge-small | 133 s | 17 ms | 默认 |
| bge-base | 669 s | 105 ms | 无明显提升 |
| bge-m3 | — | — | 未实测 |

## 评测

```bash
python scripts/eval_qa.py --out data/reports/run.json
```

33 道题的端到端评测，当前准确率 94%~100%，波动来自 LLM 生成的随机性。评测集在 `data/eval.jsonl`，可用 `scripts/build_eval.py` 重新生成。

## 限制

- PDF 扫描件需要先 OCR
- docx 中的图片不参与检索
- 单机使用，不要同时运行多个实例
