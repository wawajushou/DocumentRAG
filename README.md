# 汽车仪表盘规范文档智能问答系统

基于 RAG（检索增强生成）技术的智能问答系统，专门针对汽车仪表盘技术规格文档、操作手册和硬件协议进行优化。

## 功能特性

- **查询重写**：LLM 将用户问题拆解为 2-4 个不同视角的子查询，提高召回率
- **多路召回**：对多个子查询分别检索，分数去重合并，阈值过滤噪声
- **混合检索**：稠密向量（语义相似）+ BM25（关键词精确匹配）+ RRF 融合
- **语义感知分割**：保留表格、代码块、列表结构，不被暴力切断
- **章节标题提取**：自动提取文档章节标题存入元数据
- **结构化输出**：Markdown 标题分层（结论 → 关键依据 → 推理过程 → 补充说明）
- **交互式选择**：启动时上下键选择 LLM 提供商，10 秒超时默认 MiMo
- **增量更新**：基于文件哈希的智能增量更新，避免重复处理
- **流式输出**：实时查看 AI 回答
- **结构化日志**：loguru 日志系统，按天轮转，ERROR 单独存储
- **API 重试**：tenacity 指数退避重试，防止网络抖动导致崩溃

## 项目结构

```
rag_tool/
├── main.py                # 入口文件（提供商选择、用户交互）
├── config.py              # 配置中心（Pydantic Settings）
├── document_manager.py    # 文档管理（加载、分割、向量化）
├── rag_engine.py          # RAG 引擎（查询重写、多路召回、生成）
│
├── providers/             # LLM 提供商层（策略+工厂模式）
│   ├── __init__.py        # 工厂函数 create_llm_provider()
│   ├── base.py            # BaseLLMProvider 抽象基类
│   ├── deepseek.py        # DeepSeek 实现
│   └── mimo.py            # 小米 MiMo 实现
│
├── splitters/             # 智能分割层
│   ├── __init__.py
│   ├── header_extractor.py   # 章节标题提取器
│   └── semantic_splitter.py  # 语义感知分割器
│
├── retrievers/            # 检索器层
│   ├── __init__.py
│   ├── base.py            # BaseRetriever 抽象基类
│   ├── dense_retriever.py # 稠密向量检索（ChromaDB）
│   ├── bm25_retriever.py  # BM25 关键词检索（jieba 分词）
│   └── hybrid_retriever.py# 混合检索（RRF 融合）
│
├── utils/                 # 工具层
│   ├── __init__.py
│   ├── logger.py          # loguru 日志配置
│   └── retry.py           # tenacity 重试装饰器
│
├── data/                  # 文档数据目录
├── chroma_db/             # 向量数据库
├── models/                # 模型缓存
├── logs/                  # 日志文件
│
├── .env                   # 环境变量（不提交）
├── .env.example           # 环境变量示例
├── .gitignore
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入 API Key：

```env
API_PROVIDER=mimo
DEEPSEEK_API_KEY=your_deepseek_api_key
MIMO_API_KEY=your_mimo_api_key
USE_MODEL_SCOPE=true
```

### 3. 添加文档

将 Word/PDF/TXT 文档放入 `./data` 目录。

### 4. 运行

```bash
python main.py
```

启动后会显示提供商选择菜单（上下键选择，回车确认，10 秒超时默认 MiMo）。

## 支持的命令

| 命令 | 功能 |
|------|------|
| 输入问题 | 执行 RAG 查询 |
| `update` | 手动更新知识库 |
| `reset` | 查看重置知识库操作指南 |
| `switch` | 切换 API 提供商 |
| `quit` / `exit` | 退出程序 |

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | DeepSeek / 小米 MiMo |
| 向量模型 | BAAI/bge-small-zh-v1.5 |
| 向量数据库 | ChromaDB |
| 框架 | LangChain |
| 配置管理 | Pydantic Settings |
| 日志 | loguru |
| 重试 | tenacity |
| 分词 | jieba |
| BM25 | rank-bm25 |
| 交互选择 | pick |
| 终端美化 | Rich |

---

## 重构记录（2026-05-15）

### 阶段一：架构解耦与设计模式引入

**新增文件：**
- `providers/__init__.py` — LLM 提供商工厂，`create_llm_provider()` 函数 + `PROVIDER_REGISTRY` 注册表
- `providers/base.py` — `BaseLLMProvider` 抽象基类 + `LLMResponse` 数据结构 + `chat_with_retry()` 重试方法
- `providers/deepseek.py` — DeepSeek 实现，集成 `@api_retry` 装饰器和 `loguru` 日志
- `providers/mimo.py` — MiMo 实现，集成 `@api_retry` 装饰器和 `loguru` 日志

**重构文件：**
- `config.py` — `Config` 类升级为 `Settings(BaseSettings)`，Pydantic 启动即校验 + 延迟初始化 `get_settings()`
- `rag_engine.py` — 构造函数注入 `llm: BaseLLMProvider`，移除 `from openai import OpenAI`
- `main.py` — 使用 `create_llm_provider()` 工厂函数创建 LLM Provider

**设计模式：** 策略模式（LLM 抽象）+ 工厂模式（Provider 创建）

---

### 阶段二：文档解析与切分策略增强

**新增文件：**
- `splitters/__init__.py` — 模块初始化
- `splitters/header_extractor.py` — 章节标题提取器，支持 Markdown/中文数字/数字编号等格式
- `splitters/semantic_splitter.py` — 语义感知分割器，保留表格/代码块/列表结构

**重构文件：**
- `document_manager.py` — 集成 `SemanticSplitter` + `HeaderExtractor`，元数据增加 `header` 和 `content_type` 字段

**关键改进：**
- 表格（`|` 分隔）和代码块（`` ``` `` 包围）作为整体保留，不被切断
- 每个文档块自动提取最近的章节标题存入 `metadata["header"]`

---

### 阶段三：检索与召回引擎进阶

**新增文件：**
- `retrievers/__init__.py` — 模块初始化
- `retrievers/base.py` — `BaseRetriever` 抽象基类 + `RetrievalResult` 封装（文档+分数）
- `retrievers/dense_retriever.py` — 稠密向量检索器，基于 ChromaDB
- `retrievers/bm25_retriever.py` — BM25 关键词检索器，使用 `rank-bm25` + `jieba` 中文分词
- `retrievers/hybrid_retriever.py` — 混合检索器，RRF（倒数排序融合）算法合并 Dense + BM25

**重构文件：**
- `rag_engine.py` — 创建 `DenseRetriever` + `BM25Retriever` + `HybridRetriever`，`retrieve()` 改用混合检索

**检索流程：**
```
单查询 → Dense(语义) + BM25(关键词) → RRF 融合(k=60) → Top-K
多查询 → 相似查询过滤 → 多路召回 → 分数去重(保留最高分) → 阈值过滤 → 最终文档
```

---

### 阶段四：健壮性、日志与容错

**新增文件：**
- `utils/__init__.py` — 模块初始化
- `utils/logger.py` — `loguru` 日志配置：控制台 WARNING+、文件 DEBUG+、ERROR 单独存储、按天轮转
- `utils/retry.py` — `tenacity` 重试装饰器：指数退避(2s→4s→8s)、最多 3 次、仅重试网络错误

**重构文件：**
- `providers/base.py` — 添加 `chat_with_retry()` 方法
- `providers/deepseek.py` — `chat()` 和 `chat_stream()` 添加 `@api_retry` 装饰器 + `logger` 日志
- `providers/mimo.py` — 同上
- `document_manager.py` — 所有 `print` 替换为 `logger.info/warning/error`
- `rag_engine.py` — 查询重写、检索、生成全流程添加日志
- `main.py` — 初始化 `setup_logger()`、`create_document_manager()` 上下文管理器、`finally` 记录退出

---

### 用户交互优化

| 改动 | 内容 |
|------|------|
| `.env` 修复 | `USE_MODEL_SCOPE=true`，解决 HuggingFace 网络超时问题 |
| emoji 移除 | 移除所有 emoji 字符，解决 Windows GBK 编码报错 |
| 启动选择 | `select_provider()` 函数：上下键选择 LLM 提供商，10 秒超时默认 MiMo |

---

### 查询重写优化

| 改动 | 内容 |
|------|------|
| Prompt 优化 | 增加同义词替换、视角扩展、关键词提取等重写策略 |
| 动态数量 | 支持 2-4 个子查询（原固定 3 个） |
| 解析改进 | 支持带编号（`1. xxx`）、带引号等多种格式 |
| 长度限制 | 每个子查询最多 50 字，超长自动截断 |
| 质量评估 | 过滤重复、过短、与原始查询相同的查询 |
| 相似过滤 | `_filter_similar_queries()` 使用 Jaccard 相似度过滤冗余查询 |

---

### Prompt 输出格式优化

| 改动 | 内容 |
|------|------|
| 结构化输出 | Prompt 新增【输出格式要求】：`## 结论` → `## 关键依据` → `## 推理过程` → `## 补充说明` |
| Markdown | 要求使用粗体标记关键术语、表格呈现优先级/对比数据 |

---

### 多路召回修复与优化

| 问题 | 修复 |
|------|------|
| **BM25 索引从未构建** | `main.py` 创建 RAGEngine 后立即调用 `update_bm25_index()` |
| **BM25 构建丢失 metadata** | `update_bm25_index()` 获取 `include=["documents", "metadatas"]` |
| **无分数去重** | `multi_query_retrieve()` 按内容哈希去重，保留最高分 |
| **无阈值过滤** | 新增 `min_score=0.01` 参数，过滤噪声文档 |
| **BM25 分词质量差** | 字符级分词升级为 `jieba` 精确模式分词 |
| **update 未触发 BM25** | `main.py` 的 `update` 命令中自动调用 `update_bm25_index()` |

---

### 新增依赖

```bash
pip install pydantic>=2.0.0 pydantic-settings>=2.0.0 rank-bm25 loguru tenacity pick jieba
```
