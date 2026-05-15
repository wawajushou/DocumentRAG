"""
rag_engine.py

RAG 引擎 - 核心检索与生成模块

改造要点：
1. 通过构造函数注入 BaseLLMProvider，而非直接创建 OpenAI 客户端
2. 通过构造函数注入配置参数，而非引用全局 config
3. 所有 LLM 调用都通过 provider 接口，与具体实现解耦
4. 使用混合检索器（稠密向量 + BM25）提升检索准确率
5. 集成日志记录，便于调试和监控
"""

from typing import List, Generator, Tuple, Optional
from langchain_core.documents import Document
import hashlib

from providers.base import BaseLLMProvider
from document_manager import DocumentManager
from retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from utils.logger import get_logger

logger = get_logger()


class RAGEngine:
    """
    RAG 引擎 - 核心检索与生成模块（支持查询重写与多路召回）

    依赖注入：
    - doc_manager: 文档管理器实例
    - llm: LLM 提供商实例（抽象接口）
    - retrieve_top_k: 检索返回的文档数量
    - temperature: LLM 温度参数

    检索策略：
    - 使用混合检索器（稠密向量 + BM25）提升准确率
    - 稠密向量：理解语义相似性
    - BM25：精确匹配专有名词（错误码、芯片型号等）
    """

    def __init__(
        self,
        doc_manager: DocumentManager,
        llm: BaseLLMProvider,
        retrieve_top_k: int = 5,
        temperature: float = 0.1,
    ):
        """
        初始化 RAG 引擎

        Args:
            doc_manager: 文档管理器实例
            llm: LLM 提供商实例（通过依赖注入）
            retrieve_top_k: 检索返回的文档数量
            temperature: LLM 温度参数（0-2，越低越确定）
        """
        self.doc_manager = doc_manager
        self.llm = llm
        self.retrieve_top_k = retrieve_top_k
        self.temperature = temperature

        # 创建检索器
        self.dense_retriever = DenseRetriever(doc_manager.vectorstore)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )

        logger.info(f"RAGEngine initialized: provider={llm.provider_name}, model={llm.model_name}")

    def update_bm25_index(self) -> None:
        """
        更新 BM25 索引

        当文档更新后，需要调用此方法重建 BM25 索引。
        从向量库中获取所有文档（包含 metadata），构建 BM25 索引。
        """
        if self.doc_manager.vectorstore:
            try:
                # 获取文档内容和元数据
                all_docs = self.doc_manager.vectorstore.get(
                    include=["documents", "metadatas"]
                )
                if all_docs and "documents" in all_docs:
                    documents = [
                        Document(page_content=content, metadata=meta or {})
                        for content, meta in zip(
                            all_docs["documents"],
                            all_docs.get("metadatas", [{}] * len(all_docs["documents"]))
                        )
                    ]
                    self.bm25_retriever.build_index(documents)
                    logger.info(f"BM25 index updated: {len(documents)} documents")
            except Exception as e:
                logger.error(f"Failed to update BM25 index: {e}")

    def rewrite_query(self, original_query: str) -> List[str]:
        """
        查询重写：将原始问题拆解为多个不同侧重点的子查询

        通过 LLM 将用户问题扩展为多个角度的子查询，
        提高后续检索的召回率。

        优化点：
        1. 优化 Prompt，增加技术文档特定的重写策略
        2. 支持动态子查询数量（2-4 个）
        3. 改进解析逻辑，支持多种格式
        4. 添加查询长度限制
        5. 添加查询质量评估

        Args:
            original_query: 用户的原始问题

        Returns:
            包含重写后查询的列表（包含原始问题+重写问题，或仅原始问题如果重写失败）
        """
        logger.info(f"Query rewrite started: '{original_query}'")

        rewrite_prompt = f"""你是一位专业的汽车仪表盘开发领域专家。
请将以下问题拆解、扩写为 2-4 个不同侧重点的、适合向量检索的独立子查询。

重写策略：
1. 同义词替换：将口语化表达替换为专业术语
   - 例："车速" → "车速信号"、"速度传感器"
2. 视角扩展：从硬件、软件、协议等不同角度提问
3. 关键词提取：提取问题中的核心关键词，形成简短查询
4. 补充上下文：为模糊问题补充具体场景

要求：
- 每个子查询应该简洁明了，不超过 30 个字
- 子查询之间应该有明显差异，避免重复
- 仅返回子查询，每行一个，不要有任何其他说明

原始问题：
{original_query}

子查询：
"""

        try:
            # 通过 LLM Provider 调用
            response = self.llm.chat(
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.3,
            )

            rewritten_text = response.content
            if not rewritten_text:
                logger.warning("Query rewrite returned empty result")
                return [original_query]

            rewritten_text = rewritten_text.strip()

            # 解析重写后的查询（改进版）
            rewritten_queries = self._parse_rewritten_queries(rewritten_text)

            # 查询质量评估
            quality_queries = self._evaluate_query_quality(original_query, rewritten_queries)

            if len(quality_queries) >= 1:
                all_queries = [original_query] + quality_queries[:3]
                logger.info(f"Query rewrite completed: {len(all_queries)} sub-queries")
                logger.debug(f"Rewritten queries: {all_queries}")
                return all_queries

            logger.warning("Query rewrite failed to parse result")
            return [original_query]

        except Exception as e:
            # 查询重写失败时降级使用原始查询
            logger.error(f"Query rewrite failed: {e}, using original query")
            return [original_query]

    def _parse_rewritten_queries(self, text: str) -> List[str]:
        """
        解析重写后的查询，支持多种格式

        支持的格式：
        - 每行一个查询
        - 带编号的查询（1. xxx 或 1) xxx）
        - 带引号的查询

        Args:
            text: LLM 返回的文本

        Returns:
            解析后的查询列表
        """
        import re
        queries = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 移除编号（1. 2. 3. 或 1) 2) 3)）
            line = re.sub(r'^\d+[\.\)]\s*', '', line)

            # 移除引号
            line = line.strip('"\'')

            # 过滤过短的查询（少于 3 个字）
            if line and len(line) >= 3:
                queries.append(line)

        return queries

    def _limit_query_length(self, query: str, max_length: int = 50) -> str:
        """
        限制查询长度，过长的查询截断

        Args:
            query: 原始查询
            max_length: 最大长度

        Returns:
            截断后的查询
        """
        if len(query) <= max_length:
            return query

        # 尝试在自然断点截断
        for sep in ['，', '、', '的', '与', '和']:
            idx = query.find(sep, max_length // 2)
            if idx != -1 and idx < max_length:
                return query[:idx]

        # 强制截断
        return query[:max_length]

    def _evaluate_query_quality(self, original: str, rewritten: List[str]) -> List[str]:
        """
        评估重写查询质量，过滤低质量查询

        过滤规则：
        - 与原始查询完全相同的
        - 过短的查询（少于 3 个字）
        - 与已有查询高度相似的

        Args:
            original: 原始查询
            rewritten: 重写后的查询列表

        Returns:
            过滤后的高质量查询列表
        """
        quality_queries = []

        for query in rewritten:
            # 限制查询长度
            query = self._limit_query_length(query)

            # 过滤与原始查询完全相同的
            if query == original:
                continue

            # 过滤过短的查询（少于 3 个字）
            if len(query) < 3:
                continue

            # 过滤与已有查询高度相似的（简单的字符串包含检查）
            if any(query in q or q in query for q in quality_queries):
                continue

            quality_queries.append(query)

        return quality_queries

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        文档去重：基于内容哈希值去重

        使用 MD5 哈希算法对文档内容进行哈希，
        确保相同内容的文档不会被重复添加。

        Args:
            docs: 需要去重的文档列表

        Returns:
            去重后的文档列表
        """
        seen = set()
        unique_docs = []

        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def _retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List:
        """
        从向量数据库中检索相关文档（带分数）

        使用混合检索器（稠密向量 + BM25）提升准确率。

        Args:
            query: 用户查询
            k: 返回的文档数量，默认使用配置中的值

        Returns:
            RetrievalResult 列表（包含 document 和 score）
        """
        effective_k = k if k is not None else self.retrieve_top_k

        logger.debug(f"Retrieving: query='{query}', k={effective_k}")

        results = self.hybrid_retriever.retrieve(query, k=effective_k)

        for i, result in enumerate(results):
            logger.debug(f"Result {i+1}: score={result.score:.4f}, source={result.document.metadata.get('filename')}")

        return results

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        从向量数据库中检索相关文档（单查询，仅返回文档）

        Args:
            query: 用户查询
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        results = self._retrieve_with_scores(query, k)
        return [result.document for result in results]

    def _filter_similar_queries(self, queries: List[str], threshold: float = 0.7) -> List[str]:
        """
        过滤与已有查询过于相似的查询，减少冗余检索

        使用字符级 Jaccard 相似度判断查询间的相似性。

        Args:
            queries: 查询列表
            threshold: 相似度阈值（0-1），超过此值视为相似

        Returns:
            过滤后的查询列表
        """
        if len(queries) <= 1:
            return queries

        filtered = [queries[0]]  # 保留原始查询

        for q in queries[1:]:
            q_chars = set(q)
            is_similar = False
            for existing in filtered:
                existing_chars = set(existing)
                # Jaccard 相似度 = 交集 / 并集
                intersection = len(q_chars & existing_chars)
                union = len(q_chars | existing_chars)
                similarity = intersection / union if union > 0 else 0

                if similarity > threshold:
                    is_similar = True
                    logger.debug(f"Filtered similar query: '{q}' ~ '{existing}' (similarity={similarity:.2f})")
                    break

            if not is_similar:
                filtered.append(q)

        logger.info(f"Query filtering: {len(queries)} -> {len(filtered)} queries")
        return filtered

    def multi_query_retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None,
        min_score: float = 0.01,
    ) -> List[Document]:
        """
        多路召回：对多个查询分别检索，分数去重合并，阈值过滤

        优化流程：
        1. 过滤相似查询，减少冗余检索
        2. 对每个查询执行混合检索（Dense + BM25 + RRF）
        3. 按内容哈希去重，相同文档保留最高分数
        4. 过滤低于阈值的噪声文档
        5. 按分数降序排列

        Args:
            queries: 查询列表
            k: 每个查询返回的文档数量
            min_score: 最小 RRF 分数阈值，低于此值的文档被视为噪声

        Returns:
            去重、过滤后的文档列表
        """
        # 1. 过滤相似查询
        filtered_queries = self._filter_similar_queries(queries)

        # 2. 对每个查询执行检索
        all_results = []
        effective_k = k if k is not None else self.retrieve_top_k

        for query in filtered_queries:
            results = self._retrieve_with_scores(query, effective_k)
            all_results.extend(results)

        # 3. 按内容哈希去重，保留最高分
        unique_results = {}
        for result in all_results:
            doc_id = hashlib.md5(result.document.page_content.encode('utf-8')).hexdigest()
            if doc_id not in unique_results or result.score > unique_results[doc_id].score:
                unique_results[doc_id] = result

        # 4. 过滤低分文档
        filtered_results = [
            r for r in unique_results.values()
            if r.score >= min_score
        ]

        # 5. 按分数降序排列
        filtered_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            f"Multi-query retrieve: {len(queries)} queries -> {len(filtered_queries)} filtered -> "
            f"{len(all_results)} raw results -> {len(unique_results)} unique -> "
            f"{len(filtered_results)} after threshold (min_score={min_score})"
        )

        return [result.document for result in filtered_results]

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        """
        构建 Prompt（推理型 + 结构化输出）

        构建一个要求 LLM 进行逻辑推理的 Prompt，
        确保答案基于文档内容而非外部知识，
        并要求使用 Markdown 格式化输出以提升可读性。

        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档

        Returns:
            完整的 Prompt 字符串
        """
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = f"""你是一个高级的汽车仪表盘项目开发工程师。请根据以下【参考文档】的内容，来解答用户的【问题】。

【核心指令】：
1. 你的回答必须 100% 基于参考文档中提供的事实，绝对不能引入外部知识或凭空捏造。
2. 允许逻辑推理：如果答案没有在文档中直接写明，你需要对文档中提供的多个信息片段进行交叉对比、状态机推导和综合逻辑分析，以得出结论。
3. 只有在经过严谨的推理后，发现文档中确实缺少必要的关键条件时，才回答"根据当前文档提供的信息，无法推导出明确答案"，并指出缺少了什么条件。

【输出格式要求】（必须严格遵守）：
- 使用 Markdown 格式输出
- 开头用 "## 结论" 标题，用 1-2 句话给出核心答案
- 用 "## 关键依据" 标题，列出文档中的直接证据（用 **粗体** 标记关键术语和关键字段）
- 用 "## 推理过程" 标题，用编号列表详细说明推导逻辑，每步独立成行
- 涉及优先级、状态对比、参数对照等内容时，使用 Markdown 表格呈现
- 如有补充说明或文档未覆盖的内容，用 "## 补充说明" 标题单独列出
- 段落之间留空行，避免出现大段连续文字

【参考文档】:
{context_text}

【问题】:
{query}
"""
        return prompt

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        生成答案（非流式）

        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档

        Returns:
            生成的答案
        """
        prompt = self._build_prompt(query, context_docs)

        # 通过 LLM Provider 调用
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # 确保返回值不是 None
        return response.content if response.content is not None else ""

    def generate_stream(self, query: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """
        流式生成答案

        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档

        Yields:
            逐块生成的答案文本
        """
        logger.info(f"Generation started: query='{query}', context_docs={len(context_docs)}")

        prompt = self._build_prompt(query, context_docs)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # 通过 LLM Provider 调用流式接口
        token_count = 0
        for chunk in self.llm.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        ):
            token_count += 1
            yield chunk

        logger.info(f"Generation completed: {token_count} chunks")

    def query(self, query: str) -> Tuple[str, List[Document], List[str]]:
        """
        完整的 RAG 查询流程（非流式，带查询重写与多路召回）

        Args:
            query: 用户查询

        Returns:
            (答案, 检索到的文档列表, 重写的查询列表)
        """
        all_queries = self.rewrite_query(query)
        docs = self.multi_query_retrieve(all_queries)

        if not docs:
            return "知识库中没有相关文档，请先添加文档。", [], all_queries

        answer = self.generate(query, docs)
        return answer, docs, all_queries

    def query_stream(self, query: str) -> Tuple[Generator[str, None, None], List[Document], List[str]]:
        """
        完整的 RAG 查询流程（流式，带查询重写与多路召回）

        Args:
            query: 用户查询

        Returns:
            (流式答案生成器, 检索到的文档列表, 重写的查询列表)
        """
        all_queries = self.rewrite_query(query)
        docs = self.multi_query_retrieve(all_queries)

        if not docs:
            def empty_generator():
                yield "知识库中没有相关文档，请先添加文档。"
            return empty_generator(), [], all_queries

        stream = self.generate_stream(query, docs)
        return stream, docs, all_queries
