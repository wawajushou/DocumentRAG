from typing import List, Generator, Tuple
from openai import OpenAI
from langchain_core.documents import Document
import hashlib

from config import config
from document_manager import DocumentManager


class RAGEngine:
    """
    RAG引擎 - 核心检索与生成模块（支持查询重写与多路召回）
    """

    def __init__(self, doc_manager: DocumentManager):
        self.doc_manager = doc_manager
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
        )

    def rewrite_query(self, original_query: str) -> List[str]:
        """
        查询重写：将原始问题拆解为3个不同侧重点的子查询
        
        Args:
            original_query: 用户的原始问题
            
        Returns:
            包含重写后查询的列表（包含原始问题+3个重写问题，或仅原始问题如果重写失败）
        """
        rewrite_prompt = f"""你是一位专业的汽车仪表盘开发领域专家。
请将以下问题拆解、扩写为 3 个不同侧重点的、适合向量检索的独立子查询。

要求：
- 每个子查询应该关注问题的不同方面（如：硬件层面、软件层面、协议层面、同义词替换等）
- 子查询应该尽可能提取文档中可能出现的专业术语和关键词
- 仅返回这 3 个查询，每行一个，不要有任何其他说明

原始问题：
{original_query}

3 个子查询：
"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.3,
            )
            
            rewritten_text = response.choices[0].message.content.strip()
            
            rewritten_queries = []
            for line in rewritten_text.split("\n"):
                line = line.strip()
                if line and not line.startswith("原始问题") and not line.startswith("3 个"):
                    rewritten_queries.append(line)
            
            if len(rewritten_queries) >= 1:
                all_queries = [original_query] + rewritten_queries[:3]
                return all_queries
            
            return [original_query]
            
        except Exception as e:
            print(f"[yellow]查询重写失败: {e}，将使用原始查询[/yellow]")
            return [original_query]

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        文档去重：基于内容哈希值去重
        
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

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        从向量数据库中检索相关文档（单查询）
        
        Args:
            query: 用户查询
            k: 返回的文档数量，默认使用配置中的值
            
        Returns:
            相关文档列表
        """
        if k is None:
            k = config.RETRIEVE_TOP_K
        
        if self.doc_manager.vectorstore is None:
            return []
        
        return self.doc_manager.vectorstore.similarity_search(query, k=k)

    def multi_query_retrieve(self, queries: List[str], k: int = None) -> List[Document]:
        """
        多路召回：对多个查询分别检索，然后去重合并
        
        Args:
            queries: 查询列表
            k: 每个查询返回的文档数量
            
        Returns:
            去重合并后的文档列表
        """
        all_docs = []
        
        for query in queries:
            docs = self.retrieve(query, k)
            all_docs.extend(docs)
        
        unique_docs = self._deduplicate_docs(all_docs)
        return unique_docs

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        """
        构建Prompt
        
        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档
            
        Returns:
            完整的Prompt字符串
        """
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""你是一个专业的汽车仪表盘项目开发助手。请严格根据提供的参考文档回答，不得编造。若文档未提及，请回答'根据当前文档无法得知'。

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
        
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMPERATURE,
        )
        
        return response.choices[0].message.content

    def generate_stream(self, query: str, context_docs: List[Document]) -> Generator[str, None, None]:
        """
        流式生成答案
        
        Args:
            query: 用户查询
            context_docs: 检索到的上下文文档
            
        Yields:
            逐块生成的答案文本
        """
        prompt = self._build_prompt(query, context_docs)
        
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMPERATURE,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def query(self, query: str) -> Tuple[str, List[Document], List[str]]:
        """
        完整的RAG查询流程（非流式，带查询重写与多路召回）
        
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
        完整的RAG查询流程（流式，带查询重写与多路召回）
        
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
