import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config import get_settings
from splitters.semantic_splitter import SemanticSplitter
from splitters.header_extractor import HeaderExtractor
from utils.logger import get_logger

logger = get_logger()


def get_embeddings():
    """获取嵌入模型，支持 ModelScope 国内镜像"""
    settings = get_settings()

    if settings.use_model_scope:
        try:
            from modelscope import snapshot_download

            local_model_path = snapshot_download(
                settings.embedding_model_name,
                cache_dir=settings.model_scope_cache_dir,
            )

            logger.info(f"Model downloaded to: {local_model_path}")

            return HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs={'device': 'cpu'},
            )
        except ImportError:
            logger.warning("modelscope not installed, using HuggingFace")
        except Exception as e:
            logger.warning(f"ModelScope download failed: {e}, trying HuggingFace...")

    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def preprocess_text(text: str) -> str:
    """预处理文本，清理多余空白"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class DocumentManager:
    """
    文档管理器 - 负责文档扫描、增量更新、向量化存储

    改进：
    1. 使用语义感知分割器，保留表格和代码块结构
    2. 使用章节标题提取器，增强文档元数据
    """

    def __init__(self):
        self.settings = get_settings()
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = get_embeddings()
        logger.info("DocumentManager initialized")

        # 使用语义感知分割器（针对技术文档优化）
        self.text_splitter = SemanticSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # 章节标题提取器
        self.header_extractor = HeaderExtractor()

        self.file_hashes: Dict[str, str] = {}
        self._load_file_hashes()

    def _load_file_hashes(self) -> None:
        """加载已记录的文件哈希值"""
        file_hash_path = os.path.join(self.settings.vector_db_path, "file_hashes.json")
        if os.path.exists(file_hash_path):
            try:
                with open(file_hash_path, "r", encoding="utf-8") as f:
                    self.file_hashes = json.load(f)
            except Exception:
                self.file_hashes = {}

    def _save_file_hashes(self) -> None:
        """保存文件哈希值"""
        file_hash_path = os.path.join(self.settings.vector_db_path, "file_hashes.json")
        os.makedirs(os.path.dirname(file_hash_path), exist_ok=True)
        with open(file_hash_path, "w", encoding="utf-8") as f:
            json.dump(self.file_hashes, f, ensure_ascii=False, indent=2)

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的哈希值（使用修改时间和大小）"""
        stat = os.stat(file_path)
        return f"{stat.st_mtime}_{stat.st_size}"

    def _get_loader(self, file_path: str):
        """根据文件扩展名获取对应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            if self.settings.use_unstructured and HAS_UNSTRUCTURED:
                try:
                    return UnstructuredWordDocumentLoader(file_path, mode="single")
                except Exception as e:
                    logger.warning(f"Unstructured loader failed: {e}, falling back to Docx2txtLoader")
            return Docx2txtLoader(file_path)
        elif ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _enhance_metadata(self, docs: List[Document], file_path: str) -> List[Document]:
        """
        增强文档元数据

        添加：
        - source: 文件路径
        - filename: 文件名
        - chunk_index: 块索引
        - header: 章节标题（从文本中提取）
        """
        filename = os.path.basename(file_path)

        for i, doc in enumerate(docs):
            doc.metadata["source"] = file_path
            doc.metadata["filename"] = filename
            doc.metadata["chunk_index"] = i

            # 预处理文本
            doc.page_content = preprocess_text(doc.page_content)

            # 提取章节标题
            # 使用文本开头作为参考位置，查找最近的标题
            header = self.header_extractor.find_nearest_header(
                doc.page_content,
                position=0
            )
            doc.metadata["header"] = header if header else "未知章节"

        return docs

    def scan_documents(self) -> List[str]:
        """扫描数据目录下所有受支持的文档"""
        if not os.path.exists(self.settings.data_dir):
            return []

        documents = []
        for root, _, files in os.walk(self.settings.data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.settings.supported_extensions:
                    documents.append(os.path.join(root, file))
        return documents

    def get_changed_documents(self, all_documents: List[str]) -> List[str]:
        """获取需要更新的文档（新增或修改的）"""
        changed = []
        for doc_path in all_documents:
            file_hash = self._calculate_file_hash(doc_path)
            if doc_path not in self.file_hashes or self.file_hashes[doc_path] != file_hash:
                changed.append(doc_path)
        return changed

    def process_documents(self, document_paths: List[str], progress: Progress) -> List[Document]:
        """
        处理文档：加载、切分

        使用语义感知分割器，保留表格和代码块结构。
        为每个分割添加章节标题元数据。
        """
        all_splits = []
        task = progress.add_task("[cyan]处理文档...", total=len(document_paths))

        for doc_path in document_paths:
            try:
                progress.update(task, description=f"[cyan]正在处理: {os.path.basename(doc_path)}")

                # 加载文档
                loader = self._get_loader(doc_path)
                docs = loader.load()

                # 增强元数据（添加章节标题）
                docs = self._enhance_metadata(docs, doc_path)

                # 使用语义感知分割器
                splits = self.text_splitter.split_documents(docs)

                # 为每个分割添加 chunk_subindex 和页码估算
                cumulative_chars = 0
                chars_per_page = 1500

                for j, split in enumerate(splits):
                    split.metadata["chunk_subindex"] = j

                    if "page" not in split.metadata:
                        split_length = len(split.page_content)

                        mid_point = cumulative_chars + split_length // 2
                        estimated_page = mid_point // chars_per_page + 1
                        split.metadata["page"] = estimated_page

                        cumulative_chars += split_length

                all_splits.extend(splits)

                file_hash = self._calculate_file_hash(doc_path)
                self.file_hashes[doc_path] = file_hash

                progress.advance(task)
            except Exception as e:
                progress.console.print(f"[red]处理文档 {doc_path} 时出错: {e}")

        progress.update(task, description="[green]文档处理完成!")
        return all_splits

    def update_vectorstore(self, splits: List[Document], progress: Progress) -> None:
        """更新向量数据库"""
        if not splits:
            return

        task = progress.add_task("[cyan]向量化存储中...", total=len(splits))

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.settings.vector_db_path,
            )
        else:
            self.vectorstore.add_documents(splits)

        progress.update(task, advance=len(splits))
        self._save_file_hashes()
        progress.update(task, description="[green]向量化完成!")

    def close(self) -> None:
        """关闭向量数据库连接，释放文件锁"""
        if self.vectorstore is not None:
            try:
                if hasattr(self.vectorstore, '_client'):
                    del self.vectorstore._client
                if hasattr(self.vectorstore, '_collection'):
                    del self.vectorstore._collection
            except:
                pass
            self.vectorstore = None

    def initialize(self) -> None:
        """初始化向量数据库"""
        if os.path.exists(self.settings.vector_db_path) and os.listdir(self.settings.vector_db_path):
            self.vectorstore = Chroma(
                persist_directory=self.settings.vector_db_path,
                embedding_function=self.embeddings,
            )

    def run_update(self) -> None:
        """执行增量更新流程"""
        logger.info("Document update started")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:

            self.initialize()

            all_docs = self.scan_documents()
            if not all_docs:
                logger.warning("No documents found in ./data directory")
                progress.console.print("[yellow]未找到任何文档，请将文档放入 ./data 目录")
                return

            logger.info(f"Found {len(all_docs)} documents")

            changed_docs = self.get_changed_documents(all_docs)
            if not changed_docs:
                logger.info("All documents are up to date")
                progress.console.print("[green]所有文档都是最新的，无需更新")
                return

            logger.info(f"Found {len(changed_docs)} changed documents")
            progress.console.print(f"[blue]发现 {len(changed_docs)} 个文档需要更新")

            splits = self.process_documents(changed_docs, progress)
            if splits:
                self.update_vectorstore(splits, progress)
                logger.info(f"Successfully added {len(splits)} text chunks to knowledge base")
                progress.console.print(f"[green]成功添加 {len(splits)} 个文本块到知识库")
