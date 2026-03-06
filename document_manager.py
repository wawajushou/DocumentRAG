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

from config import config


def get_embeddings():
    """获取嵌入模型，支持 ModelScope 国内镜像"""
    if config.USE_MODEL_SCOPE:
        try:
            from modelscope import snapshot_download
            
            local_model_path = snapshot_download(
                config.EMBEDDING_MODEL_NAME,
                cache_dir=config.MODEL_SCOPE_CACHE_DIR,
            )
            
            print(f"[green]✓ 模型已下载到: {local_model_path}[/green]")
            
            return HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs={'device': 'cpu'},
            )
        except ImportError:
            print("[yellow]警告: modelscope 未安装，将使用 HuggingFace 下载模型[/yellow]")
        except Exception as e:
            print(f"[yellow]警告: 从 ModelScope 下载失败: {e}[/yellow]")
            print("[yellow]将尝试使用 HuggingFace...[/yellow]")
    
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def preprocess_text(text: str) -> str:
    """预处理文本，清理多余空白"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class DocumentManager:
    """
    文档管理器 - 负责文档扫描、增量更新、向量化存储
    """

    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = get_embeddings()
        
        separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            "! ",
            "? ",
            " ",
            "",
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=separators,
            length_function=len,
        )
        
        self.file_hashes: Dict[str, str] = {}
        self._load_file_hashes()

    def _load_file_hashes(self) -> None:
        """加载已记录的文件哈希值"""
        if os.path.exists(config.FILE_HASH_PATH):
            try:
                with open(config.FILE_HASH_PATH, "r", encoding="utf-8") as f:
                    self.file_hashes = json.load(f)
            except Exception:
                self.file_hashes = {}

    def _save_file_hashes(self) -> None:
        """保存文件哈希值"""
        os.makedirs(os.path.dirname(config.FILE_HASH_PATH), exist_ok=True)
        with open(config.FILE_HASH_PATH, "w", encoding="utf-8") as f:
            json.dump(self.file_hashes, f, ensure_ascii=False, indent=2)

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的哈希值（使用修改时间和大小）"""
        stat = os.stat(file_path)
        return f"{stat.st_mtime}_{stat.st_size}"

    def _get_loader(self, file_path: str):
        """根据文件扩展名获取对应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            if config.USE_UNSTRUCTURED and HAS_UNSTRUCTURED:
                try:
                    return UnstructuredWordDocumentLoader(file_path, mode="single")
                except Exception as e:
                    print(f"[yellow]Unstructured 加载失败: {e}，回退到 Docx2txtLoader[/yellow]")
            return Docx2txtLoader(file_path)
        elif ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _enhance_metadata(self, docs: List[Document], file_path: str) -> List[Document]:
        """增强文档元数据"""
        filename = os.path.basename(file_path)
        
        for i, doc in enumerate(docs):
            doc.metadata["source"] = file_path
            doc.metadata["filename"] = filename
            doc.metadata["chunk_index"] = i
            
            doc.page_content = preprocess_text(doc.page_content)
        
        return docs

    def scan_documents(self) -> List[str]:
        """扫描数据目录下所有受支持的文档"""
        if not os.path.exists(config.DATA_DIR):
            return []
        
        documents = []
        for root, _, files in os.walk(config.DATA_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in config.SUPPORTED_EXTENSIONS:
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
        """处理文档：加载、切分"""
        all_splits = []
        task = progress.add_task("[cyan]处理文档...", total=len(document_paths))
        
        for doc_path in document_paths:
            try:
                progress.update(task, description=f"[cyan]正在处理: {os.path.basename(doc_path)}")
                
                loader = self._get_loader(doc_path)
                docs = loader.load()
                
                docs = self._enhance_metadata(docs, doc_path)
                
                splits = self.text_splitter.split_documents(docs)
                
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
                persist_directory=config.VECTOR_DB_PATH,
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
        if os.path.exists(config.VECTOR_DB_PATH) and os.listdir(config.VECTOR_DB_PATH):
            self.vectorstore = Chroma(
                persist_directory=config.VECTOR_DB_PATH,
                embedding_function=self.embeddings,
            )

    def run_update(self) -> None:
        """执行增量更新流程"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            
            self.initialize()
            
            all_docs = self.scan_documents()
            if not all_docs:
                progress.console.print("[yellow]未找到任何文档，请将文档放入 ./data 目录")
                return
            
            changed_docs = self.get_changed_documents(all_docs)
            if not changed_docs:
                progress.console.print("[green]所有文档都是最新的，无需更新")
                return
            
            progress.console.print(f"[blue]发现 {len(changed_docs)} 个文档需要更新")
            
            splits = self.process_documents(changed_docs, progress)
            if splits:
                self.update_vectorstore(splits, progress)
                progress.console.print(f"[green]成功添加 {len(splits)} 个文本块到知识库")
