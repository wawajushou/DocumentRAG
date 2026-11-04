"""命令行入口。

  docrag                     交互问答（默认）
  docrag sync                同步 data 目录到知识库
  docrag ask "问题"          单次提问
  docrag status              查看索引和配置状态
  docrag --provider mimo     临时切换提供商
"""

from __future__ import annotations

import argparse
import re
import time

import chromadb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from docrag.config import Settings, get_settings
from docrag.embedding import create_embedder
from docrag.ingest.indexer import Indexer, build_manifest
from docrag.llm import create_llm
from docrag.logging import setup_logging
from docrag.rag import Prepared, RAGEngine
from docrag.retrieve.bm25 import BM25Index
from docrag.retrieve.rerank import Reranker
from docrag.store import ChromaStore

_CITE_RE = re.compile(r"\[(\d+)\]")

HELP_TEXT = """[bold]可用命令[/bold]
  /update   重新同步 data 目录
  /status   查看索引状态
  /quit     退出
"""


class App:
    def __init__(self, settings: Settings, console: Console):
        self.settings = settings
        self.console = console
        self._embedder = None
        self._store: ChromaStore | None = None
        self._manifest = None
        self._indexer: Indexer | None = None
        self._bm25: BM25Index | None = None
        self._engine: RAGEngine | None = None

    # ---------- 懒加载 ----------

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = create_embedder(self.settings)
        return self._embedder

    @property
    def store(self) -> ChromaStore:
        if self._store is None:
            self._store = ChromaStore(self.settings.storage_dir, self.embedder)
        return self._store

    @property
    def indexer(self) -> Indexer:
        if self._indexer is None:
            self._manifest = build_manifest(self.settings, self.embedder)
            self._indexer = Indexer(self.settings, self.embedder, self.store, self._manifest)
        return self._indexer

    def _build_bm25(self) -> BM25Index:
        if self._bm25 is None:
            chunks = list(self.store.iter_chunks())
            userdict = self.settings.data_dir / "dict.txt"
            self._bm25 = BM25Index(chunks, userdict=userdict)
            logger.info(f"BM25 索引: {len(self._bm25)} 块")
        return self._bm25

    def engine(self) -> RAGEngine:
        if self._engine is None:
            reranker = Reranker(self.settings) if self.settings.rerank_enabled else None
            self._engine = RAGEngine(
                settings=self.settings,
                llm=create_llm(self.settings),
                embedder=self.embedder,
                store=self.store,
                bm25=self._build_bm25(),
                reranker=reranker,
            )
        return self._engine

    # ---------- 命令 ----------

    def sync(self) -> None:
        with self.console.status("[cyan]同步文档...[/cyan]"):
            report = self.indexer.sync(progress=lambda msg: logger.info(msg))
        self._bm25 = None
        self._engine = None
        self.console.print(
            f"[green]同步完成[/green] 新增 {len(report.added)} / 更新 {len(report.updated)} / "
            f"删除 {len(report.removed)} / 跳过 {report.skipped} / "
            f"{report.chunks} 块 / {report.seconds:.1f}s"
        )

    def status(self) -> None:
        table = Table(title="docrag 状态", show_header=False)
        table.add_column("项", style="cyan")
        table.add_column("值")
        table.add_row("提供商", f"{self.settings.llm_provider} ({self.settings.model})")
        table.add_row("Embedding", self.settings.embedding_model)
        table.add_row("切分", f"{self.settings.chunk_size} / overlap {self.settings.chunk_overlap}")
        table.add_row("检索", f"top_k={self.settings.retrieve_top_k} rerank={self.settings.rerank_enabled}")
        table.add_row("数据目录", str(self.settings.data_dir.resolve()))
        table.add_row("向量块数", str(_count_chunks(self.settings.storage_dir)))

        if self._manifest is None:
            from docrag.ingest.manifest import Manifest

            manifest = Manifest(self.settings.storage_dir / "manifest.json", "")
            files = len(manifest.files)
        else:
            files = len(self._manifest.files)
        table.add_row("已索引文件", str(files))
        self.console.print(table)

    def ask_stream(self, question: str) -> None:
        engine = self.engine()
        with self.console.status("[cyan]分析问题并检索...[/cyan]"):
            prepared = engine.prepare(question)

        extra = prepared.sub_queries[1:]
        if extra:
            self.console.print("[dim]改写查询: " + " | ".join(extra) + "[/dim]")

        if not prepared.hits:
            self.console.print("[yellow]没有检索到相关内容。[/yellow]")
            self.console.print("[dim]先把文档放进 data 目录，然后执行 docrag sync[/dim]")
            return

        buffer: list[str] = []
        if not self.console.is_terminal:
            # 管道 / 重定向场景下没有 Live 可用，直接攒完再输出
            buffer = list(engine.stream(prepared))
            self.console.print(Markdown("".join(buffer)))
            self._show_sources("".join(buffer), prepared)
            return

        last = 0.0
        with Live(console=self.console, refresh_per_second=8, vertical_overflow="visible") as live:
            for token in engine.stream(prepared):
                buffer.append(token)
                now = time.monotonic()
                if now - last > 0.08:  # 节流，避免每个 token 都重排整篇 Markdown
                    live.update(Markdown("".join(buffer)))
                    last = now
            live.update(Markdown("".join(buffer)))

        self._show_sources("".join(buffer), prepared)

    def _show_sources(self, text: str, prepared: Prepared) -> None:
        total = len(prepared.hits)
        cited = sorted({int(n) for n in _CITE_RE.findall(text) if 1 <= int(n) <= total})

        table = Table(title="信息来源" + ("" if cited else "（模型未标注引用，列出全部召回）"))
        table.add_column("#", width=3)
        table.add_column("文件")
        table.add_column("位置")
        table.add_column("章节")

        for index in cited or range(1, total + 1):
            chunk = prepared.hits[index - 1].chunk
            position = f"第{chunk.page}页" if chunk.page else f"块{chunk.chunk_index}"
            table.add_row(str(index), chunk.file, position, chunk.header or "-")
        self.console.print(table)

    def chat(self) -> None:
        self.console.print(
            Panel(
                f"[bold]本地文档 RAG 问答[/bold]\n"
                f"提供商 [cyan]{self.settings.llm_provider}[/cyan] · "
                f"模型 [cyan]{self.settings.model}[/cyan]\n\n{HELP_TEXT}",
                border_style="blue",
            )
        )
        while True:
            try:
                line = self.console.input("[bold magenta]问题 > [/bold magenta]").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]已退出[/yellow]")
                return

            if not line:
                continue
            lowered = line.lower()
            if lowered in {"/quit", "/exit", "quit", "exit"}:
                return
            if lowered in {"/update", "update"}:
                self.sync()
                continue
            if lowered in {"/status", "status"}:
                self.status()
                continue

            try:
                self.ask_stream(line)
            except Exception as exc:  # noqa: BLE001
                logger.exception("问答失败")
                self.console.print(f"[red]出错了：{exc}[/red]")


def _count_chunks(storage_dir) -> int:
    try:
        client = chromadb.PersistentClient(path=str(storage_dir / "chroma"))
        return client.get_collection("docrag").count()
    except Exception:  # noqa: BLE001
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docrag", description="本地文档 RAG 问答")
    parser.add_argument("--provider", choices=["deepseek", "mimo"], help="临时切换 LLM 提供商")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("sync", help="同步文档到知识库")
    subparsers.add_parser("chat", help="交互问答（默认）")
    subparsers.add_parser("status", help="查看索引状态")
    ask_parser = subparsers.add_parser("ask", help="单次提问")
    ask_parser.add_argument("question", help="要问的问题")

    args = parser.parse_args(argv)

    settings = get_settings()
    if args.provider:
        settings = settings.model_copy(update={"llm_provider": args.provider})

    setup_logging(settings.log_dir)
    settings.ensure_dirs()
    console = Console()
    app = App(settings, console)

    command = args.command or "chat"
    try:
        if command == "sync":
            app.sync()
        elif command == "ask":
            app.ask_stream(args.question)
        elif command == "status":
            app.status()
        else:
            app.chat()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0
