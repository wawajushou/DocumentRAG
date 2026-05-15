"""
main.py

主程序入口 - 汽车仪表盘规范文档智能问答系统

改造要点：
1. 使用 get_settings() 获取配置
2. 使用 create_llm_provider() 工厂函数创建 LLM Provider
3. 将 Provider 注入到 RAGEngine
4. 集成日志记录，便于调试和监控
5. 使用上下文管理器确保资源正确释放
"""

import sys
import os
import shutil
import threading
import time
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.status import Status
from rich import print as rprint
from pick import pick

from config import get_settings
from providers import create_llm_provider
from document_manager import DocumentManager
from rag_engine import RAGEngine
from utils.logger import setup_logger, get_logger

# 初始化日志系统
setup_logger(log_dir="./logs", log_level="DEBUG")
logger = get_logger()


@contextmanager
def create_document_manager(settings):
    """
    文档管理器上下文管理器

    确保文档管理器在使用完毕后正确关闭，释放资源。

    Args:
        settings: 配置对象

    Yields:
        DocumentManager: 文档管理器实例
    """
    doc_manager = DocumentManager()
    try:
        yield doc_manager
    finally:
        doc_manager.close()
        logger.info("Document manager closed")


def print_welcome_panel(console: Console, settings) -> None:
    """
    打印欢迎面板

    Args:
        console: Rich Console 实例
        settings: 配置对象
    """
    welcome_text = f"""
[bold blue]汽车仪表盘规范文档智能问答系统[/bold blue]

[cyan]功能特性:[/cyan]
- 支持 Word/PDF/TXT 文档
- 智能增量更新知识库
- 基于 BGE 中文向量模型的语义检索
- DeepSeek 大模型智能回答
- 流式输出，实时查看
- 进阶 RAG: 查询重写 + 多路召回

[cyan]使用的模型:[/cyan]
  [bold]API 提供商:[/bold]       [cyan]{'小米 MiMo' if settings.api_provider == 'mimo' else 'DeepSeek'}[/cyan]
  [bold]向量嵌入模型:[/bold]    [green]{settings.embedding_model_name}[/green]
  [bold]聊天生成模型:[/bold]    [green]{settings.get_active_model_name()}[/green]
  [bold]API 服务:[/bold]        [green]{settings.get_active_base_url()}[/green]

[cyan]配置参数:[/cyan]
  [bold]Chunk Size:[/bold]      {settings.chunk_size}
  [bold]Chunk Overlap:[/bold]   {settings.chunk_overlap}
  [bold]检索 Top-K:[/bold]     {settings.retrieve_top_k}
  [bold]使用 ModelScope:[/bold]  {'是' if settings.use_model_scope else '否'}
  [bold]使用 Unstructured:[/bold] {'是' if settings.use_unstructured else '否'}

[yellow]使用说明:[/yellow]
1. 将文档放入 ./data 目录
2. 输入问题开始查询
3. 输入 'quit' 或 'exit' 退出
4. 输入 'update' 更新知识库
5. 输入 'reset' 查看重置知识库的操作指南
6. 输入 'switch' 切换 API 提供商 (DeepSeek <-> MiMo)
"""
    panel = Panel(
        welcome_text,
        title="[bold green]RAG 问答助手[/bold green]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def select_provider(console: Console, settings) -> str:
    """
    让用户通过上下键选择 LLM 提供商

    如果超过10秒未选择，默认使用小米MiMo模型

    Args:
        console: Rich Console 实例
        settings: 配置对象

    Returns:
        选择的提供商名称
    """
    # 定义选项列表
    options = [
        f"DeepSeek ({settings.deepseek_model_name})",
        f"小米 MiMo ({settings.mimo_model_name})",
        f"使用配置文件默认值 ({settings.api_provider})",
    ]

    # 存储用户选择结果
    result = {"provider": None, "timeout": False}

    def user_select():
        """用户选择线程"""
        try:
            title = "请选择 LLM 提供商 (上下键选择，回车确认，10秒后默认使用MiMo):"
            option, index = pick(options, title, indicator=">")

            if not result["timeout"]:  # 如果还没超时
                if index == 0:
                    result["provider"] = "deepseek"
                elif index == 1:
                    result["provider"] = "mimo"
                else:
                    result["provider"] = settings.api_provider
        except Exception:
            pass  # 超时后 pick 可能抛出异常，忽略即可

    def timeout_handler():
        """超时处理线程"""
        time.sleep(10)
        if result["provider"] is None:  # 如果用户还没选择
            result["timeout"] = True
            result["provider"] = "mimo"  # 默认使用 MiMo
            console.print("\n[yellow]选择超时，自动使用小米 MiMo 模型[/yellow]")

    # 启动两个线程
    select_thread = threading.Thread(target=user_select, daemon=True)
    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)

    select_thread.start()
    timeout_thread.start()

    # 等待两个线程中任意一个完成
    while result["provider"] is None:
        time.sleep(0.1)

    # 确保线程清理
    select_thread.join(timeout=0.1)
    timeout_thread.join(timeout=0.1)

    console.print(f"[green]已选择: {result['provider']}[/green]")
    return result["provider"]


def display_sources(console: Console, docs) -> None:
    """
    显示信息来源

    Args:
        console: Rich Console 实例
        docs: 检索到的文档列表
    """
    if not docs:
        return

    source_text = "\n[bold cyan]信息来源:[/bold cyan]\n"
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "未知")
        category = doc.metadata.get("category", "Text")

        source_text += f"  {i}. [blue]{filename}[/blue] (第 {page} 页)"
        if category != "Text":
            source_text += f" - [yellow]{category}[/yellow]"
        source_text += "\n"

    console.print(source_text)


def main() -> None:
    """主程序入口"""
    console = Console()
    logger.info("Application started")

    try:
        # 1. 加载配置
        settings = get_settings()

        print_welcome_panel(console, settings)

        # 2. 让用户选择 LLM 提供商
        selected_provider = select_provider(console, settings)
        settings.api_provider = selected_provider
        logger.info(f"User selected provider: {selected_provider}")

        console.print("[cyan]正在初始化系统...[/cyan]")

        # 3. 创建 LLM Provider（工厂模式）
        llm = create_llm_provider(
            provider_name=settings.api_provider,
            api_key=settings.get_active_api_key(),
            base_url=settings.get_active_base_url(),
            model=settings.get_active_model_name(),
        )
        logger.info(f"LLM provider created: {llm.provider_name}")

        # 4. 创建文档管理器（使用上下文管理器确保资源正确释放）
        with create_document_manager(settings) as doc_manager:
            doc_manager.run_update()

            # 5. 创建 RAG 引擎（依赖注入）
            rag_engine = RAGEngine(
                doc_manager=doc_manager,
                llm=llm,
                retrieve_top_k=settings.retrieve_top_k,
                temperature=settings.temperature,
            )

            # 6. 构建 BM25 索引（必须在文档更新后调用）
            rag_engine.update_bm25_index()

            console.print("[green]系统初始化完成！[/green]\n")

            while True:
                try:
                    user_input = console.input("[bold magenta]请输入问题: [/bold magenta]").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ["quit", "exit"]:
                        logger.info("User requested exit")
                        console.print("[yellow]感谢使用，再见！[/yellow]")
                        break

                    if user_input.lower() == "update":
                        logger.info("User requested knowledge base update")
                        console.print("[cyan]正在更新知识库...[/cyan]")
                        doc_manager.run_update()
                        rag_engine.update_bm25_index()
                        continue

                    if user_input.lower() == "reset":
                        console.print()
                        console.print("[red][bold]=============================================[/bold][/red]")
                        console.print("[red][bold]重置知识库操作指南[/bold][/red]")
                        console.print("[red][bold]=============================================[/bold][/red]")
                        console.print()
                        console.print("[yellow]由于 Windows 文件锁机制，程序无法自动删除正在使用的数据库。[/yellow]")
                        console.print()
                        console.print("[cyan]请按以下步骤手动操作：[/cyan]")
                        console.print("  1. [bold]先退出当前程序[/bold]（输入 'quit'）")
                        console.print(f"  2. [bold]手动删除文件夹：[/bold] [blue]{os.path.abspath(settings.vector_db_path)}[/blue]")
                        console.print("  3. [bold]重新运行程序：[/bold] python main.py")
                        console.print()
                        console.print("[green]程序重新启动时会自动用新策略重新处理所有文档！[/green]")
                        console.print()
                        console.print("[red][bold]=============================================[/bold][/red]")
                        console.print()
                        continue

                    if user_input.lower() == "switch":
                        current_provider = settings.api_provider
                        new_provider = "mimo" if current_provider == "deepseek" else "deepseek"

                        console.print(f"\n[cyan]当前 API 提供商: [bold]{'小米 MiMo' if current_provider == 'mimo' else 'DeepSeek'}[/bold][/cyan]")
                        console.print(f"[cyan]将切换到: [bold]{'小米 MiMo' if new_provider == 'mimo' else 'DeepSeek'}[/bold][/cyan]")

                        confirm = console.input("[yellow]确认切换? (yes/no): [/yellow]").strip().lower()
                        if confirm == "yes":
                            console.print("\n[cyan]正在更新配置...[/cyan]")

                            env_path = os.path.join(os.path.dirname(__file__), ".env")

                            try:
                                with open(env_path, "r", encoding="utf-8") as f:
                                    content = f.read()

                                content = content.replace(
                                    f"API_PROVIDER={current_provider}",
                                    f"API_PROVIDER={new_provider}"
                                )

                                with open(env_path, "w", encoding="utf-8") as f:
                                    f.write(content)

                                console.print("[green]配置已更新！[/green]")
                                console.print("[yellow]请重启程序以生效新配置[/yellow]\n")
                            except Exception as e:
                                console.print(f"[red]更新配置失败: {e}[/red]\n")
                        else:
                            console.print("[yellow]已取消切换[/yellow]\n")
                        continue

                    console.print()

                    with Status("[cyan]正在深度思考并拆解你的问题...[/cyan]", spinner="dots"):
                        stream, docs, all_queries = rag_engine.query_stream(user_input)

                    if len(all_queries) > 1:
                        console.print("[dim]AI 拆解的查询:[/dim]")
                        for i, q in enumerate(all_queries):
                            if i == 0:
                                console.print(f"  [dim]{i+1}. {q} (原始)[/dim]")
                            else:
                                console.print(f"  [dim]{i+1}. {q}[/dim]")
                        console.print()

                    with Status("[cyan]正在执行多路知识库检索...[/cyan]", spinner="dots"):
                        pass

                    console.print("\n[bold green]AI 回答:[/bold green]\n")

                    full_response = ""
                    with Live(console=console, refresh_per_second=10) as live:
                        for chunk in stream:
                            full_response += chunk
                            live.update(Markdown(full_response))

                    console.print()
                    display_sources(console, docs)
                    console.print("\n" + "=" * 60 + "\n")

                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt received")
                    console.print("\n[yellow]输入 'quit' 或 'exit' 退出程序[/yellow]\n")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    console.print(f"\n[red]发生错误: {str(e)}[/red]\n")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        console.print(f"[red]{str(e)}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Application startup failed: {e}")
        console.print(f"[red]系统启动失败: {str(e)}[/red]")
        sys.exit(1)
    finally:
        logger.info("Application exited")


if __name__ == "__main__":
    main()
