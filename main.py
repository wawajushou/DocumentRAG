import sys
import os
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.status import Status
from rich import print as rprint

from config import config
from document_manager import DocumentManager
from rag_engine import RAGEngine


def print_welcome_panel(console: Console) -> None:
    """打印欢迎面板"""
    welcome_text = f"""
[bold blue]🚗 汽车仪表盘规范文档智能问答系统[/bold blue]

[cyan]功能特性:[/cyan]
• 支持 Word/PDF/TXT 文档
• 智能增量更新知识库
• 基于 BGE 中文向量模型的语义检索
• DeepSeek 大模型智能回答
• 流式输出，实时查看
• ✨ 进阶 RAG: 查询重写 + 多路召回

[cyan]🤖 使用的模型:[/cyan]
  [bold]向量嵌入模型:[/bold]    [green]{config.EMBEDDING_MODEL_NAME}[/green]
  [bold]聊天生成模型:[/bold]    [green]{config.MODEL_NAME}[/green]
  [bold]API 服务:[/bold]        [green]{config.API_BASE_URL}[/green]

[cyan]📦 配置参数:[/cyan]
  [bold]Chunk Size:[/bold]      {config.CHUNK_SIZE}
  [bold]Chunk Overlap:[/bold]   {config.CHUNK_OVERLAP}
  [bold]检索 Top-K:[/bold]     {config.RETRIEVE_TOP_K}
  [bold]使用 ModelScope:[/bold]  {'是' if config.USE_MODEL_SCOPE else '否'}
  [bold]使用 Unstructured:[/bold] {'是' if config.USE_UNSTRUCTURED else '否'}

[yellow]使用说明:[/yellow]
1. 将文档放入 ./data 目录
2. 输入问题开始查询
3. 输入 'quit' 或 'exit' 退出
4. 输入 'update' 更新知识库
5. 输入 'reset' 查看重置知识库的操作指南
"""
    panel = Panel(
        welcome_text,
        title="[bold green]RAG 问答助手[/bold green]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def display_sources(console: Console, docs) -> None:
    """显示信息来源"""
    if not docs:
        return
    
    source_text = "\n[bold cyan]📚 信息来源:[/bold cyan]\n"
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
    
    try:
        print_welcome_panel(console)
        
        console.print("[cyan]正在初始化系统...[/cyan]")
        
        doc_manager = DocumentManager()
        doc_manager.run_update()
        
        rag_engine = RAGEngine(doc_manager)
        
        console.print("[green]✓ 系统初始化完成！[/green]\n")
        
        while True:
            try:
                user_input = console.input("[bold magenta]请输入问题: [/bold magenta]").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    console.print("[yellow]感谢使用，再见！[/yellow]")
                    break
                
                if user_input.lower() == "update":
                    console.print("[cyan]正在更新知识库...[/cyan]")
                    doc_manager.run_update()
                    continue
                
                if user_input.lower() == "reset":
                    console.print()
                    console.print("[red][bold]=============================================[/bold][/red]")
                    console.print("[red][bold]⚠️  重置知识库操作指南[/bold][/red]")
                    console.print("[red][bold]=============================================[/bold][/red]")
                    console.print()
                    console.print("[yellow]由于 Windows 文件锁机制，程序无法自动删除正在使用的数据库。[/yellow]")
                    console.print()
                    console.print("[cyan]请按以下步骤手动操作：[/cyan]")
                    console.print("  1. [bold]先退出当前程序[/bold]（输入 'quit'）")
                    console.print(f"  2. [bold]手动删除文件夹：[/bold] [blue]{os.path.abspath(config.VECTOR_DB_PATH)}[/blue]")
                    console.print(f"  3. [bold]同时删除文件：[/bold] [blue]{os.path.abspath(config.FILE_HASH_PATH)}[/blue]（如果存在）")
                    console.print("  4. [bold]重新运行程序：[/bold] python main.py")
                    console.print()
                    console.print("[green]程序重新启动时会自动用新策略重新处理所有文档！[/green]")
                    console.print()
                    console.print("[red][bold]=============================================[/bold][/red]")
                    console.print()
                    continue
                
                console.print()
                
                with Status("[cyan]正在深度思考并拆解你的问题...[/cyan]", spinner="dots"):
                    stream, docs, all_queries = rag_engine.query_stream(user_input)
                
                if len(all_queries) > 1:
                    console.print("[dim]💡 AI 拆解的查询:[/dim]")
                    for i, q in enumerate(all_queries):
                        if i == 0:
                            console.print(f"  [dim]{i+1}. {q} (原始)[/dim]")
                        else:
                            console.print(f"  [dim]{i+1}. {q}[/dim]")
                    console.print()
                
                with Status("[cyan]正在执行多路知识库检索...[/cyan]", spinner="dots"):
                    pass
                
                console.print("\n[bold green]🤖 AI 回答:[/bold green]\n")
                
                full_response = ""
                with Live(console=console, refresh_per_second=10) as live:
                    for chunk in stream:
                        full_response += chunk
                        live.update(Markdown(full_response))
                
                console.print()
                display_sources(console, docs)
                console.print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]输入 'quit' 或 'exit' 退出程序[/yellow]\n")
            except Exception as e:
                console.print(f"\n[red]发生错误: {str(e)}[/red]\n")
    
    except ValueError as e:
        console.print(f"[red]{str(e)}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]系统启动失败: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
