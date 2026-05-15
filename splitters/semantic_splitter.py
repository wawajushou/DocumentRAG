"""
splitters/semantic_splitter.py

语义感知分割器 - 针对技术文档优化的文本分割策略

特点：
1. 识别并保留表格结构（以 | 分隔的表格行）
2. 识别并保留代码块（``` 包围的代码）
3. 识别并保留列表结构（- 或 1. 开头的列表项）
4. 基于章节标题进行分割（优先在标题处断开）
"""

import re
from typing import List, Optional, Callable
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticSplitter:
    """
    语义感知分割器

    针对技术文档的特点进行优化：
    - 表格：以 | 分隔的行被视为一个整体，不会被切断
    - 代码块：``` 包围的代码被视为一个整体
    - 列表：连续的列表项会被尽量保持在一起
    - 标题：优先在标题处断开

    使用方法：
        splitter = SemanticSplitter(chunk_size=800, chunk_overlap=200)
        splits = splitter.split_documents(documents)
    """

    # 技术文档专用分隔符（按优先级排序）
    # 优先级从高到低：标题 > 段落 > 表格 > 代码块 > 列表 > 句子 > 单词
    TECH_SEPARATORS = [
        "\n# ",           # Markdown 一级标题
        "\n## ",          # Markdown 二级标题
        "\n### ",         # Markdown 三级标题
        "\n\n",           # 段落分隔
        "\n|",            # 表格行开始
        "\n```",          # 代码块边界
        "\n- ",           # 无序列表项
        "\n1. ",          # 有序列表项
        "。", "！", "？",  # 中文句号
        ". ", "! ", "? ",  # 英文句号
        " ", "",
    ]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        """
        初始化分割器

        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            length_function: 长度计算函数（支持 token 计数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

        # 创建基础分割器
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.TECH_SEPARATORS,
            length_function=length_function,
        )

    def _is_table_block(self, text: str) -> bool:
        """
        判断文本块是否是表格

        通过检查是否包含 | 分隔符来判断。
        """
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False

        # 检查是否包含表格分隔符 |
        table_lines = [line for line in lines if '|' in line]
        return len(table_lines) >= len(lines) * 0.5

    def _is_code_block(self, text: str) -> bool:
        """
        判断文本块是否是代码块

        通过检查是否以 ``` 包围来判断。
        """
        return text.strip().startswith('```') and text.strip().endswith('```')

    def _is_list_block(self, text: str) -> bool:
        """
        判断文本块是否是列表

        通过检查是否以 - 或 1. 等开头来判断。
        """
        lines = text.strip().split('\n')
        list_pattern = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+')

        list_lines = [line for line in lines if list_pattern.match(line)]
        return len(list_lines) >= len(lines) * 0.5

    def _split_by_structure(self, text: str) -> List[str]:
        """
        基于结构进行预分割

        识别表格、代码块、列表等结构，将它们作为独立的块保留。
        """
        blocks = []
        current_block = []
        in_code_block = False

        lines = text.split('\n')

        for line in lines:
            # 代码块边界检测
            if line.strip().startswith('```'):
                if in_code_block:
                    # 代码块结束
                    current_block.append(line)
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # 代码块开始，先保存之前的块
                    if current_block:
                        blocks.append('\n'.join(current_block))
                        current_block = []
                    current_block.append(line)
                    in_code_block = True
                continue

            if in_code_block:
                current_block.append(line)
                continue

            # 表格行检测（包含 | 的行）
            if '|' in line:
                if current_block and not any('|' in l for l in current_block):
                    # 之前的块不是表格，先保存
                    blocks.append('\n'.join(current_block))
                    current_block = []
                current_block.append(line)
                continue

            # 普通文本行
            if current_block and '|' in current_block[-1]:
                # 之前的块是表格，先保存
                blocks.append('\n'.join(current_block))
                current_block = []

            current_block.append(line)

        # 保存最后一个块
        if current_block:
            blocks.append('\n'.join(current_block))

        return blocks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表

        Args:
            documents: 文档列表

        Returns:
            分割后的文档块列表
        """
        all_splits = []

        for doc in documents:
            # 1. 基于结构预分割
            structural_blocks = self._split_by_structure(doc.page_content)

            # 2. 对每个结构块进行细粒度分割
            for block in structural_blocks:
                # 表格和代码块保持完整，不进行分割
                if self._is_table_block(block) or self._is_code_block(block):
                    if self.length_function(block) <= self.chunk_size * 1.5:
                        # 允许表格/代码块略微超过 chunk_size
                        content_type = "table" if self._is_table_block(block) else "code"
                        new_doc = Document(
                            page_content=block,
                            metadata={**doc.metadata, "content_type": content_type}
                        )
                        all_splits.append(new_doc)
                        continue

                # 普通文本使用基础分割器
                splits = self.base_splitter.split_text(block)

                for split_text in splits:
                    content_type = "text"
                    if self._is_list_block(split_text):
                        content_type = "list"

                    new_doc = Document(
                        page_content=split_text,
                        metadata={**doc.metadata, "content_type": content_type}
                    )
                    all_splits.append(new_doc)

        return all_splits

    def split_text(self, text: str) -> List[str]:
        """
        分割纯文本

        Args:
            text: 原始文本

        Returns:
            分割后的文本块列表
        """
        # 1. 基于结构预分割
        structural_blocks = self._split_by_structure(text)

        # 2. 对每个结构块进行细粒度分割
        all_splits = []
        for block in structural_blocks:
            if self._is_table_block(block) or self._is_code_block(block):
                if self.length_function(block) <= self.chunk_size * 1.5:
                    all_splits.append(block)
                    continue

            splits = self.base_splitter.split_text(block)
            all_splits.extend(splits)

        return all_splits
