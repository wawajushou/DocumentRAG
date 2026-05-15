"""
splitters/header_extractor.py

章节标题提取器 - 从技术文档中提取章节标题

支持的标题格式：
1. Markdown 标题：# ## ###
2. 中文数字标题：一、二、三、
3. 数字编号标题：1. 2. 3. 或 1.1 1.2 1.3
4. 带括号的编号：(1) (2) (3) 或 (一) (二) (三)
"""

import re
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class HeaderInfo:
    """
    章节标题信息

    Attributes:
        level: 标题级别（1=一级，2=二级，3=三级）
        title: 标题文本
        position: 在原文中的位置（字符索引）
    """

    level: int
    title: str
    position: int


class HeaderExtractor:
    """
    章节标题提取器

    使用正则表达式匹配多种标题格式，提取文档的章节结构。

    使用方法：
        extractor = HeaderExtractor()
        headers = extractor.extract_headers(text)
        nearest = extractor.find_nearest_header(text, position=1000)
    """

    # 标题匹配模式（按优先级排序）
    # 每个模式是一个元组：(正则表达式, 标题级别)
    HEADER_PATTERNS = [
        # Markdown 标题：# ## ###
        (r'^#{1}\s+(.+)$', 1),
        (r'^#{2}\s+(.+)$', 2),
        (r'^#{3}\s+(.+)$', 3),

        # 中文数字标题：一、二、三、
        (r'^[一二三四五六七八九十]+[、.]?\s*(.+)$', 1),

        # 数字编号标题：1. 2. 3.
        (r'^(\d+)\.\s+(.+)$', 2),

        # 多级数字编号：1.1 1.2 1.3
        (r'^(\d+\.\d+)\s+(.+)$', 2),

        # 带括号的编号：(1) (2) (3)
        (r'^\((\d+)\)\s*(.+)$', 3),

        # 带括号的中文数字：(一) (二) (三)
        (r'^([一二三四五六七八九十]+)\)\s*(.+)$', 2),
    ]

    def extract_headers(self, text: str) -> List[HeaderInfo]:
        """
        从文本中提取所有章节标题

        Args:
            text: 原始文本

        Returns:
            HeaderInfo 列表，按位置排序
        """
        headers = []
        lines = text.split('\n')
        current_pos = 0

        for line in lines:
            line_stripped = line.strip()

            # 尝试匹配每种标题模式
            for pattern, level in self.HEADER_PATTERNS:
                match = re.match(pattern, line_stripped)
                if match:
                    # 提取标题文本（取最后一个捕获组）
                    title = match.group(match.lastindex) if match.lastindex >= 1 else line_stripped
                    headers.append(HeaderInfo(
                        level=level,
                        title=title.strip(),
                        position=current_pos,
                    ))
                    break  # 匹配到一种模式就停止

            current_pos += len(line) + 1  # +1 for newline

        return headers

    def find_nearest_header(self, text: str, position: int) -> Optional[str]:
        """
        查找指定位置之前最近的章节标题

        Args:
            text: 原始文本
            position: 字符位置

        Returns:
            最近的章节标题文本，如果没有找到则返回 None
        """
        headers = self.extract_headers(text)

        # 查找 position 之前最近的标题
        nearest = None
        for header in headers:
            if header.position <= position:
                nearest = header
            else:
                break

        return nearest.title if nearest else None

    def get_header_hierarchy(self, text: str, position: int) -> List[str]:
        """
        获取指定位置的完整标题层级（从一级到当前级别）

        例如：["第一章 总则", "1.1 目的"]

        Args:
            text: 原始文本
            position: 字符位置

        Returns:
            标题层级列表
        """
        headers = self.extract_headers(text)

        # 找到 position 之前的所有标题
        hierarchy = []
        for header in headers:
            if header.position <= position:
                # 清理同级别的旧标题
                while hierarchy and len(hierarchy) >= header.level:
                    hierarchy.pop()
                hierarchy.append(header.title)
            else:
                break

        return hierarchy
