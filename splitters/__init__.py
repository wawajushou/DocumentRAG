"""
splitters/__init__.py

智能分割器模块 - 针对技术文档优化的文本分割策略

包含：
- HeaderExtractor: 章节标题提取器
- SemanticSplitter: 语义感知分割器
"""

from splitters.header_extractor import HeaderExtractor, HeaderInfo
from splitters.semantic_splitter import SemanticSplitter

__all__ = [
    "HeaderExtractor",
    "HeaderInfo",
    "SemanticSplitter",
]
