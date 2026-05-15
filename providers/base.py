"""
providers/base.py

LLM 提供商抽象基类 - 定义统一的 LLM 调用接口

设计模式：策略模式 (Strategy Pattern)
- 所有 LLM 提供商都实现相同的接口
- RAGEngine 依赖抽象而非具体实现
- 新增提供商只需实现此基类，无需修改 RAGEngine
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional
from pydantic import BaseModel
from utils.retry import api_retry


class LLMResponse(BaseModel):
    """
    LLM 响应的统一数据结构

    使用 Pydantic BaseModel 确保类型安全和数据验证。
    """

    content: str                    # 生成的文本内容
    model: str                      # 使用的模型名称
    usage: Optional[Dict] = None    # token 使用统计（可选）


class BaseLLMProvider(ABC):
    """
    LLM 提供商抽象基类

    所有 LLM 提供商（DeepSeek、MiMo、OpenAI、Claude 等）都必须实现此接口。
    这确保了 RAGEngine 与具体 LLM 实现的完全解耦。

    使用方法：
        class MyCustomProvider(BaseLLMProvider):
            @property
            def provider_name(self) -> str:
                return "MyCustom"

            @property
            def model_name(self) -> str:
                return "my-model"

            def chat(self, messages, temperature=0.1, max_tokens=None):
                # 实现非流式聊天
                ...

            def chat_stream(self, messages, temperature=0.1, max_tokens=None):
                # 实现流式聊天
                ...
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回提供商名称，用于日志和显示"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        ...

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        非流式聊天接口

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性（0-2，越低越确定）
            max_tokens: 最大生成 token 数（可选）

        Returns:
            LLMResponse: 统一的响应对象，包含 content、model、usage
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        流式聊天接口

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制随机性（0-2，越低越确定）
            max_tokens: 最大生成 token 数（可选）

        Yields:
            str: 逐块生成的文本片段
        """
        ...

    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        带重试的非流式聊天接口

        使用 tenacity 实现指数退避重试，防止网络抖动导致程序崩溃。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            LLMResponse: 统一的响应对象
        """
        @api_retry
        def _call():
            return self.chat(messages, temperature, max_tokens)
        return _call()
