"""
providers/mimo.py

小米 MiMo LLM 提供商实现

MiMo API 同样兼容 OpenAI 格式，使用 OpenAI SDK 调用。
"""

from typing import List, Dict, Generator, Optional
from openai import OpenAI

from providers.base import BaseLLMProvider, LLMResponse
from utils.retry import api_retry
from utils.logger import get_logger

logger = get_logger()


class MiMoProvider(BaseLLMProvider):
    """
    小米 MiMo LLM 提供商

    MiMo 是小米自研的大模型，提供 OpenAI 兼容的 API 接口。
    支持 mimo-v2-pro 等模型。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.xiaomimimo.com/v1", model: str = "mimo-v2-pro"):
        """
        初始化 MiMo 提供商

        Args:
            api_key: MiMo API 密钥
            base_url: API 基础 URL（默认 https://api.xiaomimimo.com/v1）
            model: 模型名称（默认 mimo-v2-pro）
        """
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"MiMo provider initialized: model={model}")

    @property
    def provider_name(self) -> str:
        """返回提供商名称"""
        return "小米 MiMo"

    @property
    def model_name(self) -> str:
        """返回当前使用的模型名称"""
        return self._model

    @api_retry
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        带重试的非流式聊天

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            LLMResponse: 统一的响应对象
        """
        logger.debug(f"MiMo chat request: model={self._model}, temperature={temperature}")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.debug(f"MiMo chat response: {len(content)} chars")

            return LLMResponse(
                content=content,
                model=self._model,
                usage=response.usage.model_dump() if response.usage else None,
            )
        except Exception as e:
            logger.error(f"MiMo chat error: {e}")
            raise

    @api_retry
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        带重试的流式聊天

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Yields:
            str: 逐块生成的文本片段
        """
        logger.debug(f"MiMo stream request: model={self._model}")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            logger.debug("MiMo stream completed")
        except Exception as e:
            logger.error(f"MiMo stream error: {e}")
            raise
