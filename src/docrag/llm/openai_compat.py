"""OpenAI 兼容协议的 LLM 实现（DeepSeek / MiMo 共用）。

与旧实现的三点区别：
1. 模型名和参数从 PROFILES 表来，不再硬编码过期的 deepseek-chat / mimo-v2-pro
2. 重试捕获的是 openai SDK 的真实异常类型（旧代码只捕获内置 ConnectionError，永远不触发）
3. 流式只在「建立连接」阶段重试，不会在已经吐出半截答案后重来
"""

from __future__ import annotations

from collections.abc import Iterator

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from docrag.config import Settings

RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    ConnectionError,
    TimeoutError,
)


class OpenAICompatLLM:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._max_field = settings.profile.max_output_field

        # MiMo 的 TokenPlan 除了 Bearer 还认 api-key 头，两个都带上最稳
        default_headers = None
        if settings.llm_provider == "mimo":
            default_headers = {"api-key": settings.api_key}

        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=0,  # 重试统一交给 tenacity，避免双重退避
            default_headers=default_headers,
        )
        logger.info(f"LLM ready: {self.name} ({settings.base_url})")

    @property
    def name(self) -> str:
        return f"{self._settings.llm_provider}/{self._settings.model}"

    def _body(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        thinking: bool | None = None,
    ) -> dict:
        body: dict = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": (
                self._settings.llm_temperature if temperature is None else temperature
            ),
        }
        limit = max_tokens or self._settings.llm_max_output_tokens
        if limit:
            body[self._max_field] = limit

        # thinking 是两家各自的扩展字段，OpenAI SDK 必须走 extra_body 才能透传
        extra: dict = {}
        enabled = thinking if thinking is not None else self._settings.llm_thinking == "on"
        if enabled:
            extra["thinking"] = {"type": "enabled"}
            extra["reasoning_effort"] = self._settings.llm_reasoning_effort
        elif thinking is False or self._settings.llm_thinking == "off":
            extra["thinking"] = {"type": "disabled"}
        # 其余情况（auto）交给服务端默认
        if extra:
            body["extra_body"] = extra
        return body

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )
    def _create(self, body: dict):
        return self._client.chat.completions.create(**body)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> str:
        response = self._create(self._body(messages, temperature, max_tokens, thinking))
        text = response.choices[0].message.content or ""
        if response.usage:
            logger.debug(
                f"{self.name} tokens: prompt={response.usage.prompt_tokens} "
                f"completion={response.usage.completion_tokens}"
            )
        return text

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> Iterator[str]:
        body = self._body(messages, temperature, max_tokens, thinking)
        body["stream"] = True
        # 这一步只负责把连接建起来，失败会重试；进入迭代后不再重试
        chunks = self._create(body)
        for item in chunks:
            if not item.choices:
                continue
            delta = item.choices[0].delta
            # 思考模式下 reasoning_content 与 content 分开，这里只取最终答案
            text = getattr(delta, "content", None)
            if text:
                yield text
