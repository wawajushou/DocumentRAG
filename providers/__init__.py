"""
providers/__init__.py

LLM 提供商工厂 - 根据配置自动创建对应的 Provider 实例

设计模式：工厂模式 (Factory Pattern)
- 封装对象创建逻辑
- 调用方无需知道具体实现类
- 集中管理提供商的注册和实例化
"""

from typing import Dict, Type
from providers.base import BaseLLMProvider
from providers.deepseek import DeepSeekProvider
from providers.mimo import MiMoProvider


# 提供商注册表：名称 -> 实现类
PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "mimo": MiMoProvider,
}


def create_llm_provider(
    provider_name: str,
    api_key: str,
    base_url: str,
    model: str,
) -> BaseLLMProvider:
    """
    LLM 提供商工厂函数

    Args:
        provider_name: 提供商名称（deepseek/mimo）
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称

    Returns:
        BaseLLMProvider: LLM 提供商实例

    Raises:
        ValueError: 不支持的提供商名称
    """
    provider_name = provider_name.lower()

    if provider_name not in PROVIDER_REGISTRY:
        supported = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(f"不支持的 LLM 提供商: '{provider_name}'，支持的提供商: {supported}")

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(api_key=api_key, base_url=base_url, model=model)


def register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """
    注册新的 LLM 提供商（扩展点）

    使用方法：
        from providers import register_provider
        from providers.claude import ClaudeProvider
        register_provider("claude", ClaudeProvider)

    Args:
        name: 提供商名称
        provider_class: 提供商实现类
    """
    PROVIDER_REGISTRY[name.lower()] = provider_class
