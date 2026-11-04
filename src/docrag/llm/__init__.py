"""LLM 接入层（延迟导入）。"""

from docrag.llm.base import LLMClient

__all__ = ["LLMClient", "OpenAICompatLLM", "create_llm"]


def create_llm(settings):
    from docrag.llm.openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(settings)


def __getattr__(name: str):
    if name == "OpenAICompatLLM":
        from docrag.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM
    raise AttributeError(name)
