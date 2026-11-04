"""配置中心。

所有配置只有一个来源优先级：环境变量 > .env > 代码默认值。
API Key 不落任何代码文件，.env 已在 .gitignore 中。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["deepseek", "mimo"]


class ProviderProfile:
    """各提供商的默认接入参数（协议都是 OpenAI 兼容）。"""

    __slots__ = ("base_url", "model", "supports_thinking", "max_output_field")

    def __init__(self, base_url: str, model: str, supports_thinking: bool, max_output_field: str):
        self.base_url = base_url
        self.model = model
        self.supports_thinking = supports_thinking
        # DeepSeek 用 max_tokens，MiMo 用 max_completion_tokens
        self.max_output_field = max_output_field


# 2026-11 官方在售模型，旧名 deepseek-chat / mimo-v2-pro 均已下线
PROFILES: dict[str, ProviderProfile] = {
    "deepseek": ProviderProfile(
        base_url="https://api.deepseek.com",
        model="deepseek-flash",          # DeepSeek-V4.1-Flash-1017
        supports_thinking=True,
        max_output_field="max_tokens",
    ),
    "mimo": ProviderProfile(
        # TokenPlan 套餐地址；按量付费用 https://api.xiaomimimo.com/v1
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.6-pro",
        supports_thinking=True,
        max_output_field="max_completion_tokens",
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- LLM ----------
    llm_provider: ProviderName = "deepseek"
    deepseek_api_key: str | None = None
    mimo_api_key: str | None = None

    # 按提供商分别覆盖，留空则用 PROFILES 默认值
    deepseek_base_url: str | None = None
    deepseek_model: str | None = None
    mimo_base_url: str | None = None
    mimo_model: str | None = None

    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    # 思考模式：off 最快，适合 RAG 问答；complex 时可开 on
    llm_thinking: Literal["off", "on", "auto"] = "off"
    llm_reasoning_effort: Literal["low", "medium", "high", "max"] = "low"
    llm_max_output_tokens: int | None = Field(default=None, ge=64)

    # ---------- Embedding ----------
    # 实测：这份 24 万字规范上 bge-small 的 top-1 命中不输 Qwen3，但快 40 倍
    # 长文档/多语言场景可换 Qwen/Qwen3-Embedding-0.6B（CPU 上较慢）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_batch_size: int = Field(default=8, ge=1, le=128)
    embedding_dimensions: int | None = Field(default=None, ge=64)  # MRL 降维，None 用模型默认

    # ---------- 切分 ----------
    chunk_size: int = Field(default=400, ge=100, le=4000)
    chunk_overlap: int = Field(default=60, ge=0)
    max_chunk_chars: int = Field(default=500, ge=200)  # 表格/代码块允许的上限

    # ---------- 检索 ----------
    retrieve_top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=5, le=200)      # 每路召回候选数
    dense_weight: float = Field(default=1.0, gt=0)
    bm25_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=1)
    rewrite_enabled: bool = True                            # 查询重写（多一次 LLM 调用）
    expand_neighbors: bool = True                           # 长表格分片时把相邻块一起带上

    # ---------- 重排（可选） ----------
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---------- 路径 ----------
    data_dir: Path = Path("./data")
    storage_dir: Path = Path("./.docrag")       # 向量库 + manifest
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")

    # ---------- 联网 ----------
    model_source: Literal["modelscope", "huggingface"] = "modelscope"

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """把相对路径固定到启动目录，避免换工作目录后索引"消失"。"""
        base = Path.cwd()
        for name in ("data_dir", "storage_dir", "model_dir", "log_dir"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, (base / value).resolve())
        return self

    @property
    def profile(self) -> ProviderProfile:
        return PROFILES[self.llm_provider]

    @property
    def api_key(self) -> str:
        key = self.deepseek_api_key if self.llm_provider == "deepseek" else self.mimo_api_key
        if not key:
            env = "DEEPSEEK_API_KEY" if self.llm_provider == "deepseek" else "MIMO_API_KEY"
            raise RuntimeError(f"未配置 {env}，请在 .env 或环境变量中设置")
        return key

    @property
    def base_url(self) -> str:
        override = self.deepseek_base_url if self.llm_provider == "deepseek" else self.mimo_base_url
        return override or self.profile.base_url

    @property
    def model(self) -> str:
        override = self.deepseek_model if self.llm_provider == "deepseek" else self.mimo_model
        return override or self.profile.model

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.storage_dir, self.model_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """测试用：清除缓存，让下次读取重新加载 .env。"""
    global _settings
    _settings = None
